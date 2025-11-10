//===-- GCNPreRAOptimizations.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass combines split register tuple initialization into a single pseudo:
///
///   undef %0.sub1:sreg_64 = S_MOV_B32 1
///   %0.sub0:sreg_64 = S_MOV_B32 2
/// =>
///   %0:sreg_64 = S_MOV_B64_IMM_PSEUDO 0x200000001
///
/// This is to allow rematerialization of a value instead of spilling. It is
/// supposed to be done after register coalescer to allow it to do its job and
/// before actual register allocation to allow rematerialization.
///
/// Right now the pass only handles 64 bit SGPRs with immediate initializers,
/// although the same shall be possible with other register classes and
/// instructions if necessary.
///
/// This pass also adds register allocation hints to COPY.
/// The hints will be post-processed by SIRegisterInfo::getRegAllocationHints.
/// When using True16, we often see COPY moving a 16-bit value between a VGPR_32
/// and a VGPR_16. If we use the VGPR_16 that corresponds to the lo16 bits of
/// the VGPR_32, the COPY can be completely eliminated.
///
//===----------------------------------------------------------------------===//

#include "GCNPreRAOptimizations.h"
#include "AMDGPU.h"
#include "GCNHazardRecognizer.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-pre-ra-optimizations"

static cl::opt<bool>
    EnableAntiHintsForMFMARegs("amdgpu-anti-hints-for-mfma", cl::Hidden,
                               cl::desc("Enable Anti-Hints for "
                                        "MFMA in GCNPreRAOptimizations stage."),
                               cl::init(true));

static cl::opt<bool>
    EnableAntiHintsForVAVDST("amdgpu-anti-hints-for-va-vdst", cl::Hidden,
                             cl::init(true));

static cl::opt<bool>
    EnableAntiHintsForAddr("amdgpu-anti-hints-for-addr", cl::Hidden,
                             cl::init(true));

static cl::opt<unsigned> VAVDSTLookbackWindow(
    "amdgpu-va-vdst-lookback-window", cl::Hidden,
    cl::desc("Lookback window for VA_VDST anti-hints"), cl::init(32));

static cl::opt<unsigned> AddrLookbackWindow(
    "amdgpu-addr-lookback-window", cl::Hidden,
    cl::desc("Lookback window for VA_VDST anti-hints"), cl::init(200));

static cl::opt<bool>
    InsertPreftechInstructions("amdgpu-inst-prefetch-64kb", cl::Hidden,
                             cl::init(false));

namespace {

class GCNPreRAOptimizationsImpl {
private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;
  std::unique_ptr<GCNHazardRecognizer> HazardRec;

  bool processReg(Register Reg);

public:
  GCNPreRAOptimizationsImpl(LiveIntervals *LS) : LIS(LS) {}
  bool run(MachineFunction &MF);
};

class GCNPreRAOptimizationsLegacy : public MachineFunctionPass {
public:
  static char ID;

  GCNPreRAOptimizationsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Pre-RA optimizations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                      "AMDGPU Pre-RA optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                    "Pre-RA optimizations", false, false)

char GCNPreRAOptimizationsLegacy::ID = 0;

char &llvm::GCNPreRAOptimizationsID = GCNPreRAOptimizationsLegacy::ID;

FunctionPass *llvm::createGCNPreRAOptimizationsLegacyPass() {
  return new GCNPreRAOptimizationsLegacy();
}

bool GCNPreRAOptimizationsImpl::processReg(Register Reg) {
  MachineInstr *Def0 = nullptr;
  MachineInstr *Def1 = nullptr;
  uint64_t Init = 0;
  bool Changed = false;
  SmallSet<Register, 32> ModifiedRegs;
  bool IsAGPRDst = TRI->isAGPRClass(MRI->getRegClass(Reg));

  for (MachineInstr &I : MRI->def_instructions(Reg)) {
    switch (I.getOpcode()) {
    default:
      return false;
    case AMDGPU::V_ACCVGPR_WRITE_B32_e64:
      break;
    case AMDGPU::COPY: {
      // Some subtargets cannot do an AGPR to AGPR copy directly, and need an
      // intermdiate temporary VGPR register. Try to find the defining
      // accvgpr_write to avoid temporary registers.

      if (!IsAGPRDst)
        return false;

      Register SrcReg = I.getOperand(1).getReg();

      if (!SrcReg.isVirtual())
        break;

      // Check if source of copy is from another AGPR.
      bool IsAGPRSrc = TRI->isAGPRClass(MRI->getRegClass(SrcReg));
      if (!IsAGPRSrc)
        break;

      // def_instructions() does not look at subregs so it may give us a
      // different instruction that defines the same vreg but different subreg
      // so we have to manually check subreg.
      Register SrcSubReg = I.getOperand(1).getSubReg();
      for (auto &Def : MRI->def_instructions(SrcReg)) {
        if (SrcSubReg != Def.getOperand(0).getSubReg())
          continue;

        if (Def.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32_e64) {
          const MachineOperand &DefSrcMO = Def.getOperand(1);

          // Immediates are not an issue and can be propagated in
          // postrapseudos pass. Only handle cases where defining
          // accvgpr_write source is a vreg.
          if (DefSrcMO.isReg() && DefSrcMO.getReg().isVirtual()) {
            // Propagate source reg of accvgpr write to this copy instruction
            I.getOperand(1).setReg(DefSrcMO.getReg());
            I.getOperand(1).setSubReg(DefSrcMO.getSubReg());

            // Reg uses were changed, collect unique set of registers to update
            // live intervals at the end.
            ModifiedRegs.insert(DefSrcMO.getReg());
            ModifiedRegs.insert(SrcReg);

            Changed = true;
          }

          // Found the defining accvgpr_write, stop looking any further.
          break;
        }
      }
      break;
    }
    case AMDGPU::S_MOV_B32:
      if (I.getOperand(0).getReg() != Reg || !I.getOperand(1).isImm() ||
          I.getNumOperands() != 2)
        return false;

      switch (I.getOperand(0).getSubReg()) {
      default:
        return false;
      case AMDGPU::sub0:
        if (Def0)
          return false;
        Def0 = &I;
        Init |= Lo_32(I.getOperand(1).getImm());
        break;
      case AMDGPU::sub1:
        if (Def1)
          return false;
        Def1 = &I;
        Init |= static_cast<uint64_t>(I.getOperand(1).getImm()) << 32;
        break;
      }
      break;
    }
  }

  // For AGPR reg, check if live intervals need to be updated.
  if (IsAGPRDst) {
    if (Changed) {
      for (Register RegToUpdate : ModifiedRegs) {
        LIS->removeInterval(RegToUpdate);
        LIS->createAndComputeVirtRegInterval(RegToUpdate);
      }
    }

    return Changed;
  }

  // For SGPR reg, check if we can combine instructions.
  if (!Def0 || !Def1 || Def0->getParent() != Def1->getParent())
    return Changed;

  LLVM_DEBUG(dbgs() << "Combining:\n  " << *Def0 << "  " << *Def1
                    << "    =>\n");

  if (SlotIndex::isEarlierInstr(LIS->getInstructionIndex(*Def1),
                                LIS->getInstructionIndex(*Def0)))
    std::swap(Def0, Def1);

  LIS->RemoveMachineInstrFromMaps(*Def0);
  LIS->RemoveMachineInstrFromMaps(*Def1);
  auto NewI = BuildMI(*Def0->getParent(), *Def0, Def0->getDebugLoc(),
                      TII->get(AMDGPU::S_MOV_B64_IMM_PSEUDO), Reg)
                  .addImm(Init);

  Def0->eraseFromParent();
  Def1->eraseFromParent();
  LIS->InsertMachineInstrInMaps(*NewI);
  LIS->removeInterval(Reg);
  LIS->createAndComputeVirtRegInterval(Reg);

  LLVM_DEBUG(dbgs() << "  " << *NewI);

  return true;
}

bool GCNPreRAOptimizationsLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return GCNPreRAOptimizationsImpl(LIS).run(MF);
}

PreservedAnalyses
GCNPreRAOptimizationsPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  GCNPreRAOptimizationsImpl(LIS).run(MF);
  return PreservedAnalyses::all();
}

bool GCNPreRAOptimizationsImpl::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = ST.getRegisterInfo();
  HazardRec = std::make_unique<GCNHazardRecognizer>(
      MF, GCNHazardRecognizer::OperatingMode::PostRA);

  bool Changed = false;
  bool Added = false;


  for (auto &MBB : MF) {
    if (MBB.isEntryBlock() && !Added && InsertPreftechInstructions) {
      Added=true;

      auto buildPrefetch = [&MBB, this](unsigned offset) {
        MachineInstrBuilder Prefetch =
            BuildMI(MBB, MBB.getFirstNonPHI(),MBB.getFirstNonPHI()->getDebugLoc(), TII->get(AMDGPU::S_PREFETCH_INST_PC_REL));
        Prefetch.addImm(offset);
        Prefetch.addReg(AMDGPU::SGPR_NULL);
        Prefetch.addImm(31);
      };

      buildPrefetch(57280);
      buildPrefetch(53184);
      buildPrefetch(49088);
      buildPrefetch(45024);
      buildPrefetch(40928);



      for (auto I = 0; I < 10; I++) {
        MachineInstrBuilder Prefetch =
            BuildMI(MBB, MBB.getFirstNonPHI(),MBB.getFirstNonPHI()->getDebugLoc(), TII->get(AMDGPU::S_PREFETCH_INST_PC_REL));

        Prefetch.addImm(4096*(9-I));
        Prefetch.addReg(AMDGPU::SGPR_NULL);
        Prefetch.addImm(31);
      }
      Changed = true;
      break;
    }
  }

  // Add RA anti-hints to reduce MFMA hazard NOPs
  if (EnableAntiHintsForMFMARegs) {
    // Max lookback window for RAW or WAW hazard
    constexpr unsigned MaxLookbackWindow = 2;
    for (const MachineBasicBlock &MBB : MF) {
      SmallVector<SmallVector<Register, 4>, 16> RecentMFMAs;
      SmallVector<SmallVector<Register, 4>, 16> RecentEXPs;
      unsigned InstrsSinceMFMA = 0;
      unsigned InstrsSinceEXP = 0;
      unsigned VALUsSinceMFMA = 0;
      unsigned VALUsSinceEXP = 0;
      unsigned WMMANeededVALU = 0;
      unsigned WMMANeededInstr = 0;
      for (const MachineInstr &MI : MBB) {
        if (MI.isDebugInstr())
          continue;

        if (SIInstrInfo::isVALU(MI) && !SIInstrInfo::isMFMAorWMMA(MI) && SIInstrInfo::isTRANS(MI)) {
          ++VALUsSinceMFMA;
          ++VALUsSinceEXP;
        }

        ++InstrsSinceMFMA;
        ++InstrsSinceEXP;
        // Handle MFMA instructions
        if (SIInstrInfo::isMFMAorWMMA(MI)) {
          ++VALUsSinceEXP;
          VALUsSinceMFMA = 0;
          InstrsSinceMFMA = 0;
          SmallVector<GCNHazardRecognizer::WMMASlotType, 8> WMMAPipeline;
          HazardRec->getWMMASlots(MI, WMMAPipeline);

          if (WMMAPipeline.size()) {
            WMMANeededInstr =
                std::max(WMMANeededInstr, (unsigned)WMMAPipeline.size());

            unsigned VALUSlots = 0;

            for (auto Slot : WMMAPipeline) {
              if (Slot == GCNHazardRecognizer::WMMASlotType::ValuCoExec0 ||
                  Slot == GCNHazardRecognizer::WMMASlotType::ValuCoExec1 ||
                  Slot == GCNHazardRecognizer::WMMASlotType::ValuCoExec2 ||
                  Slot ==
                      GCNHazardRecognizer::WMMASlotType::ValuCoExecLdScale ||
                  Slot == GCNHazardRecognizer::WMMASlotType::
                              ValuCoexecLastLdScale) {
                ++VALUSlots;
              }
              if (Slot == GCNHazardRecognizer::WMMASlotType::Execute) {
                --WMMANeededInstr;
              }
            }

            WMMANeededVALU = std::max(WMMANeededVALU, VALUSlots);
          }

          if (WMMAPipeline.empty()) {
            WMMANeededInstr = 16;
            WMMANeededVALU = 16;
          }

          SmallVector<Register, 4> MFMARegisters;
          // Helper to get named operand
          auto collectNamedOperand = [&](AMDGPU::OpName OpName,
                                         const char *OpNameStr) {
            const MachineOperand *MO = TII->getNamedOperand(MI, OpName);
            if (!MO) {
              LLVM_DEBUG(dbgs() << "    Named operand " << OpNameStr
                                << " not found\n");
              return;
            }
            if (MO->isReg() && MO->getReg().isVirtual()) {
              Register Reg = MO->getReg();
              const TargetRegisterClass *RC = MRI->getRegClass(Reg);
              // Only consider VGPRs
              if (TRI->hasVGPRs(RC))
                MFMARegisters.push_back(Reg);
              LLVM_DEBUG(dbgs() << "    Collected " << OpNameStr << " : "
                                << printReg(Reg, TRI) << "\n");
            }
          };

          // Collect destination and source C registers
          // collectNamedOperand(AMDGPU::OpName::vdst, "vdst"); // Destination
          collectNamedOperand(AMDGPU::OpName::src0, "src0");
          collectNamedOperand(AMDGPU::OpName::src1, "src1");
          // collectNamedOperand(AMDGPU::OpName::src2,
          //                     "src2"); // Matrix C (accumulator)
          if (!MFMARegisters.empty()) {
            RecentMFMAs.emplace_back(std::move(MFMARegisters));
            // Maintain window
            if (RecentMFMAs.size() > MaxLookbackWindow)
              RecentMFMAs.erase(RecentMFMAs.begin());
          }
          continue;
        }

        // Handle EXP instructions
        if (SIInstrInfo::isTRANS(MI)) {
          VALUsSinceEXP = 0;
          ++VALUsSinceMFMA;
          InstrsSinceEXP = 0;
          SmallVector<Register, 4> TransRegisters;
          // Helper to get named operand
          auto collectNamedOperand = [&](AMDGPU::OpName OpName,
                                         const char *OpNameStr) {
            const MachineOperand *MO = &MI.getOperand(1);
            if (!MO) {
              LLVM_DEBUG(dbgs() << "    Named operand " << OpNameStr
                                << " not found\n");
              return;
            }
            if (MO->isReg() && MO->getReg().isVirtual()) {
              Register Reg = MO->getReg();
              const TargetRegisterClass *RC = MRI->getRegClass(Reg);
              // Only consider VGPRs
              if (TRI->hasVGPRs(RC))
                TransRegisters.push_back(Reg);
              LLVM_DEBUG(dbgs() << "    Collected " << OpNameStr << " : "
                                << printReg(Reg, TRI) << "\n");
            }
          };

          // Collect destination and source C registers
          collectNamedOperand(AMDGPU::OpName::vdst, "vdst"); // Destination
          if (!TransRegisters.empty()) {
            RecentEXPs.emplace_back(std::move(TransRegisters));
            // Maintain window
            if (RecentEXPs.size() > 1)
              RecentEXPs.erase(RecentEXPs.begin());
          }
          continue;
        }

        bool ShouldCheckReuse = MI.mayLoad() || MI.mayStore() || MI.isCopy() ||
                                SIInstrInfo::isVALU(MI);
        // Skip non-relevant instructions, or skip until at least one MFMA is
        // encountered
        if (!ShouldCheckReuse || (RecentMFMAs.empty() && RecentEXPs.empty()))
          continue;

        // Process operands that might reuse MFMA registers

        for (const MachineOperand &MO : MI.operands()) {
          if (!MO.isReg() || !MO.getReg().isVirtual())
            continue;

          const Register CandidateReg = MO.getReg();
          const TargetRegisterClass *CandidateRC =
              MRI->getRegClass(CandidateReg);

          // Only process VGPR registers
          if (!TRI->isVGPRClass(CandidateRC))
            continue;

          if (!SIInstrInfo::isMFMAorWMMA(MI) &&
              VALUsSinceMFMA < (WMMANeededVALU + 1) &&
              InstrsSinceMFMA < (WMMANeededInstr + 1)) {
            for (auto It = RecentMFMAs.rbegin(); It != RecentMFMAs.rend();
                 ++It) {
              const SmallVector<Register, 4> &MFMARegs = *It;
              for (Register MFMAReg : MFMARegs) {
                if (MFMAReg == CandidateReg)
                  continue;
                // Check if MFMA register is dead at current instruction
                const LiveInterval &MFMAInterval = LIS->getInterval(MFMAReg);
                const SlotIndex CurrentSlot =
                    LIS->getInstructionIndex(MI).getRegSlot();
                if (!MFMAInterval.liveAt(CurrentSlot)) {
                  // Add bi-directional anti-hints
                  MRI->addRegAllocationAntiHints(CandidateReg, MFMAReg);
                  MRI->addRegAllocationAntiHints(MFMAReg, CandidateReg);
                }
              }
            }
          }

          if (!MO.isDef())
            continue;
          if (!SIInstrInfo::isTRANS(MI) && VALUsSinceEXP < 2) {
            for (auto It = RecentEXPs.rbegin(); It != RecentEXPs.rend(); ++It) {
              const SmallVector<Register, 4> &EXPRegs = *It;
              for (Register EXPReg : EXPRegs) {
                // Check if MFMA register is dead at current instruction
                const LiveInterval &EXPInterval = LIS->getInterval(EXPReg);
                const SlotIndex CurrentSlot =
                    LIS->getInstructionIndex(MI).getRegSlot();
                if (!EXPInterval.liveAt(CurrentSlot)) {
                  // Add bi-directional anti-hints
                  MRI->addRegAllocationAntiHints(CandidateReg, EXPReg);
                  MRI->addRegAllocationAntiHints(EXPReg, CandidateReg);
                }
              }
            }
          }
        }
      }
    }
  }

  // Add anti-hints to reduce VA_VDST hazards between VALU sources and
  // DS_LOAD.
  if (EnableAntiHintsForVAVDST && ST.getGeneration() >= AMDGPUSubtarget::GFX12) {
    const unsigned LookbackWindow = VAVDSTLookbackWindow;

    for (const MachineBasicBlock &MBB : MF) {
      SmallVector<Register, 64> RecentVALUSrcs;

      for (const MachineInstr &MI : MBB) {
        if (MI.isDebugInstr())
          continue;

        unsigned Opc = MI.getOpcode();

        if (TII->isVALU(MI)) {
          for (const MachineOperand &MO : MI.uses()) {
            if (!MO.isReg() || !MO.getReg().isVirtual())
              continue;
            Register Reg = MO.getReg();
            const TargetRegisterClass *RC = MRI->getRegClass(Reg);
            if (TRI->hasVGPRs(RC)) {
              if (!llvm::is_contained(RecentVALUSrcs, Reg)) {
                RecentVALUSrcs.push_back(Reg);
                if (RecentVALUSrcs.size() > LookbackWindow)
                  RecentVALUSrcs.erase(RecentVALUSrcs.begin());
              }
            }
          }
        }

        if (Opc == AMDGPU::V_CVT_SCALEF32_PK8_FP8_F32_e64 ||
            Opc == AMDGPU::V_CVT_SCALEF32_PK8_BF8_F32_e64 ||
            TII->isWMMA(MI)) {
          for (const MachineOperand &MO : MI.uses()) {
            if (!MO.isReg() || !MO.getReg().isVirtual())
              continue;
            Register Reg = MO.getReg();
            const TargetRegisterClass *RC = MRI->getRegClass(Reg);
            if (TRI->hasVGPRs(RC) && TRI->getRegSizeInBits(*RC) >= 64) {
              if (!llvm::is_contained(RecentVALUSrcs, Reg)) {
                RecentVALUSrcs.push_back(Reg);
                if (RecentVALUSrcs.size() > LookbackWindow)
                  RecentVALUSrcs.erase(RecentVALUSrcs.begin());
              }
            }
          }
          continue;
        }

        if (Opc == AMDGPU::V_PK_ADD_F32 || Opc == AMDGPU::V_PK_MUL_F32 ||
            Opc == AMDGPU::V_MAXIMUM3_F32_e64) {
          for (const MachineOperand &MO : MI.uses()) {
            if (!MO.isReg() || !MO.getReg().isVirtual())
              continue;
            Register Reg = MO.getReg();
            const TargetRegisterClass *RC = MRI->getRegClass(Reg);
            if (TRI->hasVGPRs(RC) && TRI->getRegSizeInBits(*RC) <= 64) {
              if (!llvm::is_contained(RecentVALUSrcs, Reg)) {
                RecentVALUSrcs.push_back(Reg);
                if (RecentVALUSrcs.size() > LookbackWindow)
                  RecentVALUSrcs.erase(RecentVALUSrcs.begin());
              }
            }
          }
          continue;
        }

        if (TII->isDS(MI) && MI.mayLoad()) {
          if (RecentVALUSrcs.empty())
            continue;

          for (const MachineOperand &MO : MI.defs()) {
            if (!MO.isReg() || !MO.getReg().isVirtual())
              continue;
            Register DSDestReg = MO.getReg();
            const TargetRegisterClass *RC = MRI->getRegClass(DSDestReg);
            if (!TRI->hasVGPRs(RC))
              continue;

            for (Register VALUSrcReg : RecentVALUSrcs) {
              if (VALUSrcReg == DSDestReg)
                continue;
              MRI->addRegAllocationAntiHints(DSDestReg, VALUSrcReg);
              MRI->addRegAllocationAntiHints(VALUSrcReg, DSDestReg);
            }
          }
        }
      }
    }
  }


  // Add anti-hints to wait_xcnt instructions between VMEM instrs with same addr.
  if (EnableAntiHintsForAddr && ST.getGeneration() >= AMDGPUSubtarget::GFX12) {
    const unsigned LookbackWindow = VAVDSTLookbackWindow;

    for (const MachineBasicBlock &MBB : MF) {
      SmallVector<Register, 64> RecentMemInstrs;

      for (const MachineInstr &MI : MBB) {
        if (MI.isDebugInstr())
          continue;

        unsigned Opc = MI.getOpcode();

        if (TII->isVMEM(MI)) {
          if (auto Op = TII->getNamedOperand(MI, AMDGPU::OpName::vdata)) {
            if (Op->isReg() && Op->getReg().isVirtual()) {
              Register Reg = Op->getReg();
              const TargetRegisterClass *RC = MRI->getRegClass(Reg);
              for (Register PrevAddr : RecentMemInstrs) {
                if (PrevAddr == Reg)
                  continue;
                
                MRI->addRegAllocationAntiHints(PrevAddr, Reg);
                MRI->addRegAllocationAntiHints(Reg, PrevAddr);
              }

              RecentMemInstrs.push_back(Reg);
              if (RecentMemInstrs.size() > LookbackWindow)
                RecentMemInstrs.erase(RecentMemInstrs.begin());
            }
          }
        }

        else if (false) {
          for (const MachineOperand &MO : MI.defs()) {
            if (!MO.isReg() || !MO.getReg().isVirtual())
              continue;
            Register VGPRDestReg = MO.getReg();
            const TargetRegisterClass *RC = MRI->getRegClass(VGPRDestReg);
            if (!TRI->hasVGPRs(RC))
              continue;

            for (Register PrevAddr : RecentMemInstrs) {
              if (PrevAddr == VGPRDestReg)
                continue;
              MRI->addRegAllocationAntiHints(VGPRDestReg, PrevAddr);
              MRI->addRegAllocationAntiHints(PrevAddr, VGPRDestReg);
            }
          }

        }


      }
    }
  }


  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (!LIS->hasInterval(Reg))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    if ((RC->MC->getSizeInBits() != 64 || !TRI->isSGPRClass(RC)) &&
        (ST.hasGFX90AInsts() || !TRI->isAGPRClass(RC)))
      continue;

    Changed |= processReg(Reg);
  }

  if (!ST.useRealTrue16Insts())
    return Changed;

  // Add RA hints to improve True16 COPY elimination.
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (MI.getOpcode() != AMDGPU::COPY)
        continue;
      Register Dst = MI.getOperand(0).getReg();
      Register Src = MI.getOperand(1).getReg();
      const TargetRegisterClass *DstRC = TRI->getRegClassForReg(*MRI, Dst);
      bool IsDst16Bit = AMDGPU::VGPR_16RegClass.hasSubClassEq(DstRC);
      if (Dst.isVirtual() && IsDst16Bit && Src.isPhysical() &&
          TRI->getRegClassForReg(*MRI, Src) == &AMDGPU::VGPR_32RegClass)
        MRI->setRegAllocationHint(Dst, 0, TRI->getSubReg(Src, AMDGPU::lo16));
      if (Src.isVirtual() &&
          MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass &&
          Dst.isPhysical() && DstRC == &AMDGPU::VGPR_32RegClass)
        MRI->setRegAllocationHint(Src, 0, TRI->getSubReg(Dst, AMDGPU::lo16));
      if (!Dst.isVirtual() || !Src.isVirtual())
        continue;
      if (MRI->getRegClass(Dst) == &AMDGPU::VGPR_32RegClass &&
          MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass) {
        MRI->setRegAllocationHint(Dst, AMDGPURI::Size32, Src);
        MRI->setRegAllocationHint(Src, AMDGPURI::Size16, Dst);
      }
      if (IsDst16Bit && MRI->getRegClass(Src) == &AMDGPU::VGPR_32RegClass)
        MRI->setRegAllocationHint(Dst, AMDGPURI::Size16, Src);
    }
  }

  return Changed;
}

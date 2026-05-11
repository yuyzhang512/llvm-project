//===-- SITensorDescSink.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Sink loop-invariant tensor descriptor component definitions from the loop
// preheader into the loop body. After MachineLICM and register coalescing,
// descriptor registers for tensor_load_to_lds have some sub-register components
// defined in the preheader (invariant) and some in the loop body (variant).
// The entire descriptor register lives across the loop, increasing SGPR
// pressure. By creating a new descriptor register in the loop body and
// rebuilding all components there, the descriptor becomes loop-local, trading
// cheap redundant scalar ops for reduced register pressure.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "si-tensor-desc-sink"

static cl::opt<bool> EnableTensorDescSink(
    "amdgpu-tensor-desc-sink", cl::init(true), cl::Hidden,
    cl::desc("Sink loop-invariant tensor descriptor components into loop body"));

namespace {

class SITensorDescSink : public MachineFunctionPass {
public:
  static char ID;

  SITensorDescSink() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI Tensor Descriptor Sink";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const SIInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const MachineLoopInfo *MLI = nullptr;

  bool processLoop(MachineLoop *L);
  bool sinkDescriptor(Register DescReg, MachineLoop *L);
  bool mergeDescriptorCopies(MachineLoop *L);

  void cloneDepsOutsideLoop(MachineInstr *MI, MachineBasicBlock *MBB,
                            MachineBasicBlock::iterator InsertPt,
                            MachineLoop *L,
                            DenseMap<Register, Register> &ClonedRegs);
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(SITensorDescSink, DEBUG_TYPE,
                      "SI Tensor Descriptor Sink", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(SITensorDescSink, DEBUG_TYPE,
                    "SI Tensor Descriptor Sink", false, false)

char SITensorDescSink::ID = 0;

char &llvm::SITensorDescSinkID = SITensorDescSink::ID;

FunctionPass *llvm::createSITensorDescSinkPass() {
  return new SITensorDescSink();
}

static bool isTensorLoadToLDS(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case AMDGPU::TENSOR_LOAD_TO_LDS_d2:
  case AMDGPU::TENSOR_LOAD_TO_LDS_d4:
    return true;
  default:
    return false;
  }
}

void SITensorDescSink::cloneDepsOutsideLoop(
    MachineInstr *MI, MachineBasicBlock *MBB,
    MachineBasicBlock::iterator InsertPt, MachineLoop *L,
    DenseMap<Register, Register> &ClonedRegs) {

  for (const MachineOperand &MO : MI->operands()) {
    if (!MO.isReg() || !MO.isUse() || !MO.getReg().isVirtual())
      continue;
    Register SrcReg = MO.getReg();
    if (ClonedRegs.count(SrcReg))
      continue;
    if (!MRI->hasOneDef(SrcReg))
      continue;
    MachineInstr *SrcDef = MRI->getVRegDef(SrcReg);
    if (!SrcDef || L->contains(SrcDef->getParent()))
      continue;
    if (SrcDef->mayLoad() || SrcDef->mayStore() || SrcDef->isCall() ||
        SrcDef->hasUnmodeledSideEffects())
      continue;

    cloneDepsOutsideLoop(SrcDef, MBB, InsertPt, L, ClonedRegs);

    MachineInstr *Clone = MBB->getParent()->CloneMachineInstr(SrcDef);
    Register OldDef = SrcDef->getOperand(0).getReg();
    const TargetRegisterClass *RC = MRI->getRegClass(OldDef);
    Register NewSrcReg = MRI->createVirtualRegister(RC);

    for (unsigned i = 0; i < Clone->getNumOperands(); ++i) {
      MachineOperand &CO = Clone->getOperand(i);
      if (CO.isReg() && CO.isDef() && CO.getReg() == OldDef) {
        CO.setReg(NewSrcReg);
        break;
      }
    }
    for (unsigned i = 0; i < Clone->getNumOperands(); ++i) {
      MachineOperand &CO = Clone->getOperand(i);
      if (!CO.isReg() || !CO.isUse() || !CO.getReg().isVirtual())
        continue;
      auto It = ClonedRegs.find(CO.getReg());
      if (It != ClonedRegs.end())
        CO.setReg(It->second);
    }

    MBB->insert(InsertPt, Clone);
    ClonedRegs[OldDef] = NewSrcReg;
  }
}

bool SITensorDescSink::sinkDescriptor(Register DescReg, MachineLoop *L) {
  const TargetRegisterClass *RC = MRI->getRegClass(DescReg);

  // Collect all def instructions of this register
  SmallVector<MachineInstr *, 12> InvariantDefs;
  SmallVector<MachineInstr *, 12> VariantDefs;

  for (MachineInstr &MI : MRI->def_instructions(DescReg)) {
    if (L->contains(MI.getParent()))
      VariantDefs.push_back(&MI);
    else
      InvariantDefs.push_back(&MI);
  }

  if (InvariantDefs.empty())
    return false;

  // If all defs are invariant (fully loop-invariant descriptor), only sink
  // if the invariant components don't all have the same value. A uniform
  // descriptor is cheap to rematerialize and not worth sinking.
  if (VariantDefs.empty()) {
    bool AllSameValue = true;
    MachineInstr *FirstDef = InvariantDefs[0];
    for (unsigned i = 1; i < InvariantDefs.size(); ++i) {
      MachineInstr *Def = InvariantDefs[i];
      if (Def->getOpcode() != FirstDef->getOpcode() ||
          !FirstDef->isIdenticalTo(*Def, MachineInstr::IgnoreDefs)) {
        AllSameValue = false;
        break;
      }
    }
    if (AllSameValue)
      return false;
  }

  // Sort invariant defs in instruction order within their basic blocks.
  // Group by BB, then sort by position within each BB.
  // Since they're all in the preheader (or dominating blocks), we need
  // to maintain their relative order.
  // Use a stable sort based on BB ordering and position within BB.
  DenseMap<MachineInstr *, unsigned> InstrOrder;
  unsigned Idx = 0;
  if (MachineBasicBlock *Preheader = L->getLoopPreheader()) {
    for (MachineInstr &MI : *Preheader)
      InstrOrder[&MI] = Idx++;
  }
  // Also index any other blocks containing invariant defs
  for (MachineInstr *MI : InvariantDefs) {
    MachineBasicBlock *MBB = MI->getParent();
    if (InstrOrder.count(&*MBB->begin()))
      continue;
    for (MachineInstr &I : *MBB)
      InstrOrder[&I] = Idx++;
  }

  llvm::sort(InvariantDefs, [&](MachineInstr *A, MachineInstr *B) {
    return InstrOrder.lookup(A) < InstrOrder.lookup(B);
  });

  Register NewReg = MRI->createVirtualRegister(RC);

  LLVM_DEBUG(dbgs() << "  Sinking " << printReg(DescReg)
                    << " -> " << printReg(NewReg)
                    << " (" << InvariantDefs.size() << " invariant, "
                    << VariantDefs.size() << " variant defs)\n");

  // Find insertion point: first variant def if any, otherwise first use
  // of DescReg in the loop body.
  MachineInstr *InsertBeforeMI = nullptr;

  if (!VariantDefs.empty()) {
    InsertBeforeMI = VariantDefs[0];
    for (unsigned i = 1; i < VariantDefs.size(); ++i) {
      MachineInstr *MI = VariantDefs[i];
      if (MI->getParent() != InsertBeforeMI->getParent())
        continue;
      for (const MachineInstr &I : *MI->getParent()) {
        if (&I == MI) { InsertBeforeMI = MI; break; }
        if (&I == InsertBeforeMI) break;
      }
    }
  } else {
    // All defs are invariant — insert before the first use in the loop
    for (MachineInstr &MI : MRI->use_instructions(DescReg)) {
      if (!L->contains(MI.getParent()))
        continue;
      if (!InsertBeforeMI) {
        InsertBeforeMI = &MI;
        continue;
      }
      if (MI.getParent() == InsertBeforeMI->getParent()) {
        for (const MachineInstr &I : *MI.getParent()) {
          if (&I == &MI) { InsertBeforeMI = &MI; break; }
          if (&I == InsertBeforeMI) break;
        }
      }
    }
  }

  if (!InsertBeforeMI)
    return false;

  MachineBasicBlock *InsertBB = InsertBeforeMI->getParent();
  MachineBasicBlock::iterator InsertPt(InsertBeforeMI);

  DenseMap<Register, Register> ClonedRegs;

  // Clone invariant defs into the loop body, in their original order,
  // before the first variant def.
  SmallPtrSet<MachineInstr *, 4> PhysRegCopies;
  bool FirstDef = true;
  for (MachineInstr *MI : InvariantDefs) {
    // Check if this instruction uses any physical registers (except
    // DescReg itself). If so, we can't safely clone it — instead,
    // create a COPY from the original descriptor's subreg.
    bool HasPhysRegUse = false;
    for (const MachineOperand &MO : MI->operands()) {
      if (MO.isReg() && MO.isUse() && MO.getReg().isPhysical() &&
          MO.getReg() != AMDGPU::NoRegister)
        HasPhysRegUse = true;
    }

    // Find the subreg index of this def
    unsigned DefSubIdx = 0;
    for (const MachineOperand &MO : MI->operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg() == DescReg) {
        DefSubIdx = MO.getSubReg();
        break;
      }
    }

    if (HasPhysRegUse) {
      // Create COPY from original descriptor register's subreg
      auto MIB = BuildMI(*InsertBB, InsertPt, MI->getDebugLoc(),
                          TII->get(AMDGPU::COPY));
      MIB.addReg(NewReg, RegState::Define, DefSubIdx);
      MIB.addReg(DescReg, RegState(), DefSubIdx);
      if (FirstDef) {
        MIB->getOperand(0).setIsUndef(true);
        FirstDef = false;
      }
      PhysRegCopies.insert(&*MIB);
      continue;
    }

    cloneDepsOutsideLoop(MI, InsertBB, InsertPt, L, ClonedRegs);

    MachineInstr *Clone = InsertBB->getParent()->CloneMachineInstr(MI);

    // Replace def register and fix undef flags
    for (unsigned i = 0; i < Clone->getNumOperands(); ++i) {
      MachineOperand &MO = Clone->getOperand(i);
      if (MO.isReg() && MO.isDef() && MO.getReg() == DescReg) {
        MO.setReg(NewReg);
        if (FirstDef) {
          MO.setIsUndef(true);
          FirstDef = false;
        } else {
          MO.setIsUndef(false);
        }
        break;
      }
    }

    // Remap uses
    for (unsigned i = 0; i < Clone->getNumOperands(); ++i) {
      MachineOperand &MO = Clone->getOperand(i);
      if (!MO.isReg() || !MO.isUse())
        continue;
      if (MO.getReg() == DescReg) {
        MO.setReg(NewReg);
      } else if (MO.getReg().isVirtual()) {
        auto It = ClonedRegs.find(MO.getReg());
        if (It != ClonedRegs.end())
          MO.setReg(It->second);
      }
    }

    InsertBB->insert(InsertPt, Clone);
  }

  // Rewrite all uses and defs of DescReg within the loop to use NewReg
  SmallVector<MachineInstr *, 16> LoopMIs;
  for (MachineInstr &MI : MRI->reg_instructions(DescReg)) {
    if (L->contains(MI.getParent()))
      LoopMIs.push_back(&MI);
  }
  for (MachineInstr *MI : LoopMIs) {
    if (PhysRegCopies.count(MI))
      continue;
    for (unsigned i = 0; i < MI->getNumOperands(); ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg() && MO.getReg() == DescReg) {
        MO.setReg(NewReg);
        if (MO.isDef())
          MO.setIsUndef(false);
      }
    }
  }

  LLVM_DEBUG(dbgs() << "  Done\n");
  return true;
}

bool SITensorDescSink::mergeDescriptorCopies(MachineLoop *L) {
  bool Changed = false;

  // Collect descriptor registers used by tensor_load_to_lds in this loop.
  SmallSetVector<Register, 8> DescRegs;
  for (MachineBasicBlock *MBB : L->blocks()) {
    for (MachineInstr &MI : *MBB) {
      if (!isTensorLoadToLDS(MI))
        continue;
      for (unsigned OpNo = 0; OpNo < 2; ++OpNo) {
        MachineOperand &MO = MI.getOperand(OpNo);
        if (MO.isReg() && MO.getReg().isVirtual())
          DescRegs.insert(MO.getReg());
      }
    }
  }

  // Look for pattern: %B = COPY %A where both %A and %B are descriptor regs
  // used by tensor_load_to_lds. %B is defined by a full COPY of %A followed
  // by subreg overrides. We can eliminate %B by applying the subreg writes
  // directly to %A.
  for (Register BReg : DescRegs) {
    // Find the full-register COPY that defines BReg (no subreg on def).
    MachineInstr *CopyMI = nullptr;
    for (MachineInstr &MI : MRI->def_instructions(BReg)) {
      if (!L->contains(MI.getParent()))
        continue;
      if (MI.isCopy() && MI.getOperand(0).getReg() == BReg &&
          MI.getOperand(0).getSubReg() == 0 &&
          MI.getOperand(1).getSubReg() == 0) {
        CopyMI = &MI;
        break;
      }
    }
    if (!CopyMI)
      continue;

    Register AReg = CopyMI->getOperand(1).getReg();
    if (!AReg.isVirtual() || !DescRegs.contains(AReg))
      continue;
    if (MRI->getRegClass(AReg) != MRI->getRegClass(BReg))
      continue;

    LLVM_DEBUG(dbgs() << "SITensorDescSink: merging " << printReg(BReg)
                      << " into " << printReg(AReg) << "\n");

    // Replace all uses/defs of BReg with AReg (except the COPY itself).
    SmallVector<MachineInstr *, 16> BRegMIs;
    for (MachineInstr &MI : MRI->reg_instructions(BReg)) {
      if (&MI != CopyMI)
        BRegMIs.push_back(&MI);
    }
    for (MachineInstr *MI : BRegMIs) {
      for (MachineOperand &MO : MI->operands()) {
        if (MO.isReg() && MO.getReg() == BReg) {
          MO.setReg(AReg);
          if (MO.isDef())
            MO.setIsUndef(false);
        }
      }
    }

    CopyMI->eraseFromParent();
    Changed = true;
  }

  return Changed;
}

bool SITensorDescSink::processLoop(MachineLoop *L) {
  bool Changed = false;

  for (MachineLoop *InnerL : *L)
    Changed |= processLoop(InnerL);

  // Merge descriptor copies first (eliminate redundant full-reg copies
  // between descriptors used by different tensor_load_to_lds instructions).
  Changed |= mergeDescriptorCopies(L);

  SmallSetVector<Register, 8> DescRegs;
  for (MachineBasicBlock *MBB : L->blocks()) {
    for (MachineInstr &MI : *MBB) {
      if (!isTensorLoadToLDS(MI))
        continue;
      for (unsigned OpNo = 0; OpNo < 2; ++OpNo) {
        MachineOperand &MO = MI.getOperand(OpNo);
        if (MO.isReg() && MO.getReg().isVirtual())
          DescRegs.insert(MO.getReg());
      }
    }
  }

  if (DescRegs.empty())
    return Changed;

  LLVM_DEBUG(dbgs() << "SITensorDescSink: found " << DescRegs.size()
                    << " descriptor regs in loop\n");

  for (Register Reg : DescRegs)
    Changed |= sinkDescriptor(Reg, L);

  // After sinking, merge again — sinking may have created new COPY patterns.
  Changed |= mergeDescriptorCopies(L);

  return Changed;
}

bool SITensorDescSink::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableTensorDescSink)
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  bool Changed = false;
  for (MachineLoop *L : *MLI)
    Changed |= processLoop(L);

  return Changed;
}

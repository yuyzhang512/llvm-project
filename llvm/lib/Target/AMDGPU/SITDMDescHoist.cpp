//===-- SITDMDescHoist.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Post-RA pass that optimizes TDM (TENSOR_LOAD_TO_LDS) descriptor setup in
/// loops by hoisting invariant sub-register copies to preheaders, redirecting
/// computations that clobber invariant descriptor fields, and merging adjacent
/// 32-bit moves back into 64-bit moves.
//===----------------------------------------------------------------------===//

#include "SITDMDescHoist.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "si-tdm-desc-hoist"

static cl::opt<bool> DisableTDMDescHoist(
    "amdgpu-disable-tdm-desc-hoist", cl::init(true), cl::Hidden,
    cl::desc("Disable TDM descriptor hoist pass"));

namespace {

class SITDMDescHoistLegacy : public MachineFunctionPass {
public:
  static char ID;

  SITDMDescHoistLegacy() : MachineFunctionPass(ID) {
    initializeSITDMDescHoistLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "SI TDM Desc Hoist"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

class SITDMDescHoist {
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineLoopInfo *MLI;

  bool processLoop(MachineLoop *L);
  bool processTDMDescriptors(MachineLoop *L, MachineBasicBlock *Preheader);

public:
  bool run(MachineFunction &MF, MachineLoopInfo &MLI);
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(SITDMDescHoistLegacy, DEBUG_TYPE, "SI TDM Desc Hoist",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(SITDMDescHoistLegacy, DEBUG_TYPE, "SI TDM Desc Hoist",
                    false, false)

char SITDMDescHoistLegacy::ID = 0;

char &llvm::SITDMDescHoistLegacyID = SITDMDescHoistLegacy::ID;

bool SITDMDescHoistLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (DisableTDMDescHoist)
    return false;
  if (skipFunction(MF.getFunction()))
    return false;
  auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  SITDMDescHoist Impl;
  return Impl.run(MF, MLI);
}

PreservedAnalyses
SITDMDescHoistPass::run(MachineFunction &MF,
                         MachineFunctionAnalysisManager &MFAM) {
  if (DisableTDMDescHoist)
    return PreservedAnalyses::all();
  auto &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  SITDMDescHoist Impl;
  Impl.run(MF, MLI);
  return PreservedAnalyses::all();
}

bool SITDMDescHoist::processLoop(MachineLoop *L) {
  bool Changed = false;

  // Process inner loops first (innermost-out).
  for (MachineLoop *Inner : *L)
    Changed |= processLoop(Inner);

  MachineBasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader)
    return Changed;

  // Identify TDM descriptor/group registers before splitting.
  DenseSet<MCRegUnit> TDMRegUnits;
  for (MachineBasicBlock *BB : L->blocks()) {
    for (MachineInstr &MI : *BB) {
      if (MI.getOpcode() != AMDGPU::TENSOR_LOAD_TO_LDS_d2 &&
          MI.getOpcode() != AMDGPU::TENSOR_LOAD_TO_LDS_d4)
        continue;
      for (auto OpName :
           {AMDGPU::OpName::vaddr0, AMDGPU::OpName::vaddr1}) {
        int Idx = AMDGPU::getNamedOperandIdx(MI.getOpcode(), OpName);
        if (Idx < 0)
          continue;
        MCRegister Reg = MI.getOperand(Idx).getReg().asMCReg();
        for (MCRegUnit Unit : TRI->regunits(Reg))
          TDMRegUnits.insert(Unit);
      }
    }
  }

  // Profitability analysis: count TDM-related copy instructions in the loop
  // body. Each S_MOV_B64 counts as 2 (it will be split). Only proceed if
  // there are enough copies to justify the transformation overhead.
  if (TDMRegUnits.empty())
    return Changed;

  unsigned TDMCopyCount = 0;
  for (MachineBasicBlock *BB : L->blocks()) {
    for (MachineInstr &MI : *BB) {
      unsigned Opc = MI.getOpcode();
      if (Opc != AMDGPU::S_MOV_B32 && Opc != AMDGPU::S_MOV_B64 &&
          !MI.isCopy())
        continue;
      if (MI.getNumOperands() < 2 || !MI.getOperand(0).isReg())
        continue;
      MCRegister DstReg = MI.getOperand(0).getReg().asMCReg();
      for (MCRegUnit Unit : TRI->regunits(DstReg)) {
        if (TDMRegUnits.count(Unit)) {
          TDMCopyCount += (Opc == AMDGPU::S_MOV_B64) ? 2 : 1;
          break;
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "SI-TDM-DESC-HOIST: Found " << TDMCopyCount
                    << " TDM-related copies in loop body\n");
  if (TDMCopyCount < 2) {
    LLVM_DEBUG(dbgs() << "SI-TDM-DESC-HOIST: Skip — insufficient benefit\n");
    return Changed;
  }

  // Phase 0: Split TDM-related S_MOV_B64 into pairs of S_MOV_B32.
  // Only split moves whose destination overlaps with TDM descriptor registers,
  // enabling fine-grained hoisting when only one half is invariant.
  SmallVector<MachineInstr *, 4> ToSplit;
  for (MachineBasicBlock *BB : L->blocks()) {
    for (MachineInstr &MI : *BB) {
      if (MI.getOpcode() != AMDGPU::S_MOV_B64)
        continue;
      const MachineOperand &Dst = MI.getOperand(0);
      const MachineOperand &Src = MI.getOperand(1);
      if (!Dst.isReg() || !Src.isReg())
        continue;
      if (!Dst.getReg().isPhysical() || !Src.getReg().isPhysical())
        continue;
      if (!TRI->isSGPRPhysReg(Dst.getReg()) ||
          !TRI->isSGPRPhysReg(Src.getReg()))
        continue;

      bool IsTDMRelated = false;
      for (MCRegUnit Unit : TRI->regunits(Dst.getReg().asMCReg())) {
        if (TDMRegUnits.count(Unit)) {
          IsTDMRelated = true;
          break;
        }
      }
      if (!IsTDMRelated)
        continue;

      ToSplit.push_back(&MI);
    }
  }

  for (MachineInstr *MI : ToSplit) {
    MachineBasicBlock *MBB = MI->getParent();
    MCRegister DstReg = MI->getOperand(0).getReg().asMCReg();
    MCRegister SrcReg = MI->getOperand(1).getReg().asMCReg();

    MCRegister DstLo = TRI->getSubReg(DstReg, AMDGPU::sub0);
    MCRegister DstHi = TRI->getSubReg(DstReg, AMDGPU::sub1);
    MCRegister SrcLo = TRI->getSubReg(SrcReg, AMDGPU::sub0);
    MCRegister SrcHi = TRI->getSubReg(SrcReg, AMDGPU::sub1);

    LLVM_DEBUG(dbgs() << "SI-TDM-DESC-HOIST: Splitting: " << *MI);
    BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(AMDGPU::S_MOV_B32), DstLo)
        .addReg(SrcLo);
    BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(AMDGPU::S_MOV_B32), DstHi)
        .addReg(SrcHi);
    MI->eraseFromParent();
    Changed = true;
  }

  // Phase 1: Build the set of all register units defined anywhere in the loop.
  DenseMap<MCRegUnit, SmallVector<MachineInstr *, 2>> UnitToDefiners;
  for (MachineBasicBlock *BB : L->blocks()) {
    for (MachineInstr &MI : *BB) {
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.isDef())
          continue;
        Register Reg = MO.getReg();
        if (!Reg.isPhysical())
          continue;
        for (MCRegUnit Unit : TRI->regunits(Reg.asMCReg()))
          UnitToDefiners[Unit].push_back(&MI);
      }
    }
  }

  // Phase 2: Collect SGPR copy candidates (S_MOV_B32 and COPY).
  SmallVector<MachineInstr *, 8> Candidates;
  for (MachineBasicBlock *BB : L->blocks()) {
    for (MachineInstr &MI : *BB) {
      unsigned Opc = MI.getOpcode();
      if (Opc != AMDGPU::S_MOV_B32 && !MI.isCopy())
        continue;

      const MachineOperand &Dst = MI.getOperand(0);
      const MachineOperand &Src = MI.getOperand(1);

      if (!Dst.isReg() || !Src.isReg())
        continue;
      if (!Dst.getReg().isPhysical() || !Src.getReg().isPhysical())
        continue;

      if (!TRI->isSGPRPhysReg(Dst.getReg()) ||
          !TRI->isSGPRPhysReg(Src.getReg()))
        continue;

      // Only handle 32-bit SGPR copies.
      const TargetRegisterClass *DstRC = TRI->getMinimalPhysRegClass(Dst.getReg());
      if (TRI->getRegSizeInBits(*DstRC) != 32)
        continue;

      Candidates.push_back(&MI);
    }
  }

  if (Candidates.empty())
    return Changed;

  // Phase 3: Hoistability analysis with mutual-dependency resolution.
  //
  // A candidate is hoistable if:
  //   (a) All source reg units have no def in the loop except by
  //       already-hoisted instructions.
  //   (b) All dest reg units are ONLY defined by this instruction or
  //       other hoistable instructions.
  //
  // Two candidates defining the same dest register (e.g., s_mov_b32 s49, s5
  // and s_mov_b32 s49, s22) create a mutual dependency that the old
  // one-at-a-time iteration cannot resolve. We fix this with a two-step
  // approach: first identify source-invariant candidates, then find the
  // maximal subset whose dest constraints are mutually satisfied.
  DenseSet<MachineInstr *> Hoistable;
  bool Progress = true;
  while (Progress) {
    Progress = false;

    // Step 1: Identify candidates whose sources are loop-invariant.
    DenseSet<MachineInstr *> SrcInv;
    for (MachineInstr *MI : Candidates) {
      if (Hoistable.count(MI))
        continue;

      MCRegister SrcReg = MI->getOperand(1).getReg().asMCReg();
      bool Inv = true;
      for (MCRegUnit Unit : TRI->regunits(SrcReg)) {
        auto It = UnitToDefiners.find(Unit);
        if (It == UnitToDefiners.end())
          continue;
        for (MachineInstr *Definer : It->second) {
          if (!Hoistable.count(Definer)) {
            Inv = false;
            break;
          }
        }
        if (!Inv)
          break;
      }
      if (Inv)
        SrcInv.insert(MI);
    }

    if (SrcInv.empty())
      break;

    // Step 2: Find the maximal subset of SrcInv whose dest constraints
    // are satisfied. A dest is OK if all its other definers are in
    // Hoistable or in the subset itself. Iteratively remove failures.
    DenseSet<MachineInstr *> Subset(SrcInv);
    bool Shrunk = true;
    while (Shrunk) {
      Shrunk = false;
      SmallVector<MachineInstr *, 4> ToRemove;
      for (MachineInstr *MI : Subset) {
        MCRegister DstReg = MI->getOperand(0).getReg().asMCReg();
        bool DstOK = true;
        for (MCRegUnit Unit : TRI->regunits(DstReg)) {
          auto It = UnitToDefiners.find(Unit);
          if (It == UnitToDefiners.end())
            continue;
          for (MachineInstr *Definer : It->second) {
            if (Definer == MI)
              continue;
            if (Hoistable.count(Definer))
              continue;
            if (Subset.count(Definer))
              continue;
            DstOK = false;
            break;
          }
          if (!DstOK)
            break;
        }
        if (!DstOK)
          ToRemove.push_back(MI);
      }
      for (MachineInstr *MI : ToRemove) {
        Subset.erase(MI);
        Shrunk = true;
      }
    }

    for (MachineInstr *MI : Subset) {
      LLVM_DEBUG(dbgs() << "SI-TDM-DESC-HOIST: Hoistable: " << *MI);
      Hoistable.insert(MI);
      Progress = true;
    }
  }

  if (Hoistable.empty())
    return Changed;

  // Phase 4: Hoist in original program order.
  SmallVector<MachineInstr *, 8> ToHoist;
  for (MachineBasicBlock *BB : L->blocks()) {
    for (MachineInstr &MI : *BB) {
      if (Hoistable.count(&MI))
        ToHoist.push_back(&MI);
    }
  }

  MachineBasicBlock::iterator InsertPt = Preheader->getFirstTerminator();
  for (MachineInstr *MI : ToHoist) {
    LLVM_DEBUG(dbgs() << "SI-TDM-DESC-HOIST: Moving to preheader: " << *MI);
    MI->removeFromParent();
    Preheader->insert(InsertPt, MI);
  }

  Changed = true;

  // Phase 5: TDM descriptor register conflict resolution.
  //
  // When a loop contains TENSOR_LOAD_TO_LDS instructions, computations may
  // clobber invariant descriptor sub-registers (e.g., 64-bit shifts that write
  // a pair containing an invariant field). This phase:
  //   (a) Redirects computations that clobber invariant TDM descriptor
  //       sub-registers to use different destination registers.
  //   (b) Redirects computations whose results are copied to TDM2 descriptor
  //       sub-registers to write directly to those sub-registers, eliminating
  //       the copy instructions.
  //   (c) Removes dead store instructions.
  //
  // This is a targeted peephole for loops with two TENSOR_LOAD_TO_LDS_d2
  // instructions sharing a similar descriptor layout.
  Changed |= processTDMDescriptors(L, Preheader);

  // Phase 6: Merge adjacent S_MOV_B32 pairs back into S_MOV_B64.
  // After splitting and hoisting, consecutive 32-bit moves that form an
  // aligned 64-bit pair can be compacted (e.g., s_mov_b32 s48,s4 +
  // s_mov_b32 s49,s5 → s_mov_b64 s[48:49],s[4:5]).
  auto findPair64 = [&](MCRegister Lo, MCRegister Hi) -> MCRegister {
    for (MCRegister Super : TRI->superregs(Lo)) {
      if (TRI->getSubReg(Super, AMDGPU::sub0) != Lo)
        continue;
      if (TRI->getSubReg(Super, AMDGPU::sub1) != Hi)
        continue;
      if (TRI->getSubReg(Super, AMDGPU::sub2))
        continue; // >64-bit, skip
      return Super;
    }
    return MCRegister();
  };

  auto mergeMovPairs = [&](MachineBasicBlock *BB) -> bool {
    bool Merged = false;
    for (auto It = BB->begin(); It != BB->end();) {
      MachineInstr &MI1 = *It;
      auto Next = std::next(It);
      if (Next == BB->end()) {
        ++It;
        break;
      }
      MachineInstr &MI2 = *Next;

      bool IsMov1 = MI1.getOpcode() == AMDGPU::S_MOV_B32 || MI1.isCopy();
      bool IsMov2 = MI2.getOpcode() == AMDGPU::S_MOV_B32 || MI2.isCopy();
      if (!IsMov1 || !IsMov2 || MI1.getNumOperands() < 2 ||
          MI2.getNumOperands() < 2) {
        ++It;
        continue;
      }

      if (!MI1.getOperand(0).isReg() || !MI1.getOperand(1).isReg() ||
          !MI2.getOperand(0).isReg() || !MI2.getOperand(1).isReg()) {
        ++It;
        continue;
      }
      if (!MI1.getOperand(1).getReg().isPhysical() ||
          !MI2.getOperand(1).getReg().isPhysical()) {
        ++It;
        continue;
      }

      MCRegister Dst1 = MI1.getOperand(0).getReg().asMCReg();
      MCRegister Dst2 = MI2.getOperand(0).getReg().asMCReg();
      MCRegister Src1 = MI1.getOperand(1).getReg().asMCReg();
      MCRegister Src2 = MI2.getOperand(1).getReg().asMCReg();

      if (!TRI->isSGPRPhysReg(Dst1) || !TRI->isSGPRPhysReg(Src1)) {
        ++It;
        continue;
      }

      // Second move must not read first move's dest.
      if (TRI->regsOverlap(Src2, Dst1)) {
        ++It;
        continue;
      }

      // Try (lo, hi) order, then (hi, lo).
      MCRegister DstPair, SrcPair;
      DstPair = findPair64(Dst1, Dst2);
      if (DstPair)
        SrcPair = findPair64(Src1, Src2);
      if (!DstPair || !SrcPair) {
        DstPair = findPair64(Dst2, Dst1);
        if (DstPair)
          SrcPair = findPair64(Src2, Src1);
      }

      if (!DstPair || !SrcPair) {
        ++It;
        continue;
      }

      LLVM_DEBUG(dbgs() << "SI-TDM-DESC-HOIST: Merging:\n  " << MI1
                        << "  " << MI2 << "  -> S_MOV_B64 "
                        << printReg(DstPair, TRI) << ", "
                        << printReg(SrcPair, TRI) << "\n");
      BuildMI(*BB, MI1, MI1.getDebugLoc(), TII->get(AMDGPU::S_MOV_B64),
              DstPair)
          .addReg(SrcPair);
      It = std::next(MI2.getIterator());
      MI1.eraseFromParent();
      MI2.eraseFromParent();
      Merged = true;
    }
    return Merged;
  };

  if (mergeMovPairs(Preheader))
    Changed = true;
  for (MachineBasicBlock *BB : L->blocks())
    if (mergeMovPairs(BB))
      Changed = true;

  return Changed;
}

static MachineInstr *findDefBefore(MachineInstr *Before, MCRegister Reg,
                                   const SIRegisterInfo *TRI) {
  MachineBasicBlock *MBB = Before->getParent();
  for (auto It = MachineBasicBlock::reverse_iterator(Before->getIterator());
       It != MBB->rend(); ++It) {
    for (const MachineOperand &MO : It->operands()) {
      if (!MO.isReg() || !MO.isDef())
        continue;
      if (TRI->regsOverlap(MO.getReg(), Reg))
        return &*It;
    }
  }
  return nullptr;
}

static void replaceRegInRange(MachineBasicBlock::iterator Begin,
                               MachineBasicBlock::iterator End,
                               MCRegister OldReg, MCRegister NewReg) {
  for (auto It = Begin; It != End; ++It) {
    for (MachineOperand &MO : It->operands()) {
      if (MO.isReg() && MO.getReg() == OldReg)
        MO.setReg(NewReg);
    }
  }
}

bool SITDMDescHoist::processTDMDescriptors(MachineLoop *L,
                                            MachineBasicBlock *Preheader) {
  SmallVector<MachineInstr *, 4> TDMs;
  for (MachineBasicBlock *BB : L->blocks()) {
    for (MachineInstr &MI : *BB) {
      if (MI.getOpcode() == AMDGPU::TENSOR_LOAD_TO_LDS_d2 ||
          MI.getOpcode() == AMDGPU::TENSOR_LOAD_TO_LDS_d4)
        TDMs.push_back(&MI);
    }
  }
  if (TDMs.size() < 2)
    return false;

  MachineInstr *TDM1 = nullptr, *TDM2 = nullptr;
  for (unsigned i = 0; i + 1 < TDMs.size(); ++i) {
    if (TDMs[i]->getParent() == TDMs[i + 1]->getParent()) {
      TDM1 = TDMs[i];
      TDM2 = TDMs[i + 1];
      break;
    }
  }
  if (!TDM1 || !TDM2)
    return false;

  MachineBasicBlock *MBB = TDM1->getParent();

  int Idx1 = AMDGPU::getNamedOperandIdx(TDM1->getOpcode(),
                                         AMDGPU::OpName::vaddr1);
  int Idx2 = AMDGPU::getNamedOperandIdx(TDM2->getOpcode(),
                                         AMDGPU::OpName::vaddr1);
  if (Idx1 < 0 || Idx2 < 0)
    return false;

  MCRegister Desc1 = TDM1->getOperand(Idx1).getReg().asMCReg();
  MCRegister Desc2 = TDM2->getOperand(Idx2).getReg().asMCReg();
  if (Desc1 == Desc2)
    return false;

  MCRegister Desc1Sub[8], Desc2Sub[8];
  static const unsigned SubRegIdx[] = {
      AMDGPU::sub0, AMDGPU::sub1, AMDGPU::sub2, AMDGPU::sub3,
      AMDGPU::sub4, AMDGPU::sub5, AMDGPU::sub6, AMDGPU::sub7};
  for (int i = 0; i < 8; ++i) {
    Desc1Sub[i] = TRI->getSubReg(Desc1, SubRegIdx[i]);
    Desc2Sub[i] = TRI->getSubReg(Desc2, SubRegIdx[i]);
  }

  int GIdx1 = AMDGPU::getNamedOperandIdx(TDM1->getOpcode(),
                                          AMDGPU::OpName::vaddr0);
  int GIdx2 = AMDGPU::getNamedOperandIdx(TDM2->getOpcode(),
                                          AMDGPU::OpName::vaddr0);
  MCRegister Grp1[4], Grp2[4];
  if (GIdx1 >= 0 && GIdx2 >= 0) {
    MCRegister G1 = TDM1->getOperand(GIdx1).getReg().asMCReg();
    MCRegister G2 = TDM2->getOperand(GIdx2).getReg().asMCReg();
    for (int i = 0; i < 4; ++i) {
      Grp1[i] = TRI->getSubReg(G1, SubRegIdx[i]);
      Grp2[i] = TRI->getSubReg(G2, SubRegIdx[i]);
    }
  }

  // Profitability analysis: count copies into Desc2/Grp2 sub-regs that can be
  // eliminated by redirecting source computations to write directly to the
  // target descriptor registers.
  unsigned RedundantCopies = 0;
  for (MachineInstr &MI : *MBB) {
    unsigned Opc = MI.getOpcode();
    if (Opc != AMDGPU::S_MOV_B32 && !MI.isCopy())
      continue;
    if (MI.getNumOperands() < 2 || !MI.getOperand(0).isReg())
      continue;
    MCRegister DstReg = MI.getOperand(0).getReg().asMCReg();
    bool IsTarget = false;
    for (int i = 0; i < 8; ++i)
      if (DstReg == Desc2Sub[i]) { IsTarget = true; break; }
    if (!IsTarget && GIdx2 >= 0)
      for (int i = 0; i < 4; ++i)
        if (DstReg == Grp2[i]) { IsTarget = true; break; }
    if (IsTarget)
      ++RedundantCopies;
  }

  LLVM_DEBUG(dbgs() << "SI-TDM-DESC-HOIST: Phase 5 analysis: "
                    << RedundantCopies << " redundant copies\n");
  if (RedundantCopies < 2) {
    LLVM_DEBUG(
        dbgs() << "SI-TDM-DESC-HOIST: Skip Phase 5 — insufficient benefit\n");
    return false;
  }

  bool AnyChange = false;

  // === Step 1: Redirect 64-bit ops that clobber invariant Desc2 sub-regs ===
  // Pattern: S_LSHL_B64 $s44_45, ... clobbers invariant s44/s45.
  // Redirect to a free pair.

  // Find free 64-bit aligned SGPR pairs.
  // Try SGPR56_SGPR57, SGPR58_SGPR59.
  struct FreePair {
    MCRegister Lo, Hi, Pair;
  };
  auto findFreePair = [&](MachineBasicBlock *BB) -> FreePair {
    static const MCRegister PairLoHi[][2] = {
        {AMDGPU::SGPR56, AMDGPU::SGPR57},
        {AMDGPU::SGPR58, AMDGPU::SGPR59},
    };
    for (const auto &P : PairLoHi) {
      bool InUse = false;
      for (MachineInstr &MI : *BB) {
        for (const MachineOperand &MO : MI.operands()) {
          if (MO.isReg() && (TRI->regsOverlap(MO.getReg(), P[0]) ||
                             TRI->regsOverlap(MO.getReg(), P[1]))) {
            InUse = true;
            break;
          }
        }
        if (InUse) break;
      }
      if (!InUse) {
        MCRegister Pair;
        for (MCRegister Super : TRI->superregs(P[0])) {
          if (TRI->getSubReg(Super, AMDGPU::sub0) == P[0] &&
              TRI->getSubReg(Super, AMDGPU::sub1) == P[1]) {
            Pair = Super;
            break;
          }
        }
        if (Pair) return {P[0], P[1], Pair};
      }
    }
    return {MCRegister(), MCRegister(), MCRegister()};
  };

  for (auto It = MBB->begin(); It != MBB->end(); ++It) {
    MachineInstr &MI = *It;
    unsigned Opc = MI.getOpcode();
    if (Opc != AMDGPU::S_LSHL_B64 && Opc != AMDGPU::S_LSHR_B64 &&
        Opc != AMDGPU::S_ADD_U64)
      continue;

    MCRegister DstPair = MI.getOperand(0).getReg().asMCReg();
    MCRegister DstLo = TRI->getSubReg(DstPair, AMDGPU::sub0);
    MCRegister DstHi = TRI->getSubReg(DstPair, AMDGPU::sub1);

    bool ClobbersDesc2 = false;
    for (int i = 0; i < 8; ++i) {
      if (DstLo == Desc2Sub[i] || DstHi == Desc2Sub[i]) {
        ClobbersDesc2 = true;
        break;
      }
    }
    if (!ClobbersDesc2)
      continue;

    auto FP = findFreePair(MBB);
    if (!FP.Pair)
      continue;

    LLVM_DEBUG(dbgs() << "SI-TDM-DESC-HOIST: Redirecting 64-bit op from "
                      << printReg(DstPair, TRI) << " to "
                      << printReg(FP.Pair, TRI) << " in " << MI);

    MI.getOperand(0).setReg(FP.Pair);
    // Replace only USES of DstLo/DstHi with FP.Lo/FP.Hi downstream.
    // Stop replacing each sub-reg at the first instruction that re-defines it,
    // because that re-definition writes the original register intentionally
    // (e.g., setting up a TDM descriptor sub-register).
    bool LoActive = true, HiActive = true;
    for (auto UseIt = std::next(MI.getIterator()); UseIt != MBB->end();
         ++UseIt) {
      if (!LoActive && !HiActive)
        break;
      // Check for re-definitions first (before replacing uses in this instr).
      for (const MachineOperand &MO : UseIt->operands()) {
        if (!MO.isReg() || !MO.isDef()) continue;
        if (MO.getReg() == DstLo) LoActive = false;
        else if (MO.getReg() == DstHi) HiActive = false;
      }
      // Replace uses (but not defs) of the old registers.
      for (MachineOperand &MO : UseIt->operands()) {
        if (!MO.isReg() || !MO.isUse()) continue;
        if (LoActive && MO.getReg() == DstLo) MO.setReg(FP.Lo);
        else if (HiActive && MO.getReg() == DstHi) MO.setReg(FP.Hi);
        else if (MO.getReg() == DstPair) MO.setReg(FP.Pair);
      }
    }
    AnyChange = true;
  }

  // === Step 2: Redirect s_mov/COPY that feed Desc2/Grp2 sub-regs ===
  // For each s_mov that copies a computed value to a Desc2 or Grp2 sub-reg,
  // redirect the source computation to write directly to the target sub-reg.
  // Scan the entire BB up to TDM2 (not just between TDM1 and TDM2) because
  // the post-RA scheduler may reorder instructions after this pass.
  SmallVector<MachineInstr *, 8> ToErase;
  for (auto It = MBB->begin(); &*It != TDM2; ++It) {
    MachineInstr &MI = *It;
    if (MI.getOpcode() != AMDGPU::S_MOV_B32 && !MI.isCopy())
      continue;
    if (MI.getNumOperands() < 2) continue;
    if (!MI.getOperand(0).isReg() || !MI.getOperand(1).isReg()) continue;
    if (!MI.getOperand(0).getReg().isPhysical()) continue;

    MCRegister DstReg = MI.getOperand(0).getReg().asMCReg();
    MCRegister SrcReg = MI.getOperand(1).getReg().asMCReg();

    // Match Desc2 or Grp2 sub-reg destination.
    MCRegister TargetReg;
    for (int i = 0; i < 8; ++i)
      if (DstReg == Desc2Sub[i]) { TargetReg = Desc2Sub[i]; break; }
    if (!TargetReg && GIdx2 >= 0)
      for (int i = 0; i < 4; ++i)
        if (DstReg == Grp2[i]) { TargetReg = Grp2[i]; break; }
    if (!TargetReg) continue;

    MachineInstr *SrcDef = findDefBefore(&MI, SrcReg, TRI);
    if (!SrcDef || SrcDef->getParent() != MBB) continue;

    // Check SrcReg has no use after this s_mov (up to TDM2).
    bool SrcUsedLater = false;
    for (auto Check = std::next(MI.getIterator()); &*Check != TDM2; ++Check) {
      for (const MachineOperand &MO : Check->operands())
        if (MO.isReg() && MO.isUse() && MO.getReg() == SrcReg)
          SrcUsedLater = true;
      if (SrcUsedLater) break;
    }
    if (SrcUsedLater) continue;

    // Check TargetReg is not read between SrcDef and TDM2 (except this s_mov).
    bool TgtUsed = false;
    for (auto Check = SrcDef->getIterator(); &*Check != TDM2; ++Check) {
      if (&*Check == &MI) continue;
      for (const MachineOperand &MO : Check->operands())
        if (MO.isReg() && MO.isUse() &&
            TRI->regsOverlap(MO.getReg(), TargetReg))
          TgtUsed = true;
      if (TgtUsed) break;
    }
    if (TgtUsed) continue;

    // For 64-bit source defs (S_ADD_U64 etc.), redirect the whole pair
    // and update uses of both sub-regs. Only safe if the high half of the
    // target pair is NOT an invariant descriptor sub-reg that would be
    // clobbered by the 64-bit write.
    MCRegister SrcDefDst = SrcDef->getOperand(0).getReg().asMCReg();
    MCRegister SrcDefLo = TRI->getSubReg(SrcDefDst, AMDGPU::sub0);
    MCRegister SrcDefHi = TRI->getSubReg(SrcDefDst, AMDGPU::sub1);
    if (SrcDefLo && SrcDefHi && SrcReg == SrcDefLo) {
      MCRegister PairReg;
      for (MCRegister Super : TRI->superregs(TargetReg)) {
        if (TRI->getSubReg(Super, AMDGPU::sub0) == TargetReg) {
          MCRegister PairHi = TRI->getSubReg(Super, AMDGPU::sub1);
          if (PairHi) { PairReg = Super; break; }
        }
      }
      // Check that the high half is not an invariant Desc2 sub-reg.
      // If it is, the 64-bit redirect would clobber it; fall through to
      // the 32-bit path instead.
      if (PairReg) {
        MCRegister NewHi = TRI->getSubReg(PairReg, AMDGPU::sub1);
        bool HiIsInvariantDesc = false;
        for (int i = 0; i < 8; ++i) {
          if (i == 1 || i == 2) continue;
          if (NewHi == Desc2Sub[i]) { HiIsInvariantDesc = true; break; }
        }
        if (HiIsInvariantDesc)
          PairReg = MCRegister();
      }
      if (PairReg) {
        MCRegister NewHi = TRI->getSubReg(PairReg, AMDGPU::sub1);
        LLVM_DEBUG(dbgs() << "SI-TDM-DESC-HOIST: Redirect 64-bit copy "
                          << printReg(SrcDefDst, TRI) << " -> "
                          << printReg(PairReg, TRI) << "\n");
        SrcDef->getOperand(0).setReg(PairReg);
        // Update uses of old sub-regs between SrcDef and TDM2.
        for (auto Upd = std::next(SrcDef->getIterator()); &*Upd != TDM2;
             ++Upd) {
          if (&*Upd == &MI) continue;
          for (MachineOperand &MO : Upd->operands()) {
            if (!MO.isReg() || !MO.isUse()) continue;
            if (MO.getReg() == SrcDefLo) MO.setReg(TargetReg);
            else if (MO.getReg() == SrcDefHi) MO.setReg(NewHi);
          }
        }
        ToErase.push_back(&MI);
        AnyChange = true;
        continue;
      }
    }

    // 32-bit redirect: change the source def to write directly to TargetReg.
    // Only works if the source def has a matching 32-bit def operand.
    // Skip if the source def is a 64-bit op (its def operand is a pair register
    // that won't match the individual sub-reg).
    // Also skip if SrcReg is read (even via super-reg overlap) between the def
    // and this copy — redirecting would break consumers like TDM1 that read
    // a descriptor super-reg containing SrcReg.
    bool SrcUsedBetween = false;
    for (auto Check = std::next(SrcDef->getIterator()); &*Check != &MI;
         ++Check) {
      for (const MachineOperand &MO : Check->operands())
        if (MO.isReg() && MO.isUse() &&
            TRI->regsOverlap(MO.getReg(), SrcReg)) {
          SrcUsedBetween = true;
          break;
        }
      if (SrcUsedBetween)
        break;
    }
    if (SrcUsedBetween)
      continue;
    bool Redirected = false;
    for (MachineOperand &MO : SrcDef->operands())
      if (MO.isReg() && MO.isDef() && MO.getReg() == SrcReg) {
        LLVM_DEBUG(dbgs() << "SI-TDM-DESC-HOIST: Redirect copy src "
                          << printReg(SrcReg, TRI) << " -> "
                          << printReg(TargetReg, TRI) << "\n");
        MO.setReg(TargetReg);
        replaceRegInRange(std::next(SrcDef->getIterator()), MI.getIterator(),
                          SrcReg, TargetReg);
        ToErase.push_back(&MI);
        AnyChange = true;
        Redirected = true;
        break;
      }
    if (!Redirected) {
      LLVM_DEBUG(dbgs() << "SI-TDM-DESC-HOIST: Cannot redirect "
                        << printReg(SrcReg, TRI) << " -> "
                        << printReg(TargetReg, TRI)
                        << " (source is 64-bit op)\n");
    }
  }
  for (MachineInstr *MI : ToErase)
    MI->eraseFromParent();
  ToErase.clear();

  // === Step 3: Hoist now-unclobbered invariant copies to preheader ===
  // After Steps 1-2 redirected ops away from Desc2 sub-regs, some
  // s_mov_b32 copies may no longer be clobbered in the loop. If the source
  // is also loop-invariant, hoist them.
  {
    MachineBasicBlock::iterator InsertPt = Preheader->getFirstTerminator();
    SmallVector<MachineInstr *, 4> ToHoistLate;
    for (auto It = MBB->begin(); It != MBB->end(); ++It) {
      MachineInstr &MI = *It;
      if (MI.getOpcode() != AMDGPU::S_MOV_B32 && !MI.isCopy())
        continue;
      if (MI.getNumOperands() < 2 || !MI.getOperand(0).isReg() ||
          !MI.getOperand(1).isReg())
        continue;
      if (!MI.getOperand(0).getReg().isPhysical() ||
          !MI.getOperand(1).getReg().isPhysical())
        continue;

      MCRegister DstReg = MI.getOperand(0).getReg().asMCReg();
      MCRegister SrcReg = MI.getOperand(1).getReg().asMCReg();

      bool IsDesc2 = false;
      for (int i = 0; i < 8; ++i)
        if (DstReg == Desc2Sub[i]) { IsDesc2 = true; break; }
      if (!IsDesc2)
        continue;

      // Source must not be defined in the loop body.
      bool SrcDefinedInLoop = false;
      for (auto Check = MBB->begin(); Check != MBB->end(); ++Check) {
        if (&*Check == &MI)
          continue;
        for (const MachineOperand &MO : Check->operands())
          if (MO.isReg() && MO.isDef() &&
              TRI->regsOverlap(MO.getReg(), SrcReg)) {
            SrcDefinedInLoop = true;
            break;
          }
        if (SrcDefinedInLoop)
          break;
      }
      if (SrcDefinedInLoop)
        continue;

      // Dest must not be clobbered by any other instruction in the loop.
      bool DstClobbered = false;
      for (auto Check = MBB->begin(); Check != MBB->end(); ++Check) {
        if (&*Check == &MI)
          continue;
        for (const MachineOperand &MO : Check->operands())
          if (MO.isReg() && MO.isDef() &&
              TRI->regsOverlap(MO.getReg(), DstReg)) {
            DstClobbered = true;
            break;
          }
        if (DstClobbered)
          break;
      }
      if (DstClobbered)
        continue;

      LLVM_DEBUG(dbgs() << "SI-TDM-DESC-HOIST: Late-hoist to preheader: "
                        << MI);
      ToHoistLate.push_back(&MI);
    }
    for (MachineInstr *MI : ToHoistLate) {
      MCRegister DstReg = MI->getOperand(0).getReg().asMCReg();
      MCRegister SrcReg = MI->getOperand(1).getReg().asMCReg();
      // Replace COPY with S_MOV_B32 to prevent Machine Copy Propagation
      // from removing this instruction later.
      if (MI->isCopy()) {
        MI->eraseFromParent();
        BuildMI(*Preheader, InsertPt, DebugLoc(), TII->get(AMDGPU::S_MOV_B32),
                DstReg)
            .addReg(SrcReg);
      } else {
        MI->removeFromParent();
        Preheader->insert(InsertPt, MI);
      }
      MBB->addLiveIn(DstReg);
      AnyChange = true;
    }
  }

  // === Step 4: Remove dead stores and unnecessary restores ===
  for (auto It = MBB->begin(); It != MBB->end();) {
    MachineInstr &MI = *It++;
    if (MI.getOpcode() != AMDGPU::S_MOV_B32 && !MI.isCopy()) continue;
    if (MI.getNumOperands() < 2) continue;
    if (!MI.getOperand(0).isReg() || !MI.getOperand(1).isReg()) continue;
    if (!MI.getOperand(0).getReg().isPhysical()) continue;

    MCRegister DstReg = MI.getOperand(0).getReg().asMCReg();

    // Check if this is a Desc1, Desc2, Grp1, or Grp2 sub-reg write.
    bool IsDescWrite = false;
    for (int i = 0; i < 8; ++i) {
      if (DstReg == Desc1Sub[i] || DstReg == Desc2Sub[i]) {
        IsDescWrite = true;
        break;
      }
    }
    if (!IsDescWrite && GIdx2 >= 0) {
      for (int i = 0; i < 4; ++i) {
        if (DstReg == Grp2[i]) {
          IsDescWrite = true;
          break;
        }
      }
    }
    if (!IsDescWrite) continue;

    // Check if DstReg is overwritten before it's read (dead store).
    bool Dead = false;
    for (auto Check = std::next(MI.getIterator()); Check != MBB->end();
         ++Check) {
      bool Used = false, Deffed = false;
      for (const MachineOperand &MO : Check->operands()) {
        if (MO.isReg() && MO.isUse() &&
            TRI->regsOverlap(MO.getReg(), DstReg))
          Used = true;
        if (MO.isReg() && MO.isDef() &&
            TRI->regsOverlap(MO.getReg(), DstReg))
          Deffed = true;
      }
      if (Used) break;
      if (Deffed) { Dead = true; break; }
    }

    if (!Dead) {
      // Check if this restores a Desc1 or Desc2 sub-reg that's not clobbered
      // in the loop body before this instruction. If nothing in the loop body
      // overwrites DstReg before this instruction, then DstReg still holds
      // its preheader value and this write is redundant.
      bool IsDescSubReg = false;
      for (int i = 0; i < 8; ++i)
        if (DstReg == Desc1Sub[i] || DstReg == Desc2Sub[i]) {
          IsDescSubReg = true;
          break;
        }
      if (IsDescSubReg) {
        bool Clobbered = false;
        for (auto Check = MBB->begin(); &*Check != &MI; ++Check)
          for (const MachineOperand &MO : Check->operands())
            if (MO.isReg() && MO.isDef() &&
                TRI->regsOverlap(MO.getReg(), DstReg))
              Clobbered = true;
        // Only redundant if not clobbered AND defined in the preheader.
        // If not defined in the preheader, this s_mov is the initial
        // definition and must be kept.
        bool DefinedInPreheader = false;
        if (!Clobbered) {
          for (const MachineInstr &PHI : *Preheader)
            for (const MachineOperand &MO : PHI.operands())
              if (MO.isReg() && MO.isDef() &&
                  TRI->regsOverlap(MO.getReg(), DstReg))
                DefinedInPreheader = true;
        }
        if (!Clobbered && DefinedInPreheader)
          Dead = true;
      }
    }

    if (Dead) {
      LLVM_DEBUG(dbgs() << "SI-TDM-DESC-HOIST: Removing: " << MI);
      MI.eraseFromParent();
      AnyChange = true;
    }
  }

  return AnyChange;
}

bool SITDMDescHoist::run(MachineFunction &MF, MachineLoopInfo &LoopInfo) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MLI = &LoopInfo;

  bool Changed = false;
  for (MachineLoop *L : *MLI)
    Changed |= processLoop(L);

  return Changed;
}

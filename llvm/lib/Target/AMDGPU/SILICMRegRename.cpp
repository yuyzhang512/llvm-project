//===-- SILICMRegRename.cpp - Rename registers to enable post-RA LICM -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass renames destination registers of loop-invariant instructions
/// whose output register is also defined by another instruction in the loop.
/// Post-RA MachineLICM refuses to hoist such instructions because it sees the
/// register as "clobbered" (defined more than once). By renaming the invariant
/// instruction's output to a free register and rewriting its users, this pass
/// eliminates the conflicting definition and enables MachineLICM to hoist.
///
/// Currently limited to:
/// - Single-block loops
/// - VGPR register class only
/// - Exactly two defs of the same register unit in the loop
///
//===----------------------------------------------------------------------===//

#include "SILICMRegRename.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "si-licm-reg-rename"

static cl::opt<bool>
    EnableLICMRegRename("amdgpu-licm-reg-rename", cl::Hidden,
                        cl::desc("Enable register renaming to help post-RA "
                                 "MachineLICM hoist loop-invariant instructions"),
                        cl::init(true));

namespace {

class SILICMRegRename {
  MachineFunction *MF;
  const GCNSubtarget *ST;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  const MachineLoopInfo *MLI;

  unsigned NumRegUnits;

  /// Process a single loop. Returns true if any renames were made.
  bool processLoop(MachineLoop *Loop);

  /// Compute the set of register units defined in the loop and those
  /// that are clobbered (defined more than once).
  void computeLoopDefs(MachineBasicBlock *LoopBB, BitVector &RUDefs,
                       BitVector &RUClobbers);

  /// Check if an instruction is a rename candidate: loop-invariant sources,
  /// single non-dead def, def is clobbered, safe to move.
  bool isRenameCandidate(MachineInstr &MI, const BitVector &LoopRUDefs,
                         const BitVector &RUClobbers, Register &Def);

  /// Compute registers that are free across the entire loop.
  BitVector computeFreeRegUnits(MachineBasicBlock *LoopBB);

  /// Find a free register in the same register class as Def.
  MCRegister findFreeReg(Register Def, const BitVector &FreeRUs);

  /// Rename the def in MI and all its users reached before the next def
  /// of the same register. Returns true if renamed.
  bool renameDefAndUsers(MachineInstr &MI, Register OldReg, MCRegister NewReg,
                         MachineBasicBlock *LoopBB);

public:
  SILICMRegRename(MachineFunction *MF, const MachineLoopInfo *MLI)
      : MF(MF), ST(&MF->getSubtarget<GCNSubtarget>()),
        TII(ST->getInstrInfo()), TRI(&TII->getRegisterInfo()),
        MRI(&MF->getRegInfo()), MLI(MLI),
        NumRegUnits(TRI->getNumRegUnits()) {}

  bool run();
};

class SILICMRegRenameLegacy : public MachineFunctionPass {
public:
  static char ID;

  SILICMRegRenameLegacy() : MachineFunctionPass(ID) {
    initializeSILICMRegRenameLegacyPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "SI LICM Register Rename";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

// ---- New PM entry point ----
PreservedAnalyses
SILICMRegRenamePass::run(MachineFunction &MF,
                         MachineFunctionAnalysisManager &MFAM) {
  if (!EnableLICMRegRename)
    return PreservedAnalyses::all();

  auto &MLI = MFAM.getResult<MachineLoopAnalysis>(MF);
  if (!SILICMRegRename(&MF, &MLI).run())
    return PreservedAnalyses::all();

  auto PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

// ---- Legacy PM plumbing ----
INITIALIZE_PASS_BEGIN(SILICMRegRenameLegacy, DEBUG_TYPE,
                      "SI LICM Register Rename", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(SILICMRegRenameLegacy, DEBUG_TYPE,
                    "SI LICM Register Rename", false, false)

char SILICMRegRenameLegacy::ID = 0;

char &llvm::SILICMRegRenameLegacyID = SILICMRegRenameLegacy::ID;

bool SILICMRegRenameLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  if (!EnableLICMRegRename)
    return false;

  auto &MLI = getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  return SILICMRegRename(&MF, &MLI).run();
}

// ---- Implementation ----

bool SILICMRegRename::run() {
  bool Changed = false;
  for (MachineLoop *Loop : MLI->getTopLevelLoops())
    Changed |= processLoop(Loop);
  return Changed;
}

bool SILICMRegRename::processLoop(MachineLoop *Loop) {
  bool Changed = false;

  // Process inner loops first.
  for (MachineLoop *SubLoop : Loop->getSubLoops())
    Changed |= processLoop(SubLoop);

  // V1: Only handle single-block loops.
  if (Loop->getNumBlocks() != 1)
    return Changed;

  MachineBasicBlock *LoopBB = Loop->getHeader();

  // Need a preheader to hoist into (MachineLICM will also check this).
  if (!Loop->getLoopPreheader())
    return Changed;

  LLVM_DEBUG(dbgs() << "SILICMRegRename: Processing loop in "
                    << MF->getName() << " BB#" << LoopBB->getNumber() << "\n");

  // Step 1 & 2: Compute defs and clobbers.
  BitVector RUDefs(NumRegUnits);
  BitVector RUClobbers(NumRegUnits);
  computeLoopDefs(LoopBB, RUDefs, RUClobbers);

  if (RUClobbers.none())
    return Changed; // Nothing clobbered, LICM won't be blocked.

  // Step 3: Identify rename candidates.
  struct Candidate {
    MachineInstr *MI;
    Register Def;
  };
  SmallVector<Candidate, 8> Candidates;

  for (MachineInstr &MI : *LoopBB) {
    Register Def;
    if (isRenameCandidate(MI, RUDefs, RUClobbers, Def))
      Candidates.push_back({&MI, Def});
  }

  if (Candidates.empty())
    return Changed;

  LLVM_DEBUG(dbgs() << "  Found " << Candidates.size()
                    << " rename candidate(s)\n");

  // Step 4: Compute free register units.
  BitVector FreeRUs = computeFreeRegUnits(LoopBB);

  // Step 5: Rename each candidate.
  for (auto &C : Candidates) {
    MCRegister NewReg = findFreeReg(C.Def, FreeRUs);
    if (!NewReg) {
      LLVM_DEBUG(dbgs() << "  No free register for "
                        << printReg(C.Def, TRI) << "\n");
      continue;
    }

    LLVM_DEBUG(dbgs() << "  Renaming " << printReg(C.Def, TRI) << " -> "
                      << printReg(NewReg, TRI) << " in: " << *C.MI);

    if (renameDefAndUsers(*C.MI, C.Def, NewReg, LoopBB)) {
      Changed = true;
      // Mark the new register's units as used so we don't reuse it.
      for (MCRegUnit Unit : TRI->regunits(NewReg))
        FreeRUs.reset(static_cast<unsigned>(Unit));
    }
  }

  return Changed;
}

void SILICMRegRename::computeLoopDefs(MachineBasicBlock *LoopBB,
                                      BitVector &RUDefs,
                                      BitVector &RUClobbers) {
  for (const MachineInstr &MI : *LoopBB) {
    if (MI.isDebugInstr())
      continue;

    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef())
        continue;
      Register Reg = MO.getReg();
      if (!Reg || !Reg.isPhysical())
        continue;

      for (MCRegUnit Unit : TRI->regunits(Reg)) {
        unsigned U = static_cast<unsigned>(Unit);
        if (RUDefs.test(U))
          RUClobbers.set(U);
        RUDefs.set(U);
      }
    }

    // Regmask operands (calls) clobber many registers.
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isRegMask())
        continue;
      // If there's a call with a regmask, conservatively mark all clobbered
      // register units. Calls in loops are rare for AMDGPU kernels.
      for (unsigned i = 0, e = TRI->getNumRegs(); i != e; ++i) {
        if (MO.clobbersPhysReg(i)) {
          for (MCRegUnit Unit : TRI->regunits(MCRegister::from(i))) {
            RUClobbers.set(static_cast<unsigned>(Unit));
            RUDefs.set(static_cast<unsigned>(Unit));
          }
        }
      }
    }
  }
}

bool SILICMRegRename::isRenameCandidate(MachineInstr &MI,
                                        const BitVector &LoopRUDefs,
                                        const BitVector &RUClobbers,
                                        Register &Def) {
  if (MI.isDebugInstr() || MI.isTerminator())
    return false;

  // Must be safe to move (no side effects, not a call, etc.).
  bool DontMoveAcrossStore = true;
  if (!MI.isSafeToMove(DontMoveAcrossStore))
    return false;

  if (MI.isCall() || MI.isInlineAsm() || MI.isConvergent())
    return false;

  if (MI.hasExtraDefRegAllocReq() || MI.hasExtraSrcRegAllocReq())
    return false;

  // Find exactly one non-dead physical def.
  Register FoundDef;
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef())
      continue;
    Register Reg = MO.getReg();
    if (!Reg || !Reg.isPhysical())
      continue;
    if (MO.isDead())
      continue;

    // Skip implicit defs of special registers (SCC, VCC, EXEC, M0, etc.).
    if (MO.isImplicit())
      continue;

    if (FoundDef) {
      // Multiple non-dead explicit defs — too complex for V1.
      return false;
    }
    FoundDef = Reg;
  }

  if (!FoundDef)
    return false;

  // V1: Only rename VGPRs.
  if (!TRI->isVGPRPhysReg(FoundDef))
    return false;

  // Check that the def's register units are clobbered.
  bool AnyClobbered = false;
  for (MCRegUnit Unit : TRI->regunits(FoundDef)) {
    if (RUClobbers.test(static_cast<unsigned>(Unit))) {
      AnyClobbered = true;
      break;
    }
  }
  if (!AnyClobbered)
    return false; // LICM won't be blocked for this instruction.

  // Check that all source operands are loop-invariant
  // (their register units are not defined inside the loop).
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isUse())
      continue;
    Register Reg = MO.getReg();
    if (!Reg || !Reg.isPhysical())
      continue;

    // Check if this source reg is defined inside the loop.
    for (MCRegUnit Unit : TRI->regunits(Reg)) {
      if (LoopRUDefs.test(static_cast<unsigned>(Unit)))
        return false; // Source is not loop-invariant.
    }
  }

  // Check that this instruction is the only invariant def of FoundDef's
  // register units. We want exactly 2 total defs: this one (invariant)
  // and one other (non-invariant). Count defs of any of FoundDef's units.
  MachineBasicBlock *MBB = MI.getParent();
  unsigned DefCount = 0;
  for (const MachineInstr &Other : *MBB) {
    if (Other.isDebugInstr())
      continue;
    for (const MachineOperand &MO : Other.operands()) {
      if (!MO.isReg() || !MO.isDef())
        continue;
      Register Reg = MO.getReg();
      if (!Reg || !Reg.isPhysical())
        continue;
      if (TRI->regsOverlap(Reg, FoundDef)) {
        DefCount++;
        break; // Count each instruction at most once.
      }
    }
  }

  if (DefCount != 2) {
    LLVM_DEBUG(dbgs() << "  Skipping (DefCount=" << DefCount << "): " << MI);
    return false;
  }

  Def = FoundDef;
  return true;
}

BitVector SILICMRegRename::computeFreeRegUnits(MachineBasicBlock *LoopBB) {
  // Start with all register units free.
  BitVector UsedRUs(NumRegUnits);

  // Mark register units used by live-ins of the loop block.
  for (const auto &LI : LoopBB->liveins()) {
    for (MCRegUnit Unit : TRI->regunits(LI.PhysReg))
      UsedRUs.set(static_cast<unsigned>(Unit));
  }

  // Mark register units used by live-ins of all successors (live-outs).
  for (MachineBasicBlock *Succ : LoopBB->successors()) {
    for (const auto &LI : Succ->liveins()) {
      for (MCRegUnit Unit : TRI->regunits(LI.PhysReg))
        UsedRUs.set(static_cast<unsigned>(Unit));
    }
  }

  // Mark register units defined or used by any instruction in the loop.
  for (const MachineInstr &MI : *LoopBB) {
    if (MI.isDebugInstr())
      continue;
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();
      if (!Reg || !Reg.isPhysical())
        continue;
      for (MCRegUnit Unit : TRI->regunits(Reg))
        UsedRUs.set(static_cast<unsigned>(Unit));
    }
    // Handle regmasks.
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isRegMask())
        continue;
      for (unsigned i = 0, e = TRI->getNumRegs(); i != e; ++i) {
        if (MO.clobbersPhysReg(i)) {
          for (MCRegUnit Unit : TRI->regunits(MCRegister::from(i)))
            UsedRUs.set(static_cast<unsigned>(Unit));
        }
      }
    }
  }

  // Mark reserved registers.
  const BitVector &Reserved = MRI->getReservedRegs();
  for (unsigned Reg = 1, e = TRI->getNumRegs(); Reg < e; ++Reg) {
    if (Reserved.test(Reg)) {
      for (MCRegUnit Unit : TRI->regunits(MCRegister::from(Reg)))
        UsedRUs.set(static_cast<unsigned>(Unit));
    }
  }

  // Invert: free = not used.
  UsedRUs.flip();
  return UsedRUs;
}

MCRegister SILICMRegRename::findFreeReg(Register Def,
                                        const BitVector &FreeRUs) {
  // Get the minimal register class for Def.
  const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Def);
  if (!RC || !SIRegisterInfo::isVGPRClass(RC))
    return MCRegister();

  // Iterate through allocatable registers in this class.
  for (MCPhysReg CandReg : *RC) {
    if (CandReg == Def)
      continue;
    if (!MRI->isAllocatable(CandReg))
      continue;
    if (MRI->isReserved(CandReg))
      continue;
    if (!TRI->isVGPRPhysReg(CandReg))
      continue;

    // Check that all register units of the candidate are free.
    bool AllFree = true;
    for (MCRegUnit Unit : TRI->regunits(CandReg)) {
      if (!FreeRUs.test(static_cast<unsigned>(Unit))) {
        AllFree = false;
        break;
      }
    }
    if (AllFree)
      return CandReg;
  }

  return MCRegister();
}

bool SILICMRegRename::renameDefAndUsers(MachineInstr &MI, Register OldReg,
                                        MCRegister NewReg,
                                        MachineBasicBlock *LoopBB) {
  // Rewrite the def operand.
  bool DefRewritten = false;
  for (MachineOperand &MO : MI.operands()) {
    if (!MO.isReg() || !MO.isDef())
      continue;
    if (MO.getReg() == OldReg) {
      MO.setReg(NewReg);
      DefRewritten = true;
    }
  }
  if (!DefRewritten)
    return false;

  // Walk forward from MI, rewriting uses of OldReg to NewReg until we hit
  // another def of OldReg (which belongs to the non-invariant instruction).
  // Since this is a single-block loop, we also need to handle the wrap-around:
  // uses before MI in program order may also be reached by this def if the
  // other def appears between them and MI.
  //
  // Strategy for single-block loop:
  // 1. Find the other def of OldReg (the non-invariant one).
  // 2. Rewrite uses of OldReg that appear between MI (exclusive) and the
  //    other def (exclusive). These are the uses reached by MI's def.
  // 3. Also handle the wrap-around: if MI appears after the other def,
  //    uses from the other def to end-of-block + start-of-block to MI
  //    are reached by the other def (not MI). Uses from MI to end + start
  //    to other-def are reached by MI.

  // Find the other def.
  MachineInstr *OtherDef = nullptr;
  for (MachineInstr &Other : *LoopBB) {
    if (&Other == &MI)
      continue;
    if (Other.isDebugInstr())
      continue;
    for (const MachineOperand &MO : Other.operands()) {
      if (!MO.isReg() || !MO.isDef())
        continue;
      if (TRI->regsOverlap(MO.getReg(), OldReg)) {
        OtherDef = &Other;
        break;
      }
    }
    if (OtherDef)
      break;
  }

  if (!OtherDef) {
    // Shouldn't happen since we checked DefCount == 2, but be safe.
    // Revert the def rename.
    for (MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && MO.isDef() && MO.getReg() == NewReg)
        MO.setReg(OldReg);
    }
    return false;
  }

  // Determine which instructions are reached by MI's def (now NewReg)
  // vs the other def (still OldReg).
  //
  // In a single-block loop, the instruction order is cyclic. The uses
  // reached by MI's def are those between MI and OtherDef (going forward
  // with wrap-around).
  //
  // Collect instruction indices for MI and OtherDef.
  unsigned MIIdx = 0, OtherIdx = 0, Idx = 0;
  for (MachineInstr &I : *LoopBB) {
    if (&I == &MI)
      MIIdx = Idx;
    if (&I == OtherDef)
      OtherIdx = Idx;
    Idx++;
  }

  // Rewrite uses between MI and OtherDef (forward, wrapping around).
  // If MIIdx < OtherIdx: rewrite uses at positions (MIIdx, OtherIdx)
  // If MIIdx > OtherIdx: rewrite uses at positions (MIIdx, end] + [0, OtherIdx)
  Idx = 0;
  for (MachineInstr &I : *LoopBB) {
    if (&I == &MI || &I == OtherDef) {
      Idx++;
      continue;
    }

    bool InRange;
    if (MIIdx < OtherIdx) {
      InRange = (Idx > MIIdx && Idx < OtherIdx);
    } else {
      InRange = (Idx > MIIdx || Idx < OtherIdx);
    }

    if (InRange) {
      for (MachineOperand &MO : I.operands()) {
        if (!MO.isReg() || !MO.isUse())
          continue;
        if (MO.getReg() == OldReg)
          MO.setReg(NewReg);
      }
    }
    Idx++;
  }

  LLVM_DEBUG(dbgs() << "    Renamed def and users successfully\n");
  return true;
}

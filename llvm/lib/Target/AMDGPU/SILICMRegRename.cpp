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
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
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

  /// Compute registers that are free for renaming in the loop.
  /// free_regs = all_regs - bb_uses - live_through - reserved
  /// where bb_uses = defs | uses (excluding candidates),
  /// live_through = live_in - bb_uses.
  BitVector computeFreeRegUnits(
      MachineBasicBlock *LoopBB,
      const SmallPtrSetImpl<MachineInstr *> &CandidateMIs);

  /// Find a free register in the same register class as Def.
  MCRegister findFreeReg(Register Def, const BitVector &FreeRUs);

  /// Rename the def in MI and all its users reached before the next def
  /// of the same register. Returns true if renamed.
  bool renameDefAndUsers(MachineInstr &MI, Register OldReg, MCRegister NewReg,
                         MachineBasicBlock *LoopBB);

  // ---- MFMA AGPR accumulator rewrite ----

  /// An MFMA accumulator chain: a group of vgprcd MFMAs sharing the same
  /// persistent VGPR accumulator, plus a transient ping-pong temp VGPR.
  struct MFMAAccChain {
    Register PersistentVGPR; // 128-bit VGPR tuple live across loop iterations
    Register TempVGPR;       // 128-bit VGPR tuple used transiently in ping-pong
    MCRegister PersistentAGPR; // AGPR assigned for persistent VGPR (unique per chain)
    MCRegister TempAGPR;       // AGPR assigned for temp VGPR (shared across chains)
    SmallVector<MachineInstr *, 8> MFMAs;
  };

  /// Collect MFMA accumulator chains in the loop body.
  void collectMFMAAccChains(MachineBasicBlock *LoopBB, MachineLoop *Loop,
                            SmallVectorImpl<MFMAAccChain> &Chains);

  /// Find a free 128-bit AGPR block.
  MCRegister findFreeAGPRBlock(const BitVector &UsedRUs);

  /// Rewrite one MFMA chain from VGPR to AGPR accumulators.
  bool rewriteMFMAChainToAGPR(MFMAAccChain &Chain, MachineLoop *Loop);

  /// Canonicalize: replace temp VGPR with persistent VGPR in MFMAs
  /// (no opcode change, stays vgprcd). Frees the temp VGPR.
  bool canonicalizeMFMAChain(MFMAAccChain &Chain, MachineLoop *Loop);

  /// Compute register units used across the entire function.
  BitVector computeUsedRegUnitsAllBlocks();

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

  // MFMA AGPR accumulator rewrite: move VGPR accumulators to AGPRs to free
  // VGPRs for register renaming.
  if (ST->hasMAIInsts()) {
    SmallVector<MFMAAccChain, 8> Chains;
    collectMFMAAccChains(LoopBB, Loop, Chains);
    if (!Chains.empty()) {
      LLVM_DEBUG(dbgs() << "  Found " << Chains.size()
                        << " MFMA accumulator chain(s)\n");
      BitVector UsedRUs = computeUsedRegUnitsAllBlocks();

      // Phase 1: Assign AGPRs and rewrite chains that can go fully to AGPR.
      // Phase 2: Remaining chains get canonicalized (temp→persistent, no AGPR).
      DenseMap<Register, MCRegister> TempVGPRToAGPR;
      bool OutOfAGPRs = false;
      for (MFMAAccChain &Chain : Chains) {
        if (OutOfAGPRs) {
          // No more AGPRs — canonicalize: replace temp VGPR with persistent
          // VGPR in MFMAs (no opcode change, stays vgprcd). Frees temp VGPR.
          Changed |= canonicalizeMFMAChain(Chain, Loop);
          continue;
        }

        // Assign shared temp AGPR (reuse if same temp VGPR).
        auto It = TempVGPRToAGPR.find(Chain.TempVGPR);
        if (It != TempVGPRToAGPR.end()) {
          Chain.TempAGPR = It->second;
        } else {
          MCRegister TempAGPR = findFreeAGPRBlock(UsedRUs);
          if (!TempAGPR) {
            LLVM_DEBUG(dbgs() << "  No free AGPR for temp "
                              << printReg(Chain.TempVGPR, TRI) << "\n");
            OutOfAGPRs = true;
            Changed |= canonicalizeMFMAChain(Chain, Loop);
            continue;
          }
          TempVGPRToAGPR[Chain.TempVGPR] = TempAGPR;
          for (MCRegUnit U : TRI->regunits(TempAGPR))
            UsedRUs.set(static_cast<unsigned>(U));
          Chain.TempAGPR = TempAGPR;
        }

        // Assign unique persistent AGPR per chain.
        MCRegister PersAGPR = findFreeAGPRBlock(UsedRUs);
        if (!PersAGPR) {
          LLVM_DEBUG(dbgs() << "  No free AGPR for persistent "
                            << printReg(Chain.PersistentVGPR, TRI) << "\n");
          OutOfAGPRs = true;
          Changed |= canonicalizeMFMAChain(Chain, Loop);
          continue;
        }
        Chain.PersistentAGPR = PersAGPR;
        for (MCRegUnit U : TRI->regunits(PersAGPR))
          UsedRUs.set(static_cast<unsigned>(U));

        if (rewriteMFMAChainToAGPR(Chain, Loop))
          Changed = true;
      }
    }
  }

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

  // Step 4: Compute free register units, excluding candidate instructions
  // (which will be hoisted out of the loop after renaming).
  SmallPtrSet<MachineInstr *, 8> CandidateMIs;
  for (auto &C : Candidates)
    CandidateMIs.insert(C.MI);
  BitVector FreeRUs = computeFreeRegUnits(LoopBB, CandidateMIs);

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

  // Check that there are at least 2 defs of FoundDef's register units:
  // this one (invariant) and at least one other (non-invariant).
  // Count defs of any of FoundDef's units.
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

BitVector SILICMRegRename::computeFreeRegUnits(
    MachineBasicBlock *LoopBB,
    const SmallPtrSetImpl<MachineInstr *> &CandidateMIs) {
  // Compute free registers using the formula:
  //   bb_uses     = defs | uses  (excluding candidate instructions)
  //   live_through = live_in - bb_uses
  //   free_regs   = all_regs - bb_uses - live_through - reserved
  // This simplifies to: free_regs = all_regs - (bb_uses | live_in) - reserved

  // Compute bb_uses: all registers defined or used in the BB, excluding
  // candidate instructions (which will be hoisted out after renaming).
  BitVector BBUses(NumRegUnits);
  for (const MachineInstr &MI : *LoopBB) {
    if (MI.isDebugInstr())
      continue;
    if (CandidateMIs.count(const_cast<MachineInstr *>(&MI)))
      continue;
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg())
        continue;
      Register Reg = MO.getReg();
      if (!Reg || !Reg.isPhysical())
        continue;
      for (MCRegUnit Unit : TRI->regunits(Reg))
        BBUses.set(static_cast<unsigned>(Unit));
    }
  }

  // Compute live_in.
  BitVector LiveIn(NumRegUnits);
  for (const auto &LI : LoopBB->liveins()) {
    for (MCRegUnit Unit : TRI->regunits(LI.PhysReg))
      LiveIn.set(static_cast<unsigned>(Unit));
  }

  // UsedRUs = bb_uses | live_in
  BitVector UsedRUs = BBUses;
  UsedRUs |= LiveIn;

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
  // If MIIdx < OtherIdx: rewrite uses at positions (MIIdx, OtherIdx]
  // If MIIdx > OtherIdx: rewrite uses at positions (MIIdx, end] + [0, OtherIdx]
  // Note: OtherDef is INCLUDED for use-operand renaming because it may read
  // OldReg as an input (e.g. V_CNDMASK_B32 that both reads and writes $vgpr7).
  // Only its def operand must remain as OldReg.
  Idx = 0;
  for (MachineInstr &I : *LoopBB) {
    if (&I == &MI) {
      Idx++;
      continue;
    }

    bool InRange;
    if (MIIdx < OtherIdx) {
      InRange = (Idx > MIIdx && Idx <= OtherIdx);
    } else {
      InRange = (Idx > MIIdx || Idx <= OtherIdx);
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

// ---- MFMA AGPR accumulator rewrite implementation ----

void SILICMRegRename::collectMFMAAccChains(
    MachineBasicBlock *LoopBB, MachineLoop *Loop,
    SmallVectorImpl<MFMAAccChain> &Chains) {
  // Group MFMAs by their unordered register pair {dst, src2}.
  // Each unique pair forms a separate chain, even if they share one register
  // (e.g. the temp VGPR in ping-pong patterns).
  DenseMap<std::pair<unsigned, unsigned>, unsigned> PairToChain;

  for (MachineInstr &MI : *LoopBB) {
    if (AMDGPU::getMFMASrcCVDstAGPROp(MI.getOpcode()) == -1)
      continue;

    MachineOperand *Dst = TII->getNamedOperand(MI, AMDGPU::OpName::vdst);
    MachineOperand *Src2 = TII->getNamedOperand(MI, AMDGPU::OpName::src2);
    if (!Dst || !Src2 || !Dst->isReg() || !Src2->isReg())
      continue;

    Register DstReg = Dst->getReg();
    Register Src2Reg = Src2->getReg();
    if (!DstReg.isPhysical() || !Src2Reg.isPhysical())
      continue;

    // Normalize the pair so {A,B} == {B,A}.
    unsigned Lo = std::min(DstReg.id(), Src2Reg.id());
    unsigned Hi = std::max(DstReg.id(), Src2Reg.id());
    auto Key = std::make_pair(Lo, Hi);

    auto It = PairToChain.find(Key);
    unsigned ChainIdx;
    if (It != PairToChain.end()) {
      ChainIdx = It->second;
    } else {
      ChainIdx = Chains.size();
      Chains.push_back({});
      PairToChain[Key] = ChainIdx;
    }
    Chains[ChainIdx].MFMAs.push_back(&MI);
  }

  // Determine persistent vs temp VGPR for each chain.
  // Persistent = live-in to an exit block (used in epilogue).
  SmallVector<MachineBasicBlock *, 4> ExitBlocks;
  Loop->getExitBlocks(ExitBlocks);

  // Collect all live-in regs of exit blocks.
  DenseSet<MCRegister> ExitLiveIns;
  for (MachineBasicBlock *ExitBB : ExitBlocks)
    for (const auto &LI : ExitBB->liveins())
      ExitLiveIns.insert(LI.PhysReg);

  // For each chain, find which registers it uses and classify them.
  SmallVector<MFMAAccChain, 8> ValidChains;
  for (auto &Chain : Chains) {
    if (Chain.MFMAs.empty())
      continue;

    // Collect all unique VGPR tuples used as dst/src2.
    DenseSet<Register> ChainRegs;
    for (MachineInstr *MI : Chain.MFMAs) {
      MachineOperand *Dst = TII->getNamedOperand(*MI, AMDGPU::OpName::vdst);
      MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);
      ChainRegs.insert(Dst->getReg());
      ChainRegs.insert(Src2->getReg());
    }

    // Classify: persistent = in exit live-ins, temp = the other.
    Register Persistent, Temp;
    for (Register R : ChainRegs) {
      if (ExitLiveIns.count(R))
        Persistent = R;
      else
        Temp = R;
    }

    // Need exactly one persistent and one temp.
    if (!Persistent || !Temp)
      continue;

    // Verify that the persistent VGPR is not used by any non-chain instruction
    // in the loop body. If it is (e.g., as src0/src1 in other MFMAs, or read
    // by V_CVT_PK, DS_WRITE, etc.), we cannot safely rewrite to AGPR because
    // those instructions would read stale VGPR values.
    DenseSet<MachineInstr *> ChainMIs(Chain.MFMAs.begin(), Chain.MFMAs.end());
    bool PersistentUsedElsewhere = false;
    for (const MachineInstr &MI : *LoopBB) {
      if (ChainMIs.count(const_cast<MachineInstr *>(&MI)))
        continue;
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;
        Register Reg = MO.getReg();
        if (!Reg || !Reg.isPhysical())
          continue;
        if (TRI->regsOverlap(Reg, Persistent)) {
          PersistentUsedElsewhere = true;
          break;
        }
      }
      if (PersistentUsedElsewhere)
        break;
    }
    if (PersistentUsedElsewhere)
      continue;

    Chain.PersistentVGPR = Persistent;
    Chain.TempVGPR = Temp;
    ValidChains.push_back(std::move(Chain));
  }

  Chains = std::move(ValidChains);
}

BitVector SILICMRegRename::computeUsedRegUnitsAllBlocks() {
  BitVector Used(NumRegUnits);
  for (MachineBasicBlock &MBB : *MF) {
    for (const MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;
        Register Reg = MO.getReg();
        if (!Reg || !Reg.isPhysical())
          continue;
        for (MCRegUnit U : TRI->regunits(Reg))
          Used.set(static_cast<unsigned>(U));
      }
    }
  }
  const BitVector &Reserved = MRI->getReservedRegs();
  for (unsigned Reg = 1, e = TRI->getNumRegs(); Reg < e; ++Reg) {
    if (Reserved.test(Reg)) {
      for (MCRegUnit U : TRI->regunits(MCRegister::from(Reg)))
        Used.set(static_cast<unsigned>(U));
    }
  }
  return Used;
}

MCRegister SILICMRegRename::findFreeAGPRBlock(const BitVector &UsedRUs) {
  const TargetRegisterClass *RC = TRI->getAGPRClassForBitWidth(128);
  if (!RC)
    return MCRegister();

  for (MCPhysReg CandReg : *RC) {
    if (MRI->isReserved(CandReg))
      continue;
    bool AllFree = true;
    for (MCRegUnit U : TRI->regunits(CandReg)) {
      if (UsedRUs.test(static_cast<unsigned>(U))) {
        AllFree = false;
        break;
      }
    }
    if (AllFree)
      return CandReg;
  }
  return MCRegister();
}

bool SILICMRegRename::rewriteMFMAChainToAGPR(MFMAAccChain &Chain,
                                              MachineLoop *Loop) {
  MCRegister PersAGPR = Chain.PersistentAGPR;
  MCRegister TempAGPR = Chain.TempAGPR;
  MachineBasicBlock *LoopBB = Loop->getHeader();
  MachineBasicBlock *Preheader = Loop->getLoopPreheader();

  LLVM_DEBUG(dbgs() << "  Rewriting MFMA chain: "
                    << printReg(Chain.PersistentVGPR, TRI) << " -> "
                    << printReg(PersAGPR, TRI) << ", "
                    << printReg(Chain.TempVGPR, TRI) << " -> "
                    << printReg(TempAGPR, TRI) << "\n");

  // Step A: Rewrite MFMA instructions in the loop.
  for (MachineInstr *MI : Chain.MFMAs) {
    int NewOpc = AMDGPU::getMFMASrcCVDstAGPROp(MI->getOpcode());
    if (NewOpc == -1)
      continue;

    // Get operands BEFORE changing the opcode, since named operand indices
    // may differ between the vgprcd and AGPR variants.
    MachineOperand *Dst = TII->getNamedOperand(*MI, AMDGPU::OpName::vdst);
    MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);

    MI->setDesc(TII->get(NewOpc));

    // Map persistent VGPR → PersistentAGPR, temp VGPR → TempAGPR.
    if (Dst && Dst->isReg()) {
      if (Dst->getReg() == Chain.PersistentVGPR)
        Dst->setReg(PersAGPR);
      else if (Dst->getReg() == Chain.TempVGPR)
        Dst->setReg(TempAGPR);
    }
    if (Src2 && Src2->isReg()) {
      if (Src2->getReg() == Chain.PersistentVGPR)
        Src2->setReg(PersAGPR);
      else if (Src2->getReg() == Chain.TempVGPR)
        Src2->setReg(TempAGPR);
    }
  }

  // Step B: Rewrite preheader zero-init instructions for persistent VGPR.
  // The temp VGPR is transient (defined inside the loop), no preheader init.
  const TargetRegisterClass *PersRC =
      TRI->getMinimalPhysRegClass(Chain.PersistentVGPR);
  unsigned RegSize = TRI->getRegSizeInBits(*PersRC) / 32;

  for (unsigned i = 0; i < RegSize; ++i) {
    unsigned SubIdx = SIRegisterInfo::getSubRegFromChannel(i, 1);
    MCRegister VGPRSub = TRI->getSubReg(Chain.PersistentVGPR, SubIdx);
    MCRegister AGPRSub = TRI->getSubReg(PersAGPR, SubIdx);

    for (MachineInstr &MI : *Preheader) {
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.isDef())
          continue;
        if (MO.getReg() != VGPRSub)
          continue;

        // Insert V_ACCVGPR_WRITE_B32_e64 after the instruction to copy the
        // VGPR result into the AGPR. Keep the original instruction intact
        // because the persistent VGPR may still be used elsewhere (e.g., as
        // src0/src1 in other MFMA instructions).
        {
          auto InsertPt = std::next(MachineBasicBlock::iterator(MI));
          BuildMI(*Preheader, InsertPt, MI.getDebugLoc(),
                  TII->get(AMDGPU::V_ACCVGPR_WRITE_B32_e64), AGPRSub)
              .addReg(VGPRSub);
        }
        break;
      }
    }
  }
  // Step C: Insert AGPR→VGPR copies for persistent VGPR in exit blocks.
  // Temp VGPR doesn't need epilogue copies (transient).
  SmallVector<MachineBasicBlock *, 4> ExitBlocks;
  Loop->getExitBlocks(ExitBlocks);

  for (MachineBasicBlock *ExitBB : ExitBlocks) {
    MachineBasicBlock::iterator InsertPt = ExitBB->begin();
    while (InsertPt != ExitBB->end() && InsertPt->isPHI())
      ++InsertPt;

    for (unsigned i = 0; i < RegSize; ++i) {
      unsigned SubIdx = SIRegisterInfo::getSubRegFromChannel(i, 1);
      MCRegister VGPRSub = TRI->getSubReg(Chain.PersistentVGPR, SubIdx);
      MCRegister AGPRSub = TRI->getSubReg(PersAGPR, SubIdx);

      BuildMI(*ExitBB, InsertPt, DebugLoc(),
              TII->get(AMDGPU::V_ACCVGPR_READ_B32_e64), VGPRSub)
          .addReg(AGPRSub);
    }

    ExitBB->addLiveIn(PersAGPR);
    ExitBB->sortUniqueLiveIns();
  }

  // Step D: Update loop block live-ins.
  // Keep the persistent VGPR as live-in — it may still be used as src0/src1
  // in other MFMA instructions (e.g. A/B matrix input).
  LoopBB->addLiveIn(PersAGPR);
  // Note: TempAGPR is transient (defined and killed within each iteration),
  // so it is NOT live across the back-edge and must not be added as a live-in.
  LoopBB->sortUniqueLiveIns();

  return true;
}

bool SILICMRegRename::canonicalizeMFMAChain(MFMAAccChain &Chain,
                                             MachineLoop *Loop) {
  LLVM_DEBUG(dbgs() << "  Canonicalizing MFMA chain: "
                    << printReg(Chain.TempVGPR, TRI) << " -> "
                    << printReg(Chain.PersistentVGPR, TRI) << "\n");

  // Replace temp VGPR with persistent VGPR in all MFMAs.
  // No opcode change — stays vgprcd. Eliminates the ping-pong.
  for (MachineInstr *MI : Chain.MFMAs) {
    MachineOperand *Dst = TII->getNamedOperand(*MI, AMDGPU::OpName::vdst);
    MachineOperand *Src2 = TII->getNamedOperand(*MI, AMDGPU::OpName::src2);

    if (Dst && Dst->isReg() && Dst->getReg() == Chain.TempVGPR) {
      Dst->setReg(Chain.PersistentVGPR);
      Dst->setIsRenamable(true);
    }
    if (Src2 && Src2->isReg() && Src2->getReg() == Chain.TempVGPR) {
      Src2->setReg(Chain.PersistentVGPR);
      Src2->setIsRenamable(true);
    }
  }

  return true;
}

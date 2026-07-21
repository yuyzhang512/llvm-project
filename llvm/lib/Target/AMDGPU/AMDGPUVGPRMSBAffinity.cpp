//===- AMDGPUVGPRMSBAffinity.cpp - bias VGPR alloc into 256-VGPR MSB groups ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// On gfx1250 a wave may use up to 1024 VGPRs, but an instruction can only
/// directly address VGPRs 0-255. VGPRs 256-1023 are reached by setting the
/// per-operand-slot MSB bits via S_SET_VGPR_MSB. Whenever consecutive
/// instructions need a different {src0,src1,src2,dst} MSB group configuration the
/// AMDGPULowerVGPREncoding pass must emit an S_SET_VGPR_MSB.
///
/// This pass runs before register allocation (after the pre-RA scheduler has
/// fixed the instruction order). It does not change code; it records, per
/// virtual register, a desired MSB group (high bits of the HW index, index >> 8).
/// SIRegisterInfo's allocation hint hook then biases the greedy allocator
/// toward that MSB group.
///
/// Algorithm (general, schedule-driven). The number of S_SET_VGPR_MSB
/// instructions equals, per MSB slot, how often the MSB group of the value in that
/// slot changes along the scheduled stream. We model this directly:
///
///   1. Walk the scheduled MIR. For each instruction, map its VGPR operands to
///      the four MSB slots using the same table AMDGPULowerVGPREncoding uses
///      (getVGPRLoweringOperandTables). Slots a given instruction does not
///      constrain stay "sticky" (unchanged), matching the lowering semantics.
///      For each slot, when the value changes from the previously seen vreg to
///      a new vreg, those two vregs would cause a mode switch unless they share
///      a MSB group -- so we add an affinity edge between them, weighted by loop
///      depth (block frequency proxy).
///
///   2. Partition the affinity graph into at most four MSB groups by greedy
///      union-find: merge the highest-weight edges first, refusing a merge when
///      the merged cluster's simultaneously-live footprint would exceed one
///      MSB group (256 dwords, from LiveIntervals). Capacity is what forces the
///      4-way split: high internal affinity holds a region's accumulators and
///      its co-scheduled DS-load tile together, while the lighter cross-region
///      edges become the cut.
///
///   3. Pack the resulting clusters into four MSB groups (hottest first, least
///      loaded MSB group), and record each vreg's MSB group.
///
/// This recovers a hand-designed MSB group layout (e.g. accumulator co-located with
/// the same-region DS-load destination, sources in the other MSB groups) when the
/// kernel has that structure, without assuming any particular tiling or
/// requiring scheduler region markers.
///
/// The hint is soft: if a MSB group fills up the allocator falls back to the rest of
/// the order, so this can never make allocation fail.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/InitializePasses.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <functional>
#include <queue>
#include <string>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-vgpr-msb-affinity"

static cl::opt<bool> EnableVGPRMSBAffinity(
    "amdgpu-vgpr-msb-affinity", cl::Hidden, cl::init(false),
    cl::desc("Bias VGPR allocation into 256-VGPR MSB groups to reduce "
             "S_SET_VGPR_MSB insertions (gfx1250)"));

// Experimental structural GEMM planner (default off). Reconstructs loop regions
// from the LLIR-scheduled instruction order (run with the pre-RA scheduler off so
// that order is preserved) and assigns banks by role: accumulator -> region %
// groups; a ds_load tile -> its loading region's bank (data_bank); a ds_load
// address -> its region's src0 bank. Replaces the affinity path only when enabled;
// off by default, so all other cases are byte-identical.
static cl::opt<bool> RegionDataFlow(
    "amdgpu-vgpr-msb-affinity-region-dataflow", cl::Hidden, cl::init(false),
    cl::desc("Structural region + data_bank GEMM planner (experimental)"));

// Hint clusters with weight > MaxClusterWeight / HotClusterDiv; colder clusters
// are left unhinted so the allocator packs them naturally.
static cl::opt<unsigned>
    HotClusterDiv("amdgpu-vgpr-msb-affinity-hot-div", cl::Hidden, cl::init(4),
                  cl::desc("Hot-cluster filter divisor"));

// Scale an affinity edge by the operand width (dwords), capped at this value, so a
// wide value that alternates in a slot outweighs a scalar. 0/1 = off.
static cl::opt<unsigned>
    TupleWeight("amdgpu-vgpr-msb-affinity-tuple-weight", cl::Hidden, cl::init(8),
                cl::desc("Cap for edge width-weighting (0/1=off)"));

// Isolation exponent for the boundary cost. Co-locating a slot removes a switch
// only if that slot is the sole changer at its boundary; weight/k^exp with exp=2
// makes the clusterer prefer isolated (single-slot) boundaries over batched ones.
// Base 144 = 12^2 keeps per-slot weights integral for k in 1..4.
static cl::opt<unsigned>
    BoundaryIsoExp("amdgpu-vgpr-msb-affinity-boundary-iso-exp", cl::Hidden,
                   cl::init(1),
                   cl::desc("Boundary-cost isolation exponent (weight/k^exp)"));

// Self-benefit gate: commit the plan only if its predicted (freq-weighted) switch
// count is below this percent of the predicted count under the natural no-hint
// layout, guarding against an already-coherent schedule where a competing
// partition would raise the count. 0 disables the gate.
static cl::opt<unsigned> BenefitPct(
    "amdgpu-vgpr-msb-affinity-benefit-pct", cl::Hidden, cl::init(75),
    cl::desc("Commit only if predicted plan switches < this % of predicted "
             "no-hint switches (self-benefit gate; 0 disables the gate)"));

// Boost the src0-slot (a ds_read address) edge at a ds_read boundary by the
// ds_read's dst width, capped at this value. src0 is narrow, so width-weighting
// under-weights it; scaling by the dst length lets the address share a group with
// its neighbours so no s_set_vgpr_msb lands on src0. 1 = off.
static cl::opt<unsigned>
    Src0DsWeight("amdgpu-vgpr-msb-affinity-ds-src0-weight", cl::Hidden,
                 cl::init(2),
                 cl::desc("Cap for the dst-length src0 boost at a ds_read"));

// Apply the src0 boost only when the ds_read boundary changes at most this many
// (non-dst) slots. Co-locating src0 removes a switch only if src0 is near the sole
// changer; on a batched boundary the other slots still change, so the boost only
// perturbs the clustering.
static cl::opt<unsigned>
    Src0IsoMax("amdgpu-vgpr-msb-affinity-ds-src0-iso-max", cl::Hidden,
               cl::init(1),
               cl::desc("Max changed slots at which the src0 boost applies"));

// Skip when the naive baseline switch weight is below this. A small baseline means
// the loop is already near-coherent (real RA keeps each slot in one group); there
// the linear-scan baseline over-estimates and the plan's predicted win does not
// survive real allocation, so committing would regress. Only ever skips, so it
// cannot add a regression.
static cl::opt<unsigned>
    MinBaseSwitch("amdgpu-vgpr-msb-affinity-min-base-switch", cl::Hidden,
                  cl::init(500),
                  cl::desc("Skip when naive baseline switch weight is below this"));

namespace {

constexpr unsigned MSBGroupSize = 256;
constexpr unsigned NumMSBGroups = 4;
// Skip the plan when a group's planned load exceeds this percent of its cap.
// Mild overflow is realizable (RA spills a few, most hints honored); severe
// overflow is not.
constexpr unsigned OverflowPct = 125;

class AMDGPUVGPRMSBAffinity {
public:
  bool run(MachineFunction &MF, LiveIntervals *LIS, MachineLoopInfo *MLI);

private:
  // Experimental structural GEMM planner (see RegionDataFlow). Reconstructs
  // regions from the (LLIR-scheduled) order and hints banks by role. Returns true
  // if it committed any hint (so run() knows the structural path handled the fn).
  bool planDataFlowRegions(MachineFunction &MF, unsigned EffMSBGroups,
                           SIMachineFunctionInfo *MFI);

  // Build the affinity graph, cluster, pack into MSB groups and commit hints for one
  // region (a set of blocks). Vregs already in \p Assigned (hinted by a hotter
  // region) are skipped; newly hinted vregs are added to it.
  void processRegion(ArrayRef<MachineBasicBlock *> Blocks,
                     ArrayRef<Register> AllVGPRs, unsigned EffMSBGroups,
                     unsigned VGPRBudget, bool Balance, bool LoopGate,
                     DenseSet<unsigned> &Assigned, SIMachineFunctionInfo *MFI);

  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  LiveIntervals *LIS = nullptr;
  MachineLoopInfo *MLI = nullptr;
  const GCNSubtarget *STI = nullptr;

  // Per-block execution weight used to scale slot-transition edges: a loop-depth
  // proxy for trip count, so an innermost-loop transition outweighs straight-line
  // code by orders of magnitude.
  uint64_t blockFreq(const MachineBasicBlock &MBB) const {
    unsigned Depth = MLI ? MLI->getLoopDepth(&MBB) : 0;
    return 1ull << std::min(4u * Depth, 40u);
  }

  // Value-group union-find: vregs connected by a tied def/use pair are the same
  // value and coalesce to one physical register, so they must not be double-
  // counted in the MSB group pressure. (General COPYs are deliberately not unioned --
  // see buildValueGroups -- so loop-carried values connected only by a PHI/COPY
  // are not coalesced here.) Mutable for path compression.
  mutable SmallVector<unsigned, 0> VGParent;

  bool isVGPRVirtReg(Register Reg) const {
    return Reg.isVirtual() && TRI->isVGPRClass(MRI->getRegClass(Reg));
  }

  unsigned dwords(Register Reg) const {
    // Integer-divide by 32: a 16-bit (VGPR_16) vreg yields 0 here. Rounding up
    // was tried and reverted -- it overcounts lo16/hi16 pairs (two 16-bit vregs
    // share one physical dword) and perturbs the clustering into worse plans on
    // True16 kernels. The footprint is only used for soft hints with a
    // severe-overflow safety net, so the 16-bit undercount is acceptable.
    return TRI->getRegSizeInBits(*MRI->getRegClass(Reg)) / 32;
  }

  // Record both the MSB group affinity (for the hint hook) and a concrete
  // register hint to a representative physreg in the group. The latter sets
  // VirtRegMap::hasKnownPreference, which the greedy allocator's priority boosts,
  // so these vregs are colored earlier and claim their group before contended
  // values fill it. Existing (copy-coalescing) hints are preserved.
  void recordMSB(SIMachineFunctionInfo *MFI, Register Reg, unsigned MSB) {
    MFI->setVGPRMSBAffinity(Reg, MSB);
    if (MRI->getRegAllocationHint(Reg).second)
      return;
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    for (MCPhysReg P : *RC) {
      if (!MRI->isReserved(P) && (TRI->getHWRegIndex(P) >> 8) == MSB) {
        MRI->setRegAllocationHint(Reg, 0, P);
        return;
      }
    }
  }

  unsigned vgFind(unsigned X) const {
    while (VGParent[X] != X) {
      VGParent[X] = VGParent[VGParent[X]];
      X = VGParent[X];
    }
    return X;
  }

  void buildValueGroups(MachineFunction &MF) {
    unsigned N = MRI->getNumVirtRegs();
    VGParent.resize(N);
    for (unsigned I = 0; I < N; ++I)
      VGParent[I] = I;
    auto uni = [&](Register A, Register B) {
      if (!isVGPRVirtReg(A) || !isVGPRVirtReg(B))
        return;
      unsigned ra = vgFind(A.virtRegIndex()), rb = vgFind(B.virtRegIndex());
      if (ra != rb)
        VGParent[ra] = rb;
    };
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB) {
        // Coalesce tied def/use pairs (e.g. the WMMA accumulator src2 tied to
        // dst). General COPYs are intentionally *not* unioned: they connect
        // distinct values and would collapse unrelated footprints.
        for (unsigned I = 0, E = MI.getNumOperands(); I < E; ++I) {
          const MachineOperand &MO = MI.getOperand(I);
          if (MO.isReg() && MO.isUse() && MO.isTied()) {
            unsigned DefIdx = MI.findTiedOperandIdx(I);
            const MachineOperand &Def = MI.getOperand(DefIdx);
            if (Def.isReg())
              uni(MO.getReg(), Def.getReg());
          }
        }
        // Coalesce the matrix accumulator chain dst <- src2. src2 is the
        // accumulator-in and dst the accumulator-out of the *same* logical
        // accumulator; across an unrolled K-loop this chains acc0->acc1->...
        // into one value group, so the footprint counts the accumulator once
        // instead of once per unrolled step (even when the operands are not
        // marked tied at this point). Different output tiles use disjoint
        // chains, so this never merges unrelated accumulators.
        if (SIInstrInfo::isWMMA(MI) || TII->isMAI(MI)) {
          const MachineOperand *D = TII->getNamedOperand(MI, AMDGPU::OpName::vdst);
          const MachineOperand *S2 =
              TII->getNamedOperand(MI, AMDGPU::OpName::src2);
          if (D && D->isReg() && S2 && S2->isReg())
            uni(D->getReg(), S2->getReg());
        }
      }
    }
  }

  // Maximum number of VGPR dwords from \p Regs simultaneously live. Live ranges
  // of vregs in the same value group are merged first so a coalescing value is
  // counted once -- this is the accurate MSB group pressure (cf. PresCount).
  unsigned maxSimultaneousDwords(ArrayRef<Register> Regs) const {
    DenseMap<unsigned, SmallVector<std::pair<SlotIndex, SlotIndex>, 2>> ByGroup;
    DenseMap<unsigned, int> GroupSize;
    for (Register Reg : Regs) {
      if (!LIS->hasInterval(Reg))
        continue;
      unsigned G = vgFind(Reg.virtRegIndex());
      GroupSize[G] = dwords(Reg);
      auto &Segs = ByGroup[G];
      for (const LiveRange::Segment &S : LIS->getInterval(Reg))
        Segs.emplace_back(S.start, S.end);
    }
    SmallVector<std::pair<SlotIndex, int>, 64> Events;
    for (auto &[G, Segs] : ByGroup) {
      llvm::sort(Segs);
      int Sz = GroupSize[G];
      SlotIndex CurS, CurE;
      bool Open = false;
      auto flush = [&] {
        Events.emplace_back(CurS, Sz);
        Events.emplace_back(CurE, -Sz);
      };
      for (auto &[S, E] : Segs) {
        if (!Open) {
          CurS = S;
          CurE = E;
          Open = true;
        } else if (S <= CurE) {
          if (CurE < E)
            CurE = E;
        } else {
          flush();
          CurS = S;
          CurE = E;
        }
      }
      if (Open)
        flush();
    }
    llvm::sort(Events, [](const std::pair<SlotIndex, int> &A,
                          const std::pair<SlotIndex, int> &B) {
      return A.first < B.first || (A.first == B.first && A.second < B.second);
    });
    int Cur = 0, Max = 0;
    for (auto &[Idx, Delta] : Events) {
      Cur += Delta;
      Max = std::max(Max, Cur);
    }
    return Max;
  }

  // Natural (no-hint) MSB group assignment used as the self-benefit baseline: a
  // linear scan over EffMSBGroups*256 columns in first-definition order, giving each
  // vreg the lowest free column run and freeing columns when a vreg's live range
  // ends. This approximates what the allocator does without our hints -- vregs
  // close together in the schedule land in the same MSB group -- so on an already
  // MSB-coherent schedule it yields few switches (and our plan must beat it).
  DenseMap<unsigned, int> computeNaiveMSB(ArrayRef<Register> Regs,
                                            unsigned EffMSBGroups) const {
    DenseMap<unsigned, int> MSB;
    const unsigned Cols = EffMSBGroups * MSBGroupSize;
    SmallVector<Register, 0> Order(Regs.begin(), Regs.end());
    llvm::stable_sort(Order, [&](Register A, Register B) {
      return LIS->getInterval(A).beginIndex() < LIS->getInterval(B).beginIndex();
    });
    SmallVector<bool, 0> Free(Cols, true);
    // Active allocations: (endIndex, startCol, width) to reclaim columns.
    SmallVector<std::tuple<SlotIndex, unsigned, unsigned>, 0> Active;
    for (Register R : Order) {
      SlotIndex Begin = LIS->getInterval(R).beginIndex();
      // Reclaim columns of ranges that ended before this def.
      for (unsigned I = 0; I < Active.size();) {
        if (std::get<0>(Active[I]) <= Begin) {
          unsigned S = std::get<1>(Active[I]), W = std::get<2>(Active[I]);
          for (unsigned C = S; C < S + W; ++C)
            Free[C] = true;
          Active[I] = Active.back();
          Active.pop_back();
        } else
          ++I;
      }
      unsigned D = dwords(R);
      // Lowest free run of D columns.
      int Start = -1;
      for (unsigned C = 0, Run = 0; C < Cols; ++C) {
        Run = Free[C] ? Run + 1 : 0;
        if (Run == D) {
          Start = (int)(C + 1 - D);
          break;
        }
      }
      int B;
      if (Start < 0) {
        // No contiguous run fits: reserve D columns in the least-occupied MSB group
        // so this vreg's footprint stays visible (otherwise the baseline looks
        // artificially uncongested and skews the self-benefit comparison).
        unsigned BestMSB = 0, BestFree = 0;
        for (unsigned Bk = 0; Bk < EffMSBGroups; ++Bk) {
          unsigned F = 0;
          for (unsigned C = Bk * MSBGroupSize; C < (Bk + 1) * MSBGroupSize; ++C)
            F += Free[C];
          if (F >= BestFree) {
            BestFree = F;
            BestMSB = Bk;
          }
        }
        for (unsigned C = BestMSB * MSBGroupSize, Reserved = 0;
             C < (BestMSB + 1) * MSBGroupSize && Reserved < D; ++C)
          if (Free[C]) {
            Free[C] = false;
            ++Reserved;
          }
        B = (int)BestMSB;
      } else {
        for (unsigned C = Start; C < Start + D; ++C)
          Free[C] = false;
        Active.emplace_back(LIS->getInterval(R).endIndex(), (unsigned)Start, D);
        B = Start / (int)MSBGroupSize;
      }
      MSB[R.virtRegIndex()] = B;
    }
    return MSB;
  }

  // Predicted freq-weighted s_set_vgpr_msb count for a given vreg->MSB group map,
  // simulated the same way AMDGPULowerVGPREncoding counts: walk the scheduled
  // stream with sticky per-slot MSB group state (reset per block), and charge the
  // block frequency once per instruction that needs any slot's MSB group to change.
  uint64_t simSwitchWeight(ArrayRef<MachineBasicBlock *> Blocks,
                           function_ref<int(Register)> msbOf,
                           bool LoopOnly = false) const {
    uint64_t Sw = 0;
    for (MachineBasicBlock *MBBp : Blocks) {
      MachineBasicBlock &MBB = *MBBp;
      // Realizability/relevance: only in-loop switches recur every iteration and
      // dominate runtime cost; prologue/epilogue switches fire once. Scoring the
      // gate on loop blocks only keeps the plan from trading a loop win for
      // one-time out-of-loop churn (which the whole-function total misranks).
      if (LoopOnly && (!MLI || MLI->getLoopDepth(&MBB) == 0))
        continue;
      uint64_t Freq = blockFreq(MBB);
      // Mode is reset to group 0 at a block header (and again at a call /
      // terminator / VGPR inline asm), matching AMDGPULowerVGPREncoding.
      int Last[4] = {0, 0, 0, 0};
      for (MachineInstr &MI : MBB) {
        if (MI.isMetaInstruction())
          continue;
        if (MI.isTerminator() || MI.isCall() ||
            (MI.isInlineAsm() && TII->hasVGPRUses(MI))) {
          Last[0] = Last[1] = Last[2] = Last[3] = 0;
          continue;
        }
        auto Ops = AMDGPU::getVGPRLoweringOperandTables(MI.getDesc());
        if (!Ops.first)
          continue;
        int Need[4] = {-1, -1, -1, -1};
        for (unsigned S = 0; S < 4; ++S) {
          const MachineOperand *MO = TII->getNamedOperand(MI, Ops.first[S]);
          if ((!MO || !MO->isReg() || !MO->getReg()) && Ops.second)
            MO = TII->getNamedOperand(MI, Ops.second[S]);
          if (!MO || !MO->isReg() || !MO->getReg())
            continue;
          Register R = MO->getReg();
          if (isVGPRVirtReg(R))
            Need[S] = std::max(0, msbOf(R));
          else if (R.isPhysical() && TRI->isVGPR(*MRI, R))
            Need[S] = (int)(TRI->getHWRegIndex(R) >> 8);
        }
        bool Changed = false;
        for (unsigned S = 0; S < 4; ++S)
          if (Need[S] >= 0 && Last[S] != Need[S])
            Changed = true;
        if (Changed)
          Sw += Freq;
        for (unsigned S = 0; S < 4; ++S)
          if (Need[S] >= 0)
            Last[S] = Need[S];
      }
    }
    return Sw;
  }
};

class AMDGPUVGPRMSBAffinityLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUVGPRMSBAffinityLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    auto *LISW = getAnalysisIfAvailable<LiveIntervalsWrapperPass>();
    auto *MLIW = getAnalysisIfAvailable<MachineLoopInfoWrapperPass>();
    return AMDGPUVGPRMSBAffinity().run(
        MF, LISW ? &LISW->getLIS() : nullptr,
        MLIW ? &MLIW->getLI() : nullptr);
  }

  StringRef getPassName() const override { return "AMDGPU VGPR MSB Affinity"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    // The default path only adds allocation hints (no IR change). The
    // experimental region-dataflow path may insert preheader COPYs to split
    // shared addresses, so it cannot preserve analyses.
    if (!RegionDataFlow)
      AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // namespace

bool AMDGPUVGPRMSBAffinity::run(MachineFunction &MF, LiveIntervals *LISIn,
                                 MachineLoopInfo *MLIIn) {
  if (!EnableVGPRMSBAffinity)
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!ST.has1024AddressableVGPRs() || !LISIn)
    return false;

  // Only steer compute kernels; graphics shaders are out of scope for the
  // 1024-VGPR / s_set_vgpr_msb MSB grouping this pass targets.
  if (!AMDGPU::isCompute(MF.getFunction().getCallingConv()))
    return false;

  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();
  LIS = LISIn;
  MLI = MLIIn;
  STI = &ST;

  LLVM_DEBUG(dbgs() << "*** AMDGPUVGPRMSBAffinity on " << MF.getName()
                    << " ***\n");

  // Coalesce value groups so MSB group pressure is not inflated by tied/loop-carried
  // vregs that will share one physical register.
  buildValueGroups(MF);

  // Nothing to MSB group if the whole function already fits in a single 256-VGPR
  // MSB group: no S_SET_VGPR_MSB is ever required, and any partition would only
  // spread the footprint across MSB groups and inflate the VGPR count. Also use the
  // measured footprint to cap how many MSB groups we are allowed to spread into, so
  // we never push values into otherwise-unused high MSB groups.
  SmallVector<Register, 0> AllVGPRs;
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (!MRI->reg_nodbg_empty(R) && isVGPRVirtReg(R))
      AllVGPRs.push_back(R);
  }
  unsigned GlobalFP = maxSimultaneousDwords(AllVGPRs);
  LLVM_DEBUG(dbgs() << "  early-check GlobalFP=" << GlobalFP << "\n");
  if (GlobalFP <= MSBGroupSize) {
    LLVM_DEBUG(dbgs() << "  -> return: footprint fits one group\n");
    return false;
  }

  // Baseline occupancy: the min of the VGPR-limited estimate (from the measured
  // footprint) and the non-VGPR limit MFI already computed (LDS / workgroup).
  // A fractional occupancy (e.g. occ 3 ~ 1.33 groups) is handled correctly by the
  // per-group VGPR budget below -- the last group holds only its fractional share,
  // so packing/overflow-checking never silently drops a wave. (This used to be a
  // hard power-of-two skip; the fractional-budget accounting made it redundant.)
  const SIMachineFunctionInfo *MFIOcc = MF.getInfo<SIMachineFunctionInfo>();
  unsigned VOcc =
      STI->getOccupancyWithNumVGPRs(GlobalFP, MFIOcc->getDynamicVGPRBlockSize());
  unsigned BaseOcc = std::min(VOcc, MFIOcc->getOccupancy());
  LLVM_DEBUG(dbgs() << "  early-check BaseOcc=" << BaseOcc << " (VOcc=" << VOcc
                    << " MFIOcc=" << MFIOcc->getOccupancy() << ")\n");
  if (BaseOcc == 0)
    return false;

  // Use every MSB group the occupancy budget allows: NumMSBGroups/BaseOcc MSB groups fit
  // without dropping a wave (occ 1 -> 4 MSB groups, occ 2 -> 2 MSB groups). Spreading
  // into more MSB groups costs only VGPRs, which is free while occupancy is the
  // binding limit -- and the extra room lets clusters avoid over-subscribing a
  // MSB group (an occ-1 kernel can get 4 groups instead of the 3 its footprint
  // strictly needs, making an otherwise-unrealizable plan fit). Never use fewer
  // than the footprint strictly requires.
  unsigned Needed = (GlobalFP + MSBGroupSize - 1) / MSBGroupSize;
  const unsigned EffMSBGroups =
      std::min(NumMSBGroups, std::max(Needed, NumMSBGroups / BaseOcc));

  // Real VGPR budget for this occupancy. For power-of-two occupancy this is a
  // whole number of 256-groups (occ 2 -> 512, occ 1 -> 1024). For a fractional
  // occupancy (occ 3 -> ~336) the last group is partial, which processRegion's
  // per-group cap accounts for so packing never silently drops a wave.
  unsigned VGPRBudget =
      STI->getMaxNumVGPRs(BaseOcc, MFIOcc->getDynamicVGPRBlockSize());

  LLVM_DEBUG(dbgs() << "  GlobalFP(true RP)=" << GlobalFP << " BaseOcc="
                    << BaseOcc << " EffMSBGroups=" << EffMSBGroups
                    << " VGPRBudget=" << VGPRBudget << "\n");

  // One whole-function clustering: capacity / register pressure is a global
  // property, so the graph, footprint, packing and gate all stay global.
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

  // Experimental structural GEMM path (default off): reconstruct regions + assign
  // banks by role. When enabled it replaces the affinity path entirely.
  if (RegionDataFlow)
    return planDataFlowRegions(MF, EffMSBGroups, MFI);

  DenseSet<unsigned> Assigned;
  SmallVector<MachineBasicBlock *, 16> Blocks;
  for (MachineBasicBlock &MBB : MF)
    Blocks.push_back(&MBB);
  // First attempt: the default (compact) clustering, scored on the whole
  // function. If it commits nothing -- because the plan was infeasible (a fat
  // cluster overflows a group, e.g. occ-1 kernels whose crammed layout leaves an
  // empty group) or predicted no benefit -- retry with balanced packing scored on
  // in-loop switches. The balanced plan spreads clusters to use the spare group
  // and the loop-only gate values the per-iteration win; this recovers near-full
  // occ-1 and fractional occ-3 kernels the compact plan gives up on, without
  // disturbing kernels the compact plan already handles (it is only reached when
  // the first attempt committed nothing).
  processRegion(Blocks, AllVGPRs, EffMSBGroups, VGPRBudget, /*Balance=*/false,
                /*LoopGate=*/false, Assigned, MFI);
  if (Assigned.empty())
    processRegion(Blocks, AllVGPRs, EffMSBGroups, VGPRBudget, /*Balance=*/true,
                  /*LoopGate=*/true, Assigned, MFI);

  // Analysis only; never changes the function (besides allocation hints).
  return false;
}

// Experimental structural GEMM planner (soft-hint version). Runs only when
// region-dataflow is enabled, and expects the LLIR-scheduled order (run with the
// pre-RA scheduler off). It reads the ";; Region N" markers the LLIR scheduler
// emits and hints banks by role so each region tends toward one bank-context:
//   - WMMA accumulator (tied dst==src2)  -> bank = region % groups;
//   - ds_load tile (its dst)             -> its loading region's bank (data_bank:
//     the move that lets the ds_load ride the WMMA's dst context);
//   - ds_load address (its src0)         -> its region's src0 bank.
// First assignment of a vreg wins, which stays consistent when a loop is unrolled
// and an accumulator is written in more than one region. Soft hints only.
bool AMDGPUVGPRMSBAffinity::planDataFlowRegions(MachineFunction &MF,
                                                unsigned EffMSBGroups,
                                                SIMachineFunctionInfo *MFI) {
  auto slotVReg = [&](const MachineInstr &MI, unsigned S) -> Register {
    auto Ops = AMDGPU::getVGPRLoweringOperandTables(MI.getDesc());
    if (!Ops.first)
      return Register();
    const MachineOperand *MO = TII->getNamedOperand(MI, Ops.first[S]);
    if ((!MO || !MO->isReg() || !MO->getReg()) && Ops.second)
      MO = TII->getNamedOperand(MI, Ops.second[S]);
    if (MO && MO->isReg() && MO->getReg() && isVGPRVirtReg(MO->getReg()))
      return MO->getReg();
    return Register();
  };
  DenseSet<unsigned> Assigned;
  auto hint = [&](Register R, int Bank) {
    if (R && Assigned.insert(R.virtRegIndex()).second)
      recordMSB(MFI, R, (unsigned)(Bank % (int)EffMSBGroups));
  };
  auto isWMMA = [&](const MachineInstr &MI) {
    return SIInstrInfo::isWMMA(MI) || SIInstrInfo::isSWMMAC(MI);
  };
  auto isDSLoad = [&](const MachineInstr &MI) {
    return TII->isDS(MI) && MI.mayLoad();
  };
  // The LLIR scheduler emits region boundaries as inline-asm markers of the form
  //   INLINEASM &";; Region N: ..."
  // Parse N as the ground-truth region index --
  // this is what makes each region's src slot uniform, since the scheduler groups
  // a region's ds_loads and WMMAs together. Returns -1 if MI is not a loop
  // (non-epilogue) region marker.
  auto regionMarker = [&](const MachineInstr &MI) -> int {
    if (!MI.isInlineAsm())
      return -1;
    const MachineOperand &MO = MI.getOperand(InlineAsm::MIOp_AsmString);
    if (!MO.isSymbol())
      return -1;
    StringRef S(MO.getSymbolName());
    StringRef Key = ";; Region ";
    size_t P = S.find(Key);
    if (P == StringRef::npos)
      return -1; // ";; Epilogue Region" and others are excluded
    int N = -1;
    if (S.substr(P + Key.size()).consumeInteger(10, N))
      return -1;
    return N;
  };
  bool Committed = false;
  bool Modified = false;
  for (MachineBasicBlock &MBB : MF) {
    if (!MLI || MLI->getLoopDepth(&MBB) == 0)
      continue; // only loop bodies have per-iteration regions
    // Pass 1: label each instruction with the region index of the most recent
    // ";; Region N" marker above it.
    DenseMap<const MachineInstr *, int> RegionOf;
    int Region = 0;
    for (MachineInstr &MI : MBB) {
      int M = regionMarker(MI);
      if (M >= 0)
        Region = M;
      RegionOf[&MI] = Region;
    }
    // Pass 2a: assign the data_bank roles. The tile is
    // colored by its *loading* region -- not its consumer: a GEMM weight is
    // reused across many consuming bursts (different banks) but has one loading
    // region, so loading-region coloring keeps it single-banked. A WMMA reads the
    // tile vreg directly, so this also fixes the WMMA's src bank.
    DenseMap<unsigned, int> BankOf;
    auto assign = [&](Register R, int Bank) {
      if (R)
        BankOf[R.virtRegIndex()] = Bank % (int)EffMSBGroups;
    };
    for (MachineInstr &MI : MBB) {
      Register Dst = slotVReg(MI, 3), Src2 = slotVReg(MI, 2);
      if (isWMMA(MI) && Dst && Src2 && Dst == Src2)
        assign(Dst, RegionOf[&MI]); // accumulator -> its region bank
      else if (isDSLoad(MI))
        assign(slotVReg(MI, 3), RegionOf[&MI]); // tile -> loading-region bank
    }
    // Pass 2b: per-region src0/src1 bank = the bank of the tiles this region's
    // WMMAs read (already fixed in 2a). This is what the region's MSB context
    // carries in the src0/src1 slots.
    DenseMap<int, int> RegionSrc0, RegionSrc1;
    auto lookup = [&](Register R) -> int {
      auto It = R ? BankOf.find(R.virtRegIndex()) : BankOf.end();
      return It == BankOf.end() ? -1 : It->second;
    };
    for (MachineInstr &MI : MBB) {
      if (!isWMMA(MI))
        continue;
      int R = RegionOf[&MI];
      if (int B0 = lookup(slotVReg(MI, 0)); B0 >= 0)
        RegionSrc0.try_emplace(R, B0);
      if (int B1 = lookup(slotVReg(MI, 1)); B1 >= 0)
        RegionSrc1.try_emplace(R, B1);
    }
    // Pass 2c: ds_load address bank = the src0 bank of its region's WMMAs
    //, so the interleaved ds_load rides the WMMA's MSB context
    // (dst=acc, src0=addr) instead of forcing a reset to the acc bank.
    //
    // When one loop-invariant address vreg is shared by ds_loads across regions
    // that need *different* src0 banks, a single vreg cannot satisfy them all --
    // its one bank forces an MSB reset in every region but its home one. We
    // duplicate it: the first region keeps the original, and each other required
    // bank gets a fresh COPY hoisted into the preheader with the ds_loads in that
    // region rewritten to it. Keeping the original for the home region means the
    // original and the copies are simultaneously live (used in different regions),
    // so the coalescer will not merge them back. This is a general lever: any
    // per-region operand-bank specialization of a shared invariant register.
    auto slotMO = [&](MachineInstr &MI, unsigned S) -> MachineOperand * {
      auto Ops = AMDGPU::getVGPRLoweringOperandTables(MI.getDesc());
      if (!Ops.first)
        return nullptr;
      MachineOperand *MO = TII->getNamedOperand(MI, Ops.first[S]);
      if ((!MO || !MO->isReg() || !MO->getReg()) && Ops.second)
        MO = TII->getNamedOperand(MI, Ops.second[S]);
      return MO;
    };
    MachineLoop *L = MLI->getLoopFor(&MBB);
    MachineBasicBlock *Preheader = L ? L->getLoopPreheader() : nullptr;
    DenseMap<unsigned, int> AddrHome;       // addr vreg idx -> home (kept) bank
    DenseMap<uint64_t, Register> AddrSplit; // (addr idx, bank) -> copy vreg
    for (MachineInstr &MI : MBB) {
      if (!isDSLoad(MI))
        continue;
      auto It = RegionSrc0.find(RegionOf[&MI]);
      if (It == RegionSrc0.end())
        continue;
      int Want = It->second;
      MachineOperand *MO = slotMO(MI, 0);
      Register Addr =
          (MO && MO->isReg() && isVGPRVirtReg(MO->getReg())) ? MO->getReg()
                                                             : Register();
      if (!Addr)
        continue;
      auto Home = AddrHome.try_emplace(Addr.virtRegIndex(), Want);
      if (Home.first->second == Want) {
        assign(Addr, Want); // home region (or all users agree): keep original
        continue;
      }
      // A different bank is needed. Only duplicate loop-invariant addresses (def
      // outside the loop), so the copy can live for free in the preheader;
      // loop-variant addresses are left at their home bank.
      MachineInstr *Def = MRI->getVRegDef(Addr);
      if (!Preheader || !Def || (L && L->contains(Def->getParent())))
        continue;
      uint64_t K = ((uint64_t)Addr.virtRegIndex() << 8) | (unsigned)Want;
      Register Copy = AddrSplit.lookup(K);
      if (!Copy) {
        Copy = MRI->createVirtualRegister(MRI->getRegClass(Addr));
        BuildMI(*Preheader, Preheader->getFirstTerminator(), DebugLoc(),
                TII->get(TargetOpcode::COPY), Copy)
            .addReg(Addr);
        AddrSplit[K] = Copy;
        assign(Copy, Want);
        Modified = true;
      }
      MO->setReg(Copy); // rewrite this region's ds_load to the private address
    }
    // Pass 2d: commit all assigned banks as allocation hints.
    for (auto &KV : BankOf) {
      hint(Register::index2VirtReg(KV.first), KV.second);
      Committed = true;
    }
  }
  LLVM_DEBUG(dbgs() << "  region-dataflow: hinted " << Assigned.size()
                    << " vregs, committed=" << Committed
                    << ", modified=" << Modified << "\n");
  (void)Committed;
  return Modified;
}

void AMDGPUVGPRMSBAffinity::processRegion(ArrayRef<MachineBasicBlock *> Blocks,
                                           ArrayRef<Register> AllVGPRs,
                                           unsigned EffMSBGroups,
                                           unsigned VGPRBudget, bool Balance,
                                           bool LoopGate,
                                           DenseSet<unsigned> &Assigned,
                                           SIMachineFunctionInfo *MFI) {
  const unsigned N = MRI->getNumVirtRegs();

  // Per-group register cap. For power-of-two occupancy VGPRBudget is a whole
  // number of 256-groups so every group's cap is 256. For a fractional
  // occupancy (e.g. occ 3 ~ 1.33 groups) the last group holds only
  // VGPRBudget - (EffMSBGroups-1)*256 registers, not a full 256 -- packing or
  // overflow-checking it at 256 would silently drop a wave. Clamp to [1,256].
  auto groupCap = [&](unsigned B) -> unsigned {
    int cap = (int)VGPRBudget - (int)(B * MSBGroupSize);
    return (unsigned)std::max(1, std::min<int>(MSBGroupSize, cap));
  };

  // Cluster capacity: normally a full group (256). With balanced packing, when
  // the footprint leaves spare room across the groups, cap at ceil(FP/EffGroups)
  // so clusters spread and every used group keeps slack for the soft hints.
  unsigned MergeCap = MSBGroupSize;
  if (Balance) {
    unsigned FP = maxSimultaneousDwords(AllVGPRs);
    unsigned Bal = (FP + EffMSBGroups - 1) / std::max(1u, EffMSBGroups);
    MergeCap = std::min<unsigned>(MSBGroupSize, std::max(1u, Bal));
  }

  // --- Step 1: build the affinity graph from the scheduled stream. ----------
  //
  // Edge weight between two vregs = sum over the program of the block frequency
  // at points where they appear consecutively in the same MSB slot (i.e. where
  // a mode switch is paid unless they share a MSB group).
  DenseMap<uint64_t, uint64_t> Edges;
  // Earliest instruction ordinal at which each edge occurs, used as the primary
  // tie-break among equal-weight edges (see the sort below).
  DenseMap<uint64_t, unsigned> EdgePos;

  auto addEdge = [&](Register A, Register B, uint64_t W, unsigned P) {
    unsigned a = A.virtRegIndex(), b = B.virtRegIndex();
    if (a == b)
      return;
    // Scale the edge by the operand width (capped) so a wide value, e.g. the
    // WMMA accumulator, that alternates in a slot outweighs a scalar doing the
    // same. TupleWeight <= 1 keeps the width-independent weight.
    if (TupleWeight > 1) {
      unsigned Dw = std::min({dwords(A), dwords(B), TupleWeight.getValue()});
      W *= std::max(1u, Dw);
    }
    if (a > b)
      std::swap(a, b);
    uint64_t Key = (uint64_t(a) << 32) | b;
    Edges[Key] += W;
    auto It = EdgePos.find(Key);
    if (It == EdgePos.end())
      EdgePos[Key] = P; // first (earliest) occurrence in program order
  };

  unsigned Pos = 0; // monotonic instruction ordinal (program order)
  for (MachineBasicBlock *MBBp : Blocks) {
    MachineBasicBlock &MBB = *MBBp;
    uint64_t Freq = blockFreq(MBB);

    // Sticky per-slot state, reset at each block (the lowering pass resets the
    // mode at block boundaries).
    Register LastInSlot[4];
    bool PrevDsRead = false;   // previous real instr was a ds_read
    unsigned PrevDsDstLen = 0; // that ds_read's vdst tuple width (dwords)

    for (MachineInstr &MI : MBB) {
      if (MI.isMetaInstruction())
        continue;
      ++Pos;
      bool ThisDsRead = TII->isDS(MI) && MI.mayLoad();

      // A COPY between two VGPR vregs will likely be coalesced by the allocator
      // (the copy hint outranks our MSB group hint), co-assigning both to one
      // physreg -- i.e. one MSB group. Add an affinity edge so our plan places them
      // in the same MSB group, consistent with that coalescing. This is capacity-
      // bounded in Step 2; and because maxSimultaneousDwords counts a non-
      // interfering (coalesceable) pair once, it stays footprint-correct.
      if (MI.isCopy()) {
        const MachineOperand &Dst = MI.getOperand(0), &Src = MI.getOperand(1);
        if (Dst.isReg() && Src.isReg() && isVGPRVirtReg(Dst.getReg()) &&
            isVGPRVirtReg(Src.getReg()))
          // Keep the copy edge on the same scale as the slot edges (a single-slot
          // boundary weighs Freq*144), so a copy (one co-location opportunity)
          // is not under-weighted and copy-related vregs still co-locate.
          addEdge(Dst.getReg(), Src.getReg(), Freq * 144, Pos);
      }

      auto Ops = AMDGPU::getVGPRLoweringOperandTables(MI.getDesc());
      if (!Ops.first) {
        PrevDsRead = ThisDsRead;
        continue;
      }

      // Width of this instruction's dst tuple (vdst = slot 3), used to scale the
      // src0 boost at a ds_read boundary ("boost to dst len").
      unsigned ThisDstLen = 0;
      if (const MachineOperand *D = TII->getNamedOperand(MI, Ops.first[3])) {
        if ((!D->isReg() || !D->getReg()) && Ops.second)
          D = TII->getNamedOperand(MI, Ops.second[3]);
        if (D && D->isReg() && D->getReg() && isVGPRVirtReg(D->getReg()))
          ThisDstLen = dwords(D->getReg());
      }

      // Per-slot edges (not deduplicated per instruction). A tied accumulator
      // drives both src2 and dst, so it contributes 2*Freq to its pair -- this
      // is intentional: it emphasizes co-location the accumulator/dst chain (the
      // highest-value pairing) over single-slot src0/src1 edges. (Deduplicating
      // to the exact "one set per boundary" cost was tried and slightly regressed
      // accumulator-heavy kernels, so it was reverted.)
      SmallVector<std::tuple<Register, Register, unsigned>, 4> Changed;
      for (unsigned S = 0; S < 4; ++S) {
        const MachineOperand *MO = TII->getNamedOperand(MI, Ops.first[S]);
        if ((!MO || !MO->isReg() || !MO->getReg()) && Ops.second)
          MO = TII->getNamedOperand(MI, Ops.second[S]);
        if (!MO || !MO->isReg() || !MO->getReg())
          continue; // Slot not constrained: stays sticky.

        Register R = MO->getReg();
        if (isVGPRVirtReg(R)) {
          if (LastInSlot[S] && LastInSlot[S] != R)
            Changed.emplace_back(LastInSlot[S], R, S);
          LastInSlot[S] = R;
        } else if (R.isPhysical() && TRI->isVGPR(*MRI, R)) {
          // A physical VGPR pins the slot to a fixed MSB group; break the run so we
          // do not attract vregs across it.
          LastInSlot[S] = Register();
        }
        // SGPR / immediate operands leave the slot sticky.
      }
      // Charge the boundary once (one s_set_vgpr_msb covers all changed slots),
      // distributed across its changed slots by weight/k^BoundaryIsoExp. Base 144
      // keeps the per-slot weight integral for 1..4 changed slots.
      if (!Changed.empty()) {
        uint64_t Denom = 1;
        for (unsigned E = 0; E < BoundaryIsoExp; ++E)
          Denom *= Changed.size();
        uint64_t W = (Freq * 144) / Denom;
        bool DsBoundary = PrevDsRead || ThisDsRead;
        // Boost factor for the src0 edge at a ds_read boundary: the width of the
        // ds_read's dst tuple ("boost to dst len"), capped by Src0DsWeight so it
        // matches the tuple-weighting the wide operand already gets.
        unsigned DstLen = ThisDsRead ? ThisDstLen : PrevDsDstLen;
        // Isolation count for the src0 gate excludes the dst slot: a ds_read
        // writes a fresh tile in the dst slot, so a WMMA->ds_read boundary looks
        // like it changes dst (different vregs) even though both tiles usually
        // land in the same group post-RA. Counting dst would misclassify these
        // as multi-slot and skip the boost, leaving the ds_read address stranded
        // in another group (the src0 g0<->g1 flip). Count only src0/src1/src2.
        unsigned ChangedNonDst = 0;
        for (auto &[A, B, S] : Changed)
          if (S != 3)
            ++ChangedNonDst;
        for (auto &[A, B, S] : Changed) {
          uint64_t w = W;
          if (S == 0 && DsBoundary && Src0DsWeight > 1 &&
              ChangedNonDst <= Src0IsoMax)
            w *= std::max(1u, std::min(DstLen, Src0DsWeight.getValue()));
          addEdge(A, B, w, Pos);
        }
      }
      PrevDsRead = ThisDsRead;
      if (ThisDsRead)
        PrevDsDstLen = ThisDstLen;
    }
  }

  if (Edges.empty())
    return;

  // --- Step 2: capacity-aware greedy union-find into clusters. --------------
  SmallVector<unsigned, 0> Parent(N), Rank(N, 0);
  SmallVector<int, 0> Footprint(N, -1); // cached per-root footprint, lazy.
  SmallVector<SmallVector<Register, 4>, 0> Members(N);
  for (unsigned I = 0; I < N; ++I) {
    Parent[I] = I;
    Register R = Register::index2VirtReg(I);
    if (!MRI->reg_nodbg_empty(R) && isVGPRVirtReg(R))
      Members[I].push_back(R);
  }

  std::function<unsigned(unsigned)> find = [&](unsigned X) {
    while (Parent[X] != X) {
      Parent[X] = Parent[Parent[X]];
      X = Parent[X];
    }
    return X;
  };
  auto footprintOf = [&](unsigned Root) -> int {
    if (Footprint[Root] < 0)
      Footprint[Root] = (int)maxSimultaneousDwords(Members[Root]);
    return Footprint[Root];
  };

  // Exact union footprint of two clusters (time-aware peak of the members).
  auto unionFP = [&](unsigned ra, unsigned rb) -> int {
    SmallVector<Register, 16> Both(Members[ra].begin(), Members[ra].end());
    Both.append(Members[rb].begin(), Members[rb].end());
    return (int)maxSimultaneousDwords(Both);
  };
  auto doMerge = [&](unsigned ra, unsigned rb, int Merged) {
    if (Rank[ra] < Rank[rb])
      std::swap(ra, rb);
    Parent[rb] = ra;
    if (Rank[ra] == Rank[rb])
      ++Rank[ra];
    Members[ra].append(Members[rb].begin(), Members[rb].end());
    Members[rb].clear();
    Footprint[ra] = Merged;
    Footprint[rb] = -1;
  };

  {
    // Lazy-priority-queue greedy: process by weight, but among comparable-weight
    // merges prefer the one with the smallest footprint *delta* (best
    // time-multiplexing). Each cluster carries an epoch bumped on every merge; a
    // popped item whose endpoints' roots/epochs changed is re-evaluated (its
    // delta is stale) and re-pushed, so the top of the queue is always accurate.
    SmallVector<uint64_t, 0> Epoch(N, 0);
    struct Item {
      uint64_t W;
      int Delta;
      unsigned Pos;
      uint64_t Key;
      unsigned RA, RB;
      uint64_t EA, EB;
      bool operator<(const Item &O) const { // max-heap: greater = higher prio
        if (W != O.W)
          return W < O.W; // higher weight first
        if (Delta != O.Delta)
          return Delta > O.Delta; // lower delta first
        if (Pos != O.Pos)
          return Pos > O.Pos; // earlier program order first
        return Key > O.Key; // lower key first (determinism)
      }
    };
    std::priority_queue<Item> PQ;
    auto push = [&](uint64_t Key, uint64_t W) {
      unsigned ai = unsigned(Key >> 32), bi = unsigned(Key & 0xffffffff);
      unsigned ra = find(ai), rb = find(bi);
      if (ra == rb)
        return;
      int Merged = unionFP(ra, rb);
      int Delta = Merged - std::max(footprintOf(ra), footprintOf(rb));
      PQ.push({W, Delta, EdgePos.lookup(Key), Key, ra, rb, Epoch[ra], Epoch[rb]});
    };
    for (auto &[Key, W] : Edges)
      push(Key, W);
    while (!PQ.empty()) {
      Item It = PQ.top();
      PQ.pop();
      unsigned ai = unsigned(It.Key >> 32), bi = unsigned(It.Key & 0xffffffff);
      unsigned ra = find(ai), rb = find(bi);
      if (ra == rb)
        continue;
      // Stale (a touched cluster changed since push) -> re-evaluate and re-push.
      if (ra != It.RA || rb != It.RB || Epoch[ra] != It.EA ||
          Epoch[rb] != It.EB) {
        push(It.Key, It.W);
        continue;
      }
      int Merged = unionFP(ra, rb);
      if (Merged > (int)MergeCap)
        continue; // Refuse: this edge becomes a cut.
      unsigned keep = Rank[ra] < Rank[rb] ? rb : ra;
      doMerge(ra, rb, Merged);
      ++Epoch[keep];
    }
  }

  // --- Step 3: pack clusters into MSB groups (hottest first, least loaded). ------
  // Cluster weight = total internal affinity (how costly it is to split).
  DenseMap<unsigned, uint64_t> ClusterWeight;
  for (auto &[Key, W] : Edges) {
    unsigned ai = unsigned(Key >> 32), bi = unsigned(Key & 0xffffffff);
    if (find(ai) == find(bi))
      ClusterWeight[find(ai)] += W;
  }

  // Only steer clusters that carry real (loop-level) affinity. Cold registers
  // with no significant same-slot neighbours are left unhinted so the allocator
  // packs them naturally instead of being forced into a MSB group (cf. PresCount's
  // free-register handling).
  uint64_t MaxCW = 0;
  for (auto &KV : ClusterWeight)
    MaxCW = std::max(MaxCW, KV.second);
  uint64_t WeightCutoff = HotClusterDiv ? MaxCW / HotClusterDiv : 0;

  SmallVector<unsigned, 0> Roots;
  for (unsigned I = 0; I < N; ++I)
    if (find(I) == I && !Members[I].empty() &&
        ClusterWeight.lookup(I) > WeightCutoff)
      Roots.push_back(I);
  // Sort hottest-first: the most important clusters are placed first. First-fit
  // then puts them in the low MSB groups -- which matters because group 0 is special:
  // AMDGPULowerVGPREncoding resets the mode to all-zero at non-fall-through
  // block entries (including the loop header every iteration), so a value in
  // group 0 needs no switch right after a reset. Keeping the hottest clusters in
  // group 0 therefore minimizes switches.
  llvm::stable_sort(Roots, [&](unsigned A, unsigned B) {
    return ClusterWeight.lookup(A) > ClusterWeight.lookup(B);
  });

  // Pack with *time-overlap-aware* capacity: a MSB group's load is the maximum
  // simultaneously-live footprint of all clusters assigned to it, NOT the sum
  // of their peaks. Clusters whose live ranges are disjoint in time (e.g.
  // successive prefetch tiles in a software-pipelined loop) therefore share a
  // MSB group instead of each reserving 256 registers -- which is what keeps the
  // VGPR count from ballooning on pipelined kernels.
  SmallVector<SmallVector<Register, 0>, 8> MSBMembers(EffMSBGroups);
  SmallVector<int, 8> MSBLoad(EffMSBGroups, 0);
  DenseMap<unsigned, int> ClusterMSB;
  unsigned NumClusters = Roots.size();
  for (unsigned Root : Roots) {
    int Best = -1, BestLoad = 0;
    // Default: first-fit into the lowest group that fits (keeps hot clusters in
    // group 0, which is reset-free after the loop header). With balanced packing:
    // place in the group that minimizes the resulting load, so clusters spread
    // across all groups and every used group keeps slack for the soft hints.
    int BestResult = INT_MAX;
    for (unsigned B = 0; B < EffMSBGroups; ++B) {
      SmallVector<Register, 16> Combined(MSBMembers[B].begin(),
                                         MSBMembers[B].end());
      Combined.append(Members[Root].begin(), Members[Root].end());
      int L = (int)maxSimultaneousDwords(Combined);
      if (!Balance) {
        if (L <= (int)groupCap(B)) {
          Best = B;
          BestLoad = L;
          break; // lowest MSB group that fits
        }
      } else if (L <= (int)groupCap(B) && L < BestResult) {
        BestResult = L;
        Best = B;
        BestLoad = L;
      }
    }
    if (Best < 0) {
      // No MSB group fits this cluster within capacity; over-subscribe the
      // least-loaded one (the soft hint may spill).
      Best = std::min_element(MSBLoad.begin(), MSBLoad.end()) -
             MSBLoad.begin();
      MSBMembers[Best].append(Members[Root].begin(), Members[Root].end());
      BestLoad = (int)maxSimultaneousDwords(MSBMembers[Best]);
    } else {
      MSBMembers[Best].append(Members[Root].begin(), Members[Root].end());
    }
    MSBLoad[Best] = BestLoad;
    ClusterMSB[Root] = Best;
  }

  // Commit the plan unless a MSB group is *severely* over-subscribed. A MSB group can
  // hold only 256 simultaneously-live registers; a mild overflow is fine (the
  // allocator spills a few and still honors most of the plan, still a win), but a
  // severe overflow makes the plan unrealizable so the allocator discards the
  // hints and regresses. Skip when a group's load exceeds its cap * OverflowPct/100.
  // Overflow is checked per group against that group's cap (the last group is
  // smaller at fractional occupancy), not a flat 256, so a fractional group
  // cannot be silently over-packed.
  bool Infeasible = false;
  for (unsigned B = 0; B < EffMSBGroups; ++B)
    if ((uint64_t)MSBLoad[B] * 100 > (uint64_t)groupCap(B) * OverflowPct)
      Infeasible = true;

  // Self-benefit gate. The pass optimizes the *observed* schedule, but an
  // MSB-aware schedule already lays the code out so the natural allocation keeps
  // slots MSB-stable -- there our competing partition can *raise* the switch
  // count. Compare the predicted switches of
  // the plan against the natural (no-hint) layout on this exact schedule, and
  // bail if the plan does not beat it. Both sims use the same naive MSB groups for
  // unhinted vregs, so this isolates the effect of our hints.
  bool NoBenefit = false;
  uint64_t PlanSw = 0, BaseSw = 0;
  if (!Infeasible && BenefitPct) {
    DenseMap<unsigned, int> NaiveMSB = computeNaiveMSB(AllVGPRs, EffMSBGroups);
    DenseMap<unsigned, int> PlanOverride;
    for (unsigned Root : Roots)
      for (Register R : Members[Root])
        PlanOverride[R.virtRegIndex()] = ClusterMSB[Root];
    auto naiveOf = [&](Register R) { return NaiveMSB.lookup(R.virtRegIndex()); };
    auto planOf = [&](Register R) {
      auto It = PlanOverride.find(R.virtRegIndex());
      return It != PlanOverride.end() ? It->second : naiveOf(R);
    };
    BaseSw = simSwitchWeight(Blocks, naiveOf, LoopGate);
    PlanSw = simSwitchWeight(Blocks, planOf, LoopGate);
    LLVM_DEBUG(dbgs() << "  gate(" << (LoopGate ? "loop-only" : "whole-fn")
                      << ") planSw=" << PlanSw << " baseSw=" << BaseSw << "\n");
    // Commit only if the plan predicts a large enough win over the naive layout
    // on this exact schedule. Use 128-bit products so the comparison can't
    // overflow on a huge function where the freq-weighted sums are large.
    NoBenefit = (unsigned __int128)PlanSw * 100 >=
                (unsigned __int128)BaseSw * BenefitPct;
    // Require a minimum absolute baseline cost. A small baseline means the loop is
    // already near-coherent; the predictor over-estimates its few switches and the
    // plan's "win" does not survive real allocation, so committing regresses (e.g.
    // an already-coherent occ-1 GEMM loop). Leave those to the allocator.
    if (MinBaseSwitch && BaseSw < MinBaseSwitch)
      NoBenefit = true;
  }

  unsigned NumAssigned = 0;
  if (!Infeasible && !NoBenefit) {
    for (unsigned Root : Roots)
      for (Register R : Members[Root]) {
        // A hotter (deeper) region already fixed this vreg's MSB group; don't
        // re-hint it to a different MSB group.
        if (!Assigned.insert(R.virtRegIndex()).second)
          continue;
        recordMSB(MFI, R, ClusterMSB[Root]);
        ++NumAssigned;
      }
  }

  LLVM_DEBUG({
    SmallVector<int, 0> FPs;
    for (unsigned Root : Roots)
      FPs.push_back(footprintOf(Root));
    llvm::sort(FPs, std::greater<int>());
    int Over = 0;
    for (int F : FPs)
      if (F > (int)MSBGroupSize)
        ++Over;
    dbgs() << "  edges=" << Edges.size() << " clusters=" << NumClusters
           << " vregsAssigned=" << NumAssigned
           << " clustersOver256=" << Over << " planSw=" << PlanSw
           << " baseSw=" << BaseSw
           << (Infeasible ? " INFEASIBLE(skipped)"
                          : (NoBenefit ? " NO-BENEFIT(skipped)" : ""))
           << "\n  cluster FPs:";
    for (int F : FPs)
      dbgs() << ' ' << F;
    dbgs() << "\n  MSB group loads:";
    for (unsigned B = 0; B < EffMSBGroups; ++B)
      dbgs() << " [" << B << "]=" << MSBLoad[B];
    dbgs() << "\n";
  });
}


char AMDGPUVGPRMSBAffinityLegacy::ID = 0;

char &llvm::AMDGPUVGPRMSBAffinityLegacyID = AMDGPUVGPRMSBAffinityLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUVGPRMSBAffinityLegacy, DEBUG_TYPE,
                      "AMDGPU VGPR MSB Affinity", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUVGPRMSBAffinityLegacy, DEBUG_TYPE,
                    "AMDGPU VGPR MSB Affinity", false, false)

FunctionPass *llvm::createAMDGPUVGPRMSBAffinityLegacyPass() {
  return new AMDGPUVGPRMSBAffinityLegacy();
}

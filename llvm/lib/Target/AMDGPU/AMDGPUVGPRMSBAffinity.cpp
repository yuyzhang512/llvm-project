//===- AMDGPUVGPRMSBAffinity.cpp - bias VGPR alloc into 256-VGPR groups ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// On gfx1250 a wave may use up to 1024 VGPRs, but an instruction can only
/// address VGPRs 0-255 directly. VGPRs 256-1023 are reached by setting the
/// per-operand-slot MSB bits with S_SET_VGPR_MSB. When consecutive instructions
/// need a different {src0,src1,src2,dst} MSB-group configuration,
/// AMDGPULowerVGPREncoding emits an S_SET_VGPR_MSB.
///
/// This pass runs before register allocation, after the pre-RA scheduler has
/// fixed the instruction order. It does not change code. It records, per
/// virtual register, a desired MSB group (the high bits of the HW index,
/// index >> 8); SIRegisterInfo's allocation-hint hook then biases the greedy
/// allocator toward that group.
///
/// The switch count equals, per MSB slot, how often the group of the value in
/// that slot changes along the scheduled stream. The pass models this:
///
///   1. Walk the scheduled MIR. Map each instruction's VGPR operands to the
///      four MSB slots with getVGPRLoweringOperandTables. When a slot's value
///      changes from one vreg to another, add an affinity edge between them,
///      weighted by block frequency.
///
///   2. Partition the graph into MSB groups by a footprint-aware greedy
///      union-find: merge the heaviest edges first, refusing a merge when the
///      cluster's simultaneously-live footprint would exceed one group (256
///      dwords, from LiveIntervals). Capacity forces the split.
///
///   3. Pack the clusters into the groups the occupancy budget allows (hottest
///      first, least loaded), and record each vreg's group.
///
/// The hint is soft: if a group fills up the allocator falls back to the rest
/// of the order, so this can never make allocation fail.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <functional>
#include <queue>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-vgpr-msb-affinity"

static cl::opt<bool> EnableVGPRMSBAffinity(
    "amdgpu-vgpr-msb-affinity", cl::Hidden, cl::init(false),
    cl::desc("Bias VGPR allocation into 256-VGPR MSB groups to reduce "
             "S_SET_VGPR_MSB insertions (gfx1250)"));

// Hint clusters whose weight exceeds MaxClusterWeight/HotClusterDiv; colder
// clusters are left unhinted so the allocator packs them naturally.
static cl::opt<unsigned> HotClusterDiv("amdgpu-vgpr-msb-affinity-hot-div",
                                       cl::Hidden, cl::init(4),
                                       cl::desc("Hot-cluster filter divisor"));

// Scale affinity edges by operand width (dwords), capped at this value, in
// attention-flavored regions. 0/1 = width-independent weight.
static cl::opt<unsigned>
    TupleWeight("amdgpu-vgpr-msb-affinity-tuple-weight", cl::Hidden,
                cl::init(8),
                cl::desc("Cap for edge width-weighting (0/1=off)"));

// Self-benefit gate: commit the plan only if its predicted (freq-weighted)
// switch count is below this percent of the predicted count under the natural
// no-hint layout. 0 disables the gate.
static cl::opt<unsigned> BenefitPct(
    "amdgpu-vgpr-msb-affinity-benefit-pct", cl::Hidden, cl::init(75),
    cl::desc("Commit only if predicted plan switches < this % of predicted "
             "no-hint switches (self-benefit gate; 0 disables the gate)"));

// Stricter benefit threshold for GEMM regions, whose natural layout is already
// group-coherent, so only a large predicted win is worth disturbing it.
static cl::opt<unsigned>
    GemmBenefitPct("amdgpu-vgpr-msb-affinity-gemm-benefit-pct", cl::Hidden,
                   cl::init(95),
                   cl::desc("Self-benefit threshold for GEMM regions"));

namespace {

constexpr unsigned MSBGroupSize = 256;
constexpr unsigned NumMSBGroups = 4;
// Skip the plan when the planned peak group load exceeds this percent of a
// group. Mild overflow is realizable (a few spills, most hints honored); severe
// overflow is not.
constexpr unsigned OverflowPct = 125;

class AMDGPUVGPRMSBAffinity {
public:
  bool run(MachineFunction &MF, LiveIntervals *LIS, MachineLoopInfo *MLI);

private:
  // Build the affinity graph, cluster, pack into groups and commit hints for
  // one region. Vregs already in \p Assigned are skipped; newly hinted vregs
  // are added to it.
  void processRegion(ArrayRef<MachineBasicBlock *> Blocks,
                     ArrayRef<Register> AllVGPRs, unsigned EffMSBGroups,
                     DenseSet<unsigned> &Assigned, SIMachineFunctionInfo *MFI);

  const SIInstrInfo *TII = nullptr;
  const SIRegisterInfo *TRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  LiveIntervals *LIS = nullptr;
  MachineLoopInfo *MLI = nullptr;
  const GCNSubtarget *STI = nullptr;

  // Per-block execution weight: a loop-depth proxy for trip count, so an
  // innermost-loop transition outweighs straight-line code by orders of
  // magnitude.
  uint64_t blockFreq(const MachineBasicBlock &MBB) const {
    unsigned Depth = MLI ? MLI->getLoopDepth(&MBB) : 0;
    return 1ull << std::min(4u * Depth, 40u);
  }

  // Value-group union-find: vregs connected by a tied def/use pair are the same
  // value and coalesce to one physical register, so they must not be
  // double-counted in the group pressure. Mutable for path compression.
  mutable SmallVector<unsigned, 0> VGParent;

  bool isVGPRVirtReg(Register Reg) const {
    return Reg.isVirtual() && TRI->isVGPRClass(MRI->getRegClass(Reg));
  }

  unsigned dwords(Register Reg) const {
    // Integer-divide by 32: a 16-bit vreg yields 0. The footprint only feeds
    // soft hints with an overflow safety net, so the 16-bit undercount is
    // acceptable.
    return TRI->getRegSizeInBits(*MRI->getRegClass(Reg)) / 32;
  }

  // Record the group affinity for the hint hook, and a concrete physreg hint to
  // a representative register in the group. The hint sets a known preference,
  // which the greedy allocator's priority boosts, so these vregs are colored
  // earlier and claim their group before contended values fill it. An existing
  // copy-coalescing hint is preserved.
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
        // Coalesce tied def/use pairs. General COPYs are intentionally not
        // unioned: they connect distinct values and would collapse unrelated
        // footprints.
        for (unsigned I = 0, E = MI.getNumOperands(); I < E; ++I) {
          const MachineOperand &MO = MI.getOperand(I);
          if (MO.isReg() && MO.isUse() && MO.isTied()) {
            unsigned DefIdx = MI.findTiedOperandIdx(I);
            const MachineOperand &Def = MI.getOperand(DefIdx);
            if (Def.isReg())
              uni(MO.getReg(), Def.getReg());
          }
        }
        // Coalesce the matrix accumulator chain dst <- src2. Across an unrolled
        // K-loop this chains the successive accumulators into one value group,
        // so the footprint counts the accumulator once. Different output tiles
        // use disjoint chains, so unrelated accumulators are never merged.
        if (SIInstrInfo::isWMMA(MI) || TII->isMAI(MI)) {
          const MachineOperand *D =
              TII->getNamedOperand(MI, AMDGPU::OpName::vdst);
          const MachineOperand *S2 =
              TII->getNamedOperand(MI, AMDGPU::OpName::src2);
          if (D && D->isReg() && S2 && S2->isReg())
            uni(D->getReg(), S2->getReg());
        }
      }
    }
  }

  // Maximum number of VGPR dwords from \p Regs simultaneously live. Live ranges
  // in the same value group are merged first so a coalescing value is counted
  // once.
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

  // Natural no-hint group assignment used as the self-benefit baseline: a
  // linear scan over EffMSBGroups*256 columns in first-definition order, giving
  // each vreg the lowest free column run and freeing columns when a live range
  // ends. This approximates what the allocator does without hints.
  DenseMap<unsigned, int> computeNaiveMSB(ArrayRef<Register> Regs,
                                          unsigned EffMSBGroups) const {
    DenseMap<unsigned, int> MSB;
    const unsigned Cols = EffMSBGroups * MSBGroupSize;
    SmallVector<Register, 0> Order(Regs.begin(), Regs.end());
    llvm::stable_sort(Order, [&](Register A, Register B) {
      return LIS->getInterval(A).beginIndex() <
             LIS->getInterval(B).beginIndex();
    });
    SmallVector<bool, 0> Free(Cols, true);
    // Active allocations: (endIndex, startCol, width) to reclaim columns.
    SmallVector<std::tuple<SlotIndex, unsigned, unsigned>, 0> Active;
    for (Register R : Order) {
      SlotIndex Begin = LIS->getInterval(R).beginIndex();
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
        // No contiguous run fits: reserve D columns in the least-occupied group
        // so this vreg's footprint stays visible in the comparison.
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

  // Predicted freq-weighted s_set_vgpr_msb count for a given vreg->group map,
  // simulated the way AMDGPULowerVGPREncoding counts: walk the scheduled stream
  // with sticky per-slot group state (reset per block), and charge the block
  // frequency once per instruction that needs any slot's group to change.
  uint64_t simSwitchWeight(ArrayRef<MachineBasicBlock *> Blocks,
                           function_ref<int(Register)> msbOf) const {
    uint64_t Sw = 0;
    for (MachineBasicBlock *MBBp : Blocks) {
      MachineBasicBlock &MBB = *MBBp;
      uint64_t Freq = blockFreq(MBB);
      int Last[4] = {-1, -1, -1, -1};
      for (MachineInstr &MI : MBB) {
        if (MI.isMetaInstruction())
          continue;
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
    return AMDGPUVGPRMSBAffinity().run(MF, LISW ? &LISW->getLIS() : nullptr,
                                       MLIW ? &MLIW->getLI() : nullptr);
  }

  StringRef getPassName() const override { return "AMDGPU VGPR MSB Affinity"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
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

  // Only steer compute kernels; graphics shaders do not use the 1024-VGPR
  // grouping this pass targets.
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

  // Coalesce value groups so group pressure is not inflated by tied or
  // loop-carried vregs that share one physical register.
  buildValueGroups(MF);

  SmallVector<Register, 0> AllVGPRs;
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (!MRI->reg_nodbg_empty(R) && isVGPRVirtReg(R))
      AllVGPRs.push_back(R);
  }

  // Nothing to do if the whole function fits in one group: no S_SET_VGPR_MSB is
  // required, and any partition would only spread the footprint and inflate the
  // VGPR count.
  unsigned GlobalFP = maxSimultaneousDwords(AllVGPRs);
  if (GlobalFP <= MSBGroupSize)
    return false;

  // Skip when the baseline occupancy is not a power of two. Power-of-two
  // occupancies are the group-aligned ones (4/2/1 -> 1/2/4 groups), where the
  // VGPR budget is a whole number of groups and spreading across groups fits
  // without dropping a wave. A fractional group budget would push a cluster
  // past the occupancy boundary and cost a wave.
  const SIMachineFunctionInfo *MFIOcc = MF.getInfo<SIMachineFunctionInfo>();
  unsigned VOcc = STI->getOccupancyWithNumVGPRs(
      GlobalFP, MFIOcc->getDynamicVGPRBlockSize());
  unsigned BaseOcc = std::min(VOcc, MFIOcc->getOccupancy());
  if (BaseOcc == 0 || (BaseOcc & (BaseOcc - 1)) != 0)
    return false;

  // Use every group the occupancy budget allows (NumMSBGroups/BaseOcc), but
  // never fewer than the footprint strictly requires. Extra groups cost only
  // VGPRs, which is free while occupancy is the binding limit, and give
  // clusters room to avoid over-subscribing a group.
  unsigned Needed = (GlobalFP + MSBGroupSize - 1) / MSBGroupSize;
  const unsigned EffMSBGroups =
      std::min(NumMSBGroups, std::max(Needed, NumMSBGroups / BaseOcc));

  // One whole-function clustering: capacity and register pressure are global
  // properties, so the graph, footprint, packing and gate all stay global. Loop
  // structure only informs the per-edge weighting inside processRegion.
  SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();
  DenseSet<unsigned> Assigned;
  SmallVector<MachineBasicBlock *, 16> Blocks;
  for (MachineBasicBlock &MBB : MF)
    Blocks.push_back(&MBB);
  processRegion(Blocks, AllVGPRs, EffMSBGroups, Assigned, MFI);

  // Analysis only; never changes the function besides allocation hints.
  return false;
}

void AMDGPUVGPRMSBAffinity::processRegion(ArrayRef<MachineBasicBlock *> Blocks,
                                          ArrayRef<Register> AllVGPRs,
                                          unsigned EffMSBGroups,
                                          DenseSet<unsigned> &Assigned,
                                          SIMachineFunctionInfo *MFI) {
  const unsigned N = MRI->getNumVirtRegs();

  // --- Step 1: build the affinity graph from the scheduled stream. ----------
  //
  // Edge weight between two vregs = sum over the program of the block frequency
  // at points where they appear consecutively in the same MSB slot, i.e. where
  // a mode switch is paid unless they share a group.
  DenseMap<uint64_t, uint64_t> Edges;
  // Earliest instruction ordinal at which each edge occurs, the primary
  // tie-break among equal-weight edges.
  DenseMap<uint64_t, unsigned> EdgePos;

  // A region with many transcendentals is attention-shaped: its accumulator is
  // churned by softmax and worth protecting with the wide-operand weight. A
  // GEMM has only a handful (address/index rcp). Classify per innermost loop,
  // and per function for the benefit gate.
  unsigned NumTrans = 0;
  for (MachineBasicBlock *MBB : Blocks)
    for (MachineInstr &MI : *MBB)
      if (TII->isTRANS(MI))
        ++NumTrans;
  bool HasTrans = NumTrans >= 8;

  DenseMap<const MachineBasicBlock *, char> BlockAttn;
  if (MLI) {
    DenseMap<const MachineLoop *, unsigned> LoopTrans;
    for (MachineBasicBlock *MBB : Blocks)
      if (const MachineLoop *L = MLI->getLoopFor(MBB))
        for (MachineInstr &MI : *MBB)
          if (TII->isTRANS(MI))
            LoopTrans[L]++;
    for (MachineBasicBlock *MBB : Blocks) {
      const MachineLoop *L = MLI->getLoopFor(MBB);
      BlockAttn[MBB] = (L && LoopTrans.lookup(L) >= 8) ? 1 : 0;
    }
  }
  // Updated as the edge walk enters each block; the edge lambda reads it.
  bool CurWeightOn = HasTrans;

  auto addEdge = [&](Register A, Register B, uint64_t W, unsigned P) {
    unsigned a = A.virtRegIndex(), b = B.virtRegIndex();
    if (a == b)
      return;
    // Scale the edge by operand width (capped) so a wide value, e.g. a WMMA
    // accumulator, that alternates in a slot outweighs a scalar doing the same.
    // Only in attention-flavored regions; a wide GEMM accumulator is quiet.
    if (TupleWeight > 1 && CurWeightOn) {
      unsigned Dw = std::min({dwords(A), dwords(B), TupleWeight.getValue()});
      W *= std::max(1u, Dw);
    }
    if (a > b)
      std::swap(a, b);
    uint64_t Key = (uint64_t(a) << 32) | b;
    Edges[Key] += W;
    if (!EdgePos.count(Key))
      EdgePos[Key] = P;
  };

  unsigned Pos = 0; // monotonic instruction ordinal (program order)
  for (MachineBasicBlock *MBBp : Blocks) {
    MachineBasicBlock &MBB = *MBBp;
    CurWeightOn = MLI ? (bool)BlockAttn.lookup(MBBp) : HasTrans;
    uint64_t Freq = blockFreq(MBB);

    // Sticky per-slot state, reset at each block (the lowering pass resets the
    // mode at block boundaries).
    Register LastInSlot[4];

    for (MachineInstr &MI : MBB) {
      if (MI.isMetaInstruction())
        continue;
      ++Pos;

      // A COPY between two VGPR vregs is likely coalesced, co-assigning both to
      // one physreg and one group. Add an affinity edge so the plan agrees, on
      // the same scale as a single-slot boundary (Freq*12).
      if (MI.isCopy()) {
        const MachineOperand &Dst = MI.getOperand(0), &Src = MI.getOperand(1);
        if (Dst.isReg() && Src.isReg() && isVGPRVirtReg(Dst.getReg()) &&
            isVGPRVirtReg(Src.getReg()))
          addEdge(Dst.getReg(), Src.getReg(), Freq * 12, Pos);
      }

      auto Ops = AMDGPU::getVGPRLoweringOperandTables(MI.getDesc());
      if (!Ops.first)
        continue;

      // One S_SET_VGPR_MSB covers every slot that changes at a boundary, so
      // charge the boundary once, distributed across its changed slots (x12
      // keeps it integer for 1..4 slots). A tied accumulator drives both dst
      // and src2, so it contributes to both of its pairings.
      SmallVector<std::pair<Register, Register>, 4> Changed;
      for (unsigned S = 0; S < 4; ++S) {
        const MachineOperand *MO = TII->getNamedOperand(MI, Ops.first[S]);
        if ((!MO || !MO->isReg() || !MO->getReg()) && Ops.second)
          MO = TII->getNamedOperand(MI, Ops.second[S]);
        if (!MO || !MO->isReg() || !MO->getReg())
          continue; // Slot not constrained: stays sticky.

        Register R = MO->getReg();
        if (isVGPRVirtReg(R)) {
          if (LastInSlot[S] && LastInSlot[S] != R)
            Changed.emplace_back(LastInSlot[S], R);
          LastInSlot[S] = R;
        } else if (R.isPhysical() && TRI->isVGPR(*MRI, R)) {
          // A physical VGPR pins the slot to a fixed group; break the run.
          LastInSlot[S] = Register();
        }
      }
      if (!Changed.empty()) {
        uint64_t W = (Freq * 12) / Changed.size();
        for (auto &PR : Changed)
          addEdge(PR.first, PR.second, W, Pos);
      }
    }
  }

  if (Edges.empty())
    return;

  // --- Step 2: footprint-aware greedy union-find into clusters. -------------
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

  // Lazy priority queue: process by weight, but among comparable-weight merges
  // prefer the smallest footprint delta (best time-multiplexing), so clusters
  // grow balanced instead of one group over-filling and orphaning values. Each
  // cluster carries an epoch bumped on every merge; a popped item whose
  // roots/epochs changed has a stale delta and is re-evaluated and re-pushed,
  // so the queue top is always accurate.
  SmallVector<uint64_t, 0> Epoch(N, 0);
  struct Item {
    uint64_t W;
    int Delta;
    unsigned Pos;
    uint64_t Key;
    unsigned RA, RB;
    uint64_t EA, EB;
    bool operator<(const Item &O) const { // max-heap: greater = higher priority
      if (W != O.W)
        return W < O.W; // higher weight first
      if (Delta != O.Delta)
        return Delta > O.Delta; // lower delta first
      if (Pos != O.Pos)
        return Pos > O.Pos; // earlier program order first
      return Key > O.Key;   // lower key first (determinism)
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
    if (ra != It.RA || rb != It.RB || Epoch[ra] != It.EA ||
        Epoch[rb] != It.EB) {
      push(It.Key, It.W); // stale: a touched cluster changed since push.
      continue;
    }
    int Merged = unionFP(ra, rb);
    if (Merged > (int)MSBGroupSize)
      continue; // Refuse: this edge becomes a cut.
    unsigned keep = Rank[ra] < Rank[rb] ? rb : ra;
    doMerge(ra, rb, Merged);
    ++Epoch[keep];
  }

  // --- Step 3: pack clusters into groups (hottest first, least loaded). ------
  // Cluster weight = total internal affinity (how costly it is to split).
  DenseMap<unsigned, uint64_t> ClusterWeight;
  for (auto &[Key, W] : Edges) {
    unsigned ai = unsigned(Key >> 32), bi = unsigned(Key & 0xffffffff);
    if (find(ai) == find(bi))
      ClusterWeight[find(ai)] += W;
  }

  // Only steer clusters that carry real loop-level affinity; cold registers are
  // left unhinted so the allocator packs them naturally.
  uint64_t MaxCW = 0;
  for (auto &KV : ClusterWeight)
    MaxCW = std::max(MaxCW, KV.second);
  uint64_t WeightCutoff = HotClusterDiv ? MaxCW / HotClusterDiv : 0;

  SmallVector<unsigned, 0> Roots;
  for (unsigned I = 0; I < N; ++I)
    if (find(I) == I && !Members[I].empty() &&
        ClusterWeight.lookup(I) > WeightCutoff)
      Roots.push_back(I);
  // Hottest first. First-fit then puts them in the low groups, which matters
  // because group 0 is reset to all-zero at every non-fall-through block entry
  // (including the loop header), so a value in group 0 needs no switch right
  // after a reset.
  llvm::stable_sort(Roots, [&](unsigned A, unsigned B) {
    return ClusterWeight.lookup(A) > ClusterWeight.lookup(B);
  });

  // Time-overlap-aware capacity: a group's load is the maximum simultaneously-
  // live footprint of its clusters, not the sum of their peaks, so clusters
  // with disjoint live ranges (e.g. successive prefetch tiles) share a group
  // instead of each reserving 256 registers.
  SmallVector<SmallVector<Register, 0>, 8> MSBMembers(EffMSBGroups);
  SmallVector<int, 8> MSBLoad(EffMSBGroups, 0);
  DenseMap<unsigned, int> ClusterMSB;
  unsigned NumClusters = Roots.size();
  for (unsigned Root : Roots) {
    int Best = -1, BestLoad = 0;
    for (unsigned B = 0; B < EffMSBGroups; ++B) {
      SmallVector<Register, 16> Combined(MSBMembers[B].begin(),
                                         MSBMembers[B].end());
      Combined.append(Members[Root].begin(), Members[Root].end());
      int L = (int)maxSimultaneousDwords(Combined);
      if (L <= (int)MSBGroupSize) {
        Best = B;
        BestLoad = L;
        break; // lowest group that fits
      }
    }
    if (Best < 0) {
      // No group fits within capacity; over-subscribe the least loaded (the
      // soft hint may spill).
      Best = std::min_element(MSBLoad.begin(), MSBLoad.end()) - MSBLoad.begin();
      MSBMembers[Best].append(Members[Root].begin(), Members[Root].end());
      BestLoad = (int)maxSimultaneousDwords(MSBMembers[Best]);
    } else {
      MSBMembers[Best].append(Members[Root].begin(), Members[Root].end());
    }
    MSBLoad[Best] = BestLoad;
    ClusterMSB[Root] = Best;
  }

  // Commit the plan unless a group is severely over-subscribed: a mild overflow
  // is fine (a few spills, most hints honored, still a win), but a severe one
  // makes the plan unrealizable so the allocator discards the hints.
  int MaxLoad = 0;
  for (int L : MSBLoad)
    MaxLoad = std::max(MaxLoad, L);
  bool Infeasible =
      (uint64_t)MaxLoad * 100 > (uint64_t)MSBGroupSize * OverflowPct;

  // Self-benefit gate. When the schedule is already group-coherent (e.g. a
  // group-aware scheduler), a competing partition can raise the switch count.
  // Compare the predicted switches of the plan against the natural no-hint
  // layout on this schedule and bail if the plan does not beat it. Both sims
  // use the same naive groups for unhinted vregs, isolating the effect of the
  // hints.
  bool NoBenefit = false;
  uint64_t PlanSw = 0, BaseSw = 0;
  if (!Infeasible && BenefitPct) {
    DenseMap<unsigned, int> NaiveMSB = computeNaiveMSB(AllVGPRs, EffMSBGroups);
    DenseMap<unsigned, int> PlanOverride;
    for (unsigned Root : Roots)
      for (Register R : Members[Root])
        PlanOverride[R.virtRegIndex()] = ClusterMSB[Root];
    auto naiveOf = [&](Register R) {
      return NaiveMSB.lookup(R.virtRegIndex());
    };
    BaseSw = simSwitchWeight(Blocks, naiveOf);
    PlanSw = simSwitchWeight(Blocks, [&](Register R) {
      auto It = PlanOverride.find(R.virtRegIndex());
      return It != PlanOverride.end() ? It->second : naiveOf(R);
    });
    // A GEMM region's natural layout is already group-coherent, so a regression
    // is pure critical-path overhead; require a large predicted win there.
    // Attention uses the looser threshold.
    unsigned EffBenefit = !HasTrans ? GemmBenefitPct : BenefitPct;
    // 128-bit products so the comparison can't overflow on a huge function.
    NoBenefit = (unsigned __int128)PlanSw * 100 >=
                (unsigned __int128)BaseSw * EffBenefit;

    // Realizability guard: the plan is realizable only if some used group has
    // slack for the allocator to place the hinted vregs. If every non-empty
    // group is near full, the hints can't be honored and the plan regresses.
    // Only for GEMM regions; attention benefits from cutting its contended
    // switches even with full groups.
    if (!HasTrans) {
      int MinUsed = INT_MAX;
      for (int L : MSBLoad)
        if (L > 0)
          MinUsed = std::min(MinUsed, L);
      if (MinUsed != INT_MAX &&
          (uint64_t)MinUsed * 100 >= (uint64_t)MSBGroupSize * 90)
        NoBenefit = true;
    }
  }

  unsigned NumAssigned = 0;
  if (!Infeasible && !NoBenefit) {
    for (unsigned Root : Roots)
      for (Register R : Members[Root]) {
        // A hotter region already fixed this vreg's group; don't re-hint it.
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
           << " vregsAssigned=" << NumAssigned << " clustersOver256=" << Over
           << " planSw=" << PlanSw << " baseSw=" << BaseSw
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

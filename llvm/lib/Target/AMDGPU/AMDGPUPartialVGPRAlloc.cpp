//===- AMDGPUPartialVGPRAlloc.cpp - Allocate the VGPRs around DS loads ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// On gfx1250 an instruction addresses v0-v255 per operand field; the high
/// bits come from S_SET_VGPR_MSB, which AMDGPULowerVGPREncoding inserts when a
/// field's 256-VGPR window changes between instructions. Right after a DS
/// instruction it cannot co-issue and costs an issue cycle.
///
/// Once the machine scheduler has fixed the order, the register every
/// instruction in a loop uses in each field is known, and so are the size and
/// the live interval of every register. A field keeps the window of the last
/// register used in it; a register in the same field later wants that window.
/// Right after a DS load (a WMMA, a packed VALU op or another DS load comes
/// next) a switch cannot co-issue and costs an issue cycle; elsewhere it
/// co-issues but still costs an instruction to fetch. So both are affinities,
/// the co-issued ones at a fraction of the weight. Clustered heaviest first
/// under a window capacity check on the live intervals, the registers get a
/// window, then a free register in it. Registers that are already physical
/// (named register pins) fix their window. Registers that do not fit are left
/// alone, and the greedy allocator places everything else.
///
/// The choice is committed as an allocation hint (-amdgpu-partial-vgpr-alloc=1)
/// of type AMDGPURI::Window, which copy hints do not replace: greedy tries the
/// register, then others of its window, unless they are taken. Only as many
/// windows are used as the function's VGPR peak needs within its occupancy
/// budget; a function that fits in window 0 is left alone. =2
/// assigns the registers outright (experimental: rewriting a register whose
/// units are also live into the entry block does not yet keep LiveIntervals
/// consistent).
///
/// A front end that knows more than the schedule (what a tensor is for, which
/// tensors belong together, what matters most) says so with
/// llvm.amdgcn.ra.hint(value, window, group, priority, guard): a window fixes
/// the register's window, a group puts its members in one cluster, the
/// priority (percent) scales the weight of its conflicts, and a guard marks a
/// buffered value: one reloaded while the previous copy may still be read. Its
/// live range is extended back by the guard (instructions) from where it is
/// written, for the capacity check and for choosing its register, so it gets
/// room of its own, like a second buffer. With
/// -amdgpu-partial-vgpr-alloc-report=<file> the pass appends a JSON line per
/// function saying which hints it followed and what conflicts are left, for
/// the front end to adjust and compile again.
///
/// Within a window a register is preferred whose previous occupant was last
/// read more than -amdgpu-partial-vgpr-alloc-guard instructions earlier: a
/// write right after a read of the same register by an in-flight WMMA waits
/// for it (s_wait_alu depctr_va_vdst).
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cmath>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-partial-vgpr-alloc"

static cl::opt<unsigned> PartialVGPRAlloc(
    "amdgpu-partial-vgpr-alloc", cl::Hidden, cl::init(0),
    cl::desc("Allocate the VGPRs around DS loads in loops before greedy, so "
             "a DS load shares its windows with the next instruction "
             "(gfx1250). 1: as allocation hints, 2: assign (experimental)"));

static cl::opt<unsigned> PartialVGPRAllocCoIssue(
    "amdgpu-partial-vgpr-alloc-coissue-pct", cl::Hidden, cl::init(10),
    cl::desc("Weight of a window change that co-issues, in percent of one "
             "that cannot (right after a DS load): it costs no cycle but "
             "still an instruction to fetch"));

static cl::opt<unsigned> PartialVGPRAllocGuard(
    "amdgpu-partial-vgpr-alloc-guard", cl::Hidden, cl::init(8),
    cl::desc("Prefer registers not read within this many instructions before "
             "they are written again"));

static cl::opt<std::string> PartialVGPRAllocReport(
    "amdgpu-partial-vgpr-alloc-report", cl::Hidden, cl::init(""),
    cl::desc("Append one JSON line per function to this file: what the "
             "partial VGPR allocation did with the front end's hints"));

static cl::opt<unsigned> PartialVGPRAllocSlack(
    "amdgpu-partial-vgpr-alloc-slack", cl::Hidden, cl::init(32),
    cl::desc("VGPRs per window kept out of reach of the partial allocation"));

namespace {

constexpr unsigned WindowSize = 256;
constexpr unsigned NumWindows = 4;

class AMDGPUPartialVGPRAlloc {
  MachineFunction &MF;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo &MRI;
  LiveIntervals &LIS;
  const MachineLoopInfo &MLI;
  RegisterClassInfo RCI;

  // Registers wanting a window: virtual ones by union-find, with the window
  // their cluster is fixed to (by a physical partner) if any.
  EquivalenceClasses<Register> Clusters;
  DenseMap<Register, int> FixedWindow; // keyed by cluster leader
  DenseMap<Register, double> Weight;   // keyed by register
  // Peak number of VGPRs of each window that are physical before this pass
  // (pins), counted once over the function.
  unsigned FixedPeak[NumWindows] = {};
  // Register units of the physical VGPRs the function refers to.
  BitVector UsedUnits;
  // Windows the function may use: as many as its VGPR peak needs, within its
  // occupancy budget.
  unsigned NumUsable = NumWindows;
  // Live intervals assigned here, per register unit.
  DenseMap<unsigned, SmallVector<const LiveRange *, 4>> Assigned;
  // A buffered value (a front-end hint with a guard): its live range extended
  // back by the guard from where it is first written, so a reload does not
  // take registers an in-flight WMMA may still be reading. Used for capacity
  // and for choosing its register, like a second buffer.
  VNInfo::Allocator VNAlloc;
  DenseMap<Register, std::unique_ptr<LiveRange>> Reserved;
  const LiveRange &rangeOf(Register R) const {
    auto It = Reserved.find(R);
    return It != Reserved.end() ? *It->second
                                : static_cast<const LiveRange &>(LIS.getInterval(R));
  }
  void reserveBuffers();

  struct Edge {
    double W;
    Register A, B;
    unsigned Exposed = 0, CoIssued = 0; // window changes this pair can cause
  };

  // Facts from the front end, per register (a `; plumb.ra_hint` inline asm).
  struct Hint {
    int Window = -1;
    unsigned Group = 0, Priority = 100, Guard = 0;
  };
  DenseMap<Register, Hint> Hints;
  SmallVector<std::string> Conflicts;
  unsigned Hinted = 0, NoRoom = 0, Lost = 0, Unplaced = 0;
  DenseMap<Register, int> WindowOf; // the window chosen for each register

  void collectHints();
  std::string name(Register R) const {
    std::string S;
    raw_string_ostream(S) << printReg(R, TRI);
    return S;
  }
  void report(ArrayRef<Edge> Edges) const;
  unsigned priority(Register R) const {
    auto It = Hints.find(R);
    return It == Hints.end() ? 100 : It->second.Priority;
  }

  bool isVGPROperand(const MachineOperand &MO) const {
    if (!MO.isReg() || !MO.getReg())
      return false;
    Register R = MO.getReg();
    if (R.isPhysical())
      return TRI->isVGPR(MRI, R);
    return TRI->isVectorRegister(MRI, R);
  }

  int windowOf(MCRegister Phys) const {
    return TRI->getHWRegIndex(Phys) / WindowSize;
  }

  unsigned dwords(Register R) const {
    return std::max<unsigned>(1, TRI->getRegSizeInBits(*MRI.getRegClass(R)) / 32);
  }

  // A register unit's live range if the function refers to it, else null:
  // computing one for every VGPR unit would leave LiveIntervals with ranges
  // that later passes (spill lowering) do not update.
  const LiveRange *unitRange(MCRegUnit U) const {
    return UsedUnits.test(static_cast<unsigned>(U)) ? &LIS.getRegUnit(U)
                                                    : nullptr;
  }

  // The register in an operand field of MI, or none.
  Register field(const MachineInstr &MI, AMDGPU::OpName Name) const {
    const MachineOperand *MO = TII->getNamedOperand(MI, Name);
    return MO && isVGPROperand(*MO) ? MO->getReg() : Register();
  }

  bool isDSLoad(const MachineInstr &MI) const {
    return TII->isDS(MI) && MI.mayLoad() && !MI.mayStore();
  }

  // The registers in the dst, src0, src1 and src2 fields of an instruction,
  // as the encoding sees them (a DS load: its address in src0).
  std::array<Register, 4> fields(const MachineInstr &MI) const {
    if (TII->isDS(MI))
      return {field(MI, AMDGPU::OpName::vdst), field(MI, AMDGPU::OpName::addr),
              field(MI, AMDGPU::OpName::data0),
              field(MI, AMDGPU::OpName::data1)};
    if (!SIInstrInfo::isVALU(MI, /*AllowLDSDMA=*/false))
      return {};
    Register Dst = field(MI, AMDGPU::OpName::vdst);
    if (!Dst && MI.getNumExplicitDefs() && isVGPROperand(MI.getOperand(0)))
      Dst = MI.getOperand(0).getReg();
    return {Dst, field(MI, AMDGPU::OpName::src0),
            field(MI, AMDGPU::OpName::src1), field(MI, AMDGPU::OpName::src2)};
  }

  // Largest number of VGPRs live at once among Regs.
  unsigned footprint(ArrayRef<Register> Regs) const;
  bool fits(ArrayRef<Register> Regs, int Window) const {
    unsigned Cap = WindowSize - PartialVGPRAllocSlack;
    if (Window >= 0)
      Cap = Cap > FixedPeak[Window] ? Cap - FixedPeak[Window] : 0;
    return footprint(Regs) <= Cap;
  }
  SmallVector<Register> members(Register Leader) const {
    SmallVector<Register> M;
    for (auto It = Clusters.findLeader(Leader); It != Clusters.member_end();
         ++It)
      M.push_back(*It);
    return M;
  }

  void collectEdges(SmallVectorImpl<Edge> &Edges);
  void cluster(ArrayRef<Edge> Edges);
  bool allocate(Register R, int Window, MCRegister &Out);
  void assign(Register R, MCRegister Phys);

public:
  AMDGPUPartialVGPRAlloc(MachineFunction &MF, LiveIntervals &LIS,
                         const MachineLoopInfo &MLI)
      : MF(MF), MRI(MF.getRegInfo()), LIS(LIS), MLI(MLI) {
    const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
    TII = ST.getInstrInfo();
    TRI = ST.getRegisterInfo();
  }
  bool run();
};

} // end anonymous namespace

unsigned AMDGPUPartialVGPRAlloc::footprint(ArrayRef<Register> Regs) const {
  SmallVector<std::pair<SlotIndex, int>> Events;
  for (Register R : Regs) {
    if (!LIS.hasInterval(R) || LIS.getInterval(R).empty())
      continue;
    int D = dwords(R);
    for (const LiveRange::Segment &S : rangeOf(R)) {
      Events.push_back({S.start, D});
      Events.push_back({S.end, -D});
    }
  }
  llvm::sort(Events, [](auto &L, auto &R) {
    return L.first < R.first || (L.first == R.first && L.second < R.second);
  });
  int Live = 0, Peak = 0;
  for (auto &[_, D] : Events)
    Peak = std::max(Peak, Live += D);
  return Peak;
}

// Every operand field keeps the window of the last register an instruction
// used in it (that is what S_SET_VGPR_MSB sets). An instruction using the field
// with a different register wants the same window; if the previous instruction
// is a DS load, the switch cannot co-issue and costs an issue cycle, otherwise
// it only costs an instruction. Both become affinities, the second at
// -amdgpu-partial-vgpr-alloc-coissue-pct percent of the first.
void AMDGPUPartialVGPRAlloc::collectEdges(SmallVectorImpl<Edge> &Edges) {
  DenseMap<std::pair<Register, Register>, Edge> Sum;
  auto Add = [&](Register A, Register B, double W, bool Exposed) {
    if (A == B)
      return;
    if (B < A)
      std::swap(A, B);
    Edge &E = Sum[{A, B}];
    E.A = A;
    E.B = B;
    E.W += W * priority(A) / 100.0 * priority(B) / 100.0;
    ++(Exposed ? E.Exposed : E.CoIssued);
  };
  const double CoIssue = PartialVGPRAllocCoIssue / 100.0;
  for (MachineBasicBlock &MBB : MF) {
    const MachineLoop *L = MLI.getLoopFor(&MBB);
    if (!L)
      continue;
    double W = std::pow(8.0, L->getLoopDepth());
    Register Last[4] = {};
    MachineInstr *Prev = nullptr;
    for (MachineInstr &MI : MBB) {
      if (MI.isMetaInstruction() || MI.isDebugInstr())
        continue;
      std::array<Register, 4> F = fields(MI);
      bool Exposed = Prev && isDSLoad(*Prev);
      for (unsigned I = 0; I < 4; ++I) {
        if (!F[I])
          continue;
        if (Last[I])
          Add(Last[I], F[I], Exposed ? W : W * CoIssue, Exposed);
        Last[I] = F[I];
      }
      Prev = &MI;
    }
  }
  for (auto &[K, E] : Sum)
    Edges.push_back(E);
  // Heaviest first; ties in a fixed order (Register numbers).
  llvm::stable_sort(Edges, [](const Edge &L, const Edge &R) {
    if (L.W != R.W)
      return L.W > R.W;
    return std::make_pair(L.A, L.B) < std::make_pair(R.A, R.B);
  });
}

// llvm.amdgcn.ra.hint left a FAKE_USE marker on each hinted value at
// selection; PHI elimination and the coalescer have rewritten its operand to
// the register now holding the value. Record the facts per register and
// remove the markers before the machine scheduler, so they do not change the
// schedule. Run by AMDGPUCollectRAHints, right after coalescing.
static bool collectRAHintMarkers(MachineFunction &MF, LiveIntervals &LIS) {
  auto *MFI = MF.getInfo<SIMachineFunctionInfo>();
  const auto &Markers = MFI->getRAHintMarkers();
  if (Markers.empty())
    return false;
  const SIRegisterInfo *TRI = MF.getSubtarget<GCNSubtarget>().getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  auto &Hints = MFI->getRAHints();
  auto &Conflicts = MFI->getRAHintConflicts();
  SmallVector<MachineInstr *> Dead;
  SmallVector<Register> Shrink;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB) {
      if (!MI.isFakeUse())
        continue;
      auto It = Markers.find(&MI);
      if (It == Markers.end())
        continue;
      Dead.push_back(&MI);
      const MachineOperand &MO = MI.getOperand(0);
      if (!MO.isReg() || !MO.getReg().isVirtual() ||
          !TRI->isVectorRegister(MRI, MO.getReg())) {
        ++MFI->getRAHintsLost();
        continue;
      }
      Register R = MO.getReg();
      Shrink.push_back(R);
      const SIMachineFunctionInfo::RAHint &H = It->second;
      auto [Old, New] = Hints.try_emplace(R, H);
      if (New)
        continue;
      // Several 8-VGPR groups, or several hints on one value, that ended up
      // in one register: one window.
      SIMachineFunctionInfo::RAHint &Prev = Old->second;
      if (H.Window >= 0 && Prev.Window >= 0 && H.Window != Prev.Window) {
        std::string S;
        raw_string_ostream(S) << printReg(R, TRI) << ": parts want windows "
                              << Prev.Window << " and " << H.Window;
        Conflicts.push_back(S);
      } else if (Prev.Window < 0) {
        Prev.Window = H.Window;
      }
      Prev.Group = Prev.Group ? Prev.Group : H.Group;
      Prev.Priority = std::max(Prev.Priority, H.Priority);
      Prev.Guard = std::max(Prev.Guard, H.Guard);
    }
  MFI->getRAHintsLost() += Markers.size() - Dead.size();
  for (MachineInstr *MI : Dead) {
    LIS.RemoveMachineInstrFromMaps(*MI);
    MI->eraseFromParent();
  }
  for (Register R : Shrink)
    if (LIS.hasInterval(R))
      LIS.shrinkToUses(&LIS.getInterval(R));
  MFI->clearRAHintMarkers();
  return true;
}

// The facts recorded before scheduling, for the registers still in use.
void AMDGPUPartialVGPRAlloc::collectHints() {
  auto *MFI = MF.getInfo<SIMachineFunctionInfo>();
  for (auto &[R, H] : MFI->getRAHints())
    if (R.isVirtual() && !MRI.reg_nodbg_empty(R) && LIS.hasInterval(R) &&
        TRI->isVectorRegister(MRI, R))
      Hints[R] = {H.Window, H.Group, H.Priority, H.Guard};
  append_range(Conflicts, MFI->getRAHintConflicts());
  Lost = MFI->getRAHintsLost();
}

void AMDGPUPartialVGPRAlloc::cluster(ArrayRef<Edge> Edges) {
  // The front end's groups and windows come first: groups, then the windows
  // of the clusters they form.
  DenseMap<unsigned, Register> GroupFirst;
  for (auto &[R, H] : Hints) {
    Clusters.getOrInsertLeaderValue(R);
    Weight[R] += 0; // take part in the allocation even without edges
    if (!H.Group)
      continue;
    auto [It, New] = GroupFirst.try_emplace(H.Group, R);
    if (!New)
      Clusters.unionSets(It->second, R);
  }
  for (auto &[R, H] : Hints) {
    if (H.Window < 0)
      continue;
    if (H.Window >= int(NumUsable)) {
      Conflicts.push_back((Twine(name(R)) + ": window " +
                           Twine(H.Window) + " is beyond the " +
                           Twine(NumUsable) + " the function needs")
                              .str());
      continue;
    }
    int &W = FixedWindow.try_emplace(Clusters.getLeaderValue(R), H.Window)
                 .first->second;
    if (W != H.Window)
      Conflicts.push_back((Twine(name(R)) + ": window " +
                           Twine(H.Window) + " conflicts with its group's " +
                           Twine(W))
                              .str());
  }
  for (const Edge &E : Edges) {
    Register A = E.A, B = E.B;
    if (A.isPhysical())
      std::swap(A, B);
    if (A.isPhysical())
      continue; // both physical: nothing to decide
    Weight[A] += E.W;
    Register LA = Clusters.getOrInsertLeaderValue(A);
    if (B.isPhysical()) {
      int W = windowOf(B);
      auto It = FixedWindow.find(LA);
      if (It != FixedWindow.end()) {
        if (It->second != W)
          LLVM_DEBUG(dbgs() << "conflict: " << printReg(A, TRI) << " wants "
                            << W << ", has " << It->second << '\n');
        continue;
      }
      if (fits(members(LA), W))
        FixedWindow[LA] = W;
      continue;
    }
    Weight[B] += E.W;
    Register LB = Clusters.getOrInsertLeaderValue(B);
    if (LA == LB)
      continue;
    int WA = FixedWindow.lookup_or(LA, -1), WB = FixedWindow.lookup_or(LB, -1);
    if (WA >= 0 && WB >= 0 && WA != WB)
      continue;
    SmallVector<Register> M = members(LA);
    append_range(M, members(LB));
    int W = WA >= 0 ? WA : WB;
    if (!fits(M, W))
      continue;
    FixedWindow.erase(LA);
    FixedWindow.erase(LB);
    Clusters.unionSets(LA, LB);
    Register L = Clusters.getLeaderValue(LA);
    if (W >= 0)
      FixedWindow[L] = W;
  }
}

void AMDGPUPartialVGPRAlloc::reserveBuffers() {
  for (auto &[R, H] : Hints) {
    if (!H.Guard || !LIS.hasInterval(R) || LIS.getInterval(R).empty())
      continue;
    const LiveInterval &LI = LIS.getInterval(R);
    MachineInstr *Def = LIS.getInstructionFromIndex(LI.beginIndex());
    if (!Def)
      continue; // live into a block: nothing is written here
    // Go back Guard instructions within the block.
    MachineBasicBlock::iterator I = Def->getIterator();
    for (unsigned N = 0; N < H.Guard && I != Def->getParent()->begin();) {
      --I;
      if (!I->isDebugOrPseudoInstr() && !I->isMetaInstruction())
        ++N;
    }
    SlotIndex From = LIS.getInstructionIndex(*I).getRegSlot();
    auto LR = std::make_unique<LiveRange>();
    VNInfo *VN = LR->getNextValue(From, VNAlloc);
    if (From < LI.beginIndex())
      LR->addSegment(LiveRange::Segment(From, LI.beginIndex(), VN));
    for (const LiveRange::Segment &S : LI)
      LR->addSegment(LiveRange::Segment(S.start, S.end, VN));
    Reserved[R] = std::move(LR);
  }
}

bool AMDGPUPartialVGPRAlloc::allocate(Register R, int Window, MCRegister &Out) {
  if (!LIS.hasInterval(R) || LIS.getInterval(R).empty())
    return false;
  const LiveRange &LI = rangeOf(R);
  const TargetRegisterClass *RC = MRI.getRegClass(R);
  SlotIndex Start = LI.beginIndex(), End = LI.endIndex();
  // 0: interferes; 1: free, but last read too recently; 2: free.
  auto Rate = [&](MCPhysReg Phys) {
    int Best = 2;
    auto Check = [&](const LiveRange &Other) {
      if (Other.overlaps(LI))
        return 0;
      // A write within the guard after a read of the same register, in
      // either order: Other read last just before R is written, or R read
      // last just before Other is written.
      int G = Hints.count(R) && Hints.lookup(R).Guard
                  ? int(Hints.lookup(R).Guard)
                  : int(PartialVGPRAllocGuard);
      for (const LiveRange::Segment &S : Other)
        if ((S.end <= Start && S.end.getApproxInstrDistance(Start) <= G) ||
            (S.start >= End && End.getApproxInstrDistance(S.start) <= G))
          return 1;
      return 2;
    };
    for (MCRegUnit U : TRI->regunits(Phys)) {
      if (const LiveRange *UR = unitRange(U))
        Best = std::min(Best, Check(*UR));
      for (const LiveRange *Other : Assigned.lookup(static_cast<unsigned>(U)))
        Best = std::min(Best, Check(*Other));
      if (!Best)
        return 0;
    }
    return Best;
  };
  MCRegister Fallback;
  for (MCPhysReg Phys : RCI.getOrder(RC)) {
    const TargetRegisterClass *PRC = TRI->getPhysRegBaseClass(Phys);
    if (!PRC || !SIRegisterInfo::isVGPRClass(PRC) ||
        windowOf(Phys) != Window ||
        windowOf(Phys) != int((TRI->getHWRegIndex(Phys) + dwords(R) - 1) / WindowSize) ||
        MRI.isReserved(Phys))
      continue;
    int Rating = Rate(Phys);
    if (Rating == 2) {
      Out = Phys;
      break;
    }
    if (Rating == 1 && !Fallback)
      Fallback = Phys;
  }
  if (!Out)
    Out = Fallback;
  if (!Out)
    return false;
  for (MCRegUnit U : TRI->regunits(Out))
    Assigned[static_cast<unsigned>(U)].push_back(&LI);
  return true;
}

// Rewrite R to Phys everywhere, as the allocator's rewriter would, and keep
// LiveIntervals right: R's interval goes, Phys's units are recomputed, and
// blocks R is live into get Phys as a live-in.
void AMDGPUPartialVGPRAlloc::assign(Register R, MCRegister Phys) {
  const LiveInterval &LI = LIS.getInterval(R);
  for (MachineBasicBlock &MBB : MF)
    if (LI.liveAt(LIS.getMBBStartIdx(&MBB)) && !MBB.isLiveIn(Phys))
      MBB.addLiveIn(Phys);
  for (MachineOperand &MO : make_early_inc_range(MRI.reg_operands(R))) {
    MCRegister P = MO.getSubReg() ? TRI->getSubReg(Phys, MO.getSubReg()) : Phys;
    MO.setReg(P);
    MO.setSubReg(0);
    if (MO.isDef())
      MO.setIsUndef(false);
    if (MO.isUse())
      MO.setIsKill(false);
  }
  LIS.removeInterval(R);
  for (MCRegUnit U : TRI->regunits(Phys))
    LIS.removeRegUnit(U);
  // Copies that became identities.
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : make_early_inc_range(MBB))
      if (MI.isCopy() && MI.getOperand(0).getReg() == MI.getOperand(1).getReg() &&
          !MI.getOperand(0).getSubReg() && !MI.getOperand(1).getSubReg() &&
          MI.getOperand(0).getReg() == Phys) {
        LIS.RemoveMachineInstrFromMaps(MI);
        MI.eraseFromParent();
      }
}

bool AMDGPUPartialVGPRAlloc::run() {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  if (!PartialVGPRAlloc || !ST.hasVGPRMSBs())
    return false;
  RCI.runOnMachineFunction(MF);
  bool HadMarkers = false;
  collectHints();

  // Physical VGPRs the function refers to (pins, ABI inputs), counted per
  // window.
  UsedUnits.resize(TRI->getNumRegUnits());
  auto Use = [&](MCRegister Phys) {
    for (MCRegUnit U : TRI->regunits(Phys))
      UsedUnits.set(static_cast<unsigned>(U));
  };
  for (MachineBasicBlock &MBB : MF) {
    for (const auto &LI : MBB.liveins())
      if (TRI->isVGPR(MRI, LI.PhysReg))
        Use(LI.PhysReg);
    for (MachineInstr &MI : MBB)
      for (const MachineOperand &MO : MI.operands())
        if (MO.isReg() && MO.getReg().isPhysical() && TRI->isVGPR(MRI, MO.getReg()))
          Use(MO.getReg().asMCReg());
  }
  for (unsigned I = 0; I < NumWindows * WindowSize; ++I) {
    MCRegister Phys = AMDGPU::VGPR0 + I;
    if (MRI.isReserved(Phys))
      continue;
    for (MCRegUnit U : TRI->regunits(Phys))
      if (const LiveRange *UR = unitRange(U); UR && !UR->empty()) {
        ++FixedPeak[I / WindowSize];
        break;
      }
  }

  // Use only the windows the function needs. If its VGPRs fit in window 0,
  // no instruction needs S_SET_VGPR_MSB and there is nothing to do.
  SmallVector<Register> AllVGPRs;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (!MRI.reg_nodbg_empty(R) && TRI->isVectorRegister(MRI, R))
      AllVGPRs.push_back(R);
  }
  unsigned Peak = footprint(AllVGPRs) + FixedPeak[0] + FixedPeak[1] +
                  FixedPeak[2] + FixedPeak[3];
  unsigned Budget = ST.getMaxNumVGPRs(MF);
  NumUsable = std::min({NumWindows, divideCeil(Budget, WindowSize),
                        std::max(1u, unsigned(divideCeil(Peak, WindowSize)))});
  bool FixedHigh = FixedPeak[1] || FixedPeak[2] || FixedPeak[3];
  if (NumUsable <= 1 && !FixedHigh)
    return HadMarkers;

  reserveBuffers();
  SmallVector<Edge> Edges;
  collectEdges(Edges);
  if (Edges.empty() && Hints.empty()) {
    report(Edges);
    return HadMarkers;
  }
  cluster(Edges);

  // Clusters with no physical partner go to the window with most room.
  SmallVector<Register> Leaders;
  for (auto It = Clusters.begin(); It != Clusters.end(); ++It)
    if ((*It)->isLeader())
      Leaders.push_back((*It)->getData());
  DenseMap<Register, double> ClusterWeight;
  for (Register L : Leaders)
    for (Register R : members(L))
      ClusterWeight[L] += Weight.lookup(R);
  llvm::stable_sort(Leaders, [&](Register A, Register B) {
    return ClusterWeight[A] > ClusterWeight[B];
  });
  SmallVector<Register> InWindow[NumWindows];
  for (Register L : Leaders) {
    SmallVector<Register> M = members(L);
    int W = FixedWindow.lookup_or(L, -1);
    if (W < 0) {
      unsigned Best = ~0u;
      for (int Cand = 0; Cand < int(NumUsable); ++Cand) {
        SmallVector<Register> All(InWindow[Cand]);
        append_range(All, M);
        unsigned F = footprint(All) + FixedPeak[Cand];
        if (fits(All, Cand) && F < Best) {
          Best = F;
          W = Cand;
        }
      }
      if (W < 0) {
        for (Register R : M)
          if (Hints.count(R)) {
            ++Unplaced;
            Conflicts.push_back(name(R) + ": its cluster (" +
                                std::to_string(footprint(M)) +
                                " VGPRs at its peak) fits no window");
          }
        continue;
      }
    }
    append_range(InWindow[W], M);
  }

  bool Changed = false;
  for (int W = 0; W < int(NumWindows); ++W) {
    SmallVector<Register> Regs(InWindow[W]);
    // Widest first, then the most weight.
    llvm::stable_sort(Regs, [&](Register A, Register B) {
      return std::make_pair(dwords(A), Weight.lookup(A)) >
             std::make_pair(dwords(B), Weight.lookup(B));
    });
    SmallVector<std::pair<Register, MCRegister>> Done;
    for (Register R : Regs) {
      MCRegister Phys;
      if (!allocate(R, W, Phys)) {
        LLVM_DEBUG(dbgs() << "no room for " << printReg(R, TRI) << " in window "
                          << W << '\n');
        ++NoRoom;
        if (Hints.count(R))
          Conflicts.push_back(name(R) + ": no free register in window " +
                              std::to_string(W));
        continue;
      }
      WindowOf[R] = W;
      ++Hinted;
      LLVM_DEBUG(dbgs() << printReg(R, TRI) << " -> " << printReg(Phys, TRI)
                        << '\n');
      Done.push_back({R, Phys});
    }
    for (auto [R, Phys] : Done) {
      if (PartialVGPRAlloc == 2)
        assign(R, Phys);
      else
        MRI.setRegAllocationHint(R, AMDGPURI::Window, Phys);
      Changed = true;
    }
    Assigned.clear();
  }
  report(Edges);
  return Changed;
}

// One JSON line: the hints followed, and the window changes the chosen
// windows still leave (counted per pair where both windows are known).
void AMDGPUPartialVGPRAlloc::report(ArrayRef<Edge> Edges) const {
  if (PartialVGPRAllocReport.empty())
    return;
  // "unknown": pairs where a register's window was not chosen here (left to
  // greedy), so whether they cause a change is not known yet.
  auto Win = [&](Register R) -> int {
    if (R.isPhysical())
      return windowOf(R.asMCReg());
    return WindowOf.lookup_or(R, -1);
  };
  unsigned CutExposed = 0, CutCoIssued = 0, Unknown = 0;
  for (const Edge &E : Edges) {
    int A = Win(E.A), B = Win(E.B);
    if (A < 0 || B < 0)
      Unknown += E.Exposed + E.CoIssued;
    else if (A != B) {
      CutExposed += E.Exposed;
      CutCoIssued += E.CoIssued;
    }
  }
  unsigned WithWindow = 0, Followed = 0;
  for (auto &[R, H] : Hints)
    if (H.Window >= 0) {
      ++WithWindow;
      Followed += WindowOf.lookup_or(R, -1) == H.Window;
    }
  std::error_code EC;
  raw_fd_ostream OS(PartialVGPRAllocReport, EC, sys::fs::OF_Append);
  if (EC)
    return;
  OS << "{\"function\": \"" << MF.getName() << "\", \"windows\": " << NumUsable
     << ", \"hints\": " << Hints.size() << ", \"hints_lost\": " << Lost
     << ", \"hints_with_window\": " << WithWindow
     << ", \"windows_followed\": " << Followed
     << ", \"hints_unplaced\": " << Unplaced
     << ", \"placed\": " << Hinted << ", \"no_room\": " << NoRoom
     << ", \"changes_left\": {\"exposed\": " << CutExposed
     << ", \"coissued\": " << CutCoIssued << ", \"unknown\": " << Unknown
     << "}, \"conflicts\": [";
  for (auto [I, C] : enumerate(Conflicts))
    OS << (I ? ", " : "") << '"' << C << '"';
  OS << "]}\n";
}

namespace {
class AMDGPUPartialVGPRAllocLegacy : public MachineFunctionPass {
public:
  static char ID;
  AMDGPUPartialVGPRAllocLegacy() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override {
    return "AMDGPU Partial VGPR Allocation";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (skipFunction(MF.getFunction()))
      return false;
    return AMDGPUPartialVGPRAlloc(
               MF, getAnalysis<LiveIntervalsWrapperPass>().getLIS(),
               getAnalysis<MachineLoopInfoWrapperPass>().getLI())
        .run();
  }
};
} // end anonymous namespace

char AMDGPUPartialVGPRAllocLegacy::ID = 0;
char &llvm::AMDGPUPartialVGPRAllocLegacyID = AMDGPUPartialVGPRAllocLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUPartialVGPRAllocLegacy, DEBUG_TYPE,
                      "AMDGPU Partial VGPR Allocation", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUPartialVGPRAllocLegacy, DEBUG_TYPE,
                    "AMDGPU Partial VGPR Allocation", false, false)

bool llvm::isAMDGPUPartialVGPRAllocEnabled() { return PartialVGPRAlloc != 0; }

namespace {
class AMDGPUCollectRAHintsLegacy : public MachineFunctionPass {
public:
  static char ID;
  AMDGPUCollectRAHintsLegacy() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override {
    return "AMDGPU Collect Register Allocation Hints";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.addPreserved<LiveIntervalsWrapperPass>();
    AU.addPreserved<SlotIndexesWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    return collectRAHintMarkers(
        MF, getAnalysis<LiveIntervalsWrapperPass>().getLIS());
  }
};
} // end anonymous namespace

char AMDGPUCollectRAHintsLegacy::ID = 0;
char &llvm::AMDGPUCollectRAHintsLegacyID = AMDGPUCollectRAHintsLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUCollectRAHintsLegacy, "amdgpu-collect-ra-hints",
                      "AMDGPU Collect Register Allocation Hints", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(AMDGPUCollectRAHintsLegacy, "amdgpu-collect-ra-hints",
                    "AMDGPU Collect Register Allocation Hints", false, false)

FunctionPass *llvm::createAMDGPUPartialVGPRAllocLegacyPass() {
  return new AMDGPUPartialVGPRAllocLegacy();
}

//===-- AMDGPUMLSchedStrategy.h - ML-focused Scheduler Strategy -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// ML-focused scheduling strategy for AMDGPU.
//
//===----------------------------------------------------------------------===//

#include "GCNHazardRecognizer.h"
#include "GCNSchedStrategy.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/MachineCycleAnalysis.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// Instruction Flavor Classification
//===----------------------------------------------------------------------===//

enum class InstructionFlavor : uint8_t {
  WMMA,            // WMMA/MFMA matrix operations
  SingleCycleVALU, // Single-cycle VALU (not TRANS32, not multi-cycle CVT)
  TRANS,           // Transcendental ops (v_exp, v_log, etc.) - 2 cycles
  MultiCycleVALU,  // 4-cycle CVT instructions (v_cvt_scalef32_pk8_fp8_f32)
  VMEM,            // FLAT/GLOBAL memory operations
  DS,              // LDS/GDS operations
  SALU,            // Scalar ALU
  DMA,             // Tensor DMA operations
  Fence,           // Fences and waits
  Other,           // Everything else
  NUM_FLAVORS
};

inline StringRef getFlavorName(InstructionFlavor F) {
  switch (F) {
  case InstructionFlavor::WMMA:           return "WMMA";
  case InstructionFlavor::SingleCycleVALU:return "VALU(1c)";
  case InstructionFlavor::TRANS:
    return "TRANS";
  case InstructionFlavor::MultiCycleVALU:
    return "VALU(Nc)";
  case InstructionFlavor::VMEM:           return "VMEM";
  case InstructionFlavor::DS:             return "DS";
  case InstructionFlavor::SALU:           return "SALU";
  case InstructionFlavor::DMA:            return "DMA";
  case InstructionFlavor::Fence:          return "Fence";
  case InstructionFlavor::Other:          return "Other";
  case InstructionFlavor::NUM_FLAVORS:    return "???";
  }
  llvm_unreachable("Unknown InstructionFlavor");
}

inline StringRef getFlavorShortName(InstructionFlavor F) {
  switch (F) {
  case InstructionFlavor::WMMA:           return "W";
  case InstructionFlavor::SingleCycleVALU:return "V";
  case InstructionFlavor::TRANS:
    return "T";
  case InstructionFlavor::MultiCycleVALU:
    return "C";
  case InstructionFlavor::VMEM:           return "M";
  case InstructionFlavor::DS:             return "D";
  case InstructionFlavor::SALU:           return "S";
  case InstructionFlavor::DMA:            return "X";
  case InstructionFlavor::Fence:          return "F";
  case InstructionFlavor::Other:          return "O";
  case InstructionFlavor::NUM_FLAVORS:    return "?";
  }
  llvm_unreachable("Unknown InstructionFlavor");
}

InstructionFlavor classifyFlavor(const MachineInstr *MI, const SIInstrInfo *SII);

using FlavorGroup = SmallVector<InstructionFlavor, 4>;

namespace FlavorGroups {
  inline FlavorGroup allVALU() {
    return {InstructionFlavor::SingleCycleVALU, InstructionFlavor::TRANS,
            InstructionFlavor::MultiCycleVALU};
  }
  inline FlavorGroup allMem() {
    return {InstructionFlavor::VMEM, InstructionFlavor::DS,
            InstructionFlavor::DMA};
  }
  inline FlavorGroup individual(InstructionFlavor F) {
    return {F};
  }
  inline FlavorGroup all() {
    FlavorGroup G;
    for (unsigned I = 0; I < static_cast<unsigned>(InstructionFlavor::NUM_FLAVORS); ++I)
      G.push_back(static_cast<InstructionFlavor>(I));
    return G;
  }
}

//===----------------------------------------------------------------------===//
// FlavorMask - Bitmask representation of InstructionFlavor combinations
//===----------------------------------------------------------------------===//

/// Bitmask representation of InstructionFlavor combinations.
/// Each bit corresponds to an InstructionFlavor enum value.
using FlavorMask = uint16_t;

namespace FlavorMasks {
constexpr FlavorMask None = 0;
constexpr FlavorMask WMMA =
    1 << static_cast<unsigned>(InstructionFlavor::WMMA);
constexpr FlavorMask VALU1c =
    1 << static_cast<unsigned>(InstructionFlavor::SingleCycleVALU);
constexpr FlavorMask TRANS =
    1 << static_cast<unsigned>(InstructionFlavor::TRANS);
constexpr FlavorMask CVT =
    1 << static_cast<unsigned>(InstructionFlavor::MultiCycleVALU);
constexpr FlavorMask VMEM =
    1 << static_cast<unsigned>(InstructionFlavor::VMEM);
constexpr FlavorMask DS = 1 << static_cast<unsigned>(InstructionFlavor::DS);
constexpr FlavorMask SALU =
    1 << static_cast<unsigned>(InstructionFlavor::SALU);
constexpr FlavorMask DMA = 1 << static_cast<unsigned>(InstructionFlavor::DMA);
constexpr FlavorMask Fence =
    1 << static_cast<unsigned>(InstructionFlavor::Fence);
constexpr FlavorMask Other =
    1 << static_cast<unsigned>(InstructionFlavor::Other);
constexpr FlavorMask All =
    (1 << static_cast<unsigned>(InstructionFlavor::NUM_FLAVORS)) - 1;

// WMMA slot combinations (matching GCNHazardRecognizer WMMASlotType)
// MemCoExec slots: can co-issue VMEM, DS, or SALU
constexpr FlavorMask MemCoExec = VMEM | DS | SALU;
// ValuCoExec slots: can co-issue VMEM, DS, SALU, SingleCycleVALU, or TRANS
constexpr FlavorMask ValuCoExec = VMEM | DS | SALU | VALU1c | TRANS;
// ValuBlocked slots: can co-issue VMEM, DS, SALU, or next WMMA (no VALU/TRANS)
constexpr FlavorMask ValuBlocked = VMEM | DS | SALU | WMMA;
} // namespace FlavorMasks

inline FlavorMask flavorToMask(InstructionFlavor F) {
  return 1 << static_cast<unsigned>(F);
}

inline bool maskContainsFlavor(FlavorMask Mask, InstructionFlavor F) {
  return Mask & flavorToMask(F);
}

std::string getMaskName(FlavorMask Mask);

//===----------------------------------------------------------------------===//
// SlotRequirement - Coexecution slot requirement
//===----------------------------------------------------------------------===//

/// Represents a single coexecution slot requirement.
/// A slot can be satisfied by any instruction matching the FlavorMask.
struct SlotRequirement {
  FlavorMask AcceptedFlavors; // Bitmask of flavors that can fill this slot
  unsigned RequiredCount;     // Number of instructions needed
  unsigned ReadyCount;        // Number of ready instructions matching mask

  SlotRequirement(FlavorMask Mask = FlavorMasks::None, unsigned Count = 0)
      : AcceptedFlavors(Mask), RequiredCount(Count), ReadyCount(0) {}

  bool isSatisfied() const { return ReadyCount >= RequiredCount; }
  unsigned getDeficit() const {
    return ReadyCount >= RequiredCount ? 0 : RequiredCount - ReadyCount;
  }
};

/// AMDGPU-specific scheduling decision reasons. These provide more granularity
/// than the generic CandReason enum for debugging purposes.
enum class AMDGPUSchedReason : uint8_t {
  None,
  WMMACoexec,              // tryVALUCoexecSlot chose based on WMMA coexecution
  CritResourceBalance,     // tryCriticalResource chose based on resource pressure
  CritResourceDep,         // tryCriticalResourceDependency chose based on enabling
  // Shadow Mix: defer until shadow-filling instructions ready
  ShadowDeferWMMA,         // Deferred WMMA waiting for co-exec (VALU/DS)
  ShadowDeferTRANS32,      // Deferred TRANS32 waiting for VALU
  // Shadow Priority: prefer long-latency so short ones fill shadow
  ShadowPriorityWMMAOverDS,    // Prefer WMMA over DS (DS fills shadow)
  ShadowPriorityWMMAOverSALU,  // Prefer WMMA over SALU (SALU fills shadow)
  ShadowPriorityCVTOverDS,     // Prefer CVT over DS
  ShadowPriorityCVTOverSALU,   // Prefer CVT over SALU
  ShadowPriorityTRANS32OverVALU, // Prefer TRANS32 over 1c VALU
  ShadowPreferVALU1cOverSALUForTRANS, // Prefer VALU1c over SALU for TRANS shadow
  // Shadow Mix: prefer instruction that enables co-exec candidates
  ShadowEnableDirect,      // Directly enables needed co-exec flavor
  ShadowEnableLookahead,   // On path to enabling co-exec via lookahead
  NUM_REASONS
};

inline StringRef getReasonName(AMDGPUSchedReason R) {
  switch (R) {
  case AMDGPUSchedReason::None:              return "None";
  case AMDGPUSchedReason::WMMACoexec:        return "WMMACoexec";
  case AMDGPUSchedReason::CritResourceBalance: return "CritResource";
  case AMDGPUSchedReason::CritResourceDep:   return "CritResourceDep";
  case AMDGPUSchedReason::ShadowDeferWMMA:   return "ShadowDeferWMMA";
  case AMDGPUSchedReason::ShadowDeferTRANS32: return "ShadowDeferTRANS32";
  case AMDGPUSchedReason::ShadowPriorityWMMAOverDS:   return "ShadowWMMA>DS";
  case AMDGPUSchedReason::ShadowPriorityWMMAOverSALU: return "ShadowWMMA>SALU";
  case AMDGPUSchedReason::ShadowPriorityCVTOverDS:    return "ShadowCVT>DS";
  case AMDGPUSchedReason::ShadowPriorityCVTOverSALU:  return "ShadowCVT>SALU";
  case AMDGPUSchedReason::ShadowPriorityTRANS32OverVALU: return "ShadowTRANS32>VALU";
  case AMDGPUSchedReason::ShadowPreferVALU1cOverSALUForTRANS: return "ShadowVALU>SALU(TRANS)";
  case AMDGPUSchedReason::ShadowEnableDirect:    return "ShadowEnableDirect";
  case AMDGPUSchedReason::ShadowEnableLookahead: return "ShadowEnableLookahead";
  case AMDGPUSchedReason::NUM_REASONS:       return "???";
  }
  llvm_unreachable("Unknown AMDGPUSchedReason");
}

class RegionMixInfo {
public:
  static constexpr unsigned NumFlavors =
      static_cast<unsigned>(InstructionFlavor::NUM_FLAVORS);

private:
  SmallVector<SmallVector<SUnit *, 8>, NumFlavors> AllSUs;

  SmallVector<SmallSetVector<SUnit *, 8>, NumFlavors> ScheduledSUs;

  SmallVector<DenseMap<SUnit *, unsigned>, NumFlavors> SUCycles;

  SmallVector<unsigned, NumFlavors> ReadyCounts;

  SmallVector<unsigned, NumFlavors> TotalCycles;

  SmallVector<unsigned, NumFlavors> ScheduledCycles;

public:
  void reset() {
    AllSUs.clear();
    AllSUs.resize(NumFlavors);
    ScheduledSUs.clear();
    ScheduledSUs.resize(NumFlavors);
    SUCycles.clear();
    SUCycles.resize(NumFlavors);
    ReadyCounts.assign(NumFlavors, 0);
    TotalCycles.assign(NumFlavors, 0);
    ScheduledCycles.assign(NumFlavors, 0);
  }

  void addSU(SUnit *SU, InstructionFlavor F, unsigned Cycles = 1) {
    unsigned Idx = static_cast<unsigned>(F);
    AllSUs[Idx].push_back(SU);
    TotalCycles[Idx] += Cycles;
    SUCycles[Idx][SU] = Cycles;
    if (SU->isTopReady())
      ReadyCounts[Idx]++;
  }

  void markScheduled(SUnit *SU, InstructionFlavor F) {
    unsigned Idx = static_cast<unsigned>(F);
    ScheduledSUs[Idx].insert(SU);
    ScheduledCycles[Idx] += SUCycles[Idx].lookup(SU);
    if (ReadyCounts[Idx] > 0)
      ReadyCounts[Idx]--;
  }

  void updateReadyCounts() {
    for (unsigned I = 0; I < NumFlavors; ++I) {
      ReadyCounts[I] = 0;
      for (SUnit *SU : AllSUs[I]) {
        if (!ScheduledSUs[I].contains(SU) && SU->isTopReady())
          ReadyCounts[I]++;
      }
    }
  }

  unsigned getReadyCount(InstructionFlavor F) const {
    return ReadyCounts[static_cast<unsigned>(F)];
  }

  unsigned getReadyCount(const FlavorGroup &G) const {
    unsigned Count = 0;
    for (InstructionFlavor F : G)
      Count += getReadyCount(F);
    return Count;
  }

  unsigned getPendingCount(InstructionFlavor F) const {
    unsigned Idx = static_cast<unsigned>(F);
    unsigned Total = AllSUs[Idx].size();
    unsigned Scheduled = ScheduledSUs[Idx].size();
    unsigned Ready = ReadyCounts[Idx];
    return Total - Scheduled - Ready;
  }

  unsigned getTotalCount(InstructionFlavor F) const {
    return AllSUs[static_cast<unsigned>(F)].size();
  }

  unsigned getRemainingCount(InstructionFlavor F) const {
    unsigned Idx = static_cast<unsigned>(F);
    return AllSUs[Idx].size() - ScheduledSUs[Idx].size();
  }

  unsigned getTotalCycles(InstructionFlavor F) const {
    return TotalCycles[static_cast<unsigned>(F)];
  }

  unsigned getRemainingCycles(InstructionFlavor F) const {
    unsigned Idx = static_cast<unsigned>(F);
    return TotalCycles[Idx] - ScheduledCycles[Idx];
  }

  ArrayRef<SUnit *> getSUs(InstructionFlavor F) const {
    return AllSUs[static_cast<unsigned>(F)];
  }

  void dumpMix(raw_ostream &OS, bool Detailed = false) const;

  void dumpReadyPending(raw_ostream &OS) const;
};

class HardwareUnitInfo {
private:
  // Ideally these would be sorted on how much they enable a secondary resource,
  // but that creates a chicken and egg problem and compile time explosion.
  SmallSetVector<SUnit *, 16> PrioritySUs;
  SmallSetVector<SUnit *, 16> AllSUs;
  unsigned TotalCycles = 0;
  InstructionFlavor Type;
  unsigned Exposed = 0;
  unsigned RemainingExposed = 0;

public:
  // TODO -- handle this better.
  bool IsAsync = false;
  unsigned Idx;
  bool ProducesCoexecWindow = false;
  unsigned CoexecWindowSize = 0;

  HardwareUnitInfo() {}


  unsigned size() { return AllSUs.size(); }
  SUnit *getTargetSU() { return *PrioritySUs.begin(); }

  // TODO -- should we allow looking past the a single depth?
  SUnit *getNextTargetSU(bool LookDeep = false) {
    for (auto *PrioritySU : PrioritySUs) {
      if (!PrioritySU->isTopReady())
        return PrioritySU;
    }

    if (!LookDeep)
      return nullptr;

    // TODO -- we may want to think about more advance strategies here.
    // For example, for GEMMs, we may want to target WMMAs by using the
    // same A operand for exmaple, leading to even better DS_READ -> WMMA
    // patterns.
    unsigned MinDepth = std::numeric_limits<unsigned int>::max();
    SUnit *TargetSU = nullptr;
    for (auto *SU : AllSUs) {
      if (SU->isScheduled)
        continue;

      if (SU->isTopReady())
        continue;

      if (SU->getDepth() < MinDepth) {
        MinDepth = SU->getDepth();
        TargetSU = SU;
      }
    }
    return TargetSU;
  }

  void fixupFIFO(unsigned FIFOSize) {TotalCycles /= FIFOSize;}

  unsigned getTotalCycles() { return TotalCycles; }

  void setType(unsigned TheType) {
    assert(TheType < (unsigned)InstructionFlavor::NUM_FLAVORS);
    Type = (InstructionFlavor)(TheType);
  }

  InstructionFlavor getType() const { return Type; }

  void setExposedCount(unsigned ExposedCount) {
    Exposed = ExposedCount;
    RemainingExposed = ExposedCount;
  }

  unsigned getRemainingExposed() { return RemainingExposed; }

  void reduceRemainingExposed() {
    if (RemainingExposed > 0)
      --RemainingExposed;
  }

  void insert(SUnit *SU, unsigned ReleaseAtCycle) {
    bool Inserted = AllSUs.insert(SU);
    TotalCycles += ReleaseAtCycle;

    assert(Inserted);
    if (PrioritySUs.empty()) {
      PrioritySUs.insert(SU);
      return;
    }
    unsigned SUDepth = SU->getDepth();
    unsigned CurrDepth = (*PrioritySUs.begin())->getDepth();
    if (SUDepth > CurrDepth)
      return;

    if (SUDepth == CurrDepth) {
      PrioritySUs.insert(SU);
      return;
    }

    // SU is lower depth and should be prioritized.
    PrioritySUs.clear();
    PrioritySUs.insert(SU);
  }

  bool contains(SUnit *SU) { return AllSUs.contains(SU); }

  bool isHigherPriority(SUnit *SU, SUnit *Other) {
    for (auto *SUOrder : PrioritySUs) {
      if (SUOrder == SU)
        return true;
      if (SUOrder == Other)
        return false;
    }

    return false;
  }

  void schedule(SUnit *SU, unsigned ReleaseAtCycle, const SIInstrInfo *TII) {
    AllSUs.remove(SU);
    PrioritySUs.remove(SU);

    if (getType() != InstructionFlavor::DS) {
      if (TotalCycles != 0)
        assert(ReleaseAtCycle <= TotalCycles);
      if (TotalCycles > ReleaseAtCycle)
        TotalCycles -= ReleaseAtCycle;
      else TotalCycles = 0;
    }
    else {
      TotalCycles -= ReleaseAtCycle / 16;
    }



    if (AllSUs.empty())
      return;
    if (PrioritySUs.empty()) {
      SmallVector<SUnit *, 16> NewPrioritySUs;
      for (auto SU : AllSUs) {
        if (NewPrioritySUs.empty()) {
          NewPrioritySUs.push_back(SU);
          continue;
        }
        unsigned SUDepth = SU->getDepth();
        unsigned CurrDepth = (*NewPrioritySUs.begin())->getDepth();
        if (SUDepth > CurrDepth)
          continue;

        if (SUDepth == CurrDepth) {
          NewPrioritySUs.push_back(SU);
          continue;
        }

        // SU is lower depth and should be prioritized.
        NewPrioritySUs.clear();
        NewPrioritySUs.push_back(SU);
      }

      if (getType() == InstructionFlavor::WMMA) {
        SIInstrInfo *SII = const_cast<SIInstrInfo *>(TII);
        sort(NewPrioritySUs, [SII](SUnit *A, SUnit *B){
          auto ASrc1 = SII->getNamedOperand(*A->getInstr(), AMDGPU::OpName::src0);
          auto BSrc1 = SII->getNamedOperand(*B->getInstr(), AMDGPU::OpName::src0);

          if (!ASrc1->isReg() || !BSrc1->isReg())
            return !ASrc1->isReg();
          
          auto AReg = ASrc1->getReg();
          auto BReg = BSrc1->getReg();

          if (!AReg.isVirtual() || !BReg.isVirtual())
            return !AReg.isVirtual();
          
          return AReg.id() < BReg.id();

        }); 

        PrioritySUs.clear();
        for (auto SU : NewPrioritySUs) {
          PrioritySUs.insert(SU);
        }
      }
    }
  }

  void reset() {
    AllSUs.clear();
    PrioritySUs.clear();
    TotalCycles = 0;
    IsAsync = false;
    Exposed = 0;
    RemainingExposed = 0;
    ProducesCoexecWindow = false;
    CoexecWindowSize = 0;
  }

  void printPriorities() {
    errs() << "New Priority WMMAs: \n";
    for (auto SU : PrioritySUs) {
      SU->getInstr()->dump();
    }
  }

  void print() {
    errs() << "HWUI: " << getFlavorName(Type) << "\n";
    errs() << "Count: " << AllSUs.size() << "\n";
    errs() << "TotalCycles: " << getTotalCycles() << "\n";
    errs() << "RemainingExposed: " << RemainingExposed << "\n";
    errs() << "ProducesCoexecWindow: " << ProducesCoexecWindow << "\n";
  }
};

class CoexecWindow {
private:
  static constexpr unsigned NumFlavors =
      static_cast<unsigned>(InstructionFlavor::NUM_FLAVORS);

public:
  InstructionFlavor WindowProducer = InstructionFlavor::Other;

  unsigned ReadyCost = 0;
  bool IsPopulated = false;
  bool IsActive = false;
  bool IsReady = false;
  bool ProducerIsReady = false;

  // Slot-based requirements (indexed by slot number, not flavor).
  // Each slot has a bitmask of acceptable flavors and a required count.
  SmallVector<SlotRequirement, 8> Slots;

  unsigned StartCycle = 0;
  unsigned EndCycle = 0;

  void printStatus() {
    errs() << "CoexecWindow status (Producer: " << getFlavorName(WindowProducer)
           << ")\n";
    for (unsigned I = 0; I < Slots.size(); ++I) {
      errs() << "  Slot " << I << ": Mask=" << getMaskName(Slots[I].AcceptedFlavors)
             << ", Required=" << Slots[I].RequiredCount
             << ", Ready=" << Slots[I].ReadyCount
             << (Slots[I].isSatisfied() ? " [OK]" : " [NEED]") << "\n";
    }
  }

  // New constructor taking slot requirements as bitmasks
  CoexecWindow(InstructionFlavor ProducerFlavor,
               ArrayRef<SlotRequirement> SlotReqs, RegionMixInfo &MixInfo)
      : WindowProducer(ProducerFlavor),
        Slots(SlotReqs.begin(), SlotReqs.end()) {
    // Add producer requirement as a slot
    Slots.push_back(SlotRequirement(flavorToMask(ProducerFlavor), 1));
    refreshMixInfo(MixInfo);
    IsPopulated = true;
  }

  // Legacy constructor for backward compatibility
  // UseSingularFlavor: when true, use individual flavor masks (DS only, SALU only)
  //                    when false, use combined masks (DS|SALU|VMEM for E slots)
  CoexecWindow(InstructionFlavor ProducerFlavor, unsigned RequiredVALU1c,
               unsigned RequiredSALU, unsigned RequiredDS, unsigned RequiredVMEM,
               RegionMixInfo &MixInfo, bool UseSingularFlavor = false)
      : WindowProducer(ProducerFlavor) {
    if (UseSingularFlavor) {
      // Singular flavor mode: each slot requires exactly one flavor type
      if (RequiredDS > 0)
        Slots.push_back(SlotRequirement(FlavorMasks::DS, RequiredDS));
      if (RequiredSALU > 0)
        Slots.push_back(SlotRequirement(FlavorMasks::SALU, RequiredSALU));
      if (RequiredVMEM > 0)
        Slots.push_back(SlotRequirement(FlavorMasks::VMEM, RequiredVMEM));
      if (RequiredVALU1c > 0)
        Slots.push_back(SlotRequirement(FlavorMasks::VALU1c, RequiredVALU1c));
    } else {
      // Combined mask mode: E slots can hold DS OR SALU OR VMEM
      // Combine DS, SALU, and VMEM requirements into MemCoExec slots
      unsigned MemCoExecCount = RequiredDS + RequiredSALU + RequiredVMEM;
      if (MemCoExecCount > 0)
        Slots.push_back(SlotRequirement(FlavorMasks::MemCoExec, MemCoExecCount));
      // VALU slots: SingleCycleVALU only
      if (RequiredVALU1c > 0)
        Slots.push_back(SlotRequirement(FlavorMasks::VALU1c, RequiredVALU1c));
    }
    // Producer slot
    Slots.push_back(SlotRequirement(flavorToMask(ProducerFlavor), 1));

    refreshMixInfo(MixInfo);
    IsPopulated = true;
  }

  void clear() {
    Slots.clear();
    ReadyCost = 0;
    IsPopulated = false;
    EndCycle = 0;
    IsActive = false;
    IsReady = false;
    WindowProducer = InstructionFlavor::Other;
    ProducerIsReady = false;
  }

  void copy(CoexecWindow &Other) {
    Slots = Other.Slots;
    ReadyCost = Other.ReadyCost;
    IsPopulated = Other.IsPopulated;
    EndCycle = Other.EndCycle;
    IsActive = Other.IsActive;
    IsReady = Other.IsReady;
    WindowProducer = Other.WindowProducer;
    ProducerIsReady = Other.ProducerIsReady;
  }

  CoexecWindow() = default;

  void refreshMixInfo(RegionMixInfo &MixInfo) {
    ReadyCost = 0;
    ProducerIsReady = true;

    for (auto &Slot : Slots) {
      Slot.ReadyCount = 0;
      // Sum ready counts for all flavors in the mask
      for (unsigned I = 0; I < NumFlavors; ++I) {
        InstructionFlavor F = static_cast<InstructionFlavor>(I);
        if (maskContainsFlavor(Slot.AcceptedFlavors, F))
          Slot.ReadyCount += MixInfo.getReadyCount(F);
      }

      if (!Slot.isSatisfied()) {
        ReadyCost += Slot.getDeficit();
        // Check if this is the producer slot
        if (Slot.AcceptedFlavors == flavorToMask(WindowProducer))
          ProducerIsReady = false;
      }
    }
    IsReady = (ReadyCost == 0);
  }

  // Get the first flavor needed to satisfy an unsatisfied slot
  InstructionFlavor getNeededFlavor() {
    if (!ProducerIsReady)
      return WindowProducer;

    for (const auto &Slot : Slots) {
      if (!Slot.isSatisfied()) {
        // Return first flavor in the mask (prioritize certain flavors)
        for (auto CandidateFlavor :
             {InstructionFlavor::SingleCycleVALU, InstructionFlavor::DS,
              InstructionFlavor::SALU, InstructionFlavor::VMEM}) {
          if (maskContainsFlavor(Slot.AcceptedFlavors, CandidateFlavor))
            return CandidateFlavor;
        }
      }
    }
    return InstructionFlavor::Other;
  }

  // Get all flavors that could help satisfy unsatisfied slots
  void getNeededFlavors(SmallVectorImpl<InstructionFlavor> &NeededFlavors) {
    if (!ProducerIsReady) {
      NeededFlavors.push_back(WindowProducer);
      return;
    }

    for (const auto &Slot : Slots) {
      if (!Slot.isSatisfied()) {
        for (unsigned I = 0; I < NumFlavors; ++I) {
          InstructionFlavor F = static_cast<InstructionFlavor>(I);
          if (maskContainsFlavor(Slot.AcceptedFlavors, F) &&
              !llvm::is_contained(NeededFlavors, F))
            NeededFlavors.push_back(F);
        }
      }
    }
  }

  // Get the bitmask of all flavors that can satisfy any unsatisfied slot
  FlavorMask getNeededMask() {
    if (!ProducerIsReady)
      return flavorToMask(WindowProducer);

    FlavorMask Mask = FlavorMasks::None;
    for (const auto &Slot : Slots) {
      if (!Slot.isSatisfied())
        Mask |= Slot.AcceptedFlavors;
    }
    return Mask;
  }

  // Get unsatisfied slots with their deficits
  void getUnsatisfiedSlots(
      SmallVectorImpl<std::pair<FlavorMask, unsigned>> &UnsatisfiedSlots) {
    for (const auto &Slot : Slots) {
      if (!Slot.isSatisfied()) {
        UnsatisfiedSlots.push_back({Slot.AcceptedFlavors, Slot.getDeficit()});
      }
    }
  }

  void schedule(MachineInstr *MI) {
    // Placeholder for future implementation
  }

  bool isReady() {
    for (const auto &Slot : Slots) {
      if (!Slot.isSatisfied())
        return false;
    }
    return true;
  }
};

class CandidateHeuristics {
public:
  CandidateHeuristics() = default;

  ScheduleDAGMI *DAG;
  const SIInstrInfo *SII;
  const SIRegisterInfo *SRI;

  const TargetSchedModel *SchedModel;

  SmallVector<SUnit *, 16> SchedDSR;

  SmallVector<SUnit *, 16> SchedMFMA;

  SmallVector<HardwareUnitInfo, 8> HWUInfo;

  SmallVector<SUnit *, 16> SchedTDM;

  SmallVector<SUnit *, 16> SchedEXP;

  RegionMixInfo MixInfo;

  AMDGPUSchedReason LastAMDGPUReason = AMDGPUSchedReason::None;

  CoexecWindow CurrentWindow;
  CoexecWindow NextWindow;

  unsigned FencedDSRLatency = 0;
  unsigned ScheduledSUCount = 0;
  unsigned SpaceBetweenFence = 0;

  bool IsPrologue = false;
  bool IsEpilogue = false;

  bool CollectedUse = false;

  bool IsPostRA;
  bool IsMemoryBound;

  bool ResourcePriorityToProducerVal;
  bool ResourcePriorityCoexecWindowSizeVal;
  bool ResourcePriorityExposedCyclesVal;
  bool EnableShadowMixVal;
  unsigned ShadowMixWMMAMinVALU1cVal;
  unsigned ShadowMixWMMAMinDSVal;
  unsigned ShadowMixWMMAMinSALUVal;
  unsigned ShadowMixWMMAMinVMEMVal;
  bool ShadowMixRulesVal;
  bool ShadowPriorityWMMAOverDSVal;
  bool ShadowPriorityWMMAOverSALUVal;
  bool ShadowPriorityCVTOverDSVal;
  bool ShadowPriorityCVTOverSALUVal;
  bool ShadowPriorityTRANS32OverVALU1cVal;
  bool ShadowPreferVALU1cOverSALUForTRANSVal;
  unsigned ShadowMixLookaheadDepthVal;
  unsigned ShadowMixMaxBlockingCostVal;
  unsigned ShadowMixMaxVisitedVal;
  unsigned ShadowMixMaxCandidatesVal;
  bool IgnoreVALUVal;
  unsigned DSFIFOSizeVal;
  unsigned DSLatencyFIFOVal;
  unsigned DSLatencySplitVal;
  unsigned LatencyForSignalVal;
  unsigned DSLatencyForFenceVal;
  unsigned ResourceToBalanceVal;

  unsigned CurrCycle;

  void initialize(ScheduleDAGMI *DAG, GCNHazardRecognizer *HazardRec,
                  const TargetSchedModel *SchedModel,
                  const TargetRegisterInfo *TRI, bool IsMemoryBound = false,
                  bool IsPostRA = false);

  void setParams();
  void collectUse(GCNHazardRecognizer *HazardRec);

  unsigned getHWUICyclesForInst(SUnit *SU, unsigned ReleaseAtCycle);
  void sortResources();
  void calculateHiddenLatency(GCNHazardRecognizer *HazardRec);

  void schedNode(SUnit *SU, GCNHazardRecognizer *HazardRec);
  void bumpNode(SUnit *SU, SchedBoundary *Zone);

  unsigned getLatencyStallCycles(SUnit *SU, SchedBoundary *Zone);
  bool tryAsyncPipe(GenericSchedulerBase::SchedCandidate &TryCand,
                    GenericSchedulerBase::SchedCandidate &Cand,
                    SchedBoundary *Zone);
  bool tryVALUCoexecSlot(GenericSchedulerBase::SchedCandidate &TryCand,
                         GenericSchedulerBase::SchedCandidate &Cand,
                         SchedBoundary *Zone);
  bool tryShadowMix(GenericSchedulerBase::SchedCandidate &TryCand,
                    GenericSchedulerBase::SchedCandidate &Cand,
                    SchedBoundary *Zone,
                    AMDGPUSchedReason &OutReason);

  bool tryWMMACoolOff(GenericSchedulerBase::SchedCandidate &TryCand,
                      GenericSchedulerBase::SchedCandidate &Cand,
                      SchedBoundary *Zone);

  bool
  tryCriticalResourceDependency(GenericSchedulerBase::SchedCandidate &TryCand,
                                GenericSchedulerBase::SchedCandidate &Cand,
                                SchedBoundary *Zone, bool IsAsync);

  bool tryCriticalResource(GenericSchedulerBase::SchedCandidate &TryCand,
                           GenericSchedulerBase::SchedCandidate &Cand,
                           SchedBoundary *Zone);

  void populateCandidateWindow(
      CoexecWindow &Window,
      InstructionFlavor Flavor = InstructionFlavor::NUM_FLAVORS);

  bool coexecWindowIsReady(CoexecWindow *Window, SchedBoundary *Zone, unsigned &MaxStall);

  void dumpRegionSummary();
};


class AMDGPUMLSchedStrategy final : public GCNSchedStrategy {
protected:
  bool tryCandidateBalanced(SchedCandidate &Cand, SchedCandidate &TryCand,
                            SchedBoundary *Zone);

  SmallVector<SUnit *, 16> SchedDSR;

  SmallVector<SUnit *, 16> SchedMFMA;

  SmallVector<HardwareUnitInfo, 8> HWUInfo;

  SmallVector<SUnit *, 16> SchedTDM;

  SmallVector<SUnit *, 16> SchedEXP;

  RegionMixInfo MixInfo;

  CandidateHeuristics Heurs;

  AMDGPUSchedReason LastAMDGPUReason = AMDGPUSchedReason::None;

  unsigned FencedDSRLatency = 0;

  void collectUse();

  bool IsPrologue = false;
  bool IsEpilogue = false;

  unsigned getHWUICyclesForInst(SUnit *SU, const SIInstrInfo *SII, unsigned ReleaseAtCycle);

  bool tryPendingCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                           SchedBoundary *Zone) override;

  void pickNodeFromQueue(SchedBoundary &Zone, const CandPolicy &ZonePolicy,
                         const RegPressureTracker &RPTracker,
                         SchedCandidate &Cand, bool &IsPending,
                         bool IsBottomUp);

  SUnit *pickNode(bool &IsTopNode) override;

  void dumpPickSummary(SUnit *SU, bool IsTopNode, SchedCandidate &Cand);

public:
  AMDGPUMLSchedStrategy(const MachineSchedContext *C);

  void initialize(ScheduleDAGMI *DAG) override;

  void schedNode(SUnit *SU, bool IsTopNode) override;

  MachineCycleInfo CI;
};

class AMDGPUMLPostSchedStrategy : public PostGenericScheduler {
protected:
  unsigned FencedDSRLatency = 0;

  SmallVector<SUnit *, 16> SchedDSR;

  SmallVector<SUnit *, 16> SchedMFMA;

  SmallVector<HardwareUnitInfo, 8> HWUInfo;

  SmallVector<SUnit *, 16> SchedTDM;

  SmallVector<SUnit *, 16> SchedEXP;

  RegionMixInfo MixInfo;

  CandidateHeuristics Heurs;

  AMDGPUSchedReason LastAMDGPUReason = AMDGPUSchedReason::None;

  bool IsPrologue = false;
  bool IsEpilogue = false;

  void dumpPickSummary(SUnit *SU, bool IsTopNode, SchedCandidate &Cand);

public:
  AMDGPUMLPostSchedStrategy(const MachineSchedContext *C);

  void schedNode(SUnit *SU, bool IsTopNode) override;

  bool tryCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                    SchedBoundary *Zone) override;

  bool tryPendingCandidate(SchedCandidate &Cand, SchedCandidate &TryCand,
                           SchedBoundary *Zone);

  void collectUse();

  void enterRegion(MachineBasicBlock *bb, MachineBasicBlock::iterator begin,
                   MachineBasicBlock::iterator end, unsigned regioninstrs);

  unsigned getHWUICyclesForInst(SUnit *SU, const SIInstrInfo *SII,
                                unsigned ReleaseAtCycle);

  void initialize(ScheduleDAGMI *DAG) override;

  SUnit *pickNode(bool &IsTopNode) override;

  void pickNodeFromQueue(SchedBoundary &Zone, SchedCandidate &Cand,
                         bool &IsPending);
};

} // End namespace llvm
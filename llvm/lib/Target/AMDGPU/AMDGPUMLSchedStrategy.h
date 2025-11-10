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

  // TODO -- should we be using lookahead to more accurately define costs?
  unsigned ReadyCost = 0;
  bool IsPopulated = false;
  bool IsActive = false;
  bool IsReady = false;
  bool ProducerIsReady = false;
  SmallVector<unsigned, NumFlavors> RequiredCounts;
  SmallVector<unsigned, NumFlavors> ReadyCounts;

  unsigned StartCycle = 0;
  unsigned EndCycle = 0;

  void printStatus() {
    errs() << "printing status\n";
    for (unsigned I = 0; I < NumFlavors; I++) {
      if (RequiredCounts[I]) {
        InstructionFlavor Flavor = static_cast<InstructionFlavor>(I);
        errs() << "Flavor: " << getFlavorName(Flavor)
               << ", Required: " << RequiredCounts[I]
               << ", Ready: " << ReadyCounts[I] << "\n";
      }
    }
  }

  CoexecWindow(InstructionFlavor ProducerFlavor, unsigned RequiredVALU1c,
               unsigned RequiredSALU, unsigned RequiredDS,
               RegionMixInfo MixInfo)
      : WindowProducer(ProducerFlavor), RequiredCounts(NumFlavors),
        ReadyCounts(NumFlavors) {
    ProducerIsReady = true;
    for (unsigned I = 0; I < NumFlavors; I++) {
      InstructionFlavor Flavor = static_cast<InstructionFlavor>(I);
      ReadyCounts[I] = MixInfo.getReadyCount(Flavor);
      if (Flavor == InstructionFlavor::DS)
        ReadyCounts[I] += MixInfo.getReadyCount(InstructionFlavor::SALU);
      RequiredCounts[I] = 0;
      if (Flavor == InstructionFlavor::SALU) {
        RequiredCounts[I] = RequiredSALU;
      }
      if (Flavor == InstructionFlavor::SingleCycleVALU) {
        RequiredCounts[I] = RequiredVALU1c;
      }
      if (Flavor == InstructionFlavor::DS) {
        RequiredCounts[I] = RequiredDS;
      }
      if (Flavor == WindowProducer) {
        RequiredCounts[I] = 1;
      }

      if (RequiredCounts[I] > ReadyCounts[I]) {
        if (Flavor == ProducerFlavor)
          ProducerIsReady = false;
        ReadyCost += RequiredCounts[I] - ReadyCounts[I];
      }
    }
    IsPopulated = true;
    IsReady = ReadyCost == 0;
  }

  void clear() {
    RequiredCounts.clear();
    ReadyCounts.clear();
    ReadyCost = 0;
    IsPopulated = false;
    EndCycle = 0;
    IsActive = false;
    IsReady = false;
    WindowProducer = InstructionFlavor::Other;
    ProducerIsReady = false;
  }

  void copy(CoexecWindow &Other) {
    RequiredCounts = Other.RequiredCounts;
    ReadyCounts = Other.ReadyCounts;
    ReadyCost = Other.ReadyCost;
    IsPopulated = Other.IsPopulated;
    EndCycle = Other.EndCycle;
    IsActive = Other.IsActive;
    IsReady = Other.IsReady;
    WindowProducer = Other.WindowProducer;
    ProducerIsReady = Other.ProducerIsReady;
  }

  CoexecWindow() = default;

  void refreshMixInfo(RegionMixInfo MixInfo) {
    if (!IsPopulated) {
      RequiredCounts.resize(NumFlavors);
      ReadyCounts.resize(NumFlavors);
    }
    ProducerIsReady = true;

    unsigned ReadyCost = 0;
    for (unsigned I = 0; I < NumFlavors; I++) {
      InstructionFlavor Flavor = static_cast<InstructionFlavor>(I);
      ReadyCounts[I] = MixInfo.getReadyCount(Flavor);
      if (RequiredCounts[I] > ReadyCounts[I]) {
        if (Flavor == WindowProducer)
          ProducerIsReady = false;
        ReadyCost += RequiredCounts[I] - ReadyCounts[I];
      }
    }
    IsReady = ReadyCost == 0;
  }

  // TODO -- should we be prioritizing based on some heuruistic?
  // Currently, using hardcoded order.
  InstructionFlavor getNeededFlavor() {
    if (!ProducerIsReady)
      return WindowProducer;
    for (auto CandidateFlavor :
         {InstructionFlavor::SingleCycleVALU, InstructionFlavor::DS,
          InstructionFlavor::SALU}) {
      unsigned Index = static_cast<unsigned>(CandidateFlavor);
      if (RequiredCounts[Index] > ReadyCounts[Index])
        return CandidateFlavor;
    }
    return InstructionFlavor::Other;
  }

  void getNeededFlavors(SmallVectorImpl<InstructionFlavor> &NeededFlavors) {
    if (!ProducerIsReady) {
      NeededFlavors.push_back(WindowProducer);
      return;
    }

    for (auto CandidateFlavor :
         {InstructionFlavor::SingleCycleVALU, InstructionFlavor::DS,
          InstructionFlavor::SALU}) {
      unsigned Index = static_cast<unsigned>(CandidateFlavor);
      if (RequiredCounts[Index] > ReadyCounts[Index])
        NeededFlavors.push_back(CandidateFlavor);
    }
  }

  void schedule(MachineInstr *MI) {
    /*
    if (MI == WindowProducer) {
      assert(!IsActive);
      WindowProducer = nullptr;
      IsActive = true;
      return;
    }*/
  }

  bool isReady() {
    for (unsigned I = 0; I < NumFlavors; I++) {
      if (RequiredCounts[I] > ReadyCounts[I])
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
//===-- AMDGPUMLSchedStrategy.cpp - ML-focused Scheduler Strategy ---------===//
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

#include "AMDGPUMLSchedStrategy.h"
#include "GCNHazardRecognizer.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "machine-scheduler"

using namespace llvm;

static cl::opt<unsigned> ResourcesToBalance(
    "amdgpu-resource-balancing", cl::Hidden,
    cl::desc("Number of resources we will try to balance during scheduling."),
    cl::init(200));

static cl::opt<unsigned>
    DSLatencySplit("amdgpu-ds-latency-split", cl::Hidden,
                   cl::desc("Latency between neighboring DS_LOAD."),
                   cl::init(0));

static cl::opt<unsigned>
    DSLatencyFIFO("amdgpu-ds-fifo-latency", cl::Hidden,
                  cl::desc("Hazard latency DS_LOAD FIFO full."), cl::init(60));

static cl::opt<unsigned> LatencyForSignal(
    "amdgpu-signal-latency", cl::Hidden,
    cl::desc("Hazard latency between BARRIER_SIGNAL and BARRIER_WAIT."),
    cl::init(35));

static cl::opt<unsigned>
    DSLatencyForFence("amdgpu-ds-fence-latency", cl::Hidden,
                      cl::desc("Hazard latency between DS_LOAD and FENCE."),
                      cl::init(60));

static cl::opt<unsigned> DSFIFOSize("amdgpu-ds-fifo-size", cl::Hidden,
                                    cl::desc("DS_LOAD FIFO size."),
                                    cl::init(16));
static cl::opt<unsigned>
    DSLatency("amdgpu-ds-latency", cl::Hidden,
              cl::desc("Latency of DS_LOAD for resource usage."), cl::init(60));

static cl::opt<bool> IgnoreVALU(
    "amdgpu-ignore-valu-resource-balancing", cl::Hidden,
    cl::desc(
        "Whether or not to ignore VALU unit when balancing HW resoiurces."),
    cl::init(false));

static cl::opt<bool> AvoidEXP(
  "amdgpu-avoid-exp-final-islot", cl::Hidden,
  cl::desc("Whether or not to try avoiding putting v_exp in final I slot of WMMA."),
  cl::init(true));

static cl::opt<bool> EnableShadowMix(
  "amdgpu-shadow-mix", cl::Hidden,
  cl::desc("Enable shadow mix lookahead scheduling to ensure co-execution "
           "opportunities (e.g., WMMA with VALU/DS) are available before "
           "scheduling long-latency instructions."),
  cl::init(true));

static cl::opt<unsigned> ShadowMixWMMAMinVALU1c(
    "amdgpu-shadow-mix-wmma-min-valu1c", cl::Hidden,
    cl::desc("Minimum number of ready single-cycle VALU instructions required "
             "before scheduling a WMMA instruction. Setting to 0 disables "
             "VALU check. WMMA has 8 co-execution slots that can be filled "
             "with 1-cycle VALU, so 2 ensures interleaving opportunity."),
    cl::init(2));

static cl::opt<unsigned> ShadowMixWMMAMinDS(
    "amdgpu-shadow-mix-wmma-min-ds", cl::Hidden,
    cl::desc(
        "Minimum number of ready DS (LDS load/store) instructions required "
        "before scheduling a WMMA instruction. Setting to 0 disables "
        "DS check. WMMA's first co-exec slot can accommodate a DS_LOAD."),
    cl::init(1));

static cl::opt<unsigned> ShadowMixWMMAMinSALU(
    "amdgpu-shadow-mix-wmma-min-salu", cl::Hidden,
    cl::desc("Minimum number of ready SALU instructions required "
             "before scheduling a WMMA instruction. Setting to 0 disables "
             "SALU check. SALU can fill WMMA co-exec slots."),
    cl::init(0));

static cl::opt<unsigned> ShadowMixLookaheadDepth(
    "amdgpu-shadow-mix-lookahead-depth", cl::Hidden,
    cl::desc("Maximum dependency depth to search when looking for pending "
             "co-execution candidates. Higher values find more opportunities "
             "but increase compile time. 0 disables lookahead (direct enable "
             "only)."),
    cl::init(15));

static cl::opt<unsigned> ShadowMixMaxBlockingCost(
    "amdgpu-shadow-mix-max-blocking-cost", cl::Hidden,
    cl::desc(
        "Maximum number of blocking instructions acceptable when searching "
        "for pending co-execution candidates. Targets with higher cost are "
        "ignored as too expensive to reach."),
    cl::init(10));

static cl::opt<unsigned> ShadowMixMaxVisited(
  "amdgpu-shadow-mix-max-visited", cl::Hidden,
  cl::desc("Maximum number of nodes to visit during blocking count BFS. "
           "Limits compile time for large DAGs."),
  cl::init(1000));

static cl::opt<unsigned> ShadowMixMaxCandidates(
    "amdgpu-shadow-mix-max-candidates", cl::Hidden,
    cl::desc(
        "Maximum number of pending candidates to examine during lookahead. "
        "Limits compile time when many pending instructions exist."),
    cl::init(13));

// Shadow priority rules: prefer long-latency instruction so short ones fill shadow.
// These are toggleable for debugging the increasingly specific heuristics.
static cl::opt<bool> ShadowPriorityWMMAOverDS(
    "amdgpu-shadow-priority-wmma-over-ds", cl::Hidden,
    cl::desc("Prefer WMMA over DS when both ready (DS fills WMMA shadow)."),
    cl::init(true));

static cl::opt<bool> ShadowPriorityWMMAOverSALU(
    "amdgpu-shadow-priority-wmma-over-salu", cl::Hidden,
    cl::desc("Prefer WMMA over SALU when both ready (SALU fills WMMA shadow)."),
    cl::init(false));

static cl::opt<bool> ShadowPriorityCVTOverDS(
  "amdgpu-shadow-priority-cvt-over-ds", cl::Hidden,
  cl::desc("Prefer CVT over DS when both ready (DS fills CVT shadow)."),
  cl::init(true));

static cl::opt<bool> ShadowPriorityCVTOverSALU(
  "amdgpu-shadow-priority-cvt-over-salu", cl::Hidden,
  cl::desc("Prefer CVT over SALU when both ready (SALU fills CVT shadow)."),
  cl::init(true));

static cl::opt<bool> ShadowPriorityTRANS32OverVALU1c(
    "amdgpu-shadow-priority-trans32-over-valu1c", cl::Hidden,
    cl::desc("Prefer TRANS32 (v_exp etc) over 1-cycle VALU when both ready "
             "(VALU fills TRANS32 shadow)."),
    cl::init(true));

static cl::opt<bool> ShadowDeferTRANS32(
  "amdgpu-shadow-defer-trans32", cl::Hidden,
  cl::desc("Defer TRANS32 instructions until enough VALU ready to fill shadow."),
  cl::init(true));

static cl::opt<unsigned> ShadowMixTRANS32MinVALU1c(
    "amdgpu-shadow-mix-trans32-min-valu1c", cl::Hidden,
    cl::desc(
        "Minimum 1-cycle VALU instructions ready before scheduling TRANS32 "
        "(when -amdgpu-shadow-defer-trans32 enabled)."),
    cl::init(1));

static cl::opt<bool> ShadowPreferVALU1cOverSALUForTRANS(
  "amdgpu-shadow-prefer-valu-over-salu-for-trans", cl::Hidden,
  cl::desc("When filling TRANS32 shadow, prefer VALU1c over SALU "
           "(reserve SALU for WMMA/CVT shadows)."),
  cl::init(true));

// Flag seems universally beneficial, may make sense to delete
static cl::opt<bool> ResourcePriorityToProducer(
    "amdgpu-resource-priority-coexec-producer", cl::Hidden,
    cl::desc("When sorting critical resources, whether to give more priortiy "
             "to coexecution producers over exposed latency."),
    cl::init(true));

// Flag seems universally beneficial, may make sense to delete
static cl::opt<bool> ResourcePriorityCoexecWindowSize(
    "amdgpu-resource-priority-coexec-windows-size", cl::Hidden,
    cl::desc(
        "When sorting critical resources, whether to sort by window size."),
    cl::init(true));

// Flag seems universally beneficial, may make sense to delete
static cl::opt<bool> ResourcePriorityExposedCycles(
    "amdgpu-resource-priority-coexec-exposed-cycles", cl::Hidden,
    cl::desc(
        "When sorting critical resources, whether to sort by exposed cycles."),
    cl::init(true));

static cl::opt<bool> ShadowMixRules(
    "amdgpu-use-shadow-mix-rules", cl::Hidden,
    cl::desc("Whether to use instruction type rules in tryShadowMix."),
    cl::init(false));

static cl::opt<unsigned> ResourcesToBalancePro(
    "amdgpu-resource-balancing-pro", cl::Hidden,
    cl::desc("Number of resources we will try to balance during scheduling."),
    cl::init(90));

static cl::opt<unsigned>
    DSLatencySplitPro("amdgpu-ds-latency-split-pro", cl::Hidden,
                      cl::desc("Latency between neighboring DS_LOAD."),
                      cl::init(0));

static cl::opt<unsigned>
    DSLatencyFIFOPro("amdgpu-ds-fifo-latency-pro", cl::Hidden,
                     cl::desc("Hazard latency DS_LOAD FIFO full."),
                     cl::init(48));

static cl::opt<unsigned> LatencyForSignalPro(
    "amdgpu-signal-latency-pro", cl::Hidden,
    cl::desc("Hazard latency between BARRIER_SIGNAL and BARRIER_WAIT."),
    cl::init(33));

static cl::opt<unsigned>
    DSLatencyForFencePro("amdgpu-ds-fence-latency-pro", cl::Hidden,
                         cl::desc("Hazard latency between DS_LOAD and FENCE."),
                         cl::init(60));

static cl::opt<unsigned> DSFIFOSizePro("amdgpu-ds-fifo-size-pro", cl::Hidden,
                                       cl::desc("DS_LOAD FIFO size."),
                                       cl::init(10));
static cl::opt<unsigned>
    DSLatencyPro("amdgpu-ds-latency-pro", cl::Hidden,
                 cl::desc("Latency of DS_LOAD for resource usage."),
                 cl::init(53));

static cl::opt<bool> IgnoreVALUPro(
    "amdgpu-ignore-valu-resource-balancing-pro", cl::Hidden,
    cl::desc(
        "Whether or not to ignore VALU unit when balancing HW resoiurces."),
    cl::init(true));

static cl::opt<bool>
    AvoidEXPPro("amdgpu-avoid-exp-final-islot-pro", cl::Hidden,
                cl::desc("Whether or not to try avoiding putting v_exp in "
                         "final I slot of WMMA."),
                cl::init(true));

static cl::opt<bool> EnableShadowMixPro(
    "amdgpu-shadow-mix-pro", cl::Hidden,
    cl::desc("Enable shadow mix lookahead scheduling to ensure co-execution "
             "opportunities (e.g., WMMA with VALU/DS) are available before "
             "scheduling long-latency instructions."),
    cl::init(true));

static cl::opt<unsigned> ShadowMixWMMAMinVALU1cPro(
    "amdgpu-shadow-mix-wmma-min-valu1c-pro", cl::Hidden,
    cl::desc("Minimum number of ready single-cycle VALU instructions required "
             "before scheduling a WMMA instruction. Setting to 0 disables "
             "VALU check. WMMA has 8 co-execution slots that can be filled "
             "with 1-cycle VALU, so 2 ensures interleaving opportunity."),
    cl::init(3));

static cl::opt<unsigned> ShadowMixWMMAMinDSPro(
    "amdgpu-shadow-mix-wmma-min-ds-pro", cl::Hidden,
    cl::desc(
        "Minimum number of ready DS (LDS load/store) instructions required "
        "before scheduling a WMMA instruction. Setting to 0 disables "
        "DS check. WMMA's first co-exec slot can accommodate a DS_LOAD."),
    cl::init(1));

static cl::opt<unsigned> ShadowMixWMMAMinSALUPro(
    "amdgpu-shadow-mix-wmma-min-salu-pro", cl::Hidden,
    cl::desc("Minimum number of ready SALU instructions required "
             "before scheduling a WMMA instruction. Setting to 0 disables "
             "SALU check. SALU can fill WMMA co-exec slots."),
    cl::init(0));

static cl::opt<unsigned> ShadowMixLookaheadDepthPro(
    "amdgpu-shadow-mix-lookahead-depth-pro", cl::Hidden,
    cl::desc("Maximum dependency depth to search when looking for pending "
             "co-execution candidates. Higher values find more opportunities "
             "but increase compile time. 0 disables lookahead (direct enable "
             "only)."),
    cl::init(8));

static cl::opt<unsigned> ShadowMixMaxBlockingCostPro(
    "amdgpu-shadow-mix-max-blocking-cost-pro", cl::Hidden,
    cl::desc(
        "Maximum number of blocking instructions acceptable when searching "
        "for pending co-execution candidates. Targets with higher cost are "
        "ignored as too expensive to reach."),
    cl::init(13));

static cl::opt<unsigned> ShadowMixMaxVisitedPro(
    "amdgpu-shadow-mix-max-visited-pro", cl::Hidden,
    cl::desc("Maximum number of nodes to visit during blocking count BFS. "
             "Limits compile time for large DAGs."),
    cl::init(1000));

static cl::opt<unsigned> ShadowMixMaxCandidatesPro(
    "amdgpu-shadow-mix-max-candidates-pro", cl::Hidden,
    cl::desc(
        "Maximum number of pending candidates to examine during lookahead. "
        "Limits compile time when many pending instructions exist."),
    cl::init(13));

// Shadow priority rules: prefer long-latency instruction so short ones fill
// shadow. These are toggleable for debugging the increasingly specific
// heuristics.
static cl::opt<bool> ShadowPriorityWMMAOverDSPro(
    "amdgpu-shadow-priority-wmma-over-ds-pro", cl::Hidden,
    cl::desc("Prefer WMMA over DS when both ready (DS fills WMMA shadow)."),
    cl::init(true));

static cl::opt<bool> ShadowPriorityWMMAOverSALUPro(
    "amdgpu-shadow-priority-wmma-over-salu-pro", cl::Hidden,
    cl::desc("Prefer WMMA over SALU when both ready (SALU fills WMMA shadow)."),
    cl::init(true));

static cl::opt<bool> ShadowPriorityCVTOverDSPro(
    "amdgpu-shadow-priority-cvt-over-ds-pro", cl::Hidden,
    cl::desc("Prefer CVT over DS when both ready (DS fills CVT shadow)."),
    cl::init(false));

static cl::opt<bool> ShadowPriorityCVTOverSALUPro(
    "amdgpu-shadow-priority-cvt-over-salu-pro", cl::Hidden,
    cl::desc("Prefer CVT over SALU when both ready (SALU fills CVT shadow)."),
    cl::init(true));

static cl::opt<bool> ShadowPriorityTRANS32OverVALU1cPro(
    "amdgpu-shadow-priority-trans32-over-valu1c-pro", cl::Hidden,
    cl::desc("Prefer TRANS32 (v_exp etc) over 1-cycle VALU when both ready "
             "(VALU fills TRANS32 shadow)."),
    cl::init(true));

static cl::opt<bool> ShadowDeferTRANS32Pro(
    "amdgpu-shadow-defer-trans32-pro", cl::Hidden,
    cl::desc(
        "Defer TRANS32 instructions until enough VALU ready to fill shadow."),
    cl::init(true));

static cl::opt<unsigned> ShadowMixTRANS32MinVALU1cPro(
    "amdgpu-shadow-mix-trans32-min-valu1c-pro", cl::Hidden,
    cl::desc(
        "Minimum 1-cycle VALU instructions ready before scheduling TRANS32 "
        "(when -amdgpu-shadow-defer-trans32 enabled)."),
    cl::init(0));

static cl::opt<bool> ShadowPreferVALU1cOverSALUForTRANSPro(
    "amdgpu-shadow-prefer-valu-over-salu-for-trans-pro", cl::Hidden,
    cl::desc("When filling TRANS32 shadow, prefer VALU1c over SALU "
             "(reserve SALU for WMMA/CVT shadows)."),
    cl::init(true));

// Flag seems universally beneficial, may make sense to delete
static cl::opt<bool> ResourcePriorityToProducerPro(
    "amdgpu-resource-priority-coexec-producer-pro", cl::Hidden,
    cl::desc("When sorting critical resources, whether to give more priortiy "
             "to coexecution producers over exposed latency."),
    cl::init(true));

// Flag seems universally beneficial, may make sense to delete
static cl::opt<bool> ResourcePriorityCoexecWindowSizePro(
    "amdgpu-resource-priority-coexec-windows-size-pro", cl::Hidden,
    cl::desc(
        "When sorting critical resources, whether to sort by window size."),
    cl::init(true));

// Flag seems universally beneficial, may make sense to delete
static cl::opt<bool> ResourcePriorityExposedCyclesPro(
    "amdgpu-resource-priority-coexec-exposed-cycles-pro", cl::Hidden,
    cl::desc(
        "When sorting critical resources, whether to sort by exposed cycles."),
    cl::init(true));

static cl::opt<bool> ShadowMixRulesPro(
    "amdgpu-use-shadow-mix-rules-pro", cl::Hidden,
    cl::desc("Whether to use instruction type rules in tryShadowMix."),
    cl::init(false));

static cl::opt<unsigned> ResourcesToBalanceEpi(
    "amdgpu-resource-balancing-epi", cl::Hidden,
    cl::desc("Number of resources we will try to balance during scheduling."),
    cl::init(50));

static cl::opt<unsigned>
    DSLatencySplitEpi("amdgpu-ds-latency-split-epi", cl::Hidden,
                      cl::desc("Latency between neighboring DS_LOAD."),
                      cl::init(0));

static cl::opt<unsigned>
    DSLatencyFIFOEpi("amdgpu-ds-fifo-latency-epi", cl::Hidden,
                     cl::desc("Hazard latency DS_LOAD FIFO full."),
                     cl::init(60));

static cl::opt<unsigned> LatencyForSignalEpi(
    "amdgpu-signal-latency-epi", cl::Hidden,
    cl::desc("Hazard latency between BARRIER_SIGNAL and BARRIER_WAIT."),
    cl::init(35));

static cl::opt<unsigned>
    DSLatencyForFenceEpi("amdgpu-ds-fence-latency-epi", cl::Hidden,
                         cl::desc("Hazard latency between DS_LOAD and FENCE."),
                         cl::init(60));

static cl::opt<unsigned> DSFIFOSizeEpi("amdgpu-ds-fifo-size-epi", cl::Hidden,
                                       cl::desc("DS_LOAD FIFO size."),
                                       cl::init(16));
static cl::opt<unsigned>
    DSLatencyEpi("amdgpu-ds-latency-epi", cl::Hidden,
                 cl::desc("Latency of DS_LOAD for resource usage."),
                 cl::init(60));

static cl::opt<bool> IgnoreVALUEpi(
    "amdgpu-ignore-valu-resource-balancing-epi", cl::Hidden,
    cl::desc(
        "Whether or not to ignore VALU unit when balancing HW resoiurces."),
    cl::init(false));

static cl::opt<bool>
    AvoidEXPEpi("amdgpu-avoid-exp-final-islot-epi", cl::Hidden,
                cl::desc("Whether or not to try avoiding putting v_exp in "
                         "final I slot of WMMA."),
                cl::init(true));

static cl::opt<bool> EnableShadowMixEpi(
    "amdgpu-shadow-mix-epi", cl::Hidden,
    cl::desc("Enable shadow mix lookahead scheduling to ensure co-execution "
             "opportunities (e.g., WMMA with VALU/DS) are available before "
             "scheduling long-latency instructions."),
    cl::init(true));

static cl::opt<unsigned> ShadowMixWMMAMinVALU1cEpi(
    "amdgpu-shadow-mix-wmma-min-valu1c-epi", cl::Hidden,
    cl::desc("Minimum number of ready single-cycle VALU instructions required "
             "before scheduling a WMMA instruction. Setting to 0 disables "
             "VALU check. WMMA has 8 co-execution slots that can be filled "
             "with 1-cycle VALU, so 2 ensures interleaving opportunity."),
    cl::init(2));

static cl::opt<unsigned> ShadowMixWMMAMinDSEpi(
    "amdgpu-shadow-mix-wmma-min-ds-epi", cl::Hidden,
    cl::desc(
        "Minimum number of ready DS (LDS load/store) instructions required "
        "before scheduling a WMMA instruction. Setting to 0 disables "
        "DS check. WMMA's first co-exec slot can accommodate a DS_LOAD."),
    cl::init(1));

static cl::opt<unsigned> ShadowMixWMMAMinSALUEpi(
    "amdgpu-shadow-mix-wmma-min-salu-epi", cl::Hidden,
    cl::desc("Minimum number of ready SALU instructions required "
             "before scheduling a WMMA instruction. Setting to 0 disables "
             "SALU check. SALU can fill WMMA co-exec slots."),
    cl::init(0));

static cl::opt<unsigned> ShadowMixLookaheadDepthEpi(
    "amdgpu-shadow-mix-lookahead-depth-epi", cl::Hidden,
    cl::desc("Maximum dependency depth to search when looking for pending "
             "co-execution candidates. Higher values find more opportunities "
             "but increase compile time. 0 disables lookahead (direct enable "
             "only)."),
    cl::init(15));

static cl::opt<unsigned> ShadowMixMaxBlockingCostEpi(
    "amdgpu-shadow-mix-max-blocking-cost-epi", cl::Hidden,
    cl::desc(
        "Maximum number of blocking instructions acceptable when searching "
        "for pending co-execution candidates. Targets with higher cost are "
        "ignored as too expensive to reach."),
    cl::init(10));

static cl::opt<unsigned> ShadowMixMaxVisitedEpi(
    "amdgpu-shadow-mix-max-visited-epi", cl::Hidden,
    cl::desc("Maximum number of nodes to visit during blocking count BFS. "
             "Limits compile time for large DAGs."),
    cl::init(1000));

static cl::opt<unsigned> ShadowMixMaxCandidatesEpi(
    "amdgpu-shadow-mix-max-candidates-epi", cl::Hidden,
    cl::desc(
        "Maximum number of pending candidates to examine during lookahead. "
        "Limits compile time when many pending instructions exist."),
    cl::init(13));

// Shadow priority rules: prefer long-latency instruction so short ones fill
// shadow. These are toggleable for debugging the increasingly specific
// heuristics.
static cl::opt<bool> ShadowPriorityWMMAOverDSEpi(
    "amdgpu-shadow-priority-wmma-over-ds-epi", cl::Hidden,
    cl::desc("Prefer WMMA over DS when both ready (DS fills WMMA shadow)."),
    cl::init(true));

static cl::opt<bool> ShadowPriorityWMMAOverSALUEpi(
    "amdgpu-shadow-priority-wmma-over-salu-epi", cl::Hidden,
    cl::desc("Prefer WMMA over SALU when both ready (SALU fills WMMA shadow)."),
    cl::init(false));

static cl::opt<bool> ShadowPriorityCVTOverDSEpi(
    "amdgpu-shadow-priority-cvt-over-ds-epi", cl::Hidden,
    cl::desc("Prefer CVT over DS when both ready (DS fills CVT shadow)."),
    cl::init(true));

static cl::opt<bool> ShadowPriorityCVTOverSALUEpi(
    "amdgpu-shadow-priority-cvt-over-salu-epi", cl::Hidden,
    cl::desc("Prefer CVT over SALU when both ready (SALU fills CVT shadow)."),
    cl::init(true));

static cl::opt<bool> ShadowPriorityTRANS32OverVALU1cEpi(
    "amdgpu-shadow-priority-trans32-over-valu1c-epi", cl::Hidden,
    cl::desc("Prefer TRANS32 (v_exp etc) over 1-cycle VALU when both ready "
             "(VALU fills TRANS32 shadow)."),
    cl::init(true));

static cl::opt<bool> ShadowDeferTRANS32Epi(
    "amdgpu-shadow-defer-trans32-epi", cl::Hidden,
    cl::desc(
        "Defer TRANS32 instructions until enough VALU ready to fill shadow."),
    cl::init(true));

static cl::opt<unsigned> ShadowMixTRANS32MinVALU1cEpi(
    "amdgpu-shadow-mix-trans32-min-valu1c-epi", cl::Hidden,
    cl::desc(
        "Minimum 1-cycle VALU instructions ready before scheduling TRANS32 "
        "(when -amdgpu-shadow-defer-trans32 enabled)."),
    cl::init(1));

static cl::opt<bool> ShadowPreferVALU1cOverSALUForTRANSEpi(
    "amdgpu-shadow-prefer-valu-over-salu-for-trans-epi", cl::Hidden,
    cl::desc("When filling TRANS32 shadow, prefer VALU1c over SALU "
             "(reserve SALU for WMMA/CVT shadows)."),
    cl::init(true));

// Flag seems universally beneficial, may make sense to delete
static cl::opt<bool> ResourcePriorityToProducerEpi(
    "amdgpu-resource-priority-coexec-producer-epi", cl::Hidden,
    cl::desc("When sorting critical resources, whether to give more priortiy "
             "to coexecution producers over exposed latency."),
    cl::init(true));

// Flag seems universally beneficial, may make sense to delete
static cl::opt<bool> ResourcePriorityCoexecWindowSizeEpi(
    "amdgpu-resource-priority-coexec-windows-size-epi", cl::Hidden,
    cl::desc(
        "When sorting critical resources, whether to sort by window size."),
    cl::init(true));

// Flag seems universally beneficial, may make sense to delete
static cl::opt<bool> ResourcePriorityExposedCyclesEpi(
    "amdgpu-resource-priority-coexec-exposed-cycles-epi", cl::Hidden,
    cl::desc(
        "When sorting critical resources, whether to sort by exposed cycles."),
    cl::init(true));

static cl::opt<bool> ShadowMixRulesEpi(
    "amdgpu-use-shadow-mix-rules-epi", cl::Hidden,
    cl::desc("Whether to use instruction type rules in tryShadowMix."),
    cl::init(false));


static cl::opt<bool> EnableWMMACooloff(
    "amdgpu-use-wmma-cooloff", cl::Hidden,
    cl::desc("Whether or not to enable WMMA cooloff."),
    cl::init(true));


namespace {

struct IncomingDSLatencyPercentParser : public cl::parser<unsigned> {
  IncomingDSLatencyPercentParser(cl::Option &O) : cl::parser<unsigned>(O) {}

  bool parse(cl::Option &O, StringRef ArgName, StringRef Arg, unsigned &Value) {
    if (Arg.getAsInteger(0, Value))
      return O.error("'" + Arg + "' value invalid for uint argument!");

    if (Value > 100)
      return O.error("'" + Arg + "' value must be in the range [0, 100]!");

    return false;
  }
};

} // end anonymous namespace

static cl::opt<unsigned, false, IncomingDSLatencyPercentParser> IncomingLoadLatencyPercent(
    "amdgpu-loop-carried-load-percent", cl::init(100), cl::Hidden,
    cl::desc(
        "Percent of maximum load latency we should try to cover for loop carried loads"));

//===----------------------------------------------------------------------===//
// Shadow Mix Lookahead Helpers
//===----------------------------------------------------------------------===//

/// Count how many successors of SU would become ready (NumPredsLeft == 1)\n/// and match the target flavor.
// FIXME - should we adjust cost for non-hideable instructions as determined by collectUse?
static unsigned countDirectlyEnabledByFlavor(SUnit *SU, InstructionFlavor TargetF,
                                              const SIInstrInfo *SII) {
  unsigned Count = 0;
  for (const SDep &Succ : SU->Succs) {
    if (Succ.isWeak())
      continue;
    SUnit *SuccSU = Succ.getSUnit();
    if (SuccSU->isBoundaryNode() || SuccSU->isScheduled)
      continue;
    // Would become ready if SU is scheduled
    if (SuccSU->NumPredsLeft == 1) {
      InstructionFlavor F = classifyFlavor(SuccSU->getInstr(), SII);
      if (F == TargetF)
        ++Count;
    }
  }
  return Count;
}

/// Compute bounded blocking depth: how many unscheduled instructions
/// transitively block TargetSU, up to MaxDepth levels and MaxVisited nodes.
/// Returns {instruction count, whether search was truncated}.
static std::pair<unsigned, bool>
computeBoundedBlockingCount(SUnit *TargetSU, unsigned MaxDepth, unsigned MaxVisited) {
  if (TargetSU->isTopReady() || MaxDepth == 0)
    return {0, false};

  unsigned Count = 0;
  bool Truncated = false;
  SmallVector<std::pair<SUnit *, unsigned>, 16> WorkList;
  DenseSet<SUnit *> Visited;

  WorkList.push_back({TargetSU, 0});

  while (!WorkList.empty()) {
    auto [SU, Depth] = WorkList.pop_back_val();
    if (Depth >= MaxDepth || Visited.size() >= MaxVisited) {
      Truncated = true;
      continue;
    }
    if (!Visited.insert(SU).second)
      continue;

    for (const SDep &Pred : SU->Preds) {
      if (Pred.isWeak())
        continue;
      SUnit *PredSU = Pred.getSUnit();
      if (PredSU->isBoundaryNode() || PredSU->isScheduled)
        continue;
      ++Count;
      WorkList.push_back({PredSU, Depth + 1});
    }
  }
  return {Count, Truncated};
}

/// Find the nearest pending single-cycle VALU and compute its blocking cost.
/// Returns {SUnit*, blocking count} or {nullptr, UINT_MAX} if none found.
static std::pair<SUnit *, unsigned>
findNearestPendingByFlavor(const RegionMixInfo &MixInfo, InstructionFlavor Flavor,
                           unsigned MaxDepth, unsigned MaxCost,
                           unsigned MaxVisited, unsigned MaxCandidates) {
  SUnit *BestSU = nullptr;
  unsigned BestCost = UINT_MAX;
  unsigned CandidatesChecked = 0;

  for (SUnit *SU : MixInfo.getSUs(Flavor)) {
    if (SU->isScheduled || SU->isTopReady())
      continue;

    if (++CandidatesChecked > MaxCandidates)
      break;

    auto [Cost, Truncated] = computeBoundedBlockingCount(SU, MaxDepth, MaxVisited);
    // Skip if too expensive or search was truncated (likely too deep)
    if (Truncated || Cost > MaxCost)
      continue;

    if (Cost < BestCost) {
      BestCost = Cost;
      BestSU = SU;
    }
  }
  return {BestSU, BestCost};
}

/// Check if scheduling SU would help enable a target SU (is on path to it).
static bool wouldHelpEnable(SUnit *SU, SUnit *TargetSU,
                            ScheduleDAGInstrs *DAG) {
  if (!TargetSU || SU == TargetSU)
    return false;
  return DAG->IsReachable(TargetSU, SU);
}

InstructionFlavor llvm::classifyFlavor(const MachineInstr *MI,
                                       const SIInstrInfo *SII) {
  if (!MI || MI->isDebugInstr())
    return InstructionFlavor::Other;

  unsigned Opc = MI->getOpcode();

  // Check for specific opcodes first.

  if (const_cast<SIInstrInfo *>(SII)->isLDSDMA(Opc))
    return InstructionFlavor::DMA;

  if (Opc == AMDGPU::ATOMIC_FENCE ||
      Opc == AMDGPU::S_WAIT_ASYNCCNT ||
      Opc == AMDGPU::S_WAIT_TENSORCNT ||
      Opc == AMDGPU::S_WAIT_DSCNT ||
      Opc == AMDGPU::S_BARRIER_WAIT ||
      Opc == AMDGPU::S_BARRIER_SIGNAL_IMM)
    return InstructionFlavor::Fence;

  unsigned RepeatRate = SII->getRepeatRate(*MI);

  if (RepeatRate > 1 && SII->isVALU(*MI) && !SII->isMFMAorWMMA(*MI) &&
      !SII->isTRANS(*MI))
    return InstructionFlavor::MultiCycleVALU;

  // Check instruction categories.

  if (SII->isMFMAorWMMA(*MI))
    return InstructionFlavor::WMMA;

  if (SII->isTRANS(*MI))
    return InstructionFlavor::TRANS;

  if (SII->isVALU(*MI))
    return InstructionFlavor::SingleCycleVALU;

  if (SII->isDS(*MI))
    return InstructionFlavor::DS;

  if (SII->isFLAT(*MI) || SII->isFLATGlobal(*MI) || SII->isFLATScratch(*MI))
    return InstructionFlavor::VMEM;

  if (SII->isSALU(*MI))
    return InstructionFlavor::SALU;

  return InstructionFlavor::Other;
}

static unsigned getFlavorCycles(const MachineInstr *MI, InstructionFlavor F,
                                const SIInstrInfo *SII) {
  // WMMA: hardcoded to 8 cycles for now (gfx1250)
  // Note: Adding this to getRepeatRate() causes regressions elsewhere.
  if (F == InstructionFlavor::WMMA)
    return 8;

  return SII->getRepeatRate(*MI);
}

void RegionMixInfo::dumpMix(raw_ostream &OS, bool Detailed) const {
  OS << "Instruction Mix:\n";
  for (unsigned I = 0; I < NumFlavors; ++I) {
    InstructionFlavor F = static_cast<InstructionFlavor>(I);
    unsigned Total = getTotalCount(F);
    if (Total == 0)
      continue;
    OS << "  " << getFlavorName(F) << ": " << Total;
    if (Detailed)
      OS << " (cycles: " << getTotalCycles(F) << ")";
    OS << "\n";
  }
}

void RegionMixInfo::dumpReadyPending(raw_ostream &OS) const {
  OS << "Ready: ";
  bool First = true;
  for (unsigned I = 0; I < NumFlavors; ++I) {
    InstructionFlavor F = static_cast<InstructionFlavor>(I);
    unsigned Ready = getReadyCount(F);
    if (Ready == 0)
      continue;
    if (!First)
      OS << ", ";
    First = false;
    OS << Ready << " " << getFlavorName(F);
  }
  OS << "\nBlocked: ";
  First = true;
  for (unsigned I = 0; I < NumFlavors; ++I) {
    InstructionFlavor F = static_cast<InstructionFlavor>(I);
    unsigned Blocked = getPendingCount(F);
    if (Blocked == 0)
      continue;
    if (!First)
      OS << ", ";
    First = false;
    unsigned RemCycles = getRemainingCycles(F);
    OS << Blocked << " " << getFlavorName(F) << "(" << RemCycles << "c)";
  }
  OS << "\n";
}

AMDGPUMLSchedStrategy::AMDGPUMLSchedStrategy(const MachineSchedContext *C)
    : GCNSchedStrategy(C) {
  SchedStages.push_back(GCNSchedStageID::ILPInitialSchedule);
  SchedStages.push_back(GCNSchedStageID::PreRARematerialize);
  // Use more accurate GCN pressure trackers.
  UseGCNTrackers = true;
}

void AMDGPUMLSchedStrategy::initialize(ScheduleDAGMI *DAG) {
  // ML scheduling strategy is only done top-down to support new resource
  // balancing heuristics.
  RegionPolicy.OnlyTopDown = true;
  RegionPolicy.OnlyBottomUp = false;
  GCNSchedStrategy::initialize(DAG);

  CI.clear();
  CI.compute(DAG->MF);

  if (Top.HazardRec) {
    delete Top.HazardRec;
    Top.HazardRec = nullptr;
  }
  Top.HazardRec = new GCNHazardRecognizer(
      DAG->MF, GCNHazardRecognizer::OperatingMode::PreRA);

  Heurs.initialize(DAG, static_cast<GCNHazardRecognizer *>(Top.HazardRec),
                   SchedModel, TRI, false, false);
}


// FIXME -- can we better consolodate pending + available?
static bool shouldCheckPending(SchedBoundary &Zone,
                               const TargetSchedModel *SchedModel) {
  return true;

  // FIXME -- enable this method, need to share flag
  // bool HasBufferedModel =
  //    SchedModel->hasInstrSchedModel() && SchedModel->getMicroOpBufferSize();
  // unsigned Combined = Zone.Available.size() + Zone.Pending.size();
  // return true; //Combined <= PendingQueueLimit && HasBufferedModel;
}

static SUnit *pickOnlyChoice(SchedBoundary &Zone,
                             const TargetSchedModel *SchedModel) {
  // pickOnlyChoice() releases pending instructions and checks for new hazards.
  SUnit *OnlyChoice = Zone.pickOnlyChoice();
  if (!shouldCheckPending(Zone, SchedModel) || Zone.Pending.empty())
    return OnlyChoice;

  return nullptr;
}

unsigned AMDGPUMLSchedStrategy::getHWUICyclesForInst(SUnit *SU,
                                                     const SIInstrInfo *SII,
                                                     unsigned ReleaseAtCycle) {
  auto Opc = SU->getInstr()->getOpcode();

  bool IsDMA = const_cast<SIInstrInfo *>(SII)->isLDSDMA(Opc);

  unsigned Latency = IsDMA ? SU->Latency : ReleaseAtCycle;
  // FIXME -- harcoded?
  // This is used to determine hardware unit balancing between LDS and other
  // resources, if we use a high cycle count to more accurately reflect LDS
  // latency, then we become LDS bound in most cases. The problem is that LDS
  // latency is usually hidden across loops, whereas other latency (e.g. WMMA)
  // are not hidden in this way.
  if (SII->isDS(*SU->getInstr()) && SU->getInstr()->mayLoad())
    Latency = 8;

  MachineInstr *MI = SU->getInstr();
  unsigned RepeatRate = SII->getRepeatRate(*MI);

  if (RepeatRate > 1 && SII->isVALU(*MI) && !SII->isMFMAorWMMA(*MI) &&
      !SII->isTRANS(*MI)) {
    Latency = RepeatRate;
  }

  return Latency;
}

void AMDGPUMLSchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
  DEBUG_WITH_TYPE("machine-scheduler-verbose", {
    dbgs() << "Scheduling: ";
    SU->getInstr()->dump();
    dbgs() << "\n\n";
  });

  Heurs.schedNode(SU, static_cast<GCNHazardRecognizer *>(Top.HazardRec));
  GCNSchedStrategy::schedNode(SU, IsTopNode);
  Heurs.bumpNode(SU, &Top);
}

// TODO - should this logic depend on the pipeline?
void CandidateHeuristics::calculateHiddenLatency(
    GCNHazardRecognizer *HazardRec) {
  unsigned WMMACycles = HWUInfo[(int)InstructionFlavor::WMMA].getTotalCycles();

  unsigned DSCycles = HWUInfo[(int)InstructionFlavor::DS].getTotalCycles();
  unsigned MultiVALUCycles =
      HWUInfo[(int)InstructionFlavor::MultiCycleVALU].getTotalCycles();
  unsigned SingleCycleVALUCycles =
      HWUInfo[(int)InstructionFlavor::SingleCycleVALU].getTotalCycles();
  unsigned SALUCount = HWUInfo[(int)InstructionFlavor::SALU].size();
  unsigned TRANSCycles =
      HWUInfo[(int)InstructionFlavor::TRANS].getTotalCycles();

  unsigned DSCount = HWUInfo[(int)InstructionFlavor::DS].size();
  unsigned WMMACount = HWUInfo[(int)InstructionFlavor::WMMA].size();
  unsigned MultiVALUCount =
      HWUInfo[(int)InstructionFlavor::MultiCycleVALU].size();
  unsigned EXPCount = HWUInfo[(int)InstructionFlavor::DS].size();
  unsigned SingleCycleVALUCount =
      HWUInfo[(int)InstructionFlavor::SingleCycleVALU].size();

  // FIXME -- what if we have multiple types of WMMA?
  unsigned ESlotCount = 0;
  unsigned ISlotCount = 0;
  unsigned ESlotForDS = 0;
  unsigned ISlotForTrans = 0;
  if (WMMACount) {
    SUnit *NextWMMA = HWUInfo[(int)InstructionFlavor::WMMA].getTargetSU();
    assert(NextWMMA);
    SmallVector<GCNHazardRecognizer::WMMASlotType, 8> WMMACoexecSlots;
    HazardRec->getWMMASlots(*NextWMMA->getInstr(), WMMACoexecSlots);
    for (auto &Slot : WMMACoexecSlots) {
      switch (Slot) {
      default:
        break;
      case GCNHazardRecognizer::WMMASlotType::MemCoExec0:
      case GCNHazardRecognizer::WMMASlotType::MemCoExec2: {
        ++ESlotForDS;
        ++ESlotCount;
        break;
      }
      case GCNHazardRecognizer::WMMASlotType::MemCoExec1:
      case GCNHazardRecognizer::WMMASlotType::MemCoExec3: {
        ++ESlotCount;
        break;
      }
      case GCNHazardRecognizer::WMMASlotType::ValuCoExec0: {
        ++ISlotForTrans;
        ++ISlotCount;
        break;
      }
      case GCNHazardRecognizer::WMMASlotType::ValuCoExec1:
      case GCNHazardRecognizer::WMMASlotType::ValuCoExec2:
      case GCNHazardRecognizer::WMMASlotType::ValuCoexecLastLdScale: {
        ++ISlotCount;
        break;
      }
      }
    }
  }

  unsigned WMMAESlot = WMMACount * ESlotCount;
  unsigned WMMAISlot = WMMACount * ISlotCount;
  unsigned WMMAESlotForDS = WMMACount * ESlotForDS;
  unsigned WMMAISlotForTRANS = WMMACount * ISlotForTrans;

  unsigned CoexecWithMultiVALU =
      MultiVALUCycles - HWUInfo[(int)InstructionFlavor::MultiCycleVALU].size();

  bool IsDSBound = DSCycles > WMMACycles + DSCycles + SingleCycleVALUCycles +
                                  TRANSCycles - 2 * WMMACount;

  // TODO -- properly model DSBound & MemBound
  if (!IsDSBound) {
    // If we are not DS Bound, then we cannot hide any of the WMMA or multi
    // cycle VALU
    HWUInfo[(int)InstructionFlavor::WMMA].setExposedCount(WMMACount);
    HWUInfo[(int)InstructionFlavor::MultiCycleVALU].setExposedCount(
        MultiVALUCount);

    unsigned DSWMMACoexecution = std::min(WMMAESlotForDS, DSCount);

    // Three main types of latency hiding:
    // 1. SALU / DS / VALU1c / EXP behinmg WMMA
    // 2. SALU / DS behind multi cycle VALU
    // 3. VALU1c behind EXP

    WMMAESlot -= DSWMMACoexecution;
    DSCount -= DSWMMACoexecution;

    if (DSCount) {
      unsigned DSMultiCoexecution = std::min(CoexecWithMultiVALU, DSCount);
      DSCount -= DSMultiCoexecution;
      CoexecWithMultiVALU -= DSMultiCoexecution;
    }

    HWUInfo[(int)InstructionFlavor::DS].setExposedCount(DSCount);

    unsigned SALUWMMACoexecution = std::min(WMMAESlot, SALUCount);
    SALUCount -= SALUWMMACoexecution;


    ShadowMixWMMAMinDSVal = WMMACount ? (DSWMMACoexecution + SALUWMMACoexecution) / WMMACount : 0;

    //errs() << "ShadowMixWMMAMinDSVal: " << ShadowMixWMMAMinDSVal << "\n";

    ShadowMixWMMAMinSALUVal = 0;//WMMACount ? SALUWMMACoexecution / WMMACount : 0;

    if (SALUCount) {
      unsigned SALUMultiCoexecution = std::min(CoexecWithMultiVALU, SALUCount);
      SALUCount -= SALUMultiCoexecution;
      CoexecWithMultiVALU -= SALUCount;
      WMMAESlot -= SALUMultiCoexecution;
    }

    HWUInfo[(int)InstructionFlavor::SALU].setExposedCount(SALUCount);

    unsigned EXPWMMACoexecution =
        true ? std::min(WMMAISlotForTRANS, EXPCount) : 0;

    WMMAISlot -= EXPWMMACoexecution;
    EXPCount -= EXPWMMACoexecution;

    HWUInfo[(int)InstructionFlavor::TRANS].setExposedCount(EXPCount);

    unsigned VALUWMMACoexecution = std::min(WMMAISlot, SingleCycleVALUCount);

    WMMAISlot -= VALUWMMACoexecution;

    ShadowMixWMMAMinVALU1cVal = WMMACount ? (WMMACount * ISlotCount - WMMAISlot) / WMMACount : 0;

    SingleCycleVALUCount -= VALUWMMACoexecution;

    unsigned VALUEXPCoexecution = std::min(EXPCount, SingleCycleVALUCount);

    SingleCycleVALUCount -= VALUEXPCoexecution;

    if (SingleCycleVALUCount)
      ShadowMixTRANS32MinVALU1c = 1;


    HWUInfo[(int)InstructionFlavor::SingleCycleVALU].setExposedCount(
        SingleCycleVALUCount);
  }
}

bool CandidateHeuristics::tryWMMACoolOff(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary *Zone) {

  GCNHazardRecognizer *HazardRec =
      static_cast<GCNHazardRecognizer *>(Zone->HazardRec);
  int CoexecSlot =
      HazardRec->getWMMACoexecSlot();

  bool UnderWMMAShadow = CoexecSlot != -1;

  if (UnderWMMAShadow) {
    GCNHazardRecognizer::WMMASlotType CurrentSlot =
        (GCNHazardRecognizer::WMMASlotType)CoexecSlot;

    if (CurrentSlot != GCNHazardRecognizer::WMMASlotType::ValuBlocked0 &&
        CurrentSlot != GCNHazardRecognizer::WMMASlotType::ValuBlocked1)
      return false;
  }

  bool TryIsWMMA = classifyFlavor(TryCand.SU->getInstr(), SII) == InstructionFlavor::WMMA;
  bool CandIsWMMA = classifyFlavor(Cand.SU->getInstr(), SII) == InstructionFlavor::WMMA;;

  if (TryIsWMMA == CandIsWMMA)
    return false;

  CoexecWindow TempWindow;
  MixInfo.updateReadyCounts();
  TempWindow.refreshMixInfo(MixInfo);
  populateCandidateWindow(TempWindow, InstructionFlavor::WMMA);
  bool WMMAWindowIsReady = TempWindow.IsReady;

  if (!TempWindow.IsReady) {
    SmallVector<InstructionFlavor, 4> NeededFlavors;
    TempWindow.getNeededFlavors(NeededFlavors);
    // Neither candidate directly enables any of the needed flavors, look eahd.
    for (InstructionFlavor &NeededFlavor : NeededFlavors) {

      auto [NearestTarget, Cost] = findNearestPendingByFlavor(
          MixInfo, NeededFlavor, ShadowMixLookaheadDepthVal,
          ShadowMixMaxBlockingCostVal, ShadowMixMaxVisitedVal,
          ShadowMixMaxCandidatesVal);

      // It is too much effort to try to make the window ready, proceed with the
      // WMMA
      if (!NearestTarget) {
        return false;
      }
    }

    // The consumers are not available, and it is not much effort to make them
    // available
    if (TryIsWMMA) {
      if (Cand.Reason > GenericSchedulerBase::RegCritical) {
        Cand.Reason = GenericSchedulerBase::RegCritical;
      }
      return true;
    }

    TryCand.Reason = GenericSchedulerBase::RegCritical;
    return true;
  }

  // It is possible we have all the coexecution consumers, but they need long
  // stalls
  unsigned MaxStall = 0;
  if (!coexecWindowIsReady(&TempWindow, Zone, MaxStall)) {
    if (MaxStall < 8) {
    if (TryIsWMMA) {
      if (Cand.Reason > GenericSchedulerBase::RegCritical) {
        Cand.Reason = GenericSchedulerBase::RegCritical;
      }
      return true;
    }

    TryCand.Reason = GenericSchedulerBase::RegCritical;
    return true;
  }
  }

  return false;
}


unsigned CandidateHeuristics::getHWUICyclesForInst(SUnit *SU,
                                                   unsigned ReleaseAtCycle) {
  auto Opc = SU->getInstr()->getOpcode();
  bool IsDMA =const_cast<SIInstrInfo *>(SII)->isLDSDMA(Opc);

  unsigned Latency = IsDMA ? 400 : ReleaseAtCycle;

  // FIXME -- harcoded?
  // This is used to determine hardware unit balancing between LDS and other
  // resources, if we use a high cycle count to more accurately reflect LDS
  // latency, then we become LDS bound in most cases. The problem is that LDS
  // latency is usually hidden across loops, whereas other latency (e.g. WMMA)
  // are not hidden in this way.
  if (SII->isDS(*SU->getInstr()) && SU->getInstr()->mayLoad())
    Latency = 60;

  MachineInstr *MI = SU->getInstr();
  unsigned RepeatRate = SII->getRepeatRate(*MI);

  if (RepeatRate > 1 && SII->isVALU(*MI) && !SII->isMFMAorWMMA(*MI) &&
      !SII->isTRANS(*MI)) {
    Latency = RepeatRate;
  }

  return Latency;
}

void CandidateHeuristics::setParams() {

  ResourcePriorityToProducerVal =
      IsPrologue    ? ResourcePriorityToProducerPro.getValue()
      : !IsEpilogue ? ResourcePriorityToProducer.getValue()
                    : ResourcePriorityToProducerEpi.getValue();

  ResourcePriorityCoexecWindowSizeVal =
      IsPrologue    ? ResourcePriorityCoexecWindowSizePro.getValue()
      : !IsEpilogue ? ResourcePriorityCoexecWindowSize.getValue()
                    : ResourcePriorityCoexecWindowSizeEpi.getValue();
  ResourcePriorityExposedCyclesVal =
      IsPrologue    ? ResourcePriorityExposedCyclesPro.getValue()
      : !IsEpilogue ? ResourcePriorityExposedCycles.getValue()
                    : ResourcePriorityExposedCyclesEpi.getValue();

  EnableShadowMixVal = IsPrologue    ? EnableShadowMixPro.getValue()
                       : !IsEpilogue ? EnableShadowMix.getValue()
                                     : EnableShadowMixEpi.getValue();

  ShadowMixWMMAMinVALU1cVal = IsPrologue ? ShadowMixWMMAMinVALU1cPro.getValue()
                              : !IsEpilogue
                                  ? ShadowMixWMMAMinVALU1c.getValue()
                                  : ShadowMixWMMAMinVALU1cEpi.getValue();
  ShadowMixWMMAMinDSVal = IsPrologue    ? ShadowMixWMMAMinDSPro.getValue()
                          : !IsEpilogue ? ShadowMixWMMAMinDS.getValue()
                                        : ShadowMixWMMAMinDSEpi.getValue();
  ShadowMixWMMAMinSALUVal = IsPrologue    ? ShadowMixWMMAMinSALUPro.getValue()
                            : !IsEpilogue ? ShadowMixWMMAMinSALU.getValue()
                                          : ShadowMixWMMAMinSALUEpi.getValue();

  ShadowMixRulesVal = IsPrologue    ? ShadowMixRulesPro.getValue()
                      : !IsEpilogue ? ShadowMixRules.getValue()
                                    : ShadowMixRulesEpi.getValue();

  ShadowPriorityWMMAOverDSVal =
      IsPrologue    ? ShadowPriorityWMMAOverDSPro.getValue()
      : !IsEpilogue ? ShadowPriorityWMMAOverDS.getValue()
                    : ShadowPriorityWMMAOverDSEpi.getValue();
  ShadowPriorityWMMAOverSALUVal =
      IsPrologue    ? ShadowPriorityWMMAOverSALUPro.getValue()
      : !IsEpilogue ? ShadowPriorityWMMAOverSALU.getValue()
                    : ShadowPriorityWMMAOverSALUEpi.getValue();
  ShadowPriorityCVTOverDSVal =
      IsPrologue    ? ShadowPriorityCVTOverDSPro.getValue()
      : !IsEpilogue ? ShadowPriorityCVTOverDS.getValue()
                    : ShadowPriorityCVTOverDSEpi.getValue();
  ShadowPriorityCVTOverSALUVal =
      IsPrologue    ? ShadowPriorityCVTOverSALUPro.getValue()
      : !IsEpilogue ? ShadowPriorityCVTOverSALU.getValue()
                    : ShadowPriorityCVTOverSALUEpi.getValue();
  ShadowPriorityTRANS32OverVALU1cVal =
      IsPrologue    ? ShadowPriorityTRANS32OverVALU1cPro.getValue()
      : !IsEpilogue ? ShadowPriorityTRANS32OverVALU1c.getValue()
                    : ShadowPriorityTRANS32OverVALU1cEpi.getValue();
  ShadowPreferVALU1cOverSALUForTRANSVal =
      IsPrologue    ? ShadowPreferVALU1cOverSALUForTRANSPro.getValue()
      : !IsEpilogue ? ShadowPreferVALU1cOverSALUForTRANS.getValue()
                    : ShadowPreferVALU1cOverSALUForTRANSEpi.getValue();

  ShadowMixLookaheadDepthVal =
      IsPrologue    ? ShadowMixLookaheadDepthPro.getValue()
      : !IsEpilogue ? ShadowMixLookaheadDepth.getValue()
                    : ShadowMixLookaheadDepthEpi.getValue();
  ShadowMixMaxBlockingCostVal =
      IsPrologue    ? ShadowMixMaxBlockingCostPro.getValue()
      : !IsEpilogue ? ShadowMixMaxBlockingCost.getValue()
                    : ShadowMixMaxBlockingCostEpi.getValue();
  ShadowMixMaxVisitedVal = IsPrologue    ? ShadowMixMaxVisitedPro.getValue()
                           : !IsEpilogue ? ShadowMixMaxVisited.getValue()
                                         : ShadowMixMaxVisitedEpi.getValue();
  ShadowMixMaxCandidatesVal = IsPrologue ? ShadowMixMaxCandidatesPro.getValue()
                              : !IsEpilogue
                                  ? ShadowMixMaxCandidates.getValue()
                                  : ShadowMixMaxCandidatesEpi.getValue();

  IgnoreVALUVal = IsPrologue   ? IgnoreVALUPro.getValue()
                  : IsEpilogue ? IgnoreVALUEpi.getValue()
                               : IgnoreVALU.getValue();

  DSFIFOSizeVal = IsPrologue    ? DSFIFOSizePro.getValue()
                  : !IsEpilogue ? DSFIFOSize.getValue()
                                : DSFIFOSizeEpi.getValue();

  DSLatencyFIFOVal = IsPrologue    ? DSLatencyFIFOPro.getValue()
                     : !IsEpilogue ? DSLatencyFIFO.getValue()
                                   : DSLatencyFIFOEpi.getValue();

  DSLatencySplitVal = IsPrologue    ? DSLatencySplitPro.getValue()
                      : !IsEpilogue ? DSLatencySplit.getValue()
                                    : DSLatencySplitEpi.getValue();

  LatencyForSignalVal = IsPrologue    ? LatencyForSignalPro.getValue()
                        : !IsEpilogue ? LatencyForSignal.getValue()
                                      : LatencyForSignalEpi.getValue();

  DSLatencyForFenceVal = IsPrologue    ? DSLatencyForFencePro.getValue()
                         : !IsEpilogue ? DSLatencyForFence.getValue()
                                       : DSLatencyForFenceEpi.getValue();

  ResourceToBalanceVal = IsPrologue    ? ResourcesToBalancePro.getValue()
                         : !IsEpilogue ? ResourcesToBalance.getValue()
                                       : ResourcesToBalanceEpi.getValue();
}

void CandidateHeuristics::initialize(ScheduleDAGMI *SchedDAG,
                                     GCNHazardRecognizer *GCNazardRec,
                                     const TargetSchedModel *TargetSchedModel,
                                     const TargetRegisterInfo *TRI,
                                     bool MemoryBound, bool PostRA) {
  DAG = SchedDAG;
  IsMemoryBound = MemoryBound;
  SchedModel = TargetSchedModel;

  SRI = static_cast<const SIRegisterInfo *>(TRI);
  SII = static_cast<const SIInstrInfo *>(DAG->TII);

  MachineBasicBlock *MBB = DAG->SUnits.begin()->getInstr()->getParent();
  IsPrologue = MBB->isEntryBlock();
  IsEpilogue = MBB->isReturnBlock();
  IsPostRA = PostRA;
  setParams();

  HWUInfo.resize((int)InstructionFlavor::NUM_FLAVORS);
  HWUInfo[(int)InstructionFlavor::DMA].IsAsync = true;

  for (unsigned I = 0; I < HWUInfo.size(); I++) {
    HWUInfo[I].setType(I);
  }

  CollectedUse = false;

  SchedDSR.clear();
  SchedMFMA.clear();
  SchedTDM.clear();

  CurrentWindow.clear();
  NextWindow.clear();

  for (auto &HWUI : HWUInfo) {
    HWUI.reset();
  }

  collectUse(GCNazardRec);
}

void CandidateHeuristics::populateCandidateWindow(CoexecWindow &Window,
                                                  InstructionFlavor Flavor) {
  SmallVector<CoexecWindow, 4> CandWindows;

  for (auto &HWUI : HWUInfo) {
    if (HWUI.ProducesCoexecWindow) {
      // TODO -- these requirements should be mapped and auto pulled
      if (HWUI.getType() == InstructionFlavor::WMMA) {
        // TODO - schednode determines the end cycle, whether we've started the
        // window.
        CandWindows.emplace_back(HWUI.getType(), ShadowMixWMMAMinVALU1cVal,
                                 ShadowMixWMMAMinSALUVal, ShadowMixWMMAMinDSVal,
                                 MixInfo);
        if (Flavor == InstructionFlavor::WMMA) {
          Window.copy(CandWindows[CandWindows.size() - 1]);
          return;
        }
      }
      if (HWUI.getType() == InstructionFlavor::TRANS) {
        CandWindows.emplace_back(HWUI.getType(),
                                 /*MinVALU1c*/ ShadowDeferTRANS32.getValue()
                                     ? ShadowMixTRANS32MinVALU1c
                                     : 0,
                                 /*MinSALU*/ 0, /*MinDS*/ 0, MixInfo);
        if (Flavor == InstructionFlavor::TRANS) {
          Window.copy(CandWindows[CandWindows.size() - 1]);
          return;
        }
      }

      // TODO - should we handle this better? MultiCycleVALU may have different
      // instructions with different repeat rate
      if (HWUI.getType() == InstructionFlavor::MultiCycleVALU) {
        CandWindows.emplace_back(HWUI.getType(), /*MinVALU1c*/ 0, /*MinSALU*/ 0,
                                 /*MinDS*/ 0, MixInfo);
        if (Flavor == InstructionFlavor::MultiCycleVALU) {
          Window.copy(CandWindows[CandWindows.size() - 1]);
          return;
        }
      }
    }
  }

  auto lookupHWUI = [this](InstructionFlavor FlavorKey) {
    for (auto &HWUI : HWUInfo) {
      if (HWUI.getType() == FlavorKey)
        return HWUI;
    }
  };

  sort(CandWindows, [this, &lookupHWUI](CoexecWindow A, CoexecWindow B) {
    auto HWUIA = lookupHWUI(A.WindowProducer);
    auto HWUIB = lookupHWUI(B.WindowProducer);

    bool HWUIAExposed = HWUIA.getRemainingExposed() > 0;
    bool HWUIBExposed = HWUIB.getRemainingExposed() > 0;

    if (HWUIAExposed != HWUIBExposed)
      return HWUIAExposed;

    if (A.ReadyCost != B.ReadyCost)
      return A.ReadyCost < B.ReadyCost;

    if (HWUIA.getRemainingExposed() != HWUIB.getRemainingExposed())
      return HWUIA.getRemainingExposed() > HWUIB.getRemainingExposed();

    return true;
  });

  Window.copy(CandWindows[0]);
}

void CandidateHeuristics::collectUse(GCNHazardRecognizer *HazardRec) {
  if (CollectedUse)
    return;

  CollectedUse = true;
  SchedDSR.clear();
  SchedMFMA.clear();
  SchedTDM.clear();
  SchedEXP.clear();
  MixInfo.reset();

  for (auto &HWUI : HWUInfo) {
    HWUI.reset();
  }

  for (unsigned I = 0; I < HWUInfo.size(); I++) {
    HWUInfo[I].reset();
    HWUInfo[I].setType(I);
  }

  HWUInfo[(int)InstructionFlavor::DMA].IsAsync = true;

  HWUInfo[(int)InstructionFlavor::WMMA].ProducesCoexecWindow = true;
  HWUInfo[(int)InstructionFlavor::MultiCycleVALU].ProducesCoexecWindow = true;
  HWUInfo[(int)InstructionFlavor::TRANS].ProducesCoexecWindow = true;

  HWUInfo[(int)InstructionFlavor::WMMA].CoexecWindowSize = 6;
  HWUInfo[(int)InstructionFlavor::MultiCycleVALU].CoexecWindowSize = 3;
  HWUInfo[(int)InstructionFlavor::TRANS].CoexecWindowSize = 1;

  if (!SchedModel || !SchedModel->hasInstrSchedModel())
    return;

  unsigned I = 0;
  unsigned PrevDSR = 0;
  unsigned PrevFence = 0;
  unsigned FencedDSRCount = 0;
  for (auto &SU : DAG->SUnits) {
    unsigned ReleaseAtCycle = 0;
    const MCSchedClassDesc *SC = DAG->getSchedClass(&SU);
    for (TargetSchedModel::ProcResIter
             PI = SchedModel->getWriteProcResBegin(SC),
             PE = SchedModel->getWriteProcResEnd(SC);
         PI != PE; ++PI) {
      ReleaseAtCycle = std::max(ReleaseAtCycle, (unsigned)PI->ReleaseAtCycle);
    }
    unsigned Latency = getHWUICyclesForInst(&SU, ReleaseAtCycle);

    auto *MI = SU.getInstr();
    InstructionFlavor Flavor = classifyFlavor(MI, SII);
    HWUInfo[(int)(Flavor)].insert(&SU, Latency);

    if (Flavor == InstructionFlavor::WMMA && Latency != 8)
      HWUInfo[(unsigned)Flavor].CoexecWindowSize = Latency-1;
    unsigned FlavorCycles = getFlavorCycles(MI, Flavor, SII);
    MixInfo.addSU(&SU, Flavor, FlavorCycles);

    if (SII->isDS(*MI) && MI->mayLoad()) {
      PrevDSR = I;
    }
    if (MI->getOpcode() == AMDGPU::ATOMIC_FENCE) {
      if (PrevFence < PrevDSR) {
        ++FencedDSRCount;
      }
      PrevFence = I;
    }
    I++;
  }

  unsigned MaxCycles = 0;
  if (FencedDSRCount) {
    for (auto HWUI : HWUInfo) {
      MaxCycles = std::max(MaxCycles, HWUI.getTotalCycles());
    }

    FencedDSRLatency = MaxCycles / FencedDSRCount;
    FencedDSRLatency = std::max(DSLatency.getValue(), FencedDSRLatency);
  }

  HWUInfo[(unsigned)InstructionFlavor::Other].reset();
  if (IgnoreVALUVal) {
    HWUInfo[(unsigned)InstructionFlavor::SingleCycleVALU].reset();
    HWUInfo[(unsigned)InstructionFlavor::SALU].reset();
  }

  HWUInfo[(unsigned)InstructionFlavor::DS].fixupFIFO(16);

  calculateHiddenLatency(HazardRec);
  LLVM_DEBUG(dumpRegionSummary());
}

void CandidateHeuristics::dumpRegionSummary() {
  MachineBasicBlock *BB = DAG->begin()->getParent();
  dbgs() << "\n=== Region: " << DAG->MF.getName() << " BB" << BB->getNumber()
         << " (" << DAG->SUnits.size() << " SUs) ===\n";

  MixInfo.dumpMix(dbgs(), /*Detailed=*/true);

  dbgs() << "\nHWUI Resource Pressure (sorted):\n";
  sortResources();
  for (auto &HWUI : HWUInfo) {
    if (HWUI.getTotalCycles() == 0)
      continue;

    StringRef Name = getFlavorName(HWUI.getType());
    dbgs() << "  [" << HWUI.Idx << "] " << Name << ": " << HWUI.getTotalCycles()
           << " cycles, " << HWUI.size() << " instrs\n";
  }
  dbgs() << "\n";
}

void CandidateHeuristics::sortResources() {
  // Highest priority should be first.
  sort(HWUInfo, [this](HardwareUnitInfo &A, HardwareUnitInfo &B) {
    // Both are either exposed or unexposed, prefer exec window producer
    if (ResourcePriorityToProducerVal) {
      if (A.ProducesCoexecWindow != B.ProducesCoexecWindow)
        return A.ProducesCoexecWindow;

      bool AIsExposed = A.getRemainingExposed() > 0;
      bool BIsExposed = B.getRemainingExposed() > 0;

      if (AIsExposed != BIsExposed)
        return AIsExposed;
    }

    else {
      bool AIsExposed = A.getRemainingExposed() > 0;
      bool BIsExposed = B.getRemainingExposed() > 0;

      if (AIsExposed != BIsExposed)
        return AIsExposed;

      if (A.ProducesCoexecWindow != B.ProducesCoexecWindow)
        return A.ProducesCoexecWindow;
    }

    if (ResourcePriorityCoexecWindowSizeVal) {
      if (A.CoexecWindowSize != B.CoexecWindowSize) {
        return A.CoexecWindowSize > B.CoexecWindowSize;
      }
    }

    if (ResourcePriorityExposedCyclesVal)
      // Give priority to the hardware unit with the most exposed cycles
      if (A.getRemainingExposed() != B.getRemainingExposed())
        return A.getRemainingExposed() > B.getRemainingExposed();

    // Less relevant tiebreakers
    // Total cycles
    if (A.getTotalCycles() != B.getTotalCycles())
      return A.getTotalCycles() > B.getTotalCycles();

    // In ties -- prefer the resource with longer latency instructions
    if (A.size() != B.size())
      return A.size() < B.size();

    // Default to HardwareUnitInfo order
    return A.Idx < B.Idx;
  });
}

static std::optional<unsigned> getMSBs(const MachineOperand &MO,
                                       const SIRegisterInfo *TRI) {
  if (!MO.isReg())
    return std::nullopt;

  MCRegister Reg = MO.getReg();
  const TargetRegisterClass *RC = TRI->getPhysRegBaseClass(Reg);
  if (!RC || !TRI->isVGPRClass(RC))
    return std::nullopt;

  unsigned Idx = TRI->getHWRegIndex(Reg);
  return Idx >> 8;
}

unsigned CandidateHeuristics::getLatencyStallCycles(SUnit *SU,
                                                    SchedBoundary *Zone) {
  unsigned CurrCycle = Zone->getCurrCycle();
  unsigned ReadyCycle = SU->TopReadyCycle;
  auto *MI = SU->getInstr();
  const GCNSubtarget &ST = DAG->MF.getSubtarget<GCNSubtarget>();

  if (SII->isDS(*MI) && MI->mayLoad()) {

    if (SchedDSR.size() >= DSFIFOSizeVal) {
      unsigned TopOfFIFO = SchedDSR.size() - DSFIFOSizeVal;
      unsigned TopOfFIFOIssue = SchedDSR[TopOfFIFO]->TopReadyCycle;
      // TODO -- should be release at cycle.
      ReadyCycle = std::max(TopOfFIFOIssue + DSLatencyFIFOVal, ReadyCycle);
    }
    if (SchedDSR.size()) {
      unsigned LastDSRIssue = SchedDSR[SchedDSR.size() - 1]->TopReadyCycle;
      // TODO -- should be release at cycle.
      ReadyCycle = std::max(LastDSRIssue + DSLatencySplitVal, ReadyCycle);
    }

    for (auto Pred : SU->Preds) {
      auto PredSU = Pred.getSUnit();
      if (SII->isVALU(*PredSU->getInstr())) {
        ReadyCycle = std::max(ReadyCycle, PredSU->TopReadyCycle +
                                              ST.getVDstThreshold(DAG->MF));
      }
    }
  }

  else if (const_cast<SIInstrInfo *>(SII)->isLDSDMA(MI->getOpcode())) {
    return 0;
  }

  else if (MI->getOpcode() == AMDGPU::S_BARRIER_WAIT && SchedTDM.size()) {
    auto PrevTDM = SchedTDM[SchedTDM.size() - 1];

    if (PrevTDM->getInstr()->getOpcode() == AMDGPU::S_BARRIER_SIGNAL_IMM) {
      ReadyCycle =
          std::max(ReadyCycle, PrevTDM->TopReadyCycle + LatencyForSignalVal);
    }

  }

  else if ((MI->getOpcode() == AMDGPU::ATOMIC_FENCE ||
            MI->getOpcode() == AMDGPU::S_WAIT_TENSORCNT ||
            MI->getOpcode() == AMDGPU::S_WAIT_DSCNT)) {
    if (SchedDSR.size()) {
      auto PrevDSR = SchedDSR[SchedDSR.size() - 1];
      ReadyCycle =
          std::max(ReadyCycle, PrevDSR->TopReadyCycle + DSLatencyForFenceVal);
    } else if (!(IsPrologue || IsEpilogue)) {
      // TODO: Can we detect CFG carried loads?
      unsigned LatencyToCover = IncomingLoadLatencyPercent * DSLatencyForFenceVal / 100;
      ReadyCycle = std::max(ReadyCycle, LatencyToCover);
    }
  }

  else if (SII->isMFMAorWMMA(*MI)) {
      // TODO: Can we detect CFG carried loads?
      unsigned LatencyToCover = IncomingLoadLatencyPercent * DSLatencyForFenceVal / 100;
      ReadyCycle = std::max(ReadyCycle, LatencyToCover);
  }

  unsigned LongLatVALU = SII->isTRANS(*MI) ? 0 : SII->getRepeatRate(*MI);
  if (LongLatVALU > 1 && (SchedMFMA.size() || SchedEXP.size())) {
    if (SchedMFMA.size()) {
      auto PrevMFMA = SchedMFMA[SchedMFMA.size() - 1];
      unsigned PrevMFMAIssue = PrevMFMA->TopReadyCycle;
      ReadyCycle = std::max(PrevMFMAIssue + PrevMFMA->Latency, ReadyCycle);
    }

    if (SchedEXP.size()) {
      auto PrevEXP = SchedEXP[SchedEXP.size() - 1];
      unsigned PrevEXPIssue = PrevEXP->TopReadyCycle;
      ReadyCycle = std::max(PrevEXPIssue + 2, ReadyCycle);
    }
  }

  if (IsPostRA) {
    if (SchedMFMA.size() && !SII->isMFMAorWMMA(*MI) &&
        (SII->isVALU(*MI) || SII->isTRANS(*MI))) {
      auto PrevMFMA = SchedMFMA[SchedMFMA.size() - 1];
      unsigned PrevMFMAIssue = PrevMFMA->TopReadyCycle;

      if (PrevMFMAIssue + PrevMFMA->Latency > ReadyCycle) {
        for (auto &MO : MI->operands()) {
          if (!MO.isReg())
            continue;
          if (!MO.getReg().isPhysical())
            continue;

          if (!SRI->isVGPR(DAG->MRI, MO.getReg()))
            continue;

          for (auto &OtherMO : PrevMFMA->getInstr()->operands()) {
            if (!OtherMO.isReg())
              continue;
            if (!OtherMO.getReg().isPhysical())
              continue;

            if (!SRI->isVGPR(DAG->MRI, OtherMO.getReg()))
              continue;

            if (SRI->regsOverlap(MO.getReg(), OtherMO.getReg())) {
              ReadyCycle =
                  std::max(PrevMFMAIssue + PrevMFMA->Latency, ReadyCycle);
              break;
            }
          }
        }
      }
    }

    if (SchedEXP.size() && SII->isVALU(*MI) && !SII->isTRANS(*MI)) {
      auto PrevEXP = SchedEXP[SchedEXP.size() - 1];
      unsigned PrevEXPIssue = PrevEXP->TopReadyCycle;

      if (PrevEXPIssue + 2 > ReadyCycle) {
        for (auto &MO : MI->operands()) {
          if (!MO.isReg())
            continue;
          if (!MO.getReg().isPhysical())
            continue;

          if (!SRI->isVGPR(DAG->MRI, MO.getReg()))
            continue;

          for (auto &OtherMO : PrevEXP->getInstr()->operands()) {
            if (!OtherMO.isReg())
              continue;
            if (!OtherMO.getReg().isPhysical())
              continue;
            // if (!OtherMO.isDef())
            //   continue;

            if (!SRI->isVGPR(DAG->MRI, OtherMO.getReg()))
              continue;

            if (SRI->regsOverlap(MO.getReg(), OtherMO.getReg())) {
              ReadyCycle = std::max(PrevEXPIssue + 2, ReadyCycle);
              break;
            }
          }
        }
      }
    }

    if (SchedDSR.size()) {
      auto PrevDSR = SchedDSR[SchedDSR.size() - 1];
      unsigned PrevDSRIssue = PrevDSR->TopReadyCycle;
      if (PrevDSRIssue + 2 > ReadyCycle) {
        std::optional<unsigned> DSRMSB;
        for (auto &MO : PrevDSR->getInstr()->operands()) {
          DSRMSB = getMSBs(MO, SRI);
          if (DSRMSB)
            break;
        }
        std::optional<unsigned> ThisMSB;
        for (auto &MO : MI->operands()) {
          ThisMSB = getMSBs(MO, SRI);
          if (ThisMSB)
            break;
        }

        if (ThisMSB && DSRMSB && ThisMSB.value() != DSRMSB.value()) {
          ReadyCycle = std::max(PrevDSRIssue + 2, ReadyCycle);
        }
      }
    }
  }

  GCNHazardRecognizer *HazardRec =
      static_cast<GCNHazardRecognizer *>(Zone->HazardRec);
  if (HazardRec) {
    unsigned HazardStates = HazardRec->getHazardWaitStates(MI);
    if (HazardStates + CurrCycle > ReadyCycle) {
      return HazardStates;
    }
  }

  if (ReadyCycle > CurrCycle) {
    SU->TopReadyCycle = ReadyCycle;
    auto Wait = ReadyCycle - CurrCycle;
    return Wait;
  }

  return 0;
}

bool CandidateHeuristics::tryAsyncPipe(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary *Zone) {

  // TODO -- can we just call the member function?
  auto getStallCycles =
      [this,
       Zone](GenericSchedulerBase::SchedCandidate &SchedCand) -> unsigned {
    SUnit *SU = SchedCand.SU;
    unsigned ReadyCycle = SU->TopReadyCycle;
    unsigned CurrCycle = Zone->getCurrCycle();
    MachineInstr *MI = SU->getInstr();
    if (const_cast<SIInstrInfo *>(SII)->isLDSDMA(MI->getOpcode())) {
      return 0;
    }

    else if (MI->getOpcode() == AMDGPU::S_BARRIER_WAIT && SchedTDM.size()) {
      auto PrevTDM = SchedTDM[SchedTDM.size() - 1];

      if (PrevTDM->getInstr()->getOpcode() == AMDGPU::S_BARRIER_SIGNAL_IMM) {
        ReadyCycle =
            std::max(ReadyCycle, PrevTDM->TopReadyCycle + LatencyForSignalVal);
      }

    }

    else if ((MI->getOpcode() == AMDGPU::ATOMIC_FENCE ||
              MI->getOpcode() == AMDGPU::S_WAIT_TENSORCNT ||
              MI->getOpcode() == AMDGPU::S_WAIT_DSCNT)) {
      if (SchedDSR.size()) {
        auto PrevDSR = SchedDSR[SchedDSR.size() - 1];
        ReadyCycle =
            std::max(ReadyCycle, PrevDSR->TopReadyCycle + DSLatencyForFenceVal);
      } else {
        ReadyCycle = std::max(ReadyCycle, DSLatencyForFenceVal);
      }
    }
    if (ReadyCycle > CurrCycle)
      return ReadyCycle - CurrCycle;

    return 0;
  };

  auto isAsyncPipe =
      [this](GenericSchedulerBase::SchedCandidate &SchedCand) -> bool {
    SUnit *SU = SchedCand.SU;
    MachineInstr *MI = SU->getInstr();
    unsigned Opc = MI->getOpcode();

    return const_cast<SIInstrInfo *>(SII)->isLDSDMA(Opc)||
           Opc == AMDGPU::S_BARRIER_WAIT ||
           Opc == AMDGPU::S_BARRIER_SIGNAL_IMM || Opc == AMDGPU::ATOMIC_FENCE ||
           Opc == AMDGPU::S_WAIT_TENSORCNT || Opc == AMDGPU::S_WAIT_DSCNT;
  };

  bool CandIsAsync = isAsyncPipe(Cand);
  bool TryIsAsync = isAsyncPipe(TryCand);

  if (CandIsAsync == TryIsAsync)
    return false;

  if (CandIsAsync) {
    unsigned Stalls = getStallCycles(Cand);
    if (!Stalls) {
      if (Cand.Reason > GenericSchedulerBase::RegCritical) {
        Cand.Reason = GenericSchedulerBase::RegCritical;
      }
      return true;
    }
    return false;
  }

  if (TryIsAsync) {
    unsigned Stalls = getStallCycles(TryCand);
    if (!Stalls) {
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }
    return false;
  }

  return false;
}

bool CandidateHeuristics::tryVALUCoexecSlot(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary *Zone) {
  GCNHazardRecognizer *HazardRec =
      static_cast<GCNHazardRecognizer *>(Zone->HazardRec);
  int CoexecSlot =
      HazardRec->getWMMACoexecSlot(getLatencyStallCycles(TryCand.SU, Zone));

  MachineInstr *TryMI = TryCand.SU->getInstr();
  MachineInstr *CandMI = Cand.SU->getInstr();

  auto PreferNonTransVALU =
      [this](GenericSchedulerBase::SchedCandidate &TryCand,
             GenericSchedulerBase::SchedCandidate &Cand) {
        MachineInstr *TryMI = TryCand.SU->getInstr();
        MachineInstr *CandMI = Cand.SU->getInstr();
        // We don't want to issue TRANS or CVT here as they (along with WMMA)
        // will clog the whole VALU unit for multiple cycles
        bool TryIsSingleCycleVALU =
            SII->isVALU(*TryMI) && !SII->isMFMAorWMMA(*TryMI) &&
            !SII->isTRANS(*TryMI) && (SII->getRepeatRate(*TryMI) <= 1);
        bool CandIsSingleCycleVALU =
            SII->isVALU(*CandMI) && !SII->isMFMAorWMMA(*CandMI) &&
            !SII->isTRANS(*CandMI) && !SII->isTRANS(*CandMI) &&
            (SII->getRepeatRate(*CandMI) <= 1);

        if (TryIsSingleCycleVALU == CandIsSingleCycleVALU) {
          return false;
        }

        if (CandIsSingleCycleVALU)
          if (Cand.Reason > GenericSchedulerBase::RegCritical) {
            Cand.Reason = GenericSchedulerBase::RegCritical;
          }

        if (TryIsSingleCycleVALU) {
          TryCand.Reason = GenericSchedulerBase::RegCritical;
        }

        return true;
      };

  if (CoexecSlot == -1) {
    unsigned TransWaits = HazardRec->getTRANS32HazardState();
    if (TransWaits) {
      return PreferNonTransVALU(TryCand, Cand);
    }

    return false;
  }

  GCNHazardRecognizer::WMMASlotType CurrentSlot =
      (GCNHazardRecognizer::WMMASlotType)CoexecSlot;

  auto PreferTransVALU =
      [this, &PreferNonTransVALU](GenericSchedulerBase::SchedCandidate &TryCand,
                                  GenericSchedulerBase::SchedCandidate &Cand) {
        MachineInstr *TryMI = TryCand.SU->getInstr();
        MachineInstr *CandMI = Cand.SU->getInstr();
        // We don't want to issue TRANS or CVT here as they (along with WMMA)
        // will clog the whole VALU unit for multiple cycles
        bool TryIsTRANS = SII->isTRANS(*TryMI);
        bool CandIsTTRANS = SII->isTRANS(*CandMI);

        if (!TryIsTRANS && !CandIsTTRANS) {
          return PreferNonTransVALU(TryCand, Cand);
        }

        if (TryIsTRANS == CandIsTTRANS) {
          return false;
        }

        if (CandIsTTRANS)
          if (Cand.Reason > GenericSchedulerBase::RegCritical) {
            Cand.Reason = GenericSchedulerBase::RegCritical;
          }

        if (TryIsTRANS) {
          TryCand.Reason = GenericSchedulerBase::RegCritical;
        }

        return true;
      };

  switch (CurrentSlot) {
  default:
    return false;

  case GCNHazardRecognizer::WMMASlotType::MemCoExec0:
  case GCNHazardRecognizer::WMMASlotType::MemCoExec2: {
    bool TryIsMem = SII->isFLATGlobal(*TryMI) || SII->isDS(*TryMI);
    bool CandIsMem = SII->isFLATGlobal(*CandMI) || SII->isDS(*CandMI);

    bool TryIsLargeCopy = TryMI->isCopy();
    bool CandIsLargeCopy = CandMI->isCopy();

    if (TryIsLargeCopy) {
      TryIsLargeCopy &= SRI->getRegSizeInBits(*DAG->MRI.getRegClass(
                            TryMI->getOperand(0).getReg())) > 64 &&
                        !TryMI->getOperand(0).getSubReg();
      ;
    }

    if (CandIsLargeCopy) {
      CandIsLargeCopy &= SRI->getRegSizeInBits(*DAG->MRI.getRegClass(
                             CandMI->getOperand(0).getReg())) > 64 &&
                         !CandMI->getOperand(0).getSubReg();
      ;
    }

    if (CandIsLargeCopy && TryIsLargeCopy)
      return false;

    if (!CandIsLargeCopy && !TryIsLargeCopy) {

      if (TryIsMem == CandIsMem)
        return false;

      if (CandIsMem)
        if (Cand.Reason > GenericSchedulerBase::RegCritical)
          Cand.Reason = GenericSchedulerBase::RegCritical;

      if (TryIsMem)
        TryCand.Reason = GenericSchedulerBase::RegCritical;

      return true;
    }

    if (CandIsLargeCopy)
      TryCand.Reason = GenericSchedulerBase::RegCritical;

    else if (Cand.Reason > GenericSchedulerBase::RegCritical)
      Cand.Reason = GenericSchedulerBase::RegCritical;

    return true;
  }

  case GCNHazardRecognizer::WMMASlotType::MemCoExec1:
  case GCNHazardRecognizer::WMMASlotType::MemCoExec3: {
    bool TryIsMem = SII->isFLATGlobal(*TryMI) || SII->isDS(*TryMI);
    bool CandIsMem = SII->isFLATGlobal(*CandMI) || SII->isDS(*CandMI);

    bool TryIsLargeCopy = TryMI->isCopy();
    bool CandIsLargeCopy = CandMI->isCopy();

    if (TryIsLargeCopy) {
      TryIsLargeCopy &= SRI->getRegSizeInBits(*DAG->MRI.getRegClass(
                            TryMI->getOperand(0).getReg())) > 64 &&
                        !TryMI->getOperand(0).getSubReg();
      ;
    }

    if (CandIsLargeCopy) {
      CandIsLargeCopy &= SRI->getRegSizeInBits(*DAG->MRI.getRegClass(
                             CandMI->getOperand(0).getReg())) > 64 &&
                         !CandMI->getOperand(0).getSubReg();
      ;
    }

    if (CandIsLargeCopy && TryIsLargeCopy)
      return false;

    if (!CandIsLargeCopy && !TryIsLargeCopy) {

      if (CandIsMem == TryIsMem) {
        return false;
      }

      if (!CandIsMem)
        if (Cand.Reason > GenericSchedulerBase::RegCritical)
          Cand.Reason = GenericSchedulerBase::RegCritical;

      if (!TryIsMem)
        TryCand.Reason = GenericSchedulerBase::RegCritical;

      return true;
    }

    if (CandIsLargeCopy)
      TryCand.Reason = GenericSchedulerBase::RegCritical;

    else if (Cand.Reason > GenericSchedulerBase::RegCritical)
      Cand.Reason = GenericSchedulerBase::RegCritical;

    return true;
  }

  case GCNHazardRecognizer::WMMASlotType::ValuBlocked0: {
    bool TryIsSALU = SII->isMFMAorWMMA(*TryMI);
    bool CandIsSALU = SII->isMFMAorWMMA(*CandMI);

    if (!TryIsSALU && !CandIsSALU)
      return false;

    if (CandIsSALU && TryIsSALU) {
      return false;
    }

    if (CandIsSALU)
      if (Cand.Reason > GenericSchedulerBase::RegCritical)
        Cand.Reason = GenericSchedulerBase::RegCritical;

    if (TryIsSALU)
      TryCand.Reason = GenericSchedulerBase::RegCritical;

    return true;
  }

  case GCNHazardRecognizer::WMMASlotType::ValuBlocked1: {
    bool TryIsWMMA = SII->isMFMAorWMMA(*TryMI);
    bool CandIsWMMA = SII->isMFMAorWMMA(*CandMI);

    if (!TryIsWMMA && !CandIsWMMA)
      return false;

    if (CandIsWMMA && TryIsWMMA) {
      return false;
    }
    if (CandIsWMMA)
      if (Cand.Reason > GenericSchedulerBase::RegCritical)
        Cand.Reason = GenericSchedulerBase::RegCritical;

    if (TryIsWMMA)
      TryCand.Reason = GenericSchedulerBase::RegCritical;

    return true;
  }

  case GCNHazardRecognizer::WMMASlotType::ValuCoExec1: {
    return PreferNonTransVALU(TryCand, Cand);
  }

  case GCNHazardRecognizer::WMMASlotType::ValuCoExec0: {
    // We prefer 2 cycle TRANS here
    // FIXME -- should check that it is 2 cycle
    return PreferTransVALU(TryCand, Cand);
  }

  case GCNHazardRecognizer::WMMASlotType::ValuCoexecLastLdScale: {
    return PreferNonTransVALU(TryCand, Cand);
  }

  case GCNHazardRecognizer::WMMASlotType::ValuCoExec2: {
    return PreferTransVALU(TryCand, Cand);
  }
  }
  return false;
}

bool CandidateHeuristics::coexecWindowIsReady(CoexecWindow *Window,
                                              SchedBoundary *Zone, unsigned &MaxStall) {
  if (!Window->IsReady)
    return false;

  unsigned MaxFlavors = static_cast<unsigned>(InstructionFlavor::NUM_FLAVORS);

  InstructionFlavor Producer = Window->WindowProducer;
  auto ProducerSUs = MixInfo.getSUs(Producer);

  unsigned MinStall = 64;

  for (auto SU : ProducerSUs) {
    unsigned SUStall = getLatencyStallCycles(SU, Zone);
    if (SUStall < MinStall)
      MinStall = SUStall;
  }

  unsigned Add = 1;

  for (unsigned I = 0; I < MaxFlavors; I++) {
    unsigned RequiredCount = Window->RequiredCounts[I];
    if (!RequiredCount)
      continue;

    InstructionFlavor Flavor = static_cast<InstructionFlavor>(I);
    auto FlavorSUs = MixInfo.getSUs(Flavor);

    unsigned ReadyCount = 0;
    for (auto SU : FlavorSUs) {
      if (Producer == InstructionFlavor::WMMA &&
          Flavor == InstructionFlavor::SingleCycleVALU) {
        Add = 3;
      }
      if (!SU->isScheduled && SU->isTopReady()) {
        auto Stall = getLatencyStallCycles(SU, Zone);
        if (Stall > MaxStall)
          MaxStall = Stall;
        if (Stall <= MinStall + Add) {
          ++ReadyCount;
        }
      }
      if (ReadyCount >= RequiredCount)
        break;
    }

    if (Flavor == InstructionFlavor::DS) {
      auto FlavorSUs = MixInfo.getSUs(InstructionFlavor::SALU);
    for (auto SU : FlavorSUs) {
      if (Producer == InstructionFlavor::WMMA &&
          Flavor == InstructionFlavor::SingleCycleVALU) {
        Add = 3;
      }
      if (!SU->isScheduled && SU->isTopReady()) {
        auto Stall = getLatencyStallCycles(SU, Zone);
        if (Stall > MaxStall)
          MaxStall = Stall;
        if (Stall <= MinStall + Add) {
          ++ReadyCount;
        }
      }
      if (ReadyCount >= RequiredCount)
        break;
    }
    }



    if (ReadyCount < RequiredCount)
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Shadow Mix Heuristic
//===----------------------------------------------------------------------===//

/// Shadow Mix scheduling ensures WMMA instructions are only scheduled when
/// sufficient co-execution candidates (VALU and/or DS) are ready.
/// This enables the interleaved WMMA+VALU+DS pattern shown in optimal
/// schedules.
///
/// Shadow Priority Rules (toggleable, prefer long-latency so short fills
/// shadow): 1a. WMMA over DS       (-amdgpu-shadow-priority-wmma-over-ds) 1b.
/// WMMA over SALU     (-amdgpu-shadow-priority-wmma-over-salu) 2a. CVT over DS
/// (-amdgpu-shadow-priority-cvt-over-ds) 2b. CVT over SALU
/// (-amdgpu-shadow-priority-cvt-over-salu)
/// 3.  TRANS32 over VALU1c (-amdgpu-shadow-priority-trans32-over-valu1c)
///
/// Co-exec enablement rules:
/// 4. If enough co-exec candidates ready -> no intervention
/// 5. Defer WMMA if not enough VALU/DS ready
/// 6. Prefer instructions that enable co-exec candidates
/// 7. Lookahead to find path to pending co-exec
///
/// Returns true if a decision was made, sets Reason on the preferred candidate.
bool CandidateHeuristics::tryShadowMix(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary *Zone, 
    AMDGPUSchedReason &OutReason) {

  if (!EnableShadowMixVal)
    return false;

  MixInfo.updateReadyCounts();
  bool CurrentIsPopulated = CurrentWindow.IsPopulated;
  bool NextIsPopulated = NextWindow.IsPopulated;

  CurrentWindow.refreshMixInfo(MixInfo);
  CoexecWindow *TargetWindow = nullptr;
  TargetWindow = &CurrentWindow;

  if (!CurrentWindow.IsActive) {

    CoexecWindow TempWindow;
    TempWindow.refreshMixInfo(MixInfo);
    populateCandidateWindow(TempWindow);

    if (TempWindow.IsPopulated) {
      CurrentWindow.copy(TempWindow);
    }
  }

  if (CurrentWindow.IsReady && CurrentWindow.IsActive) {
    assert(CurrentIsPopulated);
    bool NextIsPopulated = NextWindow.IsPopulated;
    NextWindow.refreshMixInfo(MixInfo);
    if (!NextIsPopulated) {
      populateCandidateWindow(NextWindow);
    }
    TargetWindow = &NextWindow;
  }

  assert(TargetWindow);
  unsigned ReadyVALU1c =
      MixInfo.getReadyCount(InstructionFlavor::SingleCycleVALU);
  unsigned ReadyDS = MixInfo.getReadyCount(InstructionFlavor::DS);
  unsigned ReadySALU = MixInfo.getReadyCount(InstructionFlavor::SALU);

  // FIXME: should these values be determined by calculateHiddenLatency
  unsigned RequiredVALU1c = ShadowMixWMMAMinVALU1cVal;
  unsigned RequiredDS = ShadowMixWMMAMinDSVal;
  unsigned RequiredSALU = ShadowMixWMMAMinSALUVal;

  InstructionFlavor TryFlavor = classifyFlavor(TryCand.SU->getInstr(), SII);
  InstructionFlavor CandFlavor = classifyFlavor(Cand.SU->getInstr(), SII);

  bool TryIsWMMA = (TryFlavor == InstructionFlavor::WMMA);
  bool CandIsWMMA = (CandFlavor == InstructionFlavor::WMMA);
  bool TryIsCVT = (TryFlavor == InstructionFlavor::MultiCycleVALU);
  bool CandIsCVT = (CandFlavor == InstructionFlavor::MultiCycleVALU);
  bool TryIsDS = (TryFlavor == InstructionFlavor::DS);
  bool CandIsDS = (CandFlavor == InstructionFlavor::DS);
  bool TryIsSALU = (TryFlavor == InstructionFlavor::SALU);
  bool CandIsSALU = (CandFlavor == InstructionFlavor::SALU);
  bool TryIsVALU1c = (TryFlavor == InstructionFlavor::SingleCycleVALU);
  bool CandIsVALU1c = (CandFlavor == InstructionFlavor::SingleCycleVALU);
  bool TryIsTRANS32 = (TryFlavor == InstructionFlavor::TRANS);
  bool CandIsTRANS32 = (CandFlavor == InstructionFlavor::TRANS);


  // Helper lambda for shadow priority decisions
  auto preferFirst = [&](bool TryIsFirst, AMDGPUSchedReason Reason,
                         const char *FirstName,
                         const char *SecondName) -> bool {
    if (TryIsFirst) {
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      OutReason = Reason;
      LLVM_DEBUG(dbgs() << "  ShadowMix: Prefer " << FirstName << " over "
                        << SecondName << " (will fill shadow)\n");
      return true;
    } else {
      if (Cand.Reason > GenericSchedulerBase::RegCritical)
        Cand.Reason = GenericSchedulerBase::RegCritical;
      OutReason = Reason;
      LLVM_DEBUG(dbgs() << "  ShadowMix: Prefer " << FirstName << " over "
                        << SecondName << " (will fill shadow)\n");
      return true;
    }
  };

  if (ShadowMixRulesVal) {
    // Shadow Priority Rules: prefer long-latency so short ones fill shadow
    // slots. Each rule is independently toggleable for debugging.

    // Rule 1a: WMMA over DS
    if (ShadowPriorityWMMAOverDSVal && TryIsWMMA != CandIsWMMA) {
      if ((TryIsWMMA && CandIsDS) || (TryIsDS && CandIsWMMA))
        return preferFirst(TryIsWMMA,
                           AMDGPUSchedReason::ShadowPriorityWMMAOverDS, "WMMA",
                           "DS");
    }

    // Rule 1b: WMMA over SALU
    if (ShadowPriorityWMMAOverSALUVal && TryIsWMMA != CandIsWMMA) {
      if ((TryIsWMMA && CandIsSALU) || (TryIsSALU && CandIsWMMA))
        return preferFirst(TryIsWMMA,
                           AMDGPUSchedReason::ShadowPriorityWMMAOverSALU,
                           "WMMA", "SALU");
    }

    // Rule 2a: CVT over DS
    if (ShadowPriorityCVTOverDSVal && TryIsCVT != CandIsCVT) {
      if ((TryIsCVT && CandIsDS) || (TryIsDS && CandIsCVT))
        return preferFirst(TryIsCVT, AMDGPUSchedReason::ShadowPriorityCVTOverDS,
                           "CVT", "DS");
    }

    // Rule 2b: CVT over SALU
    if (ShadowPriorityCVTOverSALUVal && TryIsCVT != CandIsCVT) {
      if ((TryIsCVT && CandIsSALU) || (TryIsSALU && CandIsCVT))
        return preferFirst(TryIsCVT,
                           AMDGPUSchedReason::ShadowPriorityCVTOverSALU, "CVT",
                           "SALU");
    }

    // Rule 3: TRANS32 (v_exp etc) over 1-cycle VALU
    if (ShadowPriorityTRANS32OverVALU1cVal && TryIsTRANS32 != CandIsTRANS32) {
      if ((TryIsTRANS32 && CandIsVALU1c) || (TryIsVALU1c && CandIsTRANS32))
        return preferFirst(TryIsTRANS32,
                           AMDGPUSchedReason::ShadowPriorityTRANS32OverVALU,
                           "TRANS32", "VALU1c");
    }

    // Rule 3b: When filling TRANS32 shadow, prefer VALU1c over SALU
    // This reserves SALU for WMMA/CVT shadows where it's more valuable.
    if (ShadowPreferVALU1cOverSALUForTRANSVal && TryIsVALU1c != CandIsVALU1c) {
      if ((TryIsVALU1c && CandIsSALU) || (TryIsSALU && CandIsVALU1c))
        return preferFirst(
            TryIsVALU1c, AMDGPUSchedReason::ShadowPreferVALU1cOverSALUForTRANS,
            "VALU1c", "SALU");
    }
  }

  if (!TargetWindow->IsReady) {
    bool TryIsProducer = TryFlavor == TargetWindow->WindowProducer;
    bool CandIsProducer = CandFlavor == TargetWindow->WindowProducer;

    if (TryIsProducer && CandIsProducer)
      return false;

    SmallVector<InstructionFlavor, 4> NeededFlavors;
    TargetWindow->getNeededFlavors(NeededFlavors);

    assert(NeededFlavors.size());

    for (InstructionFlavor &NeededFlavor : NeededFlavors) {
      unsigned TryEnables = 0;

      if (!TryIsProducer)
        TryEnables =
            countDirectlyEnabledByFlavor(TryCand.SU, NeededFlavor, SII);

      unsigned CandEnables = 0;

      if (!CandIsProducer)
        CandEnables = countDirectlyEnabledByFlavor(Cand.SU, NeededFlavor, SII);

      if (!TryEnables && !CandEnables)
        continue;

      if (TryEnables && CandEnables)
        return false;

      StringRef FlavorName = getFlavorShortName(NeededFlavor);
      if (CandEnables > TryEnables) {
        if (Cand.Reason > GenericSchedulerBase::RegCritical)
          Cand.Reason = GenericSchedulerBase::RegCritical;
        OutReason = AMDGPUSchedReason::ShadowEnableDirect;
        LLVM_DEBUG(dbgs() << "  ShadowMix: Prefer Cand (enables " << CandEnables
                          << " vs " << TryEnables << " " << FlavorName
                          << ")\n");
        return true;
      } else {
        TryCand.Reason = GenericSchedulerBase::RegCritical;
        OutReason = AMDGPUSchedReason::ShadowEnableDirect;
        LLVM_DEBUG(dbgs() << "  ShadowMix: Prefer TryCand (enables "
                          << TryEnables << " vs " << CandEnables << " "
                          << FlavorName << ")\n");
        return true;
      }
    }

    // Neither candidate directly enables any of the needed flavors, look eahd.
    for (InstructionFlavor &NeededFlavor : NeededFlavors) {

      auto [NearestTarget, Cost] = findNearestPendingByFlavor(
          MixInfo, NeededFlavor, ShadowMixLookaheadDepthVal,
          ShadowMixMaxBlockingCostVal, ShadowMixMaxVisitedVal,
          ShadowMixMaxCandidatesVal);

      if (NearestTarget) {
        bool TryHelps =
            !TryIsProducer && wouldHelpEnable(TryCand.SU, NearestTarget, DAG);
        bool CandHelps =
            !CandIsProducer && wouldHelpEnable(Cand.SU, NearestTarget, DAG);

        if (!TryHelps && !CandHelps)
          continue;

        if (TryHelps && CandHelps)
          return false;

        StringRef FlavorName = getFlavorShortName(NeededFlavor);
        if (TryHelps) {
          TryCand.Reason = GenericSchedulerBase::RegCritical;
          OutReason = AMDGPUSchedReason::ShadowEnableLookahead;
          LLVM_DEBUG(dbgs() << "  ShadowMix: Prefer TryCand (on path to "
                            << FlavorName << ", cost=" << Cost << ")\n");
          return true;
        } else {
          if (Cand.Reason > GenericSchedulerBase::RegCritical)
            Cand.Reason = GenericSchedulerBase::RegCritical;
          OutReason = AMDGPUSchedReason::ShadowEnableLookahead;
          LLVM_DEBUG(dbgs() << "  ShadowMix: Prefer Cand (on path to "
                            << FlavorName << ", cost=" << Cost << ")\n");
          return true;
        }
      }
    }

    // The window is not ready, and we cannot make progress according to our
    // lookahead analysis. Greedily take the window now.
    if (!TryIsProducer && !CandIsProducer)
      return false;

    StringRef FlavorName = getFlavorShortName(TargetWindow->WindowProducer);
    if (TryIsProducer) {
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      OutReason = AMDGPUSchedReason::ShadowEnableLookahead;
      LLVM_DEBUG(
          dbgs()
          << "  ShadowMix: Lookahead failed, Prefer TryCand Window Producer "
          << FlavorName << "\n");
      return true;
    }

    if (Cand.Reason > GenericSchedulerBase::RegCritical)
      Cand.Reason = GenericSchedulerBase::RegCritical;
    OutReason = AMDGPUSchedReason::ShadowEnableLookahead;
    LLVM_DEBUG(
        dbgs() << "  ShadowMix: Lookahead failed, Prefer Cand Window Producer "
               << FlavorName << "\n");
    return true;
  }

  // Rule 4: If we have enough co-exec candidates, and we do not need to stall
  // for them, then schedule the window.
  unsigned MaxStall = 0;
  if (coexecWindowIsReady(TargetWindow, Zone, MaxStall)) {
    bool TryIsProducer = TryFlavor == TargetWindow->WindowProducer;
    bool CandIsProducer = CandFlavor == TargetWindow->WindowProducer;
    if (TryIsProducer == CandIsProducer)
      return false;

    // TODO -- what if next is TRANS and current is active WMMA?
    if (CandIsProducer) {
      // Cand is non-WMMA, prefer it
      if (Cand.Reason > GenericSchedulerBase::RegCritical)
        Cand.Reason = GenericSchedulerBase::RegCritical;
      OutReason = AMDGPUSchedReason::ShadowDeferWMMA;

      LLVM_DEBUG(dbgs() << "  ShadowMix: Producer (Window Ready)\n");
      return true;
    } else {
      // TryCand is non-WMMA, prefer it
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      OutReason = AMDGPUSchedReason::ShadowDeferWMMA;
      LLVM_DEBUG(dbgs() << "  ShadowMix: Producer (Window Ready)\n");
      return true;
    }
  }
  bool TryIsProducer = TryFlavor == TargetWindow->WindowProducer;
  bool CandIsProducer = CandFlavor == TargetWindow->WindowProducer;
  // Rule 5: Defer WMMA if window consumers are not ready.
  if (TryIsProducer != CandIsProducer) {
    // Prefer the non-WMMA candidate
    if (TryIsProducer) {
      // Cand is non-WMMA, prefer it
      if (Cand.Reason > GenericSchedulerBase::RegCritical)
        Cand.Reason = GenericSchedulerBase::RegCritical;
      OutReason = AMDGPUSchedReason::ShadowDeferWMMA;
      LLVM_DEBUG(dbgs() << "  ShadowMix: Deferring WMMA (VALU1c=" << ReadyVALU1c
                        << "/" << RequiredVALU1c << ", DS=" << ReadyDS << "/"
                        << RequiredDS << ")\n");
      return true;
    } else {
      // TryCand is non-WMMA, prefer it
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      OutReason = AMDGPUSchedReason::ShadowDeferWMMA;
      LLVM_DEBUG(dbgs() << "  ShadowMix: Deferring WMMA (VALU1c=" << ReadyVALU1c
                        << "/" << RequiredVALU1c << ", DS=" << ReadyDS << "/"
                        << RequiredDS << ")\n");
      return true;
    }
  }

  return false;
}

bool CandidateHeuristics::tryCriticalResourceDependency(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary *Zone,
    bool IsAsync) {

  if (IsAsync)
    return false;

  auto HasPrioritySU = [this, &Cand, &TryCand,
                              IsAsync](unsigned ResourceIdx) {
    // unsigned MaxAvailableLat =
    // Zone->findMaxLatency(Zone->Available.elements());
    HardwareUnitInfo HWUI = HWUInfo[ResourceIdx];
    // unsigned CriticalUsage = HWUI.getTotalCycles();

    auto CandFlavor = classifyFlavor(Cand.SU->getInstr(), SII);
    bool LookDeep = CandFlavor == InstructionFlavor::DS &&
                    HWUI.getType() == InstructionFlavor::WMMA;
    auto *TargetSU = HWUI.getNextTargetSU(LookDeep);



    if (!TargetSU)
      return false;

    return true;
  };

  auto IsCandidateResource = [this, &Cand, &TryCand,
                              IsAsync](unsigned ResourceIdx) {
    // unsigned MaxAvailableLat =
    // Zone->findMaxLatency(Zone->Available.elements());
    HardwareUnitInfo HWUI = HWUInfo[ResourceIdx];
    // unsigned CriticalUsage = HWUI.getTotalCycles();

    if (!IsAsync && HWUI.getRemainingExposed() == 0 &&
        !HWUI.ProducesCoexecWindow)
      return false;

    return true;
  };

  auto TryEnablesResource = [&Cand, &TryCand, this](unsigned ResourceIdx) {
    HardwareUnitInfo HWUI = HWUInfo[ResourceIdx];
    auto CandFlavor = classifyFlavor(Cand.SU->getInstr(), SII);

    // We want to ensure our DS order matches WMMA order.
    bool LookDeep = CandFlavor == InstructionFlavor::DS &&
                    HWUI.getType() == InstructionFlavor::WMMA;
    auto *TargetSU = HWUI.getNextTargetSU(LookDeep);

    bool CandEnables =
        TargetSU != Cand.SU && DAG->IsReachable(TargetSU, Cand.SU);
    bool TryCandEnables =
        TargetSU != TryCand.SU && DAG->IsReachable(TargetSU, TryCand.SU);

    if (!CandEnables && !TryCandEnables)
      return false;

    if (CandEnables && !TryCandEnables) {
      if (Cand.Reason > GenericSchedulerBase::RegCritical)
        Cand.Reason = GenericSchedulerBase::RegCritical;

      return true;
    }

    if (!CandEnables && TryCandEnables) {
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }

    // Both enable, prefer the critical path.
    bool CandHeight = Cand.SU->getHeight();
    bool TryCandHeight = TryCand.SU->getHeight();

    if (CandHeight > TryCandHeight) {
      if (Cand.Reason > GenericSchedulerBase::RegCritical)
        Cand.Reason = GenericSchedulerBase::RegCritical;

      return true;
    }

    if (CandHeight < TryCandHeight) {
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }

    // Same critical path, just prefer original candidate.
    if (Cand.Reason > GenericSchedulerBase::RegCritical)
      Cand.Reason = GenericSchedulerBase::RegCritical;

    return true;
  };

  if (IsAsync) {
    for (unsigned I = 0; I < HWUInfo.size(); I++) {
      if (!HWUInfo[I].IsAsync)
        continue;
      
      if (!HasPrioritySU(I))
        continue;

      if (!IsCandidateResource(I))
        return false;

      return TryEnablesResource(I);
    }
    return false;
  }

  unsigned Cutoff = std::min(HWUInfo.size(), (size_t)ResourceToBalanceVal);
  unsigned CheckedResources = 0;

  for (unsigned I = 0; I < HWUInfo.size(); I++) {
    if (CheckedResources++ >= Cutoff)
      return false;

    if (!HasPrioritySU(I))
      continue;

    // If we have encountered a resource that is not critical, then neither
    // candidate enables a critical resource
    if (!IsCandidateResource(I))
      return false;

    bool Enabled = TryEnablesResource(I);
    // If neither has enabled the resource, continue to the next resource
    if (Enabled)
      return true;
  }
  return false;
}

bool CandidateHeuristics::tryCriticalResource(
    GenericSchedulerBase::SchedCandidate &TryCand,
    GenericSchedulerBase::SchedCandidate &Cand, SchedBoundary *Zone) {
  unsigned CandOp = Cand.SU->getInstr()->getOpcode();


  unsigned Cutoff = std::min(HWUInfo.size(), (size_t)ResourceToBalanceVal);
  unsigned CheckedResources = 0;
  for (unsigned I = 0; I < HWUInfo.size(); I++) {
    HardwareUnitInfo HWUI = HWUInfo[I];
    if (CheckedResources++ >= Cutoff)
      return false;

    // unsigned MaxAvailableLat =
    // Zone->findMaxLatency(Zone->Available.elements()); unsigned CriticalUsage
    // = HWUI.getTotalCycles();

    // if (MaxAvailableLat > CriticalUsage)
    //   return false;

    if (HWUI.getRemainingExposed() == 0 && !HWUI.ProducesCoexecWindow)
      return false;

    bool CandUsesCrit = HWUI.contains(Cand.SU);
    bool TryCandUsesCrit = HWUI.contains(TryCand.SU);

    if (!CandUsesCrit && !TryCandUsesCrit)
      continue;

    if (CandUsesCrit && !TryCandUsesCrit) {
      if (Cand.Reason > GenericSchedulerBase::RegCritical)
        Cand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }

    if (!CandUsesCrit && TryCandUsesCrit) {
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }

    if (HWUI.isHigherPriority(Cand.SU, TryCand.SU)) {
      if (Cand.Reason > GenericSchedulerBase::RegCritical)
        Cand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }

    if (HWUI.isHigherPriority(TryCand.SU, Cand.SU)) {
      TryCand.Reason = GenericSchedulerBase::RegCritical;
      return true;
    }
  }

  return false;
}

void CandidateHeuristics::schedNode(SUnit *SU, GCNHazardRecognizer *HazardRec) {
  // FIXME - is this ProcResource loop correct? Can we just use it to maximize ReleaseAtCycle?
  // Can we fold the ReleaseAtCycle logic into getHWUICYclesForInst?
  if (SchedModel && SchedModel->hasInstrSchedModel()) {
    unsigned ReleaseAtCycle = 0;
    const MCSchedClassDesc *SC = DAG->getSchedClass(SU);
    for (TargetSchedModel::ProcResIter
             PI = SchedModel->getWriteProcResBegin(SC),
             PE = SchedModel->getWriteProcResEnd(SC);
         PI != PE; ++PI) {
      ReleaseAtCycle = std::max(ReleaseAtCycle, (unsigned)PI->ReleaseAtCycle);
    }
    unsigned Latency = getHWUICyclesForInst(SU, ReleaseAtCycle);
    unsigned MaxInstrLatency = std::max(Latency, ReleaseAtCycle);

    auto *MI = SU->getInstr();
    InstructionFlavor Flavor = classifyFlavor(MI, SII);

    // FIXME
    bool IsHidden = HazardRec->inVALUShadow();

    bool FoundIt = false;
    for (auto &HWUI : HWUInfo) {
      if (HWUI.getType() == Flavor) {
        HWUI.schedule(SU, Latency, SII);
        if (!IsHidden)
          HWUI.reduceRemainingExposed();
        FoundIt = true;
        break;
      }
    }

    assert(FoundIt);

    // TODO - explore if we should base this on coexec flavor
    if (SII->isMFMAorWMMA(*MI)) {
      SchedMFMA.push_back(SU);
    }
    if (SII->isDS(*MI) && MI->mayLoad()) {
      SchedDSR.push_back(SU);
    }

    auto Opc = MI->getOpcode();
    if (Opc == AMDGPU::ATOMIC_FENCE || Opc == AMDGPU::S_WAIT_ASYNCCNT || Opc == AMDGPU::S_WAIT_TENSORCNT || Opc == AMDGPU::S_WAIT_DSCNT || Opc == AMDGPU::S_BARRIER_WAIT || Opc == AMDGPU::S_BARRIER_SIGNAL_IMM) {
      SchedTDM.push_back(SU);
    }

    if (SII->isTRANS(*MI)) {
      SchedEXP.push_back(SU);
    }
  }

  InstructionFlavor Flavor = classifyFlavor(SU->getInstr(), SII);
  MixInfo.markScheduled(SU, Flavor);
}

void CandidateHeuristics::bumpNode(SUnit *SU, SchedBoundary *Zone) {
  if (SchedModel && SchedModel->hasInstrSchedModel()) {
    unsigned ReleaseAtCycle = 0;
    const MCSchedClassDesc *SC = DAG->getSchedClass(SU);
    for (TargetSchedModel::ProcResIter
             PI = SchedModel->getWriteProcResBegin(SC),
             PE = SchedModel->getWriteProcResEnd(SC);
         PI != PE; ++PI) {
      ReleaseAtCycle = std::max(ReleaseAtCycle, (unsigned)PI->ReleaseAtCycle);
    }
    unsigned Latency = getHWUICyclesForInst(SU, ReleaseAtCycle);
    unsigned MaxInstrLatency = std::max(Latency, ReleaseAtCycle);
    InstructionFlavor Flavor = classifyFlavor(SU->getInstr(), SII);

    if (CurrentWindow.IsActive && SU->TopReadyCycle >= CurrentWindow.EndCycle) {
      CurrentWindow.clear();
      CoexecWindow TempWindow;
      MixInfo.updateReadyCounts();
      TempWindow.refreshMixInfo(MixInfo);
      populateCandidateWindow(TempWindow);

      if (TempWindow.IsPopulated) {
        CurrentWindow.copy(TempWindow);
        NextWindow.clear();
      }
    }

    // It is possible that the current instruction has clobbered the previous
    // window and started a new one.
    if (!CurrentWindow.IsActive && Flavor != InstructionFlavor::Other && Flavor == CurrentWindow.WindowProducer) {
      CurrentWindow.IsActive = true;
      CurrentWindow.StartCycle = SU->TopReadyCycle;
      CurrentWindow.EndCycle = MaxInstrLatency + CurrentWindow.StartCycle - 1;
    }
  }

  CurrCycle = Zone->getCurrCycle();
}

void AMDGPUMLSchedStrategy::dumpPickSummary(SUnit *SU, bool IsTopNode,
                                            SchedCandidate &Cand) {
  const SIInstrInfo *SII = static_cast<const SIInstrInfo *>(DAG->TII);
  unsigned Cycle = IsTopNode ? Top.getCurrCycle() : Bot.getCurrCycle();

  dbgs() << "=== Pick @ Cycle " << Cycle << " ===\n";

  Heurs.MixInfo.updateReadyCounts();
  Heurs.MixInfo.dumpReadyPending(dbgs());

  InstructionFlavor Flavor = classifyFlavor(SU->getInstr(), SII);
  dbgs() << "Picked: SU(" << SU->NodeNum << ") ";
  SU->getInstr()->print(dbgs(), /*IsStandalone=*/true, /*SkipOpers=*/false,
                        /*SkipDebugLoc=*/true);
  dbgs() << " [" << getFlavorName(Flavor) << "]\n";

  dbgs() << "  Reason: ";
  if (LastAMDGPUReason != AMDGPUSchedReason::None)
    dbgs() << getReasonName(LastAMDGPUReason);
  else if (Cand.Reason != NoCand)
    dbgs() << GenericSchedulerBase::getReasonStr(Cand.Reason);
  else
    dbgs() << "Unknown";
  dbgs() << "\n\n";

  LastAMDGPUReason = AMDGPUSchedReason::None;
}

bool AMDGPUMLSchedStrategy::tryPendingCandidate(SchedCandidate &Cand,
                                                SchedCandidate &TryCand,
                                                SchedBoundary *Zone) {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg)) {
    DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "PhysReg\n");
    return TryCand.Reason != NoCand;
  }

  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    if (EnableWMMACooloff && Heurs.tryWMMACoolOff(TryCand, Cand, Zone)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "WMMACoolOff\n");
      return TryCand.Reason != NoCand;
    }


    if (Heurs.tryAsyncPipe(TryCand, Cand, Zone)) {
      return TryCand.Reason != NoCand;
    }
  }

  // Avoid exceeding the target's limit.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                  RegExcess, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  /*
  // Avoid increasing the max critical pressure in the scheduled region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                  TryCand, Cand, RegCritical, TRI, DAG->MF))
    return TryCand.Reason != NoCand;*/

  if (SameBoundary) {

    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(Heurs.getLatencyStallCycles(TryCand.SU, Zone),
                Heurs.getLatencyStallCycles(Cand.SU, Zone), TryCand, Cand,
                Stall)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", {
        unsigned TryStall = Heurs.getLatencyStallCycles(TryCand.SU, Zone);
        unsigned CandStall = Heurs.getLatencyStallCycles(Cand.SU, Zone);
        dbgs() << "Stall, Try: " << TryStall << ", Cand: " << CandStall << "\n";
      });
      return TryCand.Reason != NoCand;
    }

    if (Heurs.tryVALUCoexecSlot(TryCand, Cand, Zone)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "ValuCoexec\n");
      LastAMDGPUReason = AMDGPUSchedReason::WMMACoexec;
      return TryCand.Reason != NoCand;
    }

    // Shadow Mix: Ensure sufficient VALU ready before scheduling WMMA.
    // This enables interleaved WMMA+VALU co-execution patterns.
    if (Heurs.tryShadowMix(TryCand, Cand, Zone, LastAMDGPUReason)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "ShadowMix\n");
      return TryCand.Reason != NoCand;
    }

    Heurs.sortResources();
    if (Heurs.tryCriticalResource(TryCand, Cand, Zone)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "CritResource\n");
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceBalance;
      return TryCand.Reason != NoCand;
    }

    if (Heurs.tryCriticalResourceDependency(TryCand, Cand, Zone, false)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs()
                                                       << "CritResourceDep\n");
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceDep;
      return TryCand.Reason != NoCand;
    }

    if (Heurs.tryCriticalResourceDependency(TryCand, Cand, Zone, true)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose",
                      dbgs() << "CritResourceDep Async\n");
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceDep;
      return TryCand.Reason != NoCand;
    }
  }

  // Avoid increasing the max critical pressure in the scheduled region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                  TryCand, Cand, RegCritical, TRI, DAG->MF))
    return TryCand.Reason != NoCand;

  return false;
}

bool AMDGPUMLSchedStrategy::tryCandidateBalanced(SchedCandidate &Cand,
                                                 SchedCandidate &TryCand,
                                                 SchedBoundary *Zone) {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = FirstValid;
    return true;
  }

  // Bias PhysReg Defs and copies to their uses and defined respectively.
  if (tryGreater(biasPhysReg(TryCand.SU, TryCand.AtTop),
                 biasPhysReg(Cand.SU, Cand.AtTop), TryCand, Cand, PhysReg)) {
    DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "PhysReg\n");
    return TryCand.Reason != NoCand;
  }
  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    if (EnableWMMACooloff && Heurs.tryWMMACoolOff(TryCand, Cand, Zone)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "WMMACoolOff\n");
      return TryCand.Reason != NoCand;
    }


    if (Heurs.tryAsyncPipe(TryCand, Cand, Zone)) {
      return TryCand.Reason != NoCand;
    }
  }

  // Avoid exceeding the target's limit.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.Excess, Cand.RPDelta.Excess, TryCand, Cand,
                  RegExcess, TRI, DAG->MF)) {
    return TryCand.Reason != NoCand;
  }

  /*
  // Avoid increasing the max critical pressure in the scheduled region.
  if (DAG->isTrackingPressure() && tryPressure(TryCand.RPDelta.CriticalMax,
                                               Cand.RPDelta.CriticalMax,
                                               TryCand, Cand, RegCritical, TRI,
                                               DAG->MF))
    return TryCand.Reason != NoCand;
*/
  // We only compare a subset of features when comparing nodes between
  // Top and Bottom boundary. Some properties are simply incomparable, in many
  // other instances we should only override the other boundary if something
  // is a clear good pick on one boundary. Skip heuristics that are more
  // "tie-breaking" in nature.

  // Keep clustered nodes together to encourage downstream peephole
  // optimizations which may reduce resource requirements.
  //
  // This is a best effort to set things up for a post-RA pass. Optimizations
  // like generating loads of multiple registers should ideally be done within
  // the scheduler pass by combining the loads during DAG postprocessing.
  /*
  unsigned CandZoneCluster = getClusterID(Cand.AtTop);
  unsigned TryCandZoneCluster = getClusterID(TryCand.AtTop);
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);

  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
                 Cluster)) {
                 DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() <<
  "Cluster\n"); return TryCand.Reason != NoCand;
                 }
  }*/

  if (SameBoundary) {
    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(Heurs.getLatencyStallCycles(TryCand.SU, Zone), Heurs.getLatencyStallCycles(Cand.SU, Zone),  TryCand, Cand, Stall)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", {
        unsigned TryStall = Heurs.getLatencyStallCycles(TryCand.SU, Zone);
        unsigned CandStall = Heurs.getLatencyStallCycles(Cand.SU, Zone);
        dbgs() << "Stall, Try: " << TryStall << ", Cand: " << CandStall << "\n";
      });
      return TryCand.Reason != NoCand;
    }

    if (Heurs.tryVALUCoexecSlot(TryCand, Cand, Zone)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "ValuCoexec\n");
      LastAMDGPUReason = AMDGPUSchedReason::WMMACoexec;
      return TryCand.Reason != NoCand;
    }

    // Shadow Mix: Ensure sufficient VALU ready before scheduling WMMA.
    // This enables interleaved WMMA+VALU co-execution patterns.
    if (Heurs.tryShadowMix(TryCand, Cand, Zone, LastAMDGPUReason)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "ShadowMix\n");
      return TryCand.Reason != NoCand;
    }

    Heurs.sortResources();
    if (Heurs.tryCriticalResource(TryCand, Cand, Zone)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "CritResource\n");
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceBalance;
      return TryCand.Reason != NoCand;
    }

    if (Heurs.tryCriticalResourceDependency(TryCand, Cand, Zone, false)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "CritResourceDep\n");
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceDep;
      return TryCand.Reason != NoCand;
    }

    if (Heurs.tryCriticalResourceDependency(TryCand, Cand, Zone, true)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "CritResourceDep Async\n");
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceDep;
      return TryCand.Reason != NoCand;
    }

    // For loops that are acyclic path limited, aggressively schedule for
    // latency. Within an single cycle, whenever CurrMOps > 0, allow normal
    // heuristics to take precedence.
    //if (Rem.IsAcyclicLatencyLimited && !Zone->getCurrMOps() &&
    //    tryLatency(TryCand, Cand, *Zone))
    //  return TryCand.Reason != NoCand;

  }
  // Avoid increasing the max critical pressure in the scheduled region.
  if (DAG->isTrackingPressure() &&
      tryPressure(TryCand.RPDelta.CriticalMax, Cand.RPDelta.CriticalMax,
                  TryCand, Cand, RegCritical, TRI, DAG->MF)) {
    return TryCand.Reason != NoCand;
  }

  // Keep clustered nodes together to encourage downstream peephole
  // optimizations which may reduce resource requirements.
  //
  // This is a best effort to set things up for a post-RA pass. Optimizations
  // like generating loads of multiple registers should ideally be done within
  // the scheduler pass by combining the loads during DAG postprocessing.
  /*
  unsigned CandZoneCluster = getClusterID(Cand.AtTop);
  unsigned TryCandZoneCluster = getClusterID(TryCand.AtTop);
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);

  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
                 Cluster)) {
                 DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "Cluster\n");
    return TryCand.Reason != NoCand;
                 }
*/
  if (SameBoundary) {
    /*
    // Weak edges are for clustering and other constraints.
    if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand, Weak)) {
                  DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "Weak\n");
      return TryCand.Reason != NoCand;
                }
                */
  }

  // Avoid increasing the max pressure of the entire region.
  //if (DAG->isTrackingPressure() &&
  //    tryPressure(TryCand.RPDelta.CurrentMax, Cand.RPDelta.CurrentMax, TryCand,
  //                Cand, RegMax, TRI, DAG->MF)) {
   //                DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "RP\n");
   // return TryCand.Reason != NoCand;
   //               }

  // Fall through to original instruction order.
  if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
      (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
    DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "NID\n");
    TryCand.Reason = NodeOrder;
    return true;
  }

  return false;
}

void AMDGPUMLSchedStrategy::pickNodeFromQueue(
    SchedBoundary &Zone, const CandPolicy &ZonePolicy,
    const RegPressureTracker &RPTracker, SchedCandidate &Cand, bool &IsPending,
    bool IsBottomUp) {
  const SIRegisterInfo *SRI = static_cast<const SIRegisterInfo *>(TRI);
  ArrayRef<unsigned> Pressure = RPTracker.getRegSetPressureAtPos();
  unsigned SGPRPressure = 0;
  unsigned VGPRPressure = 0;
  IsPending = false;
  if (DAG->isTrackingPressure()) {
    if (!UseGCNTrackers) {
      SGPRPressure = Pressure[AMDGPU::RegisterPressureSets::SReg_32];
      VGPRPressure = Pressure[AMDGPU::RegisterPressureSets::VGPR_32];
    } else {
      GCNRPTracker *T = IsBottomUp
                            ? static_cast<GCNRPTracker *>(&UpwardTracker)
                            : static_cast<GCNRPTracker *>(&DownwardTracker);
      SGPRPressure = T->getPressure().getSGPRNum();
      VGPRPressure = T->getPressure().getArchVGPRNum();
    }
  }
  LLVM_DEBUG(dbgs() << "Available Q:\n");
  DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "Checking Available:\n");
  ReadyQueue &AQ = Zone.Available;
  for (SUnit *SU : AQ) {
    DEBUG_WITH_TYPE("machine-scheduler-verbose", SU->getInstr()->dump());
    SchedCandidate TryCand(ZonePolicy);
    initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI, SGPRPressure,
                  VGPRPressure, IsBottomUp);
    // Pass SchedBoundary only when comparing nodes from the same boundary.
    SchedBoundary *ZoneArg = Cand.AtTop == TryCand.AtTop ? &Zone : nullptr;
    tryCandidateBalanced(Cand, TryCand, ZoneArg);
    if (TryCand.Reason != NoCand) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "NewBest!\n");
      // Initialize resource delta if needed in case future heuristics query it.
      if (TryCand.ResDelta == SchedResourceDelta())
        TryCand.initResourceDelta(Zone.DAG, SchedModel);
      printCandidateDecision(Cand, TryCand);
      Cand.setBest(TryCand);
    } else {
      printCandidateDecision(TryCand, Cand);
    }
  }

  if (!shouldCheckPending(Zone, SchedModel))
    return;

  LLVM_DEBUG(dbgs() << "Pending Q:\n");
  DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "Checking Pending:\n");
  ReadyQueue &PQ = Zone.Pending;
  for (SUnit *SU : PQ) {
    DEBUG_WITH_TYPE("machine-scheduler-verbose", SU->getInstr()->dump());
    SchedCandidate TryCand(ZonePolicy);
    initCandidate(TryCand, SU, Zone.isTop(), RPTracker, SRI, SGPRPressure,
                  VGPRPressure, IsBottomUp);
    // Pass SchedBoundary only when comparing nodes from the same boundary.
    SchedBoundary *ZoneArg = Cand.AtTop == TryCand.AtTop ? &Zone : nullptr;
    AMDGPUMLSchedStrategy::tryPendingCandidate(Cand, TryCand, ZoneArg);
    if (TryCand.Reason != NoCand) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "NewBest!\n");
      // Initialize resource delta if needed in case future heuristics query it.
      if (TryCand.ResDelta == SchedResourceDelta())
        TryCand.initResourceDelta(Zone.DAG, SchedModel);
      printCandidateDecision(Cand, TryCand);
      IsPending = true;
      Cand.setBest(TryCand);
    } else {
      printCandidateDecision(TryCand, Cand);
    }
  }
}

SUnit *AMDGPUMLSchedStrategy::pickNode(bool &IsTopNode) {
  if (DAG->top() == DAG->bottom()) {
    assert(Top.Available.empty() && Top.Pending.empty() &&
           Bot.Available.empty() && Bot.Pending.empty() && "ReadyQ garbage");
    return nullptr;
  }
  bool PickedPending;
  SUnit *SU;
  SchedCandidate *PickedCand = nullptr;
  do {
    PickedPending = false;
    if (RegionPolicy.OnlyTopDown) {
      SU = pickOnlyChoice(Top, SchedModel);
      if (!SU) {
        CandPolicy NoPolicy;
        TopCand.reset(NoPolicy);
        pickNodeFromQueue(Top, TopCand.Policy, DAG->getTopRPTracker(), TopCand,
                          PickedPending,
                          /*IsBottomUp=*/false);
        assert(TopCand.Reason != NoCand && "failed to find a candidate");
        SU = TopCand.SU;
        PickedCand = &TopCand;
      }
      IsTopNode = true;
    } else if (RegionPolicy.OnlyBottomUp) {
      SU = pickOnlyChoice(Bot, SchedModel);
      if (!SU) {
        CandPolicy NoPolicy;
        BotCand.reset(NoPolicy);
        pickNodeFromQueue(Bot, BotCand.Policy, DAG->getBotRPTracker(), BotCand,
                          PickedPending,
                          /*IsBottomUp=*/true);
        assert(BotCand.Reason != NoCand && "failed to find a candidate");
        SU = BotCand.SU;
        PickedCand = &BotCand;
      }
      IsTopNode = false;
    } else {
      SU = pickNodeBidirectional(IsTopNode, PickedPending);
      PickedCand = IsTopNode ? &TopCand : &BotCand;
    }
  } while (SU->isScheduled);

  LLVM_DEBUG(if (PickedCand) dumpPickSummary(SU, IsTopNode, *PickedCand));

  if (PickedPending) {
    unsigned ReadyCycle = IsTopNode ? SU->TopReadyCycle : SU->BotReadyCycle;
    SchedBoundary &Zone = IsTopNode ? Top : Bot;
    unsigned CurrentCycle = Zone.getCurrCycle();
    if (ReadyCycle > CurrentCycle)
      Zone.bumpCycle(ReadyCycle);

    // FIXME: checkHazard() doesn't give information about which cycle the
    // hazard will resolve so just keep bumping the cycle by 1. This could be
    // made more efficient if checkHazard() returned more details.
    while (Zone.checkHazard(SU))
      Zone.bumpCycle(Zone.getCurrCycle() + 1);

    Zone.releasePending();
  }

  if (SU->isTopReady())
    Top.removeReady(SU);
  if (SU->isBottomReady())
    Bot.removeReady(SU);

  return SU;
}


AMDGPUMLPostSchedStrategy::AMDGPUMLPostSchedStrategy(
    const MachineSchedContext *C)
    : PostGenericScheduler(C) {}

bool AMDGPUMLPostSchedStrategy::tryCandidate(SchedCandidate &Cand,
                                             SchedCandidate &TryCand, SchedBoundary *Zone) {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = FirstValid;
    return true;
  }

  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(Heurs.getLatencyStallCycles(TryCand.SU, Zone),
                Heurs.getLatencyStallCycles(Cand.SU, Zone),
                TryCand, Cand, Stall)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", {
        unsigned TryStall = Heurs.getLatencyStallCycles(
            TryCand.SU, Zone);
        unsigned CandStall = Heurs.getLatencyStallCycles(
            Cand.SU, Zone);
        dbgs() << "Stall, Try: " << TryStall << ", Cand: " << CandStall << "\n";
      });
      return TryCand.Reason != NoCand;
    }

    if (Heurs.tryVALUCoexecSlot(TryCand, Cand, Zone)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "ValuCoexec\n");
      LastAMDGPUReason = AMDGPUSchedReason::WMMACoexec;
      return TryCand.Reason != NoCand;
    }

    // Shadow Mix: Ensure sufficient VALU ready before scheduling WMMA.
    // This enables interleaved WMMA+VALU co-execution patterns.
    if (Heurs.tryShadowMix(TryCand, Cand, Zone, LastAMDGPUReason)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "ShadowMix\n");
      return TryCand.Reason != NoCand;
    }

    Heurs.sortResources();
    if (Heurs.tryCriticalResource(TryCand, Cand, Zone)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "CritResource\n");
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceBalance;
      return TryCand.Reason != NoCand;
    }

    if (Heurs.tryCriticalResourceDependency(TryCand, Cand, Zone, false)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "CritResourceDep\n");
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceDep;
      return TryCand.Reason != NoCand;
    }

    if (Heurs.tryCriticalResourceDependency(TryCand, Cand, Zone, true)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "CritResourceDepAsync\n");
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceDep;
      return TryCand.Reason != NoCand;
    }

    // For loops that are acyclic path limited, aggressively schedule for
    // latency. Within an single cycle, whenever CurrMOps > 0, allow normal
    // heuristics to take precedence.
    if (Rem.IsAcyclicLatencyLimited && !Zone->getCurrMOps() &&
        tryLatency(TryCand, Cand, *Zone)) {
          DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "Latency\n");
      return TryCand.Reason != NoCand;
        }
  }

  // Keep clustered nodes together to encourage downstream peephole
  // optimizations which may reduce resource requirements.
  //
  // This is a best effort to set things up for a post-RA pass. Optimizations
  // like generating loads of multiple registers should ideally be done within
  // the scheduler pass by combining the loads during DAG postprocessing.
  /*
  unsigned CandZoneCluster = getClusterID(Cand.AtTop);
  unsigned TryCandZoneCluster = getClusterID(TryCand.AtTop);
  bool CandIsClusterSucc =
      isTheSameCluster(CandZoneCluster, Cand.SU->ParentClusterIdx);
  bool TryCandIsClusterSucc =
      isTheSameCluster(TryCandZoneCluster, TryCand.SU->ParentClusterIdx);

  if (tryGreater(TryCandIsClusterSucc, CandIsClusterSucc, TryCand, Cand,
                 Cluster))
    return TryCand.Reason != NoCand;
*/
  if (SameBoundary) {
    // Weak edges are for clustering and other constraints.
    if (tryLess(getWeakLeft(TryCand.SU, TryCand.AtTop),
                getWeakLeft(Cand.SU, Cand.AtTop), TryCand, Cand, Weak)) {
                  DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "Cluster\n");
      return TryCand.Reason != NoCand;
                }
  }

  if (SameBoundary) {
    // Fall through to original instruction order.
    if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
        (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
          DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "NID\n");
      TryCand.Reason = NodeOrder;
      return true;
    }
  }

  return false;
}

bool AMDGPUMLPostSchedStrategy::tryPendingCandidate(SchedCandidate &Cand,
                                                    SchedCandidate &TryCand,
                                                    SchedBoundary *Zone) {
  // Initialize the candidate if needed.
  if (!Cand.isValid()) {
    TryCand.Reason = NodeOrder;
    return true;
  }

  bool SameBoundary = Zone != nullptr;
  if (SameBoundary) {
    // Prioritize instructions that read unbuffered resources by stall cycles.
    if (tryLess(Heurs.getLatencyStallCycles(TryCand.SU, Zone),
                Heurs.getLatencyStallCycles(Cand.SU, Zone), TryCand, Cand,
                Stall)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", {
        unsigned TryStall = Heurs.getLatencyStallCycles(TryCand.SU, Zone);
        unsigned CandStall = Heurs.getLatencyStallCycles(Cand.SU, Zone);
        dbgs() << "Stall, Try: " << TryStall << ", Cand: " << CandStall << "\n";
      });
      return TryCand.Reason != NoCand;
    }

    if (Heurs.tryVALUCoexecSlot(TryCand, Cand, Zone)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "ValuCoexec\n");
      LastAMDGPUReason = AMDGPUSchedReason::WMMACoexec;
      return TryCand.Reason != NoCand;
    }

    // Shadow Mix: Ensure sufficient VALU ready before scheduling WMMA.
    // This enables interleaved WMMA+VALU co-execution patterns.
    if (Heurs.tryShadowMix(TryCand, Cand, Zone, LastAMDGPUReason)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "ShadowMix\n");
      return TryCand.Reason != NoCand;
    }

    Heurs.sortResources();
    if (Heurs.tryCriticalResource(TryCand, Cand, Zone)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "CritResource\n");
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceBalance;
      return TryCand.Reason != NoCand;
    }

    if (Heurs.tryCriticalResourceDependency(TryCand, Cand, Zone, false)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "CritResourceDep\n");
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceDep;
      return TryCand.Reason != NoCand;
    }

    if (Heurs.tryCriticalResourceDependency(TryCand, Cand, Zone, true)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "CritResourceDep Async\n");
      LastAMDGPUReason = AMDGPUSchedReason::CritResourceDep;
      return TryCand.Reason != NoCand;
    }
  }

  if (SameBoundary) {
    // Fall through to original instruction order.
    if ((Zone->isTop() && TryCand.SU->NodeNum < Cand.SU->NodeNum) ||
        (!Zone->isTop() && TryCand.SU->NodeNum > Cand.SU->NodeNum)) {
          DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "NID\n");
      TryCand.Reason = NodeOrder;
      return true;
    }
  }

  return false;
}

void AMDGPUMLPostSchedStrategy::schedNode(SUnit *SU, bool IsTopNode) {
   DEBUG_WITH_TYPE("machine-scheduler-verbose", {
     dbgs() << "Scheduling: "; DAG->dumpNode(*SU);
     dbgs() << "\n\n";
   });

   Heurs.schedNode(SU, static_cast<GCNHazardRecognizer *>(Top.HazardRec));
   PostGenericScheduler::schedNode(SU, IsTopNode);
   Heurs.bumpNode(SU, &Top);
}

unsigned AMDGPUMLPostSchedStrategy::getHWUICyclesForInst(
    SUnit *SU, const SIInstrInfo *SII, unsigned ReleaseAtCycle) {
  auto Opc = SU->getInstr()->getOpcode();
  bool IsDMA = const_cast<SIInstrInfo *>(SII)->isLDSDMA(Opc);
  unsigned Latency = IsDMA ? SU->Latency : ReleaseAtCycle;
  if (SII->isDS(*SU->getInstr()) && SU->getInstr()->mayLoad())
    Latency = 8;

  return Latency;
}

void AMDGPUMLPostSchedStrategy::initialize(ScheduleDAGMI *DAG) {
  // ML scheduling strategy is only done top-down to support new resource
  // balancing heuristics.
  DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "New region\n");
  RegionPolicy.OnlyTopDown = true;
  RegionPolicy.OnlyBottomUp = false;
  PostGenericScheduler::initialize(DAG);

  if (Top.HazardRec) {
    delete Top.HazardRec;
    Top.HazardRec = nullptr;
  }
  Top.HazardRec = new GCNHazardRecognizer(
      DAG->MF, GCNHazardRecognizer::OperatingMode::PostRA);

  Heurs.initialize(DAG, static_cast<GCNHazardRecognizer *>(Top.HazardRec),
                   SchedModel, TRI, false, true);
}

void AMDGPUMLPostSchedStrategy::pickNodeFromQueue(SchedBoundary &Zone,
                                                  SchedCandidate &Cand,
                                                  bool &IsPending) {
  ReadyQueue &Q = Zone.Available;
  DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "Checking Available\n");
  for (SUnit *SU : Q) {
    DEBUG_WITH_TYPE("machine-scheduler-verbose", DAG->dumpNode(*SU));
    SchedCandidate TryCand(Cand.Policy);
    TryCand.SU = SU;
    TryCand.AtTop = Zone.isTop();
    TryCand.initResourceDelta(DAG, SchedModel);
    if (AMDGPUMLPostSchedStrategy::tryCandidate(Cand, TryCand, &Zone)) {
      IsPending = false;
      Cand.setBest(TryCand);
       DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "NewBest\n");
      LLVM_DEBUG(traceCandidate(Cand));
    }
  }

  ReadyQueue &PQ = Zone.Pending;
  DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "Checking Pending\n");
  for (SUnit *SU : PQ) {
    DEBUG_WITH_TYPE("machine-scheduler-verbose", DAG->dumpNode(*SU));
    SchedCandidate TryCand(Cand.Policy);
    TryCand.SU = SU;
    TryCand.AtTop = Zone.isTop();
    TryCand.initResourceDelta(DAG, SchedModel);
    // Pass SchedBoundary only when comparing nodes from the same boundary.
    SchedBoundary *ZoneArg = Cand.AtTop == TryCand.AtTop ? &Zone : nullptr;
    if (tryPendingCandidate(Cand, TryCand, ZoneArg)) {
      DEBUG_WITH_TYPE("machine-scheduler-verbose", dbgs() << "NewBest\n");
      IsPending = true;
      Cand.setBest(TryCand);
      LLVM_DEBUG(traceCandidate(Cand));
    }
  }
}

void AMDGPUMLPostSchedStrategy::dumpPickSummary(SUnit *SU, bool IsTopNode,
                                                SchedCandidate &Cand) {
  const SIInstrInfo *SII = static_cast<const SIInstrInfo *>(DAG->TII);
  unsigned Cycle = IsTopNode ? Top.getCurrCycle() : Bot.getCurrCycle();

  dbgs() << "=== PostRA Pick @ Cycle " << Cycle << " ===\n";

  Heurs.MixInfo.updateReadyCounts();
  Heurs.MixInfo.dumpReadyPending(dbgs());

  InstructionFlavor Flavor = classifyFlavor(SU->getInstr(), SII);
  dbgs() << "Picked: SU(" << SU->NodeNum << ") ";
  SU->getInstr()->print(dbgs(), /*IsStandalone=*/true, /*SkipOpers=*/false,
                        /*SkipDebugLoc=*/true);
  dbgs() << " [" << getFlavorName(Flavor) << "]\n";

  dbgs() << "  Reason: ";
  if (LastAMDGPUReason != AMDGPUSchedReason::None)
    dbgs() << getReasonName(LastAMDGPUReason);
  else if (Cand.Reason != NoCand)
    dbgs() << GenericSchedulerBase::getReasonStr(Cand.Reason);
  else
    dbgs() << "Unknown";
  dbgs() << "\n\n";

  LastAMDGPUReason = AMDGPUSchedReason::None;
}

/// Pick the next node to schedule.
SUnit *AMDGPUMLPostSchedStrategy::pickNode(bool &IsTopNode) {
  bool IsPending = false;
  if (DAG->top() == DAG->bottom()) {
    assert(Top.Available.empty() && Top.Pending.empty() &&
           Bot.Available.empty() && Bot.Pending.empty() && "ReadyQ garbage");
    return nullptr;
  }
  SUnit *SU;
  SchedCandidate *PickedCand = nullptr;
  if (RegionPolicy.OnlyBottomUp) {
    SU = pickOnlyChoice(Top, SchedModel);
    if (!SU) {
      CandPolicy NoPolicy;
      BotCand.reset(NoPolicy);
      // Set the bottom-up policy based on the state of the current bottom
      // zone and the instructions outside the zone, including the top zone.
      setPolicy(BotCand.Policy, /*IsPostRA=*/true, Bot, nullptr);
      pickNodeFromQueue(Bot, BotCand, IsPending);
      assert(BotCand.Reason != NoCand && "failed to find a candidate");
      SU = BotCand.SU;
      PickedCand = &BotCand;
    }
    IsTopNode = false;
  } else if (RegionPolicy.OnlyTopDown) {
    SU = pickOnlyChoice(Top, SchedModel);
    if (!SU) {
      CandPolicy NoPolicy;
      TopCand.reset(NoPolicy);
      // Set the top-down policy based on the state of the current top zone
      // and the instructions outside the zone, including the bottom zone.
      setPolicy(TopCand.Policy, /*IsPostRA=*/true, Top, nullptr);
      pickNodeFromQueue(Top, TopCand, IsPending);
      assert(TopCand.Reason != NoCand && "failed to find a candidate");

      SU = TopCand.SU;
      PickedCand = &TopCand;
    }
    IsTopNode = true;

  } else {
    SU = pickNodeBidirectional(IsTopNode, IsPending);
    PickedCand = IsTopNode ? &TopCand : &BotCand;
  }
  assert(!SU->isScheduled && "SUnit scheduled twice.");

  LLVM_DEBUG(if (PickedCand) dumpPickSummary(SU, IsTopNode, *PickedCand));

  if (IsPending) {
    unsigned ReadyCycle = IsTopNode ? SU->TopReadyCycle : SU->BotReadyCycle;
    SchedBoundary &Zone = IsTopNode ? Top : Bot;
    unsigned CurrentCycle = Zone.getCurrCycle();
    if (ReadyCycle > CurrentCycle)
      Zone.bumpCycle(ReadyCycle);

    // FIXME: checkHazard() doesn't give information about which cycle the
    // hazard will resolve so just keep bumping the cycle by 1. This could be
    // made more efficient if checkHazard() returned more details.
    while (Zone.checkHazard(SU))
      Zone.bumpCycle(Zone.getCurrCycle() + 1);

    Zone.releasePending();
  }

  if (SU->isTopReady())
    Top.removeReady(SU);
  if (SU->isBottomReady())
    Bot.removeReady(SU);

  LLVM_DEBUG(dbgs() << "Scheduling SU(" << SU->NodeNum << ") "
                    << *SU->getInstr());

  return SU;
}
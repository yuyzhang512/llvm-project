//===-- GCNHazardRecognizers.h - GCN Hazard Recognizers ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines hazard recognizers for scheduling on GCN processors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPUHAZARDRECOGNIZERS_H
#define LLVM_LIB_TARGET_AMDGPUHAZARDRECOGNIZERS_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "SIInstrInfo.h"
#include <list>


namespace llvm {

class MachineFunction;
class MachineInstr;
class MachineOperand;
class MachineRegisterInfo;
class SIRegisterInfo;
class GCNSubtarget;

class GCNHazardRecognizer final : public ScheduleHazardRecognizer {
public:
  typedef function_ref<bool(const MachineInstr &)> IsHazardFn;
  typedef function_ref<bool(const MachineInstr &, int WaitStates)> IsExpiredFn;
  typedef function_ref<unsigned int(const MachineInstr *)> GetNumWaitStatesFn;

  /// Operating mode for the hazard recognizer.
  /// - PreRA: Used during pre-RA scheduling (virtual registers, limited hazard checking)
  /// - PostRA: Used during post-RA scheduling (physical registers, full hazard checking)
  /// - HazardRecognizerMode: Used by the standalone hazard recognizer pass (inserts NOPs)
  enum class OperatingMode { PreRA, PostRA, HazardRecognizerMode };

  /// WMMA pipeline slot types for co-execution tracking.
  /// Based on GFX1250 WMMA pipeline behavior:
  /// - Execute: WMMA execution cycle, can only co-issue control instructions
  /// - MemCoExec: Can co-issue mem or salu (non-control)
  /// - ValuCoExec: Can co-issue mem, salu, or valu
  /// - ValuBlocked: VALU blocked after WMMA completes, can only issue wmma/mem/salu
  /// - WMMABlocked: WMMA blocked, can issue mem/salu/valu
  enum class WMMASlotType {
    Execute,
    MemCoExec0,
    MemCoExec1,
    MemCoExec2,
    MemCoExec3,
    ValuCoExec0,
    ValuCoExec1,
    ValuCoExec2,
    ValuCoexecLastLdScale,
    ValuCoExecLdScale,
    ValuBlocked0,
    ValuBlocked1,
    WMMABlocked
  };

  int getWMMACoexecSlot();
  int getWMMACoexecSlot(unsigned LookAhead);

  bool isWMMAPipelineHazard();

  bool inVALUShadow();
  void getWMMASlots(const MachineInstr &MI,
                    SmallVectorImpl<WMMASlotType> &WMMAPipelineState);
  
  unsigned getWaitStatesBetween(MachineInstr *Begin, MachineInstr *End) const;

private:
  // Operating mode determines which hazards are checked and whether fixes are applied.
  OperatingMode Mode;

  // Legacy flag for backward compatibility - true when in HazardRecognizerMode.
  bool IsHazardRecognizerMode;

  // This variable stores the instruction that has been emitted this cycle. It
  // will be added to EmittedInstrs, when AdvanceCycle() or RecedeCycle() is
  // called.
  MachineInstr *CurrCycleInstr;
  std::list<MachineInstr*> EmittedInstrs;

  /// Separate tracking for VALU pipeline hazards (WMMA coexecution).
  /// Only contains VALU/WMMA instructions. nullptr entries represent V_NOP
  /// stalls that were inserted specifically to resolve VALU-pipeline hazards.
  /// This allows WMMA hazard detection to ignore S_NOP stalls that don't help.
  std::list<MachineInstr*> EmittedVALUInstrs;

  /// Maximum depth to track in EmittedVALUInstrs.
  /// WMMA hazards can require up to 9 VALU instructions between dependent ops.
  static constexpr unsigned MaxVALULookAhead = 10;

  /// Tracks whether a WMMA coexecution hazard was detected in the last
  /// getHazardType() call. If the scheduler stalls for this hazard, we need
  /// to record a V_NOP placeholder in EmittedVALUInstrs.
  bool HasPendingWMMAHazard = false;

  const MachineFunction &MF;
  const GCNSubtarget &ST;
  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  const TargetSchedModel &TSchedModel;

  // Loop info for V_NOP hoisting, passed from the pass manager.
  MachineLoopInfo *MLI = nullptr;

  bool RunLdsBranchVmemWARHazardFixup;

  /// Current WMMA pipeline state for pre-RA scheduling.
  /// Tracks remaining cycles and their slot types.
  SmallVector<WMMASlotType, 16> WMMAPipelineState;

  /// Track if the last instruction emitted was a TRANS32 instruction.
  unsigned CyclesUntilTRANS32 = 0;

  unsigned CyclesUntilVALU = 0;

  unsigned CyclesUntilSALU = 0;

  /// Tracks whether the last WMMA scale pipeline ended with its final
  /// VALU co-exec slot being consumed by a VALU. When set, issuing another
  /// WMMA immediately should incur a one cycle stall.
  unsigned PendingWMMAScaleValuTailStall = 0;

  /// Check WMMA co-execution hazards for pre-RA scheduling.
  /// Returns the number of stall cycles needed before MI can be issued.
  unsigned checkWMMACoexecSlot(const MachineInstr &MI) const;

  /// Check for TRANS32 hazards.
  /// Returns the number of stall cycles needed before MI can be issued.
  unsigned checkTRANS32Hazard(const MachineInstr &MI) const;

  unsigned checkCVTHazard(const MachineInstr &MI) const;

  unsigned checkSSrcHazard(const MachineInstr &MI) const;

  /// Update WMMA pipeline state when a WMMA instruction is emitted.
  void updateWMMAPipelineState(const MachineInstr &MI);

  /// Update TRANS32 state when an instruction is emitted.
  void updateTRANS32State(const MachineInstr &MI, bool SubOne = false);

  void updateCVTState(const MachineInstr &MI, bool SubOne = false);

  void updateSSrcState(const MachineInstr &MI, bool SubOne = false);

  //===--------------------------------------------------------------------===//
  // Pre-RA scheduling mode wrappers.
  // These methods handle pre-RA specific state tracking and can be extended
  // for future pre-RA hazard recognition features.
  //===--------------------------------------------------------------------===//

  /// Pre-RA wrapper for EmitInstruction - updates pre-RA specific state.
  void preRAEmitInstruction(MachineInstr *MI);

  /// Pre-RA wrapper for AdvanceCycle - advances pre-RA pipeline state.
  void preRAAdvanceCycle();

  /// Pre-RA wrapper for Reset - clears pre-RA specific state.
  void preRAReset();

  /// Pre-RA hazard check - returns additional wait states for pre-RA mode.
  unsigned preRAGetHazardWaitStates(MachineInstr *MI) const;

  unsigned postRAGetHazardWaitStates(MachineInstr *MI) const;

  /// RegUnits of uses in the current soft memory clause.
  mutable BitVector ClauseUses;

  /// RegUnits of defs in the current soft memory clause.
  mutable BitVector ClauseDefs;

  void resetClause() const {
    ClauseUses.reset();
    ClauseDefs.reset();
  }

  void addClauseInst(const MachineInstr &MI) const;

  /// \returns the number of wait states before another MFMA instruction can be
  /// issued after \p MI.
  unsigned getMFMAPipelineWaitStates(const MachineInstr &MI) const;

  // Advance over a MachineInstr bundle. Look for hazards in the bundled
  // instructions.
  void processBundle();

  // Run on an individual instruction in hazard recognizer mode. This can be
  // used on a newly inserted instruction before returning from PreEmitNoops.
  void runOnInstruction(MachineInstr *MI);

  static unsigned getDefaultNumWaitStates(const MachineInstr *MI) {
    return MI ? SIInstrInfo::getNumWaitStates(*MI) : 1;
  }

using StaticGetNumWaitStatesFn =
    function_ref<unsigned int(const MachineInstr &)>;

  int getWaitStatesSince(
      IsHazardFn IsHazard, int Limit,
      StaticGetNumWaitStatesFn GetNumWaitStates) const;

int getWaitStatesSince(
    IsHazardFn IsHazard, int Limit, GetNumWaitStatesFn GetNumWaitStates) const;

  int getWaitStatesSince(
      IsHazardFn IsHazard, int Limit) const;


  /// Query the VALU-specific instruction list for hazards.
  /// This only considers VALU/WMMA instructions and V_NOP stalls.
  /// Used for WMMA coexecution hazards where S_NOPs don't resolve the hazard.
  int getWaitStatesSinceVALU(IsHazardFn IsHazard, int Limit) const;

  int getWaitStatesSinceDef(unsigned Reg, IsHazardFn IsHazardDef, int Limit) const;
  int getWaitStatesSinceSetReg(IsHazardFn IsHazard, int Limit) const;

  int checkSoftClauseHazards(MachineInstr *SMEM) const;
  int checkSMRDHazards(MachineInstr *SMRD) const;
  int checkVMEMHazards(MachineInstr *VMEM) const;
  int checkDPPHazards(MachineInstr *DPP) const;
  int checkDivFMasHazards(MachineInstr *DivFMas) const;
  int checkGetRegHazards(MachineInstr *GetRegInstr) const;
  int checkSetRegHazards(MachineInstr *SetRegInstr) const;
  int createsVALUHazard(const MachineInstr &MI) const;
  int checkVALUHazards(MachineInstr *VALU) const;
  int checkVALUHazardsHelper(const MachineOperand &Def,
                             const MachineRegisterInfo &MRI) const;
  int checkRWLaneHazards(MachineInstr *RWLane) const;
  int checkRFEHazards(MachineInstr *RFE) const;
  int checkInlineAsmHazards(MachineInstr *IA) const;
  int checkReadM0Hazards(MachineInstr *SMovRel) const;
  int checkNSAtoVMEMHazard(MachineInstr *MI) const;
  int checkFPAtomicToDenormModeHazard(MachineInstr *MI) const;
  // Emit \p WaitStatesNeeded V_NOP instructions before \p InsertPt.
  // If IsHoisting is true, uses empty DebugLoc for compiler-inserted NOPs.
  void emitVNops(MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
                 int WaitStatesNeeded, bool IsHoisting = false);
  void fixHazards(MachineInstr *MI);
  bool fixVcmpxPermlaneHazards(MachineInstr *MI);
  bool fixVMEMtoScalarWriteHazards(MachineInstr *MI);
  bool fixSMEMtoVectorWriteHazards(MachineInstr *MI);
  bool fixVcmpxExecWARHazard(MachineInstr *MI);
  bool fixLdsBranchVmemWARHazard(MachineInstr *MI);
  bool fixLdsDirectVALUHazard(MachineInstr *MI);
  bool fixLdsDirectVMEMHazard(MachineInstr *MI);
  bool fixVALUPartialForwardingHazard(MachineInstr *MI);
  bool fixVALUTransUseHazard(MachineInstr *MI);
  bool fixVALUTransCoexecutionHazards(MachineInstr *MI);
  bool fixWMMAHazards(MachineInstr *MI);
  int checkWMMACoexecutionHazards(MachineInstr *MI) const;
  bool fixWMMACoexecutionHazards(MachineInstr *MI);
  bool tryHoistWMMAVnopsFromLoop(MachineInstr *MI, int WaitStatesNeeded);
  bool hasWMMAHazardInLoop(MachineLoop *L, MachineInstr *MI,
                           bool IncludeSubloops = true);
  bool hasWMMAToWMMARegOverlap(const MachineInstr &WMMA,
                               const MachineInstr &MI) const;
  bool hasWMMAToVALURegOverlap(const MachineInstr &WMMA,
                               const MachineInstr &MI) const;
  bool isCoexecutionHazardFor(const MachineInstr &I,
                              const MachineInstr &MI) const;
  int checkTRANSCoexecutionHazards(MachineInstr *MI);

  bool fixShift64HighRegBug(MachineInstr *MI);
  bool fixVALUMaskWriteHazard(MachineInstr *MI);
  bool fixRequiredExportPriority(MachineInstr *MI);
  bool fixGetRegWaitIdle(MachineInstr *MI);
  bool fixDsAtomicAsyncBarrierArriveB64(MachineInstr *MI);
  bool fixScratchBaseForwardingHazard(MachineInstr *MI);
  bool fixSetRegMode(MachineInstr *MI);

  int checkMAIHazards(MachineInstr *MI) const;
  int checkMAIHazards908(MachineInstr *MI) const;
  int checkMAIHazards90A(MachineInstr *MI) const;
  /// Pad the latency between neighboring MFMA instructions with s_nops. The
  /// percentage of wait states to fill with s_nops is specified by the command
  /// line option '-amdgpu-mfma-padding-ratio'.
  ///
  /// For example, with '-amdgpu-mfma-padding-ratio=100':
  ///
  /// 2 pass MFMA instructions have a latency of 2 wait states. Therefore, a
  /// 'S_NOP 1' will be added between sequential MFMA instructions.
  ///
  /// V_MFMA_F32_4X4X1F32
  /// V_MFMA_F32_4X4X1F32
  ///-->
  /// V_MFMA_F32_4X4X1F32
  /// S_NOP 1
  /// V_MFMA_F32_4X4X1F32
  int checkMFMAPadding(MachineInstr *MI) const;
  int checkMAIVALUHazards(MachineInstr *MI) const;
  int checkMAILdStHazards(MachineInstr *MI) const;
  int checkPermlaneHazards(MachineInstr *MI) const;

public:
  GCNHazardRecognizer(const MachineFunction &MF, OperatingMode Mode, 
                      MachineLoopInfo *MLI = nullptr);

  /// Legacy constructor - defaults to HazardRecognizerMode for backward compatibility.
  GCNHazardRecognizer(const MachineFunction &MF);

  /// Returns the current operating mode.
  OperatingMode getOperatingMode() const { return Mode; }

  /// Returns true if running in pre-RA scheduling mode.
  bool isPreRA() const { return Mode == OperatingMode::PreRA; }

  /// Returns true if running in post-RA scheduling mode.
  bool isPostRA() const { return Mode == OperatingMode::PostRA; }

  // We can only issue one instruction per cycle.
  bool atIssueLimit() const override { return true; }
  void EmitInstruction(SUnit *SU) override;
  void EmitInstruction(MachineInstr *MI) override;
  HazardType getHazardType(SUnit *SU, int Stalls) override;

  /// Returns the number of wait states until all hazards for \p MI are
  /// resolved. This is useful for scheduling heuristics that want
  /// cycle-accurate hazard information rather than just a boolean.  Unlike
  /// PreEmitNoops, this does not modify state or fix hazards.
  unsigned getHazardWaitStates(MachineInstr *MI) const;
  void EmitNoop() override;
  unsigned PreEmitNoops(MachineInstr *) override;
  unsigned PreEmitNoopsCommon(MachineInstr *) const;
  void AdvanceCycle() override;
  void RecedeCycle() override;
  bool ShouldPreferAnother(SUnit *SU) const override;
  void Reset() override;

  unsigned getStallCount(SUnit *SU) override;

  unsigned getTRANS32HazardState() { return CyclesUntilTRANS32; };
};

} // end namespace llvm

#endif //LLVM_LIB_TARGET_AMDGPUHAZARDRECOGNIZERS_H

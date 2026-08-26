//===-- AMDGPUPrepareAGPRAlloc.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Make simple transformations to relax register constraints for cases which can
// allocate to AGPRs or VGPRs. Replace materialize of inline immediates into
// AGPR or VGPR with a pseudo with an AV_* class register constraint. This
// allows later passes to inflate the register class if necessary. The register
// allocator does not know to replace instructions to relax constraints.
//
//===----------------------------------------------------------------------===//

#include "AMDGPUPrepareAGPRAlloc.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/IR/Metadata.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-prepare-agpr-alloc"

namespace {

class AMDGPUPrepareAGPRAllocImpl {
private:
  const SIInstrInfo &TII;
  MachineRegisterInfo &MRI;

  bool isAV64Imm(const MachineOperand &MO) const;

public:
  AMDGPUPrepareAGPRAllocImpl(const GCNSubtarget &ST, MachineRegisterInfo &MRI)
      : TII(*ST.getInstrInfo()), MRI(MRI) {}
  bool run(MachineFunction &MF);
};

class AMDGPUPrepareAGPRAllocLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPrepareAGPRAllocLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "AMDGPU Prepare AGPR Alloc"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS(AMDGPUPrepareAGPRAllocLegacy, DEBUG_TYPE,
                "AMDGPU Prepare AGPR Alloc", false, false)

char AMDGPUPrepareAGPRAllocLegacy::ID = 0;

char &llvm::AMDGPUPrepareAGPRAllocLegacyID = AMDGPUPrepareAGPRAllocLegacy::ID;

static bool recordPinHints(MachineFunction &MF);

bool AMDGPUPrepareAGPRAllocLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  return AMDGPUPrepareAGPRAllocImpl(ST, MF.getRegInfo()).run(MF);
}

PreservedAnalyses
AMDGPUPrepareAGPRAllocPass::run(MachineFunction &MF,
                                MachineFunctionAnalysisManager &MFAM) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  AMDGPUPrepareAGPRAllocImpl(ST, MF.getRegInfo()).run(MF);
  return PreservedAnalyses::all();
}

bool AMDGPUPrepareAGPRAllocImpl::isAV64Imm(const MachineOperand &MO) const {
  return MO.isImm() && TII.isLegalAV64PseudoImm(MO.getImm());
}

bool AMDGPUPrepareAGPRAllocImpl::run(MachineFunction &MF) {
  // Before the bail-out below: no-AGPR targets are where VGPR pinning matters.
  bool Changed = recordPinHints(MF);

  if (MRI.isReserved(AMDGPU::AGPR0))
    return Changed;

  const MCInstrDesc &AVImmPseudo32 = TII.get(AMDGPU::AV_MOV_B32_IMM_PSEUDO);
  const MCInstrDesc &AVImmPseudo64 = TII.get(AMDGPU::AV_MOV_B64_IMM_PSEUDO);
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if ((MI.getOpcode() == AMDGPU::V_MOV_B32_e32 &&
           TII.isInlineConstant(MI, 1)) ||
          (MI.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32_e64 &&
           MI.getOperand(1).isImm())) {
        MI.setDesc(AVImmPseudo32);
        Changed = true;
        continue;
      }

      // TODO: If only half of the value is rewritable, is it worth splitting it
      // up?
      if ((MI.getOpcode() == AMDGPU::V_MOV_B64_e64 ||
           MI.getOpcode() == AMDGPU::V_MOV_B64_PSEUDO) &&
          isAV64Imm(MI.getOperand(1))) {
        MI.setDesc(AVImmPseudo64);
        Changed = true;
        continue;
      }
    }
  }

  return Changed;
}

//===----------------------------------------------------------------------===//
// Metadata-driven register pinning: carrier -> allocation hint.
//
// Must run pre-RA, in SSA form: coalescing merges the pinned value into wider
// live ranges later, and a hint recorded now rides along with it.
//===----------------------------------------------------------------------===//

// True for the PIN_{VGPR,AGPR}_B* carrier pseudos.
static bool isPinPseudo(const SIInstrInfo *TII, const MachineInstr &MI) {
  StringRef N = TII->getName(MI.getOpcode());
  return N.starts_with("PIN_VGPR_B") || N.starts_with("PIN_AGPR_B");
}

// The tuple a pin targets, or 0 if RegNo is not a legal member of RC.
static MCRegister pinPhysReg(const SIRegisterInfo *TRI,
                             const TargetRegisterClass *RC, bool WantAGPR,
                             unsigned RegNo) {
  unsigned First = (WantAGPR ? AMDGPU::AGPR0 : AMDGPU::VGPR0) + RegNo;
  MCRegister PR = TRI->getRegSizeInBits(*RC) == 32
                      ? MCRegister(First)
                      : TRI->getMatchingSuperReg(First, AMDGPU::sub0, RC);
  if (PR && RC->contains(PR))
    return PR;
  return MCRegister();
}

// An AGPR request on a value an MFMA produces only means anything if the
// instruction writes its accumulator to the AGPR file. The two encodings differ
// in exactly that, so move to the one that can hold the value where it was
// asked to be. Left alone, the request is honoured with a copy around every
// use, which costs far more than the placement is worth.
static bool useAGPRFormMFMA(MachineInstr &MI, const SIInstrInfo *TII,
                            const SIRegisterInfo *TRI,
                            MachineRegisterInfo &MRI) {
  int AGPROp = AMDGPU::getAGPRFormOp(MI.getOpcode());
  if (AGPROp == -1)
    return false;

  // The accumulator is read and written by the same instruction, so both ends
  // have to move together.
  auto toAGPR = [&](MachineOperand &MO) {
    if (!MO.isReg() || !MO.getReg().isVirtual())
      return;
    const TargetRegisterClass *RC = MRI.getRegClass(MO.getReg());
    if (TRI->isAGPRClass(RC))
      return;
    if (const TargetRegisterClass *ARC = TRI->getEquivalentAGPRClass(RC))
      MRI.setRegClass(MO.getReg(), ARC);
  };

  toAGPR(MI.getOperand(0));
  if (int Src2 =
          AMDGPU::getNamedOperandIdx(MI.getOpcode(), AMDGPU::OpName::src2);
      Src2 != -1)
    toAGPR(MI.getOperand(Src2));

  MI.setDesc(TII->get(AGPROp));
  return true;
}

static bool recordPinHints(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  const SIRegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  SmallVector<MachineInstr *, 8> Pins;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (isPinPseudo(TII, MI))
        Pins.push_back(&MI);

  for (MachineInstr *Pin : Pins) {
    Register Dst = Pin->getOperand(0).getReg();
    Register Src = Pin->getOperand(1).getReg();
    unsigned RegNo = Pin->getOperand(2).getImm();
    const TargetRegisterClass *RC = MRI.getRegClass(Dst);

    // Exact placement, not just the right 256-VGPR group: the win is fewer full
    // VALU drains (s_wait_alu depctr_va_vdst(0)) at the WMMA window.
    // A value asked for an AGPR is only worth placing there if whatever
    // produces it can write the AGPR file directly.
    if (TRI->isAGPRClass(RC) && Src.isVirtual())
      if (MachineInstr *Def = MRI.getVRegDef(Src))
        useAGPRFormMFMA(*Def, TII, TRI, MRI);

    if (MCRegister PR = pinPhysReg(TRI, RC, TRI->isAGPRClass(RC), RegNo)) {
      MRI.setRegAllocationHint(Dst, AMDGPURI::PinnedReg, PR);
      if (Src.isVirtual())
        MRI.setRegAllocationHint(Src, AMDGPURI::PinnedReg, PR);
    }

    // The carrier is an identity; forward it and drop it.
    BuildMI(*Pin->getParent(), Pin, Pin->getDebugLoc(),
            TII->get(TargetOpcode::COPY), Dst)
        .addReg(Src, RegState::NoFlags, Pin->getOperand(1).getSubReg());
    Pin->eraseFromParent();
  }
  return !Pins.empty();
}

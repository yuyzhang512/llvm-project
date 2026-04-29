--- |
  ; ModuleID = '/home/jeff/source/llvm_emu/llvm-project/llvm/test/CodeGen/AMDGPU/coexec-window-mask.mir'
  source_filename = "/home/jeff/source/llvm_emu/llvm-project/llvm/test/CodeGen/AMDGPU/coexec-window-mask.mir"
  target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
  target triple = "amdgcn"
  
  define void @test_memcoexec_slot_accepts_ds() #0 {
    ret void
  }
  
  define void @test_memcoexec_slot_accepts_salu() #0 {
    ret void
  }
  
  define void @test_memcoexec_slot_accepts_vmem() #0 {
    ret void
  }
  
  attributes #0 = { "amdgpu-waves-per-eu"="1,1" "target-cpu"="gfx1250" }
...
---
name:            test_memcoexec_slot_accepts_ds
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vreg_512_align2, preferred-register: '', flags: [  ] }
  - { id: 1, class: vreg_512_align2, preferred-register: '', flags: [  ] }
  - { id: 2, class: vreg_256_align2, preferred-register: '', flags: [  ] }
  - { id: 3, class: vgpr_32_lo256, preferred-register: '', flags: [  ] }
  - { id: 4, class: vgpr_32_lo256, preferred-register: '', flags: [  ] }
  - { id: 5, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 6, class: vreg_64_align2, preferred-register: '', flags: [  ] }
  - { id: 7, class: vreg_256_align2, preferred-register: '', flags: [  ] }
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    %0:vreg_512_align2 = IMPLICIT_DEF
    %1:vreg_512_align2 = IMPLICIT_DEF
    %4:vgpr_32_lo256 = IMPLICIT_DEF
    %3:vgpr_32_lo256 = IMPLICIT_DEF
    %2:vreg_256_align2 = IMPLICIT_DEF
    %5:vgpr_32 = IMPLICIT_DEF
    %6:vreg_64_align2 = DS_READ_B64_gfx9 %5, 0, 0, implicit $exec
    early-clobber %7:vreg_256_align2 = V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_w32_threeaddr %0, %1, 0, %2, %3, %4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, implicit $exec
    S_ENDPGM 0, implicit %6, implicit %7
...
---
name:            test_memcoexec_slot_accepts_salu
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vreg_512_align2, preferred-register: '', flags: [  ] }
  - { id: 1, class: vreg_512_align2, preferred-register: '', flags: [  ] }
  - { id: 2, class: vreg_256_align2, preferred-register: '', flags: [  ] }
  - { id: 3, class: vgpr_32_lo256, preferred-register: '', flags: [  ] }
  - { id: 4, class: vgpr_32_lo256, preferred-register: '', flags: [  ] }
  - { id: 5, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 6, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 7, class: sgpr_32, preferred-register: '', flags: [  ] }
  - { id: 8, class: vreg_256_align2, preferred-register: '', flags: [  ] }
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    %0:vreg_512_align2 = IMPLICIT_DEF
    %1:vreg_512_align2 = IMPLICIT_DEF
    %2:vreg_256_align2 = IMPLICIT_DEF
    %4:vgpr_32_lo256 = IMPLICIT_DEF
    %3:vgpr_32_lo256 = IMPLICIT_DEF
    %5:sgpr_32 = IMPLICIT_DEF
    %6:sgpr_32 = IMPLICIT_DEF
    %7:sgpr_32 = S_ADD_U32 %5, %6, implicit-def $scc
    early-clobber %8:vreg_256_align2 = V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_w32_threeaddr %0, %1, 0, %2, %3, %4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, implicit $exec
    S_ENDPGM 0, implicit %7, implicit %8
...
---
name:            test_memcoexec_slot_accepts_vmem
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vreg_512_align2, preferred-register: '', flags: [  ] }
  - { id: 1, class: vreg_512_align2, preferred-register: '', flags: [  ] }
  - { id: 2, class: vreg_256_align2, preferred-register: '', flags: [  ] }
  - { id: 3, class: vgpr_32_lo256, preferred-register: '', flags: [  ] }
  - { id: 4, class: vgpr_32_lo256, preferred-register: '', flags: [  ] }
  - { id: 5, class: vreg_64_align2, preferred-register: '', flags: [  ] }
  - { id: 6, class: vreg_64_align2, preferred-register: '', flags: [  ] }
  - { id: 7, class: vreg_256_align2, preferred-register: '', flags: [  ] }
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       16
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  numKernargPreloadSGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    %0:vreg_512_align2 = IMPLICIT_DEF
    %1:vreg_512_align2 = IMPLICIT_DEF
    %4:vgpr_32_lo256 = IMPLICIT_DEF
    %3:vgpr_32_lo256 = IMPLICIT_DEF
    %2:vreg_256_align2 = IMPLICIT_DEF
    %5:vreg_64_align2 = IMPLICIT_DEF
    %6:vreg_64_align2 = GLOBAL_LOAD_DWORDX2 %5, 0, 0, implicit $exec
    early-clobber %7:vreg_256_align2 = V_WMMA_SCALE_F32_16X16X128_F8F6F4_f8_f8_w32_threeaddr %0, %1, 0, %2, %3, %4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, implicit $exec
    S_ENDPGM 0, implicit %6, implicit %7
...

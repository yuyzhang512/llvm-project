; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -verify-machineinstrs < %s | FileCheck %s

; !amdgpu.pin.agpr on a matrix operand or accumulator. An AGPR request is only
; worth honouring where the value can live in that file without a copy at every
; use: ds_read writes an AGPR directly, and MFMA reads src0/src1 from one.
;
; The accumulator additionally needs the MFMA encoding that writes its result to
; the AGPR file, so the request switches the instruction to that form.

@lds = internal addrspace(3) global [256 x <4 x half>] undef, align 16

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)
declare i32 @llvm.amdgcn.workitem.id.x()

; src0 and src1 come straight out of LDS into the requested tuples, and the
; accumulator stays in the requested one across the whole chain.
; CHECK-LABEL: {{^}}pin_operands_and_acc:
; CHECK: ds_read_b64 a[8:9]
; CHECK: ds_read_b64 a[12:13]
; CHECK: v_mfma_f32_16x16x16_f16 a[16:19], a[8:9], a[12:13], 0
; CHECK: v_mfma_f32_16x16x16_f16 a[16:19], a[8:9], a[12:13], a[16:19]
; Nothing has to be shuttled between the two files.
; CHECK-NOT: v_accvgpr
define amdgpu_kernel void @pin_operands_and_acc(ptr addrspace(1) %out) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %pa = getelementptr [256 x <4 x half>], ptr addrspace(3) @lds, i32 0, i32 %tid
  %x = xor i32 %tid, 1
  %pb = getelementptr [256 x <4 x half>], ptr addrspace(3) @lds, i32 0, i32 %x

  %a = load <4 x half>, ptr addrspace(3) %pa, align 8, !amdgpu.pin.agpr !0
  %b = load <4 x half>, ptr addrspace(3) %pb, align 8, !amdgpu.pin.agpr !1

  %c0 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %b, <4 x float> zeroinitializer, i32 0, i32 0, i32 0), !amdgpu.pin.agpr !2
  %c1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %b, <4 x float> %c0, i32 0, i32 0, i32 0), !amdgpu.pin.agpr !2
  store <4 x float> %c1, ptr addrspace(1) %out, align 16
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,64" }

!0 = !{i32 8}
!1 = !{i32 12}
!2 = !{i32 16}

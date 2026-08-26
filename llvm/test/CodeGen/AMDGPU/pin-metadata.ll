; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -verify-machineinstrs < %s | FileCheck %s

; !amdgpu.pin.vgpr !{i32 N} on an instruction asks the allocator to place that
; instruction's result in the VGPR tuple starting at N. It is a hint, so
; allocation still succeeds when N is unavailable.
;
; Reaching v256 and above needs a VGPR budget wide enough to contain it, which
; is why these kernels ask for a small workgroup and one wave per EU.

declare i32 @llvm.amdgcn.workitem.id.x()

; CHECK-LABEL: {{^}}pin_v256:
; CHECK: s_set_vgpr_msb
; CHECK: global_load_b64 {{.*}}/*v[256:257]*/
define amdgpu_kernel void @pin_v256(ptr addrspace(1) %p, ptr addrspace(1) %q) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gp = getelementptr <2 x i32>, ptr addrspace(1) %p, i32 %tid
  %gq = getelementptr <2 x i32>, ptr addrspace(1) %q, i32 %tid
  %v = load <2 x i32>, ptr addrspace(1) %gp, !amdgpu.pin.vgpr !0
  store <2 x i32> %v, ptr addrspace(1) %gq
  ret void
}

; A second group: 512 selects a different 256-VGPR window.
; CHECK-LABEL: {{^}}pin_v512:
; CHECK: global_load_b64 {{.*}}/*v[512:513]*/
define amdgpu_kernel void @pin_v512(ptr addrspace(1) %p, ptr addrspace(1) %q) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gp = getelementptr <2 x i32>, ptr addrspace(1) %p, i32 %tid
  %gq = getelementptr <2 x i32>, ptr addrspace(1) %q, i32 %tid
  %v = load <2 x i32>, ptr addrspace(1) %gp, !amdgpu.pin.vgpr !1
  store <2 x i32> %v, ptr addrspace(1) %gq
  ret void
}

; The carrier is consumed before the emitter; no pseudo may reach the output.
; CHECK-LABEL: {{^}}no_leftover_pseudo:
; CHECK-NOT: PIN_
define amdgpu_kernel void @no_leftover_pseudo(ptr addrspace(1) %p, ptr addrspace(1) %q) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gp = getelementptr <2 x i32>, ptr addrspace(1) %p, i32 %tid
  %gq = getelementptr <2 x i32>, ptr addrspace(1) %q, i32 %tid
  %v = load <2 x i32>, ptr addrspace(1) %gp, !amdgpu.pin.vgpr !0
  store <2 x i32> %v, ptr addrspace(1) %gq
  ret void
}

; A function with no request is untouched.
; CHECK-LABEL: {{^}}no_pin:
; CHECK-NOT: s_set_vgpr_msb
; CHECK: global_load_b64 v[0:1],
define amdgpu_kernel void @no_pin(ptr addrspace(1) %p, ptr addrspace(1) %q) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gp = getelementptr <2 x i32>, ptr addrspace(1) %p, i32 %tid
  %gq = getelementptr <2 x i32>, ptr addrspace(1) %q, i32 %tid
  %v = load <2 x i32>, ptr addrspace(1) %gp
  store <2 x i32> %v, ptr addrspace(1) %gq
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,32" "amdgpu-waves-per-eu"="1,1" }

!0 = !{i32 256}
!1 = !{i32 512}

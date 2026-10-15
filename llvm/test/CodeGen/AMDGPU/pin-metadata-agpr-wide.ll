; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -verify-machineinstrs < %s | FileCheck %s

; An MFMA operand is often wider than one pinned load: two pinned 4-dword
; halves are glued into the 8-dword operand, and that wider value is what stays
; live. The request on each half still decides where the whole operand goes.

@lds = internal addrspace(3) global [512 x <4 x i32>] poison, align 16

declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, i32 immarg, i32, i32 immarg, i32)
declare i32 @llvm.amdgcn.workitem.id.x()

; CHECK-LABEL: {{^}}glued_halves:
; CHECK-DAG: ds_read_b128 a[8:11]
; CHECK-DAG: ds_read_b128 a[12:15]
; CHECK: v_mfma_scale_f32_16x16x128_f8f6f4 {{.*}}, a[8:15],
define amdgpu_kernel void @glued_halves(ptr addrspace(1) %out, i32 %sa, i32 %sb) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p0 = getelementptr [512 x <4 x i32>], ptr addrspace(3) @lds, i32 0, i32 %tid
  %t1 = add i32 %tid, 64
  %p1 = getelementptr [512 x <4 x i32>], ptr addrspace(3) @lds, i32 0, i32 %t1
  %t2 = add i32 %tid, 128
  %p2 = getelementptr [512 x <4 x i32>], ptr addrspace(3) @lds, i32 0, i32 %t2
  %lo = load <4 x i32>, ptr addrspace(3) %p0, align 16, !amdgpu.pin.agpr !0
  %hi = load <4 x i32>, ptr addrspace(3) %p1, align 16, !amdgpu.pin.agpr !1
  %b = load <8 x i32>, ptr addrspace(3) %p2, align 32
  %a = shufflevector <4 x i32> %lo, <4 x i32> %hi, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %c = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %b, <8 x i32> %a, <4 x float> zeroinitializer, i32 0, i32 0, i32 0, i32 %sa, i32 0, i32 %sb)
  store <4 x float> %c, ptr addrspace(1) %out
  ret void
}

; The same operand loaded before a loop and read on every iteration.
; CHECK-LABEL: {{^}}glued_halves_loop:
; CHECK-DAG: ds_read_b128 a[16:19]
; CHECK-DAG: ds_read_b128 a[20:23]
; CHECK: .LBB{{[0-9_]+}}:
; CHECK: v_mfma_scale_f32_16x16x128_f8f6f4 {{.*}}, a[16:23],
define amdgpu_kernel void @glued_halves_loop(ptr addrspace(1) %out, i32 %n, i32 %sa, i32 %sb) #0 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p0 = getelementptr [512 x <4 x i32>], ptr addrspace(3) @lds, i32 0, i32 %tid
  %t1 = add i32 %tid, 64
  %p1 = getelementptr [512 x <4 x i32>], ptr addrspace(3) @lds, i32 0, i32 %t1
  %lo = load <4 x i32>, ptr addrspace(3) %p0, align 16, !amdgpu.pin.agpr !2
  %hi = load <4 x i32>, ptr addrspace(3) %p1, align 16, !amdgpu.pin.agpr !3
  %a = shufflevector <4 x i32> %lo, <4 x i32> %hi, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi <4 x float> [ zeroinitializer, %entry ], [ %c, %loop ]
  %tb = add i32 %tid, %i
  %pb = getelementptr [512 x <4 x i32>], ptr addrspace(3) @lds, i32 0, i32 %tb
  %b = load <8 x i32>, ptr addrspace(3) %pb, align 32
  %c = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %b, <8 x i32> %a, <4 x float> %acc, i32 0, i32 0, i32 0, i32 %sa, i32 0, i32 %sb)
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop

exit:
  store <4 x float> %c, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,64" }

!0 = !{i32 8}
!1 = !{i32 12}
!2 = !{i32 16}
!3 = !{i32 20}

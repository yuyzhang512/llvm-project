; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 -verify-machineinstrs < %s | FileCheck %s

; llvm.{read,write}_register naming a VGPR/AGPR tuple. The named register is the
; value's home: the defining instruction writes it directly, uses read it back,
; and it is removed from the allocatable set so an unrelated value cannot be
; placed there in the gaps between a write and a later read.

declare i64 @llvm.read_register.i64(metadata)
declare void @llvm.write_register.i64(metadata, i64)
declare i32 @llvm.amdgcn.workitem.id.x()

; The load is written straight into the named tuple -- no copy around it.
; CHECK-LABEL: {{^}}named_vgpr:
; CHECK: global_load_dwordx2 v[8:9],
; CHECK-NOT: v_mov_b32{{.*}}v8
; CHECK: v_lshl_add_u64 v[{{[0-9:]+}}], v[8:9],
define amdgpu_kernel void @named_vgpr(ptr addrspace(1) %p, ptr addrspace(1) %q) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gp = getelementptr i64, ptr addrspace(1) %p, i32 %tid
  %gq = getelementptr i64, ptr addrspace(1) %q, i32 %tid
  %v = load i64, ptr addrspace(1) %gp
  call void @llvm.write_register.i64(metadata !0, i64 %v)
  %r = call i64 @llvm.read_register.i64(metadata !0)
  %s = add i64 %r, 1
  store i64 %s, ptr addrspace(1) %gq
  ret void
}

; Unrelated values live across the named range must not land in it. v8/v9 are
; reserved, so every other value is placed elsewhere.
; CHECK-LABEL: {{^}}named_vgpr_no_clash:
; CHECK-NOT: global_load_dwordx2 v[8:9], {{.*}} sc0 sc1
; CHECK: v_lshl_add_u64 v[{{[0-9:]+}}], v[8:9],
define amdgpu_kernel void @named_vgpr_no_clash(ptr addrspace(1) %p, ptr addrspace(1) %q) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gp = getelementptr i64, ptr addrspace(1) %p, i32 %tid
  %gq = getelementptr i64, ptr addrspace(1) %q, i32 %tid
  %v = load i64, ptr addrspace(1) %gp
  call void @llvm.write_register.i64(metadata !0, i64 %v)
  %a1 = load volatile i64, ptr addrspace(1) %gp
  %a2 = load volatile i64, ptr addrspace(1) %gp
  %a3 = load volatile i64, ptr addrspace(1) %gp
  %a4 = load volatile i64, ptr addrspace(1) %gp
  %r = call i64 @llvm.read_register.i64(metadata !0)
  %t1 = add i64 %a1, %a2
  %t2 = add i64 %a3, %a4
  %t3 = add i64 %t1, %t2
  %s = add i64 %r, %t3
  store i64 %s, ptr addrspace(1) %gq
  ret void
}

; A named AGPR tuple holds an MFMA accumulator across a chain. Naming the
; register selects the form of the instruction that writes its result to the
; AGPR file, so the accumulator stays put with no accvgpr moves around it --
; without the name the VGPR (vgprcd) form is chosen and every round trip costs
; a pair of moves per register.
; CHECK-LABEL: {{^}}named_agpr_acc:
; CHECK: v_mfma_f32_16x16x16_f16 a[16:19],
; CHECK: v_mfma_f32_16x16x16_f16 a[16:19], {{.*}}, a[16:19]
; CHECK-NOT: v_accvgpr
declare <4 x i32> @llvm.read_register.v4i32(metadata)
declare void @llvm.write_register.v4i32(metadata, <4 x i32>)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)
@lds = internal addrspace(3) global [256 x <4 x half>] undef, align 16

define amdgpu_kernel void @named_agpr_acc(ptr addrspace(1) %out) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %pa = getelementptr [256 x <4 x half>], ptr addrspace(3) @lds, i32 0, i32 %tid
  %a = load <4 x half>, ptr addrspace(3) %pa, align 8
  %c0 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %a, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
  %bc0 = bitcast <4 x float> %c0 to <4 x i32>
  call void @llvm.write_register.v4i32(metadata !1, <4 x i32> %bc0)
  %ri0 = call <4 x i32> @llvm.read_register.v4i32(metadata !1)
  %r0 = bitcast <4 x i32> %ri0 to <4 x float>
  %c1 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %a, <4 x half> %a, <4 x float> %r0, i32 0, i32 0, i32 0)
  %bc1 = bitcast <4 x float> %c1 to <4 x i32>
  call void @llvm.write_register.v4i32(metadata !1, <4 x i32> %bc1)
  %ri1 = call <4 x i32> @llvm.read_register.v4i32(metadata !1)
  %r1 = bitcast <4 x i32> %ri1 to <4 x float>
  store <4 x float> %r1, ptr addrspace(1) %out, align 16
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,64" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,64" }

!0 = !{!"v[8:9]"}
!1 = !{!"a[16:19]"}

; Miscompile: a loop-carried value held in a named register loses its
; accumulation when the register is initialised with a constant.
;
; SIFoldOperands folds the pre-loop constant into the MFMA's srcC, ignoring
; that write_register redefines the register on every iteration.
;
; RUN: llc -mtriple=amdgpu9.42 < %s | FileCheck %s
; CHECK: v_mfma_f32_16x16x16_f16 a[16:19], v[{{[0-9:]+}}], v[{{[0-9:]+}}], a[16:19]

declare <4 x i32> @llvm.read_register.v4i32(metadata)
declare void @llvm.write_register.v4i32(metadata, <4 x i32>)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, i32 immarg, i32 immarg, i32 immarg)
@lds = internal addrspace(3) global [256 x <4 x half>] undef, align 16

define amdgpu_kernel void @acc_loop(ptr addrspace(1) %out, i32 %n) #0 {
entry:
  %x = load <4 x half>, ptr addrspace(3) @lds, align 8
  call void @llvm.write_register.v4i32(metadata !0, <4 x i32> zeroinitializer)
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %i2, %body ]
  %cur = call <4 x i32> @llvm.read_register.v4i32(metadata !0)
  %acc = bitcast <4 x i32> %cur to <4 x float>
  %nxt = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %x, <4 x half> %x, <4 x float> %acc, i32 0, i32 0, i32 0)
  %bc = bitcast <4 x float> %nxt to <4 x i32>
  call void @llvm.write_register.v4i32(metadata !0, <4 x i32> %bc)
  %i2 = add i32 %i, 1
  %c = icmp slt i32 %i2, %n
  br i1 %c, label %body, label %done
done:
  %r = call <4 x i32> @llvm.read_register.v4i32(metadata !0)
  store <4 x i32> %r, ptr addrspace(1) %out, align 16
  ret void
}
attributes #0 = { "amdgpu-flat-work-group-size"="1,64" }
!0 = !{!"a[16:19]"}

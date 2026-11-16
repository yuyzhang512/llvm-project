; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx950 -stop-after=amdgpu-lower-pin-to-named-reg < %s | FileCheck %s

; A pinned value %p joins PHI %x, whose other incoming value %new is then written
; to the tuple at the end of %then. %p also flows on that edge into PHI %y, so the
; tuple has to be read for %y before %new overwrites it (a ring of values rotating
; through a merge, as after a pipelined loop's remainder step).

; CHECK-LABEL: @rotate(
; CHECK: then:
; CHECK: %[[OLD:.*]] = call <4 x i32> @llvm.read_register.v4i32(metadata ![[A0:[0-9]+]])
; CHECK: call void @llvm.write_register.v4i32(metadata ![[A0]], <4 x i32> %new)
; CHECK-NEXT: br label %join
; CHECK: join:
; CHECK: %y = phi <4 x i32> [ %b, %entry ], [ %[[OLD]], %then ]
; CHECK: ![[A0]] = !{!"a[0:3]"}
define amdgpu_kernel void @rotate(ptr addrspace(1) %in, ptr addrspace(1) %out, <4 x i32> %a, <4 x i32> %b, i1 %c) {
entry:
  %p = call <4 x i32> @llvm.amdgcn.pin.agpr.v4i32(<4 x i32> %a, i32 0)
  br i1 %c, label %then, label %join

then:
  %new = load <4 x i32>, ptr addrspace(1) %in, align 16
  br label %join

join:
  %x = phi <4 x i32> [ %p, %entry ], [ %new, %then ]
  %y = phi <4 x i32> [ %b, %entry ], [ %p, %then ]
  store <4 x i32> %x, ptr addrspace(1) %out, align 16
  %out1 = getelementptr <4 x i32>, ptr addrspace(1) %out, i32 1
  store <4 x i32> %y, ptr addrspace(1) %out1, align 16
  ret void
}

declare <4 x i32> @llvm.amdgcn.pin.agpr.v4i32(<4 x i32>, i32)

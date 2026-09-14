; RUN: not llc -mtriple=amdgpu9.42 < %s 2>&1 | FileCheck %s

; The type decides how many registers the value occupies, so a name that spells
; a different number of them is rejected rather than quietly placed elsewhere.

; CHECK: invalid register "v255"
; CHECK: invalid register "v[8:9]"

declare i64 @llvm.read_register.i64(metadata)
declare void @llvm.write_register.i64(metadata, i64)
declare <4 x i32> @llvm.read_register.v4i32(metadata)
declare void @llvm.write_register.v4i32(metadata, <4 x i32>)

; One register named, two needed.
define amdgpu_kernel void @too_narrow(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %v = load i64, ptr addrspace(1) %p
  call void @llvm.write_register.i64(metadata !0, i64 %v)
  %r = call i64 @llvm.read_register.i64(metadata !0)
  store i64 %r, ptr addrspace(1) %q
  ret void
}

; Two registers named, four needed.
define amdgpu_kernel void @too_wide(ptr addrspace(1) %p, ptr addrspace(1) %q) {
  %v = load <4 x i32>, ptr addrspace(1) %p
  call void @llvm.write_register.v4i32(metadata !1, <4 x i32> %v)
  %r = call <4 x i32> @llvm.read_register.v4i32(metadata !1)
  store <4 x i32> %r, ptr addrspace(1) %q
  ret void
}

!0 = !{!"v255"}
!1 = !{!"v[8:9]"}

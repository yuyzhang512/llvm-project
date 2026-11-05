; RUN: not llc -mtriple=amdgpu9.42-amd-amdhsa -filetype=null < %s 2>&1 | FileCheck %s

; CHECK: error: {{.*}}invalid register "v[8:10]" for llvm.pin.register
define amdgpu_kernel void @width(ptr addrspace(1) %p) {
  %x = load <4 x i32>, ptr addrspace(1) %p
  %z = call <4 x i32> @llvm.pin.register.v4i32(<4 x i32> %x, metadata !0, i1 false)
  store <4 x i32> %z, ptr addrspace(1) %p
  ret void
}

; CHECK: error: {{.*}}invalid register "v[8:10]" for llvm.write_register
define amdgpu_kernel void @width_required(ptr addrspace(1) %p) {
  %x = load <4 x i32>, ptr addrspace(1) %p
  %z = call <4 x i32> @llvm.pin.register.v4i32(<4 x i32> %x, metadata !0, i1 true)
  store <4 x i32> %z, ptr addrspace(1) %p
  ret void
}

!0 = !{!"v[8:10]"}

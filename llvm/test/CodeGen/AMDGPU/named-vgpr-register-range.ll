; RUN: not llc -mtriple=amdgpu9.00 < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=amdgpu9.42 < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=amdgpu11.00 < %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: llc -mtriple=amdgpu12.50 -verify-machineinstrs < %s | FileCheck -check-prefix=OK %s

; A register named by llvm.{read,write}_register has to be one the subtarget can
; address. v256 is a valid register number but only gfx1250 can address it;
; taken on a smaller file it would alias v0 and clobber it.

; ERR: invalid register "v[256:257]"
; OK: v[{{[0-9:]+}}] /*v[256:257]*/

declare i64 @llvm.read_register.i64(metadata)
declare void @llvm.write_register.i64(metadata, i64)

define amdgpu_kernel void @high_vgpr(ptr addrspace(1) %p, ptr addrspace(1) %q) #0 {
  %v = load i64, ptr addrspace(1) %p
  call void @llvm.write_register.i64(metadata !0, i64 %v)
  %r = call i64 @llvm.read_register.i64(metadata !0)
  store i64 %r, ptr addrspace(1) %q
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,64" }

!0 = !{!"v[256:257]"}

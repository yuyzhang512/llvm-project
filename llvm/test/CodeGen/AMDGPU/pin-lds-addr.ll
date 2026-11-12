; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -stop-after=amdgpu-lower-pin-to-named-reg < %s | FileCheck --check-prefix=IR %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 < %s | FileCheck --check-prefix=ISA %s

; llvm.amdgcn.pin.lds.addr asks for the LDS loads producing a value to take
; their address from a range of VGPRs. Loads whose addresses differ only by a
; constant share one base register.

@lds = external addrspace(3) global [0 x i8], align 16

; One variable term: @lds + root + constant, as in a GEMM operand load.

; IR-LABEL: @one_root(
; IR: %[[RAW:.*]] = getelementptr i8, ptr addrspace(3) @lds, i32 %root
; IR: %lds.base = ptrtoint ptr addrspace(3) %[[RAW]] to i32
; IR: call void @llvm.write_register.i32(metadata ![[V40:[0-9]+]], i32 %lds.base)
; IR: %[[B:.*]] = call i32 @llvm.read_register.i32(metadata ![[V40]])
; IR: %[[ADDR:.*]] = add i32 %[[B]], 64
; IR: inttoptr i32 %[[ADDR]] to ptr addrspace(3)
; ISA-LABEL: one_root:
; ISA-DAG: ds_load_b128 v[{{[0-9]+:[0-9]+}}], v40{{$}}
; ISA-DAG: ds_load_b128 v[{{[0-9]+:[0-9]+}}], v40 offset:64
define amdgpu_kernel void @one_root(i32 %root, ptr addrspace(1) %out) {
  %p0 = getelementptr i8, ptr addrspace(3) @lds, i32 %root
  %p1 = getelementptr i8, ptr addrspace(3) %p0, i32 64
  %a = load <4 x i32>, ptr addrspace(3) %p0, align 16
  %b = load <4 x i32>, ptr addrspace(3) %p1, align 16
  %v = shufflevector <4 x i32> %a, <4 x i32> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %pv = call <8 x i32> @llvm.amdgcn.pin.lds.addr.v8i32(<8 x i32> %v, i32 40, i32 2)
  store <8 x i32> %pv, ptr addrspace(1) %out
  ret void
}

; A double buffer read in a loop, through transposing loads: the slot term
; changes every iteration, the lane terms don't, and a [2 x i8] GEP scales its
; index by 2. The base is recomputed in the loop into the pinned register.

; IR-LABEL: @double_buffer(
; IR: loop:
; IR: %[[SLOT:.*]] = and i32
; IR: %[[S2:.*]] = shl i32 %[[SLOT]], 1
; IR: %[[SUM:.*]] = add i32 %lane, %[[S2]]
; IR: %[[RAW:.*]] = getelementptr i8, ptr addrspace(3) @lds, i32 %[[SUM]]
; IR: %lds.base = ptrtoint ptr addrspace(3) %[[RAW]] to i32
; IR: call void @llvm.write_register.i32(metadata ![[V300:[0-9]+]], i32 %lds.base)
; IR: %[[B:.*]] = call i32 @llvm.read_register.i32(metadata ![[V300]])
; IR: %[[A0:.*]] = add i32 %[[B]], 0
; IR: %[[P0:.*]] = inttoptr i32 %[[A0]] to ptr addrspace(3)
; IR: call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %[[P0]])
; IR: %[[A1:.*]] = add i32 %[[B]], 96
; IR: %[[P1:.*]] = inttoptr i32 %[[A1]] to ptr addrspace(3)
; IR: call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %[[P1]])
; ISA-LABEL: double_buffer:
; ISA-DAG: ds_load_tr16_b128 v[{{[0-9]+:[0-9]+}}], v44 /*v300*/{{$}}
; ISA-DAG: ds_load_tr16_b128 v[{{[0-9]+:[0-9]+}}], v44 /*v300*/ offset:96
define amdgpu_kernel void @double_buffer(i32 %lane, i32 %n, ptr addrspace(1) %out) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi <16 x bfloat> [ zeroinitializer, %entry ], [ %sum, %loop ]
  %shl = shl i32 %i, 14
  %slot = and i32 %shl, 16384
  %p.slot = getelementptr [2 x i8], ptr addrspace(3) @lds, i32 %slot
  %p.lane = getelementptr i8, ptr addrspace(3) %p.slot, i32 %lane
  %p1 = getelementptr i8, ptr addrspace(3) %p.lane, i32 96
  %a = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %p.lane)
  %b = call <8 x bfloat> @llvm.amdgcn.ds.load.tr16.b128.v8bf16(ptr addrspace(3) %p1)
  %v = shufflevector <8 x bfloat> %a, <8 x bfloat> %b, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %pv = call <16 x bfloat> @llvm.amdgcn.pin.lds.addr.v16bf16(<16 x bfloat> %v, i32 300, i32 4)
  %sum = fadd <16 x bfloat> %acc, %pv
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, %n
  br i1 %done, label %exit, label %loop
exit:
  store <16 x bfloat> %sum, ptr addrspace(1) %out
  ret void
}


; Offsets more than 64 KB apart need two base registers; a [3 x i8] GEP scales
; its index by 3.

; IR-LABEL: @far_and_scaled(
; IR: %[[M:.*]] = mul i32 %row, 3
; IR: %[[S:.*]] = add i32 %lane, %[[M]]
; IR: %[[S0:.*]] = add i32 %[[S]], 16
; IR: getelementptr i8, ptr addrspace(3) @lds, i32 %[[S0]]
; IR: %[[S1:.*]] = add i32 %{{.*}}, 70000
; IR: getelementptr i8, ptr addrspace(3) @lds, i32 %[[S1]]
; ISA-LABEL: far_and_scaled:
; ISA-DAG: ds_load_b128 v[{{[0-9]+:[0-9]+}}], v40{{$}}
; ISA-DAG: ds_load_b128 v[{{[0-9]+:[0-9]+}}], v41{{$}}
define amdgpu_kernel void @far_and_scaled(i32 %row, i32 %lane, ptr addrspace(1) %out) {
  %p.row = getelementptr [3 x i8], ptr addrspace(3) @lds, i32 %row
  %p = getelementptr i8, ptr addrspace(3) %p.row, i32 %lane
  %p0 = getelementptr i8, ptr addrspace(3) %p, i32 16
  %p1 = getelementptr i8, ptr addrspace(3) %p, i32 70000
  %a = load <4 x i32>, ptr addrspace(3) %p0, align 16
  %b = load <4 x i32>, ptr addrspace(3) %p1, align 16
  %v = shufflevector <4 x i32> %a, <4 x i32> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %pv = call <8 x i32> @llvm.amdgcn.pin.lds.addr.v8i32(<8 x i32> %v, i32 40, i32 2)
  store <8 x i32> %pv, ptr addrspace(1) %out
  ret void
}

; Addresses with no variable term left (a PHI of equal constants) and loads in
; unreachable blocks keep their addresses.

; IR-LABEL: @const_and_unreachable(
; IR-NOT: lds.base
; IR-LABEL: @same_terms(
define amdgpu_kernel void @const_and_unreachable(i1 %c, i32 %x, ptr addrspace(1) %out) {
entry:
  br i1 %c, label %a, label %b
a:
  br label %join
b:
  br label %join
join:
  %k = phi i32 [ 32, %a ], [ 32, %b ]
  %p = getelementptr i8, ptr addrspace(3) @lds, i32 %k
  %v = load <4 x i32>, ptr addrspace(3) %p, align 16
  %pv = call <4 x i32> @llvm.amdgcn.pin.lds.addr.v4i32(<4 x i32> %v, i32 40, i32 1)
  store <4 x i32> %pv, ptr addrspace(1) %out
  ret void
dead:
  %pd = getelementptr i8, ptr addrspace(3) @lds, i32 %x
  %y = load <4 x i32>, ptr addrspace(3) %pd, align 16
  %py = call <4 x i32> @llvm.amdgcn.pin.lds.addr.v4i32(<4 x i32> %y, i32 41, i32 1)
  store <4 x i32> %py, ptr addrspace(1) %out
  ret void
}

; The same terms in a different order share one base; a negative constant
; becomes a negative distance from the cluster start.

; IR: %lds.base = ptrtoint
; IR-NOT: %lds.base{{[0-9]+}} =
; IR: ret void
; ISA-LABEL: same_terms:
; ISA-DAG: ds_load_b128 v[{{[0-9]+:[0-9]+}}], v40{{$}}
; ISA-DAG: ds_load_b128 v[{{[0-9]+:[0-9]+}}], v40 offset:48{{$}}
define amdgpu_kernel void @same_terms(i32 %r, i32 %s, ptr addrspace(1) %out) {
  %p0.r = getelementptr i8, ptr addrspace(3) @lds, i32 %r
  %p0 = getelementptr i8, ptr addrspace(3) %p0.r, i32 %s
  %p0.c = getelementptr i8, ptr addrspace(3) %p0, i32 -16
  %p1.s = getelementptr i8, ptr addrspace(3) @lds, i32 %s
  %p1 = getelementptr i8, ptr addrspace(3) %p1.s, i32 %r
  %p1.c = getelementptr i8, ptr addrspace(3) %p1, i32 32
  %a = load <4 x i32>, ptr addrspace(3) %p0.c, align 16
  %b = load <4 x i32>, ptr addrspace(3) %p1.c, align 16
  %v = shufflevector <4 x i32> %a, <4 x i32> %b, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %pv = call <8 x i32> @llvm.amdgcn.pin.lds.addr.v8i32(<8 x i32> %v, i32 40, i32 1)
  store <8 x i32> %pv, ptr addrspace(1) %out
  ret void
}
; IR-DAG: ![[V40]] = !{!"v40"}
; IR-DAG: ![[V300]] = !{!"v300"}

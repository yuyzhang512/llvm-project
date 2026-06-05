; RUN: opt -mtriple=amdgcn -mcpu=gfx1250 -passes='amdgpu-codegenprepare' -S < %s | FileCheck %s

declare void @llvm.amdgcn.tensor.load.to.lds(<4 x i32>, <8 x i32>, <4 x i32>, <4 x i32>, <8 x i32>, i32)

; Four loads whose group0 (vaddr0) is freshly built per load, sharing
; lanes 0 / 2 / 3 (initialized from the same constant root + same lane-2/3
; insertelements) and differing only in lane 1. CodeGenPrepare should rewrite
; the 2nd/3rd/4th descriptors to chain via @llvm.amdgcn.tensor.desc.update.lane.

; CHECK-LABEL: @chain_lane1_four_loads(
; CHECK:       call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %d0_3,
; CHECK:       %[[U1:.*]] = call <4 x i32> @llvm.amdgcn.tensor.desc.update.lane.v4i32(<4 x i32> %d0_3, i32 %lds_1, i32 1)
; CHECK-NEXT:  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %[[U1]],
; CHECK:       %[[U2:.*]] = call <4 x i32> @llvm.amdgcn.tensor.desc.update.lane.v4i32(<4 x i32> %[[U1]], i32 %lds_2, i32 1)
; CHECK-NEXT:  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %[[U2]],
; CHECK:       %[[U3:.*]] = call <4 x i32> @llvm.amdgcn.tensor.desc.update.lane.v4i32(<4 x i32> %[[U2]], i32 %lds_3, i32 1)
; CHECK-NEXT:  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %[[U3]],
define void @chain_lane1_four_loads(i32 %lds_0, i32 %lds_1, i32 %lds_2, i32 %lds_3,
                                    i32 %addr_lo, i32 %addr_hi,
                                    <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3) {
  %d0 = insertelement <4 x i32> <i32 -1073741823, i32 poison, i32 poison, i32 poison>, i32 %lds_0, i64 1
  %d0_2 = insertelement <4 x i32> %d0, i32 %addr_lo, i64 2
  %d0_3 = insertelement <4 x i32> %d0_2, i32 %addr_hi, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %d0_3, <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3, <8 x i32> zeroinitializer, i32 0)

  %d1 = insertelement <4 x i32> <i32 -1073741823, i32 poison, i32 poison, i32 poison>, i32 %lds_1, i64 1
  %d1_2 = insertelement <4 x i32> %d1, i32 %addr_lo, i64 2
  %d1_3 = insertelement <4 x i32> %d1_2, i32 %addr_hi, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %d1_3, <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3, <8 x i32> zeroinitializer, i32 0)

  %d2 = insertelement <4 x i32> <i32 -1073741823, i32 poison, i32 poison, i32 poison>, i32 %lds_2, i64 1
  %d2_2 = insertelement <4 x i32> %d2, i32 %addr_lo, i64 2
  %d2_3 = insertelement <4 x i32> %d2_2, i32 %addr_hi, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %d2_3, <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3, <8 x i32> zeroinitializer, i32 0)

  %d3 = insertelement <4 x i32> <i32 -1073741823, i32 poison, i32 poison, i32 poison>, i32 %lds_3, i64 1
  %d3_2 = insertelement <4 x i32> %d3, i32 %addr_lo, i64 2
  %d3_3 = insertelement <4 x i32> %d3_2, i32 %addr_hi, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %d3_3, <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3, <8 x i32> zeroinitializer, i32 0)

  ret void
}

; Negative case: descriptors differ in two lanes AND we're not inside a loop,
; so the carrier-phi can't be set up. The pass must not chain.
; CHECK-LABEL: @no_chain_no_loop_two_lanes(
; CHECK-NOT:   call <4 x i32> @llvm.amdgcn.tensor.desc.update.lane
; CHECK:       ret void
define void @no_chain_no_loop_two_lanes(i32 %lds_0, i32 %lds_1, i32 %a0, i32 %a1,
                                        <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3) {
  %d0 = insertelement <4 x i32> <i32 0, i32 poison, i32 poison, i32 0>, i32 %lds_0, i64 1
  %d0_2 = insertelement <4 x i32> %d0, i32 %a0, i64 2
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %d0_2, <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3, <8 x i32> zeroinitializer, i32 0)

  %d1 = insertelement <4 x i32> <i32 0, i32 poison, i32 poison, i32 0>, i32 %lds_1, i64 1
  %d1_2 = insertelement <4 x i32> %d1, i32 %a1, i64 2
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %d1_2, <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3, <8 x i32> zeroinitializer, i32 0)
  ret void
}

; Multi-lane chain inside a loop: lanes 2 and 3 of group0 differ between the
; two loads. Pass should set up a carrier-phi at the loop header so the
; back-edge value of the IE chain's root is the chain result (non-trivial phi),
; preventing the IR optimizer from eliminating the rebuild IEs that would
; otherwise look redundant. Without the carrier-phi the post-RA coalescer
; couldn't tie the multi-lane chain in place.
; CHECK-LABEL: @chain_two_lanes_in_loop(
; CHECK:       loop:
; CHECK:         %{{[a-z0-9_.]*carrier[a-z0-9_.]*}} = phi <4 x i32> [
; CHECK:         %[[U1:.*]] = call <4 x i32> @llvm.amdgcn.tensor.desc.update.lane.v4i32(<4 x i32> %{{.*}}, i32 %v0_b, i32 2)
; CHECK-NEXT:    %[[U2:.*]] = call <4 x i32> @llvm.amdgcn.tensor.desc.update.lane.v4i32(<4 x i32> %[[U1]], i32 %v1_b, i32 3)
; CHECK-NEXT:    call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %[[U2]],
define void @chain_two_lanes_in_loop(i32 %lds_a, i32 %v0_a, i32 %v1_a,
                                     i32 %lds_b, i32 %v0_b, i32 %v1_b,
                                     <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3,
                                     i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %a0 = insertelement <4 x i32> <i32 -1073741823, i32 poison, i32 poison, i32 poison>, i32 %lds_a, i64 1
  %a1 = insertelement <4 x i32> %a0, i32 %v0_a, i64 2
  %a  = insertelement <4 x i32> %a1, i32 %v1_a, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %a, <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3, <8 x i32> zeroinitializer, i32 0)
  %b0 = insertelement <4 x i32> <i32 -1073741823, i32 poison, i32 poison, i32 poison>, i32 %lds_a, i64 1
  %b1 = insertelement <4 x i32> %b0, i32 %v0_b, i64 2
  %b  = insertelement <4 x i32> %b1, i32 %v1_b, i64 3
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %b, <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3, <8 x i32> zeroinitializer, i32 0)
  %i.next = add i32 %i, 1
  %cmp = icmp ne i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}

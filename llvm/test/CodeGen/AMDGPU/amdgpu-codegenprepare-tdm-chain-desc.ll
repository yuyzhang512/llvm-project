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

; Negative case: descriptors differ in two lanes (lane 1 AND lane 2). The
; pass must not chain these -- they don't fit the single-lane-update model.
; CHECK-LABEL: @no_chain_two_lanes_differ(
; CHECK-NOT:   call <4 x i32> @llvm.amdgcn.tensor.desc.update.lane
; CHECK:       ret void
define void @no_chain_two_lanes_differ(i32 %lds_0, i32 %lds_1, i32 %a0, i32 %a1,
                                       <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3) {
  %d0 = insertelement <4 x i32> <i32 0, i32 poison, i32 poison, i32 0>, i32 %lds_0, i64 1
  %d0_2 = insertelement <4 x i32> %d0, i32 %a0, i64 2
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %d0_2, <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3, <8 x i32> zeroinitializer, i32 0)

  %d1 = insertelement <4 x i32> <i32 0, i32 poison, i32 poison, i32 0>, i32 %lds_1, i64 1
  %d1_2 = insertelement <4 x i32> %d1, i32 %a1, i64 2
  call void @llvm.amdgcn.tensor.load.to.lds(<4 x i32> %d1_2, <8 x i32> %g1, <4 x i32> %g2, <4 x i32> %g3, <8 x i32> zeroinitializer, i32 0)
  ret void
}

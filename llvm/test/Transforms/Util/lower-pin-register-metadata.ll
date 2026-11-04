; RUN: opt -passes=lower-pin-register-metadata -S < %s | FileCheck %s

; CHECK-LABEL: @store(
; CHECK: %[[P:.*]] = call <4 x i32> @llvm.pin.register.v4i32(<4 x i32> %v, metadata ![[V8:[0-9]+]], i1 true)
; CHECK: store <4 x i32> %[[P]], ptr %p, align 16{{$}}
define void @store(ptr %p, <4 x i32> %v) {
  store <4 x i32> %v, ptr %p, align 16, !llvm.pin.register !0
  ret void
}

; The hint form.
; CHECK-LABEL: @hint(
; CHECK: call i32 @llvm.pin.register.i32(i32 %v, metadata ![[V4:[0-9]+]], i1 false)
define void @hint(ptr %p, i32 %v) {
  store i32 %v, ptr %p, align 4, !llvm.pin.register !1
  ret void
}

; Padding added to store a three-element vector is not pinned.
; CHECK-LABEL: @widened(
; CHECK: %[[P:.*]] = call <3 x float> @llvm.pin.register.v3f32(<3 x float> %v, metadata ![[V8_10:[0-9]+]], i1 true)
; CHECK: shufflevector <3 x float> %[[P]], <3 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
define void @widened(ptr %p, <3 x float> %v) {
  %w = shufflevector <3 x float> %v, <3 x float> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  store <4 x float> %w, ptr %p, align 16, !llvm.pin.register !2
  ret void
}

; CHECK-DAG: ![[V8]] = !{!"v[8:11]"}
; CHECK-DAG: ![[V4]] = !{!"v4"}
; CHECK-DAG: ![[V8_10]] = !{!"v[8:10]"}
!0 = !{!"v[8:11]", i1 true}
!1 = !{!"v4", i1 false}
!2 = !{!"v[8:10]", i1 true}

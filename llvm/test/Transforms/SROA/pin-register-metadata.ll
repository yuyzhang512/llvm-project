; RUN: opt -passes=sroa -S < %s | FileCheck %s
; RUN: opt -passes=mem2reg -S < %s | FileCheck %s

; A load hinted with !llvm.pin.register that promotion replaces gives the hint
; to the instruction it is replaced with.

define <4 x float> @copy(<4 x float> %a) {
; CHECK-LABEL: define <4 x float> @copy(
; CHECK:         [[M:%.*]] = fmul <4 x float> [[A:%.*]], splat (float 2.000000e+00), !llvm.pin.register [[V8:![0-9]+]]
; CHECK-NEXT:    ret <4 x float> [[M]]
  %x = alloca <4 x float>
  %m = fmul <4 x float> %a, splat (float 2.0)
  store <4 x float> %m, ptr %x
  %l = load <4 x float>, ptr %x, !llvm.pin.register !0
  ret <4 x float> %l
}

; An instruction with its own hint keeps it.
define <4 x float> @keep(<4 x float> %a) {
; CHECK-LABEL: define <4 x float> @keep(
; CHECK:         fmul <4 x float> [[A:%.*]], splat (float 2.000000e+00), !llvm.pin.register [[V9:![0-9]+]]
  %x = alloca <4 x float>
  %m = fmul <4 x float> %a, splat (float 2.0), !llvm.pin.register !1
  store <4 x float> %m, ptr %x
  %l = load <4 x float>, ptr %x, !llvm.pin.register !0
  ret <4 x float> %l
}

!0 = !{!"v[8:11]"}
!1 = !{!"v[12:15]"}
; CHECK-DAG: [[V8]] = !{!"v[8:11]"}
; CHECK-DAG: [[V9]] = !{!"v[12:15]"}

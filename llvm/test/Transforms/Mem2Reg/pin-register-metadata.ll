; RUN: opt -passes=mem2reg -S < %s | FileCheck %s
; RUN: opt -passes=sroa -S < %s | FileCheck %s

; Promoting an alloca marked !llvm.pin.register !{!"<name>"}, a hint, gives
; each instruction stored to it the same metadata. An argument has nowhere to
; carry it, and an instruction already hinted keeps its own register.

define i32 @f(i32 %a, i32 %b, i1 %c, i1 %d) {
; CHECK-LABEL: define i32 @f(
; CHECK:         [[X:%.*]] = add i32 [[A:%.*]], 1, !llvm.pin.register [[V8:![0-9]+]]
; CHECK:         [[Y:%.*]] = add i32 [[B:%.*]], 1, !llvm.pin.register [[V9:![0-9]+]]
; CHECK:         phi i32
entry:
  %v = alloca i32, align 4, !llvm.pin.register !0
  %x = add i32 %a, 1
  store i32 %x, ptr %v
  br i1 %c, label %then, label %else
then:
  %y = add i32 %b, 1, !llvm.pin.register !1
  store i32 %y, ptr %v
  br label %exit
else:
  br i1 %d, label %arg, label %exit
arg:
  store i32 %b, ptr %v
  br label %exit
exit:
  %r = load i32, ptr %v
  ret i32 %r
}

!0 = !{!"v8"}
!1 = !{!"v9"}
; CHECK-DAG: [[V8]] = !{!"v8"}
; CHECK-DAG: [[V9]] = !{!"v9"}

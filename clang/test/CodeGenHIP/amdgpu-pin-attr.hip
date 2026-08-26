// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -x hip \
// RUN:   -fcuda-is-device -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

// -disable-llvm-passes: AMDGPULowerPinMetadata rewrites the metadata into the
// carrier at the start of the pipeline, so this checks what clang itself emits.

// amdgpu_pin_{vgpr,agpr}(N) tags the instruction the statement performs with
// !amdgpu.pin.{vgpr,agpr} !{i32 N}, naming the register its result should land
// in. The tag goes on the operation rather than on the assigned variable: the
// variable's storage is removed by SROA, while the operation that computes the
// value is still there when the allocator runs.

typedef float float2 __attribute__((ext_vector_type(2)));

// CHECK-LABEL: define{{.*}}pin_vgpr
// CHECK: fadd contract <2 x float>{{.*}}!amdgpu.pin.vgpr ![[VGPR:[0-9]+]]
__attribute__((device)) void pin_vgpr(float2 *out, float2 in) {
  float2 x;
  [[clang::amdgpu_pin_vgpr(4)]] x = in + in;
  *out = x;
}

// The call is the operation, so it carries the tag even though a store follows.
// CHECK-LABEL: define{{.*}}pin_call
// CHECK: call contract <2 x float> @llvm.fma.v2f32{{.*}}!amdgpu.pin.agpr ![[AGPR:[0-9]+]]
__attribute__((device)) void pin_call(float2 *out, float2 in) {
  float2 x;
  [[clang::amdgpu_pin_agpr(8)]] x = __builtin_elementwise_fma(in, in, in);
  *out = x;
}

// An unpinned statement is left alone.
// CHECK-LABEL: define{{.*}}no_pin
// CHECK-NOT: !amdgpu.pin.
__attribute__((device)) void no_pin(float2 *out, float2 in) {
  float2 x;
  x = in + in;
  *out = x;
}

// CHECK-DAG: ![[VGPR]] = !{i32 4}
// CHECK-DAG: ![[AGPR]] = !{i32 8}

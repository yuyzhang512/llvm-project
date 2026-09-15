// RUN: %clang_cc1 -triple amdgpu9.42 -x hip -fcuda-is-device \
// RUN:   -emit-llvm -O1 -o - %s | FileCheck %s

// Two forms beyond amdgpu_pin_{vgpr,agpr}(N) on the result:
//   - no register number, which asks only for the register file;
//   - amdgpu_pin_gpr, which names the operands of the operation separately.

typedef float float4 __attribute__((ext_vector_type(4)));
typedef _Float16 half4 __attribute__((ext_vector_type(4)));

__attribute__((device)) __attribute__((shared)) half4 lds[256];

// No number: the carrier takes -1, which the backend reads as "this file, any
// register".
// CHECK-LABEL: define {{.*}}@_Z9file_onlyPKDv4_fPS_(
// CHECK: call <4 x i32> @llvm.amdgcn.pin.agpr.v4i32(<4 x i32> %{{[0-9]+}}, i32 -1)
__attribute__((global)) void file_only(const float4 *in, float4 *out) {
  float4 acc;
  [[clang::amdgpu_pin_agpr()]] acc = in[0] + in[1];
  out[0] = acc;
}

// Empty parentheses and a bare attribute are the same request.
// CHECK-LABEL: define {{.*}}@_Z4barePKDv4_fPS_(
// CHECK: call <4 x i32> @llvm.amdgcn.pin.vgpr.v4i32(<4 x i32> %{{[0-9]+}}, i32 -1)
__attribute__((global)) void bare(const float4 *in, float4 *out) {
  float4 acc;
  [[clang::amdgpu_pin_vgpr]] acc = in[0] + in[1];
  out[0] = acc;
}

// Operand 0 is the result and 1.. are the sources, so each is carried by
// whatever computes it: the two loads and the multiply itself.
// CHECK-LABEL: define {{.*}}@_Z8operandsPDv4_f(
// CHECK-DAG: call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %{{[0-9]+}}, i32 8)
// CHECK-DAG: call <2 x i32> @llvm.amdgcn.pin.agpr.v2i32(<2 x i32> %{{[0-9]+}}, i32 12)
// CHECK-DAG: call <4 x i32> @llvm.amdgcn.pin.agpr.v4i32(<4 x i32> %{{[0-9]+}}, i32 16)
__attribute__((global)) void operands(float4 *out) {
  int t = __builtin_amdgcn_workitem_id_x();
  half4 x = lds[t], y = lds[t ^ 1];
  float4 acc = {0, 0, 0, 0};
  [[clang::amdgpu_pin_gpr(0, "a[16:19]", 1, "a8", 2, "a12")]]
  acc = __builtin_amdgcn_mfma_f32_16x16x16f16(x, y, acc, 0, 0, 0);
  out[0] = acc;
}

// A VGPR name reaches the other file, and a tuple name gives the first register
// of the tuple.
// CHECK-LABEL: define {{.*}}@_Z5mixedPDv4_f(
// CHECK-DAG: call <2 x i32> @llvm.amdgcn.pin.vgpr.v2i32(<2 x i32> %{{[0-9]+}}, i32 4)
// CHECK-DAG: call <4 x i32> @llvm.amdgcn.pin.vgpr.v4i32(<4 x i32> %{{[0-9]+}}, i32 20)
__attribute__((global)) void mixed(float4 *out) {
  int t = __builtin_amdgcn_workitem_id_x();
  half4 x = lds[t], y = lds[t ^ 1];
  float4 acc = {0, 0, 0, 0};
  [[clang::amdgpu_pin_gpr(0, "v[20:23]", 1, "v4")]]
  acc = __builtin_amdgcn_mfma_f32_16x16x16f16(x, y, acc, 0, 0, 0);
  out[0] = acc;
}

// RUN: %clang_cc1 -triple amdgpu9.42 -x hip -fcuda-is-device -emit-llvm \
// RUN:   -fpin-register-lowering=intrinsic -o - %s \
// RUN:   | FileCheck --check-prefixes=CHECK,REQ %s
// RUN: %clang_cc1 -triple amdgpu9.42 -x hip -fcuda-is-device -emit-llvm \
// RUN:   -fpin-register-lowering=intrinsic -fpin-register-hint \
// RUN:   -o - %s | FileCheck --check-prefixes=CHECK,HINT %s

// With -fpin-register-lowering=intrinsic a pinned local keeps its alloca, and
// every value stored to it goes through llvm.pin.register first.

#define __global__ __attribute__((global))

typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float3 __attribute__((ext_vector_type(3)));

// CHECK-LABEL: define {{.*}}void @_Z3accPDv4_fi(
// CHECK: %acc = alloca <4 x float>
// REQ: %[[P0:.*]] = call {{.*}}<4 x float> @llvm.pin.register.v4f32(<4 x float> zeroinitializer, metadata ![[V20:[0-9]+]], i1 true)
// HINT: %[[P0:.*]] = call {{.*}}<4 x float> @llvm.pin.register.v4f32(<4 x float> zeroinitializer, metadata ![[V20:[0-9]+]], i1 false)
// CHECK: store <4 x float> %[[P0]], ptr %acc.ascast
// CHECK: %[[ADD:.*]] = fadd {{.*}}<4 x float>
// CHECK: %[[P1:.*]] = call {{.*}}<4 x float> @llvm.pin.register.v4f32(<4 x float> %[[ADD]], metadata ![[V20]]
// CHECK: store <4 x float> %[[P1]], ptr %acc.ascast
// An element write stores the whole vector.
// CHECK: %[[INS:.*]] = insertelement <4 x float>
// CHECK: %[[P2:.*]] = call {{.*}}<4 x float> @llvm.pin.register.v4f32(<4 x float> %[[INS]], metadata ![[V20]]
// CHECK: store <4 x float> %[[P2]], ptr %acc.ascast
__global__ void acc(float4 *out, int n) {
  __attribute__((amdgpu_pin_vgpr(20))) float4 acc = {0, 0, 0, 0};
  for (int i = 0; i < n; i++)
    acc += out[i];
  acc.x = 1.0f;
  out[0] = acc;
}

// A three-element vector takes three registers.
// CHECK-LABEL: define {{.*}}void @_Z5threePDv3_f(
// CHECK: %[[P:.*]] = call {{.*}}<3 x float> @llvm.pin.register.v3f32(<3 x float> %{{.*}}, metadata ![[V8:[0-9]+]]
// CHECK: store <3 x float> %[[P]], ptr %v.ascast
__global__ void three(float3 *out) {
  __attribute__((amdgpu_pin_vgpr(8))) float3 v = out[0];
  out[1] = v;
}

// The variable can be captured by reference. The alloca carries the pin, so a
// value stored through the capture is pinned when the alloca is promoted.
// CHECK-LABEL: define {{.*}}void @_Z7capturePDv4_f(
// CHECK: %acc = alloca <4 x float>, align 16, addrspace(5), !llvm.pin.register ![[ACC:[0-9]+]]
__global__ void capture(float4 *out) {
  __attribute__((amdgpu_pin_vgpr(20))) float4 acc = {0, 0, 0, 0};
  auto add = [&](float4 v) __attribute__((always_inline)) { acc += v; };
  add(out[0]);
  out[1] = acc;
}

// CHECK-DAG: ![[V20]] = !{!"v[20:23]"}
// CHECK-DAG: ![[V8]] = !{!"v[8:10]"}
// REQ-DAG: ![[ACC]] = !{!"v[20:23]", i1 true}
// HINT-DAG: ![[ACC]] = !{!"v[20:23]", i1 false}

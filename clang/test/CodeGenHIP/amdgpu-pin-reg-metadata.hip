// RUN: %clang_cc1 -triple amdgpu9.42 -x hip -fcuda-is-device -emit-llvm \
// RUN:   -disable-llvm-passes -o - %s | FileCheck --check-prefix=FE %s
// RUN: %clang_cc1 -triple amdgpu9.42 -x hip -fcuda-is-device -emit-llvm -O3 \
// RUN:   -o - %s | FileCheck --check-prefix=OPT %s

// A local with amdgpu_pin_vgpr or amdgpu_pin_agpr keeps its alloca. Each
// instruction whose value is stored to it carries !llvm.pin.register, and so
// does the alloca, for values stored through a pointer to it.

#define __global__ __attribute__((global))

typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float3 __attribute__((ext_vector_type(3)));

// FE-LABEL: define {{.*}}void @_Z3accPDv4_fi(
// FE: %acc = alloca <4 x float>, align 16, addrspace(5), !llvm.pin.register ![[V20:[0-9]+]]
// FE: store <4 x float> zeroinitializer, ptr %acc.ascast
// FE: %add = fadd {{.*}}<4 x float> %{{.*}}, %{{.*}}, !llvm.pin.register ![[V20]]
// FE: store <4 x float> %add, ptr %acc.ascast
// An element write stores the whole vector.
// FE: %[[INS:.*]] = insertelement <4 x float> %{{.*}}, float 1.000000e+00, i64 0, !llvm.pin.register ![[V20]]
// FE: store <4 x float> %[[INS]], ptr %acc.ascast

// OPT-LABEL: define {{.*}}void @_Z3accPDv4_fi(
// OPT-NOT: alloca
// OPT: fadd {{.*}}<4 x float> {{.*}}, !llvm.pin.register ![[V20:[0-9]+]]
__global__ void acc(float4 *out, int n) {
  __attribute__((amdgpu_pin_vgpr(20))) float4 acc = {0, 0, 0, 0};
  for (int i = 0; i < n; i++)
    acc += out[i];
  acc.x = 1.0f;
  out[0] = acc;
}

// A three-element vector takes three registers.
// FE-LABEL: define {{.*}}void @_Z5threePDv3_f(
// FE: %[[L:.*]] = load <3 x float>, {{.*}}, !llvm.pin.register ![[V8:[0-9]+]]
// FE: store <3 x float> %[[L]], ptr %v.ascast
__global__ void three(float3 *out) {
  __attribute__((amdgpu_pin_vgpr(8))) float3 v = out[0];
  out[1] = v;
}

// A by-reference capture stores through a pointer that is not known to be the
// variable, so the value stored in the lambda gets no hint here. The marked
// alloca gives it one when the alloca is promoted.
// FE-LABEL: define {{.*}}void @_Z7capturePDv4_f(
// FE: %acc = alloca <4 x float>, align 16, addrspace(5), !llvm.pin.register ![[V20]]
__global__ void capture(float4 *out) {
  __attribute__((amdgpu_pin_vgpr(20))) float4 acc = {0, 0, 0, 0};
  auto add = [&](float4 v) { acc += v; };
  add(out[1]);
  out[0] = acc;
}

// FE-DAG: ![[V20]] = !{!"v[20:23]"}
// FE-DAG: ![[V8]] = !{!"v[8:10]"}
// OPT: ![[V20]] = !{!"v[20:23]"}

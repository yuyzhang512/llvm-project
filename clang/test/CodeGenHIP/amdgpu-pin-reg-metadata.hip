// RUN: %clang_cc1 -triple amdgpu9.42 -x hip -fcuda-is-device -emit-llvm \
// RUN:   -disable-llvm-passes -fpin-register-lowering=metadata \
// RUN:   -fpin-register-hint -o - %s | FileCheck --check-prefix=FE %s
// RUN: %clang_cc1 -triple amdgpu9.42 -x hip -fcuda-is-device -emit-llvm -O3 \
// RUN:   -fpin-register-lowering=metadata -fpin-register-hint -o - %s \
// RUN:   | FileCheck --check-prefix=OPT %s
// RUN: %clang_cc1 -triple amdgpu9.42 -x hip -fcuda-is-device -emit-llvm \
// RUN:   -disable-llvm-passes -fpin-register-lowering=metadata -o - %s \
// RUN:   | FileCheck --check-prefix=REQ %s

// With -fpin-register-lowering=metadata -fpin-register-hint a pinned local
// keeps its alloca, and each instruction whose value is stored to it carries
// !llvm.pin.register, which instruction selection reads. A required pin still
// goes through llvm.pin.register.

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
// FE-NOT: @llvm.pin.register

// OPT-LABEL: define {{.*}}void @_Z3accPDv4_fi(
// OPT-NOT: alloca
// OPT: fadd {{.*}}<4 x float> {{.*}}, !llvm.pin.register ![[V20:[0-9]+]]
// OPT-NOT: @llvm.pin.register

// REQ-LABEL: define {{.*}}void @_Z3accPDv4_fi(
// REQ: %acc = alloca <4 x float>, align 16, addrspace(5), !llvm.pin.register ![[ACC:[0-9]+]]
// REQ: call {{.*}}<4 x float> @llvm.pin.register.v4f32(<4 x float> zeroinitializer, metadata ![[V20:[0-9]+]], i1 true)
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

// FE-DAG: ![[V20]] = !{!"v[20:23]"}
// FE-DAG: ![[V8]] = !{!"v[8:10]"}
// OPT: ![[V20]] = !{!"v[20:23]"}
// REQ-DAG: ![[ACC]] = !{!"v[20:23]", i1 true}
// REQ-DAG: ![[V20]] = !{!"v[20:23]"}

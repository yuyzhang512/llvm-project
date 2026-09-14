// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x hip \
// RUN:   -fcuda-is-device -emit-llvm -O1 -o - %s | FileCheck %s

// A local carrying amdgpu_pin_{vgpr,agpr} names the register tuple it lives in.
// It has no stack storage: every read and write becomes
// llvm.{read,write}_register on that name, so there is no alloca to promote.

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

typedef float float4 __attribute__((ext_vector_type(4)));

// A single-register value names one register, and the variable never reaches
// memory.
// CHECK-LABEL: define {{.*}}void @_Z6scalarPKiPi(
// CHECK-NOT: alloca
// CHECK: call void @llvm.write_register.i32(metadata ![[V8:[0-9]+]], i32
// CHECK: call i32 @llvm.read_register.i32(metadata ![[V8]])
__global__ void scalar(const int *in, int *out) {
  __attribute__((amdgpu_pin_vgpr(8))) int acc;
  acc = in[0];
  out[0] = acc + 1;
}

// A four-dword value occupies four consecutive registers, and travels through
// the intrinsic as 32-bit lanes because it is declared only over integers.
// CHECK-LABEL: define {{.*}}void @_Z4widePKDv4_fPS_(
// CHECK-NOT: alloca
// CHECK: call void @llvm.write_register.v4i32(metadata ![[V256:[0-9]+]], <4 x i32>
// CHECK: call <4 x i32> @llvm.read_register.v4i32(metadata ![[V256]])
__global__ void wide(const float4 *in, float4 *out) {
  __attribute__((amdgpu_pin_vgpr(256))) float4 acc;
  acc = in[0];
  out[0] = acc;
}

// An AGPR request names the other file.
// CHECK-LABEL: define {{.*}}void @_Z4agprPKDv4_fPS_(
// CHECK: call void @llvm.write_register.v4i32(metadata ![[A16:[0-9]+]], <4 x i32>
__global__ void agpr(const float4 *in, float4 *out) {
  __attribute__((amdgpu_pin_agpr(16))) float4 acc;
  acc = in[0];
  out[0] = acc;
}

// An initializer is written straight to the register.
// CHECK-LABEL: define {{.*}}void @_Z4initPi(
// CHECK-NOT: alloca
// CHECK: call void @llvm.write_register.i32(metadata ![[V4:[0-9]+]], i32 7)
__global__ void init(int *out) {
  __attribute__((amdgpu_pin_vgpr(4))) int acc = 7;
  out[0] = acc;
}

// CHECK-DAG: ![[V8]] = !{!"v8"}
// CHECK-DAG: ![[V256]] = !{!"v[256:259]"}
// CHECK-DAG: ![[A16]] = !{!"a[16:19]"}
// CHECK-DAG: ![[V4]] = !{!"v4"}

// RUN: %clang_cc1 -triple amdgpu9.42 -x hip \
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

// A dependent register number is substituted and re-checked per instantiation,
// so a template can compute the register from its arguments.
// CHECK-LABEL: define {{.*}}void @_Z6calleriPi(
// CHECK: call void @llvm.write_register.i32(metadata ![[T264:[0-9]+]], i32
// CHECK: call void @llvm.write_register.i32(metadata ![[T520:[0-9]+]], i32
template <int Base> __device__ int pin_at(int i) {
  __attribute__((amdgpu_pin_vgpr(Base + 8))) int acc;
  acc = i;
  return acc + 1;
}
__global__ void caller(int i, int *out) {
  out[0] = pin_at<256>(i);
  out[1] = pin_at<512>(i);
}

// An initializer is written straight to the register.
// CHECK-LABEL: define {{.*}}void @_Z4initPi(
// CHECK-NOT: alloca
// CHECK: call void @llvm.write_register.i32(metadata ![[V4:[0-9]+]], i32 7)
__global__ void init(int *out) {
  __attribute__((amdgpu_pin_vgpr(4))) int acc = 7;
  out[0] = acc;
}

// A const reference binds to a copy read out of the register, the way it would
// bind to a bit-field. The copy is a temporary; the variable gets no storage.
__device__ __attribute__((noinline)) float4 by_const_ref(const float4 &v) { return v * v; }

// CHECK-LABEL: define {{.*}}@_Z15const_ref_to_cpPDv4_f(
// CHECK: %ref.tmp = alloca <4 x float>
// CHECK-NOT: alloca
// CHECK: call void @llvm.write_register.v4i32(metadata ![[V64:[0-9]+]], <4 x i32>
// CHECK: %[[COPY:.*]] = tail call <4 x i32> @llvm.read_register.v4i32(metadata ![[V64]])
// CHECK: store <4 x i32> %[[COPY]], ptr addrspace(5) %ref.tmp
// CHECK: call {{.*}}@_Z12by_const_refRKDv4_f(
__global__ void const_ref_to_cp(float4 *out) {
  __attribute__((amdgpu_pin_vgpr(64))) float4 acc;
  acc = out[0];
  out[1] = by_const_ref(acc);
}

// One pinned local per tile, numbered from a generic lambda's parameter inside
// a function template: each instantiation names its own tuple.
template <int I> struct num { static constexpr int value = I; };
template <class F> __device__ void twice(F f) { f(num<0>{}); f(num<1>{}); }
template <int Base> __device__ void per_tile(float4 *out) {
  twice([&](auto i) {
    __attribute__((amdgpu_pin_vgpr(Base + i.value * 4))) auto t = out[i.value];
    out[2 + i.value] = t;
  });
}

// CHECK-LABEL: define {{.*}}@_Z5tilesPDv4_f(
// CHECK-NOT: alloca
// CHECK: call void @llvm.write_register.v4i32(metadata ![[T0:[0-9]+]], <4 x i32>
// CHECK: call void @llvm.write_register.v4i32(metadata ![[T1:[0-9]+]], <4 x i32>
__global__ void tiles(float4 *out) { per_tile<32>(out); }

// CHECK-DAG: ![[V8]] = !{!"v8"}
// CHECK-DAG: ![[V256]] = !{!"v[256:259]"}
// CHECK-DAG: ![[A16]] = !{!"a[16:19]"}
// CHECK-DAG: ![[V64]] = !{!"v[64:67]"}
// CHECK-DAG: ![[T0]] = !{!"v[32:35]"}
// CHECK-DAG: ![[T1]] = !{!"v[36:39]"}
// CHECK-DAG: ![[V4]] = !{!"v4"}
// CHECK-DAG: ![[T264]] = !{!"v264"}
// CHECK-DAG: ![[T520]] = !{!"v520"}


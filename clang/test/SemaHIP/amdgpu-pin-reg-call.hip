// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x hip \
// RUN:   -fcuda-is-device -fsyntax-only -verify=expected,gfx942 %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx908 -x hip \
// RUN:   -fcuda-is-device -fsyntax-only -verify=expected,gfx908 %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x hip \
// RUN:   -fcuda-is-device -fsyntax-only -Wno-pinned-register-call \
// RUN:   -verify=off %s

// off-no-diagnostics

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __forceinline__ inline __attribute__((always_inline))

typedef float float4 __attribute__((ext_vector_type(4)));

#define PIN(r) __attribute__((amdgpu_pin_vgpr(r)))
#define PINA(r) __attribute__((amdgpu_pin_agpr(r)))

__device__ float4 ext(float4 x);
__device__ __forceinline__ float4 inl(float4 x) { return x * x; }

// A call that is not inlined clobbers the caller-saved registers.

__global__ void caller_saved(float4 *out) {
  PIN(8) float4 a = out[0]; // expected-note {{make the callee always_inline, or pin 'a' to callee-saved registers}}
  out[1] = ext(out[2]); // expected-warning {{call may clobber 'a' (v[8:11]), which is used after it}}
  out[3] = a;
}

__global__ void straddle(float4 *out) {
  PIN(46) float4 a = out[0]; // expected-note {{make the callee always_inline}}
  out[1] = ext(out[2]); // expected-warning {{call may clobber 'a' (v[46:49]), which is used after it}}
  out[3] = a;
}

__global__ void in_loop(float4 *out, int n) {
  PIN(8) float4 acc = {0, 0, 0, 0}; // expected-note {{make the callee always_inline}}
  for (int i = 0; i < n; i++)
    acc += ext(out[i]); // expected-warning {{call may clobber 'acc' (v[8:11]), which is used after it}}
  out[0] = acc;
}

__global__ void indirect(float4 *out, float4 (*f)(float4)) {
  PIN(8) float4 a = out[0]; // expected-note {{make the callee always_inline}}
  out[1] = f(out[2]); // expected-warning {{call may clobber 'a' (v[8:11]), which is used after it}}
  out[3] = a;
}

__global__ void agpr(float4 *out) {
  PINA(0) float4 a = out[0]; // expected-note {{make the callee always_inline}}
  PINA(32) float4 b = out[1]; // gfx908-note {{make the callee always_inline}}
  out[2] = ext(out[3]); // expected-warning {{call may clobber 'a' (a[0:3]), which is used after it}} \
                        // gfx908-warning {{call may clobber 'b' (a[32:35]), which is used after it}}
  out[4] = a + b;
}

// No warning: callee-saved registers.
__global__ void callee_saved(float4 *out) {
  PIN(40) float4 a = out[0];
  PIN(56) float4 b = out[1];
  out[2] = ext(out[3]);
  out[4] = a + b;
}

// No warning: not used after the call.
__global__ void dead_after(float4 *out) {
  PIN(8) float4 a = out[0];
  out[1] = a;
  out[2] = ext(out[3]);
}

// No warning: the call result overwrites it.
__global__ void overwritten(float4 *out) {
  PIN(8) float4 a = out[0];
  a = ext(a);
  out[1] = a;
}

// No warning: always_inline callees and builtins are not calls.
__global__ void inlined(float4 *out) {
  PIN(8) float4 a = out[0];
  out[1] = inl(out[2]);
  out[2] = __builtin_amdgcn_workitem_id_x();
  out[3] = a;
}

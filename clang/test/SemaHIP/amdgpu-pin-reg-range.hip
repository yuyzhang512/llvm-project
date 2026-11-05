// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x hip \
// RUN:   -fcuda-is-device -fsyntax-only -verify=gfx942 %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -x hip \
// RUN:   -fcuda-is-device -fsyntax-only -verify=gfx1250 %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fsyntax-only \
// RUN:   -verify=host %s

// host-no-diagnostics

#define __global__ __attribute__((global))

typedef float float4 __attribute__((ext_vector_type(4)));

#define PIN(r) __attribute__((amdgpu_pin_vgpr(r)))
#define PINA(r) __attribute__((amdgpu_pin_agpr(r)))

__global__ void last_vgprs(float4 *out) {
  PIN(252) float4 a = out[0];
  out[1] = a;
}

__global__ void past_256(float4 *out) {
  PIN(254) float4 a = out[0]; // gfx942-error {{'a' is pinned to v[254:257], past the 256 VGPRs on gfx942}}
  out[1] = a;
}

__global__ void past_1024(float4 *out) {
  PIN(1022) float4 a = out[0]; // gfx942-error {{'a' is pinned to v[1022:1025], past the 256 VGPRs on gfx942}} \
                               // gfx1250-error {{'a' is pinned to v[1022:1025], past the 1024 VGPRs on gfx1250}}
  out[1] = a;
}

__global__ void agpr(float4 *out) {
  PINA(252) float4 a = out[0]; // gfx1250-error {{'a' is pinned to a[252:255], but there are no AGPRs on gfx1250}}
  PINA(253) float4 b = out[1]; // gfx942-error {{'b' is pinned to a[253:256], past the 256 AGPRs on gfx942}} \
                               // gfx1250-error {{'b' is pinned to a[253:256], but there are no AGPRs on gfx1250}}
  out[2] = a + b;
}

template <int R> __global__ void dependent(float4 *out) {
  PIN(R) float4 a = out[0]; // gfx942-error {{'a' is pinned to v[300:303], past the 256 VGPRs on gfx942}}
  out[1] = a;
}
template __global__ void dependent<300>(float4 *); // gfx942-note {{in instantiation of}}

__global__ void deduced(float4 *out) {
  PIN(255) auto a = out[0]; // gfx942-error {{'a' is pinned to v[255:258], past the 256 VGPRs on gfx942}}
  out[1] = a;
}


// A pinned value has to name a register tuple the backend supports.

typedef int int5 __attribute__((ext_vector_type(5)));
typedef int int13 __attribute__((ext_vector_type(13)));
typedef int int16 __attribute__((ext_vector_type(16)));
typedef int int32 __attribute__((ext_vector_type(32)));

__global__ void widths(int16 *out16, int5 *out5, int13 *out13, int32 *out32) {
  PIN(16) int16 a = out16[0];
  PIN(40) int5 b = out5[0]; // gfx942-error {{'b' takes 5 registers; a pinned value must take 1, 2, 3, 4, 8, or 16}} \
                            // gfx1250-error {{'b' takes 5 registers; a pinned value must take 1, 2, 3, 4, 8, or 16}}
  PIN(48) int13 c = out13[0]; // gfx942-error {{'c' takes 13 registers}} \
                              // gfx1250-error {{'c' takes 13 registers}}
  PIN(64) int32 d = out32[0]; // gfx942-error {{'d' takes 32 registers}} \
                              // gfx1250-error {{'d' takes 32 registers}}
  out16[1] = a;
  out5[1] = b;
  out13[1] = c;
  out32[1] = d;
}

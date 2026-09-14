// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x hip \
// RUN:   -fcuda-is-device -fsyntax-only -verify %s

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

typedef float float4 __attribute__((ext_vector_type(4)));

// Only an automatic local has a lifetime the register can cover.
__device__ __attribute__((amdgpu_pin_vgpr(8))) int g; // expected-warning {{'amdgpu_pin_vgpr' attribute ignored}}

__global__ void f(int p) { // expected-note {{declared here}}
  __attribute__((amdgpu_pin_vgpr(8))) int ok;
  __attribute__((amdgpu_pin_agpr(16))) float4 ok_agpr;
  (void)ok;
  (void)ok_agpr;

  static __attribute__((amdgpu_pin_vgpr(8))) int s; // expected-warning {{'amdgpu_pin_vgpr' attribute ignored}}
  (void)s;

  // A type that is not a whole number of registers cannot be placed.
  __attribute__((amdgpu_pin_vgpr(8))) char c; // expected-warning {{'amdgpu_pin_vgpr' attribute ignored}}
  (void)c;

  // The register number has to be a non-negative constant.
  __attribute__((amdgpu_pin_vgpr(-1))) int neg; // expected-error {{'amdgpu_pin_vgpr' attribute requires a non-negative integral compile time constant expression}}
  (void)neg;

  __attribute__((amdgpu_pin_vgpr(p))) int nonconst; // expected-error {{expression is not an integral constant expression}} expected-note {{function parameter 'p' with unknown value cannot be used in a constant expression}}
  (void)nonconst;
}

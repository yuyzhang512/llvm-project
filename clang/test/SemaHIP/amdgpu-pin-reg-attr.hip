// RUN: %clang_cc1 -triple amdgpu9.42 -x hip \
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

// A dependent argument is checked when the template is instantiated.
template <int N> __device__ int tmpl() {
  __attribute__((amdgpu_pin_vgpr(N))) int acc; // expected-error {{'amdgpu_pin_vgpr' attribute requires a non-negative integral compile time constant expression}}
  acc = 0;
  return acc;
}
__global__ void use(int *out) {
  out[0] = tmpl<-1>(); // expected-note {{in instantiation of function template specialization 'tmpl<-1>' requested here}}
}

// An auto type is checked once it has been deduced, not before.
__global__ void deduced(float4 *in, short *s) {
  __attribute__((amdgpu_pin_vgpr(8))) auto v = in[0];
  __attribute__((amdgpu_pin_vgpr(12))) auto h = s[0]; // expected-warning {{'amdgpu_pin_vgpr' attribute ignored}}
  in[1] = v;
  s[1] = h;
}

// A register number computed from a generic lambda's parameter, inside a
// function template: the lambda's call operator is instantiated after the
// template, and the pin has to follow it there.
template <int I> struct num { static constexpr int value = I; };
template <class F> __device__ void twice(F f) { f(num<0>{}); f(num<1>{}); }
template <int Base> __device__ void per_tile(float4 *out) {
  twice([&](auto i) {
    __attribute__((amdgpu_pin_agpr(Base + i.value * 4))) auto t = out[i.value];
    out[2 + i.value] = t;
  });
}
__global__ void tiles(float4 *out) { per_tile<16>(out); }

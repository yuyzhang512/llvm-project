// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x hip \
// RUN:   -fcuda-is-device -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x hip \
// RUN:   -fcuda-is-device -fsyntax-only -Wno-pinned-register-overlap \
// RUN:   -verify=off %s

// off-no-diagnostics

#define __global__ __attribute__((global))

typedef float float4 __attribute__((ext_vector_type(4)));

#define PIN(r) __attribute__((amdgpu_pin_vgpr(r)))
#define PINA(r) __attribute__((amdgpu_pin_agpr(r)))

// Two locals with overlapping registers cannot both be live.

__global__ void same_regs(float4 *out, const float4 *in) {
  PIN(8) float4 a = in[0]; // expected-note {{'a' is pinned to v[8:11] here}}
  PIN(8) float4 b = in[1]; // expected-warning {{writing 'b' (v[8:11]) clobbers 'a' (v[8:11]), which is used later}}
  out[0] = a + b;
}

__global__ void partial_overlap(float4 *out, const float4 *in) {
  PIN(8) float4 a = in[0]; // expected-note {{'a' is pinned to v[8:11] here}}
  PIN(10) float4 b = in[1]; // expected-warning {{writing 'b' (v[10:13]) clobbers 'a' (v[8:11]), which is used later}}
  out[0] = a + b;
}

__global__ void element_write(float4 *out, const float4 *in) {
  PIN(8) float4 a = in[0]; // expected-note {{'a' is pinned to v[8:11] here}}
  PIN(11) float b;
  b = 1.0f; // expected-warning {{writing 'b' (v11) clobbers 'a' (v[8:11]), which is used later}}
  out[0] = a + b;
}

__global__ void in_loop(float4 *out, const float4 *in, int n) {
  PIN(8) float4 acc = {0, 0, 0, 0}; // expected-note {{'acc' is pinned to v[8:11] here}}
  for (int i = 0; i < n; i++) {
    PIN(8) float4 b = in[i]; // expected-warning {{writing 'b' (v[8:11]) clobbers 'acc' (v[8:11]), which is used later}}
    acc = acc + b;
  }
  out[0] = acc;
}

// The other value is used only on a later branch.
__global__ void on_branch(float4 *out, const float4 *in, bool c) {
  PIN(8) float4 a = in[0]; // expected-note {{'a' is pinned to v[8:11] here}}
  PIN(8) float4 b = in[1]; // expected-warning {{writing 'b' (v[8:11]) clobbers 'a' (v[8:11]), which is used later}}
  if (c)
    out[0] = a;
  else
    out[0] = b;
}

// No warning: a is dead before b is written.
__global__ void reuse(float4 *out, const float4 *in) {
  PIN(8) float4 a = in[0];
  out[0] = a;
  PIN(8) float4 b = in[1];
  out[1] = b;
}

// No warning: disjoint registers.
__global__ void disjoint(float4 *out, const float4 *in) {
  PIN(8) float4 a = in[0];
  PIN(12) float4 b = in[1];
  out[0] = a + b;
}

// No warning: one accumulator carried around a loop.
__global__ void loop_acc(float4 *out, const float4 *in, int n) {
  PIN(8) float4 acc = {0, 0, 0, 0};
  for (int i = 0; i < n; i++)
    acc = acc + in[i];
  out[0] = acc;
}

// No warning: same number, but a VGPR and an AGPR.
__global__ void vgpr_agpr(float4 *out, const float4 *in) {
  PIN(8) float4 a = in[0];
  PINA(8) float4 b = in[1];
  out[0] = a + b;
}

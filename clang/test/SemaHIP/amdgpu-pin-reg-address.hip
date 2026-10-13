// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx942 -x hip \
// RUN:   -fcuda-is-device -fsyntax-only -verify %s

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))

typedef float f32x4 __attribute__((ext_vector_type(4)));

// A pinned variable is the register itself and has no storage, so every form
// that needs its address is rejected rather than reaching code generation. It
// is still an lvalue: like a bit-field, a const reference binds to a copy.

__global__ void take_address(f32x4 *out) {
  __attribute__((amdgpu_pin_vgpr(8))) f32x4 v;
  v = out[0];
  f32x4 *p = &v; // expected-error {{address of register variable requested}}
  out[1] = *p;
}

__global__ void capture_by_reference(f32x4 *out) {
  __attribute__((amdgpu_pin_agpr(0))) f32x4 a; // expected-note {{'a' declared here}}
  a = out[0];
  auto f = [&] { return a; }; // expected-error {{address of register variable requested}}
  out[1] = f();
}

// Uses that do not need an address keep working.

__device__ inline f32x4 by_value(f32x4 v) { return v + v; }

__global__ void reads_and_writes_are_fine(f32x4 *out) {
  __attribute__((amdgpu_pin_vgpr(8))) f32x4 v;
  v = out[0];
  f32x4 copy = v;
  out[1] = by_value(copy);
  out[2] = v;
}

__global__ void capture_by_value_is_fine(f32x4 *out) {
  __attribute__((amdgpu_pin_vgpr(8))) f32x4 v;
  v = out[0];
  f32x4 copy = v;
  auto f = [copy] { return copy + copy; };
  out[1] = f();
}

__device__ inline f32x4 by_const_ref(const f32x4 &v) { return v + v; }

__global__ void const_reference_binds_to_a_copy(f32x4 *out) {
  __attribute__((amdgpu_pin_vgpr(8))) f32x4 v;
  v = out[0];
  const f32x4 &r = v;
  out[1] = r;
  out[2] = by_const_ref(v);
}

__device__ inline void by_ref(f32x4 &v) { v = v + v; } // expected-note {{passing argument to parameter 'v' here}}

__global__ void non_const_reference(f32x4 *out) {
  __attribute__((amdgpu_pin_vgpr(8))) f32x4 v; // expected-note 2 {{'v' declared here}}
  v = out[0];
  f32x4 &r = v; // expected-error {{non-const reference cannot bind to register variable 'v'}}
  by_ref(v);    // expected-error {{non-const reference cannot bind to register variable 'v'}}
  out[1] = r;
}

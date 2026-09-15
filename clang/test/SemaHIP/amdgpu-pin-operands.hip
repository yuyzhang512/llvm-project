// RUN: %clang_cc1 -triple amdgpu9.42 -x hip -fcuda-is-device -fsyntax-only -verify %s

typedef float float4 __attribute__((ext_vector_type(4)));

__attribute__((global)) void f(const float4 *in, float4 *out, int n) {
  float4 acc;

  // The register number is optional; omitting it asks only for the file.
  [[clang::amdgpu_pin_vgpr]] acc = in[0] + in[1];
  [[clang::amdgpu_pin_vgpr()]] acc = in[0] + in[1];
  [[clang::amdgpu_pin_agpr(16)]] acc = in[0] + in[1];

  // amdgpu_pin_gpr takes an operand number and a register name, repeated.
  [[clang::amdgpu_pin_gpr(0, "v[20:23]")]] acc = in[0] + in[1];
  [[clang::amdgpu_pin_gpr(0, "a[16:19]", 1, "a8")]] acc = in[0] + in[1];

  // An odd number of arguments leaves an operand number with no register.
  // expected-error@+1 {{'clang::amdgpu_pin_gpr' attribute requires a string}}
  [[clang::amdgpu_pin_gpr(0, "v8", 1)]] acc = in[0] + in[1];

  // The register has to be a string naming it, not a number.
  // expected-error@+1 {{'clang::amdgpu_pin_gpr' attribute requires a string}}
  [[clang::amdgpu_pin_gpr(0, 8)]] acc = in[0] + in[1];

  // Naming one operand twice is a contradiction rather than a refinement.
  // expected-error@+1 {{'clang::amdgpu_pin_gpr' attribute}}
  [[clang::amdgpu_pin_gpr(0, "v8", 0, "v12")]] acc = in[0] + in[1];

  // Operand numbers are non-negative.
  // expected-error@+1 {{'clang::amdgpu_pin_gpr' attribute requires a non-negative}}
  [[clang::amdgpu_pin_gpr(-1, "v8")]] acc = in[0] + in[1];

  out[0] = acc;
}

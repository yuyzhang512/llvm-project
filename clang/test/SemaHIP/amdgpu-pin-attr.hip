// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx950 -x hip \
// RUN:   -fcuda-is-device -fsyntax-only -verify %s

typedef float float2 __attribute__((ext_vector_type(2)));

__attribute__((device)) void ok(void) {
  float2 a, b, c;
  [[clang::amdgpu_pin_agpr(0)]] a = a + a;
  [[clang::amdgpu_pin_vgpr(8)]] b = b + b;
  constexpr int base = 16;
  [[clang::amdgpu_pin_agpr(base + 4)]] c = c + c;
  (void)a; (void)b; (void)c;
}

__attribute__((device)) void bad(int n) { // expected-note {{declared here}}
  float2 a, b, c;
  // expected-error@+1 {{'clang::amdgpu_pin_agpr' attribute requires a non-negative integral compile time constant expression}}
  [[clang::amdgpu_pin_agpr(-1)]] a = a + a;
  // expected-error@+2 {{expression is not an integral constant expression}}
  // expected-note@+1 {{function parameter 'n' with unknown value cannot be used in a constant expression}}
  [[clang::amdgpu_pin_vgpr(n)]] b = b + b;
  // Only the result is named; an operand is pinned by pinning what produces it.
  // expected-error@+1 {{'clang::amdgpu_pin_vgpr' attribute takes one argument}}
  [[clang::amdgpu_pin_vgpr(4, 8)]] c = c + c;
  (void)a; (void)b; (void)c;
}

// The request names the result of an operation, so it has nowhere to attach on
// a declaration -- neither a local, nor a global, nor a function.
__attribute__((device)) void bad_subject(void) {
  // expected-error@+1 {{'clang::amdgpu_pin_vgpr' attribute cannot be applied to a declaration}}
  [[clang::amdgpu_pin_vgpr(4)]] float2 x;
  (void)x;
}

// expected-error@+1 {{'clang::amdgpu_pin_agpr' attribute cannot be applied to a declaration}}
[[clang::amdgpu_pin_agpr(0)]] __attribute__((device)) void func(void) {}

// expected-error@+1 {{'clang::amdgpu_pin_vgpr' attribute cannot be applied to a declaration}}
[[clang::amdgpu_pin_vgpr(0)]] __attribute__((device)) float2 g_pinned;

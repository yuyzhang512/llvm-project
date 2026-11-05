// RUN: %clang_cc1 -triple amdgpu9.42 -x hip -fcuda-is-device -emit-llvm \
// RUN:   -disable-llvm-passes -fpin-register-lowering=metadata -o - %s \
// RUN:   | FileCheck --check-prefix=FE %s
// RUN: %clang_cc1 -triple amdgpu9.42 -x hip -fcuda-is-device -emit-llvm -O3 \
// RUN:   -fpin-register-lowering=metadata -o - %s \
// RUN:   | FileCheck --check-prefix=OPT %s

// With -fpin-register-lowering=metadata a pinned local keeps its alloca, and
// every store to it carries !llvm.pin.register. The pass pipeline turns that
// into llvm.pin.register before mem2reg.

#define __global__ __attribute__((global))

typedef float float4 __attribute__((ext_vector_type(4)));

// FE-LABEL: define {{.*}}void @_Z3accPDv4_fi(
// FE: %acc = alloca <4 x float>
// FE: store <4 x float> zeroinitializer, ptr %acc.ascast, align 16, !llvm.pin.register ![[PIN:[0-9]+]]
// FE: store <4 x float> %add, ptr %acc.ascast, align 16, !llvm.pin.register ![[PIN]]
// FE: store <4 x float> %{{.*}}, ptr %acc.ascast, align 16, !llvm.pin.register ![[PIN]]
// FE: ![[PIN]] = !{!"v[20:23]", i1 true}

// OPT-LABEL: define {{.*}}void @_Z3accPDv4_fi(
// OPT-NOT: alloca
// OPT: call {{.*}}<4 x float> @llvm.pin.register.v4f32(<4 x float> zeroinitializer, metadata ![[V20:[0-9]+]], i1 true)
// OPT-NOT: !llvm.pin.register
// OPT: ![[V20]] = !{!"v[20:23]"}
__global__ void acc(float4 *out, int n) {
  __attribute__((amdgpu_pin_vgpr(20))) float4 acc = {0, 0, 0, 0};
  for (int i = 0; i < n; i++)
    acc += out[i];
  acc.x = 1.0f;
  out[0] = acc;
}

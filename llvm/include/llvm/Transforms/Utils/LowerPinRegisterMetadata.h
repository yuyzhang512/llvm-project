//===- LowerPinRegisterMetadata.h - Turn pin metadata into intrinsics -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A store carrying !llvm.pin.register asks that the stored value be placed in
// a named register. Optimizations drop metadata they do not know, so this pass
// turns the request into a call to llvm.pin.register on the stored value
// before the optimizer runs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_LOWERPINREGISTERMETADATA_H
#define LLVM_TRANSFORMS_UTILS_LOWERPINREGISTERMETADATA_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/Compiler.h"

namespace llvm {

// Required: without it the request is lost, at any optimization level.
struct LowerPinRegisterMetadataPass
    : public RequiredPassInfoMixin<LowerPinRegisterMetadataPass> {
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_LOWERPINREGISTERMETADATA_H

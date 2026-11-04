//===- LowerPinRegisterMetadata.cpp - Turn pin metadata into intrinsics ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/LowerPinRegisterMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"

using namespace llvm;

// Whether \p SV keeps its first operand and fills the remaining lanes from an
// undef second operand.
static bool isWidening(const ShuffleVectorInst *SV) {
  auto *SrcTy = cast<FixedVectorType>(SV->getOperand(0)->getType());
  int NumSrc = SrcTy->getNumElements();
  ArrayRef<int> Mask = SV->getShuffleMask();
  if (int(Mask.size()) <= NumSrc || !isa<UndefValue>(SV->getOperand(1)))
    return false;
  for (int I = 0; I != int(Mask.size()); ++I) {
    if (I < NumSrc && Mask[I] != I)
      return false;
    if (I >= NumSrc && Mask[I] != PoisonMaskElem && Mask[I] < NumSrc)
      return false;
  }
  return true;
}

PreservedAnalyses LowerPinRegisterMetadataPass::run(Function &F,
                                                    FunctionAnalysisManager &) {
  unsigned KindID = F.getContext().getMDKindID("llvm.pin.register");
  SmallVector<StoreInst *> Stores;
  for (Instruction &I : instructions(F))
    if (auto *SI = dyn_cast<StoreInst>(&I))
      if (SI->getMetadata(KindID))
        Stores.push_back(SI);
  if (Stores.empty())
    return PreservedAnalyses::all();

  for (StoreInst *SI : Stores) {
    // !{!"<register name>", i1 <required>}
    MDNode *MD = SI->getMetadata(KindID);
    SI->setMetadata(KindID, nullptr);
    if (MD->getNumOperands() != 2 || !isa<MDString>(MD->getOperand(0)))
      continue;
    auto *Required = mdconst::dyn_extract<ConstantInt>(MD->getOperand(1));
    if (!Required)
      continue;

    // Padding added to store a vector, e.g. a <3 x float> widened to
    // <4 x float>, is not part of the value.
    Value *V = SI->getValueOperand();
    Instruction *Widen = nullptr;
    if (auto *SV = dyn_cast<ShuffleVectorInst>(V))
      if (SV->hasOneUse() && isWidening(SV)) {
        Widen = SV;
        V = SV->getOperand(0);
      }

    IRBuilder<> B(Widen ? Widen : SI);
    Metadata *NameOps[] = {MD->getOperand(0).get()};
    Value *Name = MetadataAsValue::get(F.getContext(),
                                       MDNode::get(F.getContext(), NameOps));
    Value *Pinned = B.CreateIntrinsic(Intrinsic::pin_register, {V->getType()},
                                      {V, Name, Required});
    if (Widen)
      Widen->setOperand(0, Pinned);
    else
      SI->setOperand(0, Pinned);
  }
  return PreservedAnalyses::none();
}

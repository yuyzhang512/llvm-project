//===- AMDGPULowerPinToNamedReg.cpp - Pin a value through a named register ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// llvm.amdgcn.pin.{vgpr,agpr}(V, N) asks for V to live in the tuple starting
// at register N. Front ends that build SSA directly, rather than a variable
// with storage, can only say this about a value. Give it the named-register
// form: V is written to the tuple where it is pinned, and each block that uses
// it reads the tuple where it first needs the value. The same request made
// through !amdgpu.pin.{vgpr,agpr} metadata on the instruction is lowered alike.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Pass.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-pin-to-named-reg"

namespace {

class AMDGPULowerPinToNamedRegLegacy : public FunctionPass {
public:
  static char ID;
  AMDGPULowerPinToNamedRegLegacy() : FunctionPass(ID) {}
  bool runOnFunction(Function &F) override;
  StringRef getPassName() const override {
    return "AMDGPU Lower Pin To Named Register";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }
};

} // end anonymous namespace

// Write V to the tuple at Reg right before WriteAt, and give every use of Old a
// read of the tuple in its block instead.
static void lowerToNamedReg(Value *V, Value *Old, Instruction *WriteAt,
                            int64_t Reg, bool IsAGPR) {
  Type *Ty = V->getType();
  LLVMContext &Ctx = Ty->getContext();
  TypeSize Bits = WriteAt->getDataLayout().getTypeSizeInBits(Ty);
  unsigned Lanes = Bits.isScalable() ? 0 : Bits.getFixedValue() / 32;
  Type *IntTy = Type::getInt32Ty(Ctx);
  if (Lanes > 1)
    IntTy = FixedVectorType::get(IntTy, Lanes);

  // A request for the file alone, or for a value that is not whole 32-bit
  // lanes, names no tuple.
  if (Reg < 0 || !Lanes || Bits.getFixedValue() % 32 ||
      !CastInst::isBitCastable(Ty, IntTy)) {
    if (Old != V)
      Old->replaceAllUsesWith(V);
    return;
  }

  std::string Name = Lanes == 1 ? formatv("{0}{1}", IsAGPR ? 'a' : 'v', Reg).str()
                                : formatv("{0}[{1}:{2}]", IsAGPR ? 'a' : 'v',
                                          Reg, Reg + Lanes - 1)
                                      .str();
  Value *MD =
      MetadataAsValue::get(Ctx, MDNode::get(Ctx, MDString::get(Ctx, Name)));

  // A PHI that merges the pinned value with values from other edges carries it
  // around a loop. It joins the pin: the other incoming values are written to
  // the tuple at the end of their blocks, and the PHI's uses read the tuple.
  // Only when each such block leads to the PHI alone; a write on a block with
  // other successors would clobber the tuple on paths that never reach it.
  SmallSetVector<Value *, 4> Pinned;
  Pinned.insert(Old);
  SmallVector<PHINode *, 4> Joined;
  for (unsigned I = 0; I != Pinned.size(); ++I) {
    for (User *U : Pinned[I]->users()) {
      auto *PN = dyn_cast<PHINode>(U);
      if (!PN || Pinned.contains(PN) || PN->getType() != Ty)
        continue;
      bool CanJoin = true;
      for (unsigned K = 0, E = PN->getNumIncomingValues(); K != E; ++K) {
        Value *In = PN->getIncomingValue(K);
        if (Pinned.contains(In) || In == PN || isa<UndefValue>(In))
          continue;
        if (PN->getIncomingBlock(K)->getSingleSuccessor() != PN->getParent())
          CanJoin = false;
      }
      if (CanJoin) {
        Pinned.insert(PN);
        Joined.push_back(PN);
      }
    }
  }

  // Read once in each block that needs the value, ahead of its first use
  // there; a PHI takes the value at the end of the block it comes in from. The
  // value is then never carried from one block to another outside the tuple.
  MapVector<BasicBlock *, SmallVector<Use *, 4>> UsesByBlock;
  for (Value *P : Pinned)
    for (Use &U : P->uses()) {
      auto *User = cast<Instruction>(U.getUser());
      auto *PN = dyn_cast<PHINode>(User);
      if (PN && Pinned.contains(PN))
        continue;
      UsesByBlock[PN ? PN->getIncomingBlock(U) : User->getParent()].push_back(
          &U);
    }

  IRBuilder<> B(WriteAt);
  B.CreateIntrinsic(Intrinsic::write_register, {IntTy},
                    {MD, B.CreateBitCast(V, IntTy)});
  for (PHINode *PN : Joined)
    for (unsigned K = 0, E = PN->getNumIncomingValues(); K != E; ++K) {
      Value *In = PN->getIncomingValue(K);
      if (Pinned.contains(In) || isa<UndefValue>(In))
        continue;
      IRBuilder<> WB(PN->getIncomingBlock(K)->getTerminator());
      WB.CreateIntrinsic(Intrinsic::write_register, {IntTy},
                         {MD, WB.CreateBitCast(In, IntTy)});
    }

  for (auto &[BB, Uses] : UsesByBlock) {
    Instruction *At = BB->getTerminator();
    for (Use *U : Uses) {
      auto *User = cast<Instruction>(U->getUser());
      if (!isa<PHINode>(User) && User->comesBefore(At))
        At = User;
    }
    IRBuilder<> RB(At);
    Value *R = RB.CreateBitCast(
        RB.CreateIntrinsic(Intrinsic::read_register, {IntTy}, {MD}), Ty);
    for (Use *U : Uses)
      U->set(R);
  }
  for (PHINode *PN : Joined)
    PN->replaceAllUsesWith(PoisonValue::get(Ty));
  for (PHINode *PN : Joined)
    PN->eraseFromParent();
}

bool AMDGPULowerPinToNamedRegLegacy::runOnFunction(Function &F) {
  bool Changed = false;
  for (Instruction &I : make_early_inc_range(instructions(F))) {
    // llvm.amdgcn.pin.{vgpr,agpr}(V, N): the carrier's own result is the value.
    if (auto *II = dyn_cast<IntrinsicInst>(&I);
        II && (II->getIntrinsicID() == Intrinsic::amdgcn_pin_vgpr ||
               II->getIntrinsicID() == Intrinsic::amdgcn_pin_agpr)) {
      lowerToNamedReg(II->getArgOperand(0), II, II,
                      cast<ConstantInt>(II->getArgOperand(1))->getSExtValue(),
                      II->getIntrinsicID() == Intrinsic::amdgcn_pin_agpr);
      II->eraseFromParent();
      Changed = true;
      continue;
    }
    // !amdgpu.pin.{vgpr,agpr} !{i32 N} on the instruction computing the value.
    for (bool IsAGPR : {false, true}) {
      MDNode *MD = I.getMetadata(IsAGPR ? "amdgpu.pin.agpr" : "amdgpu.pin.vgpr");
      if (!MD || I.getType()->isVoidTy())
        continue;
      int64_t Reg = -1;
      if (MD->getNumOperands())
        if (auto *C = mdconst::dyn_extract<ConstantInt>(MD->getOperand(0)))
          Reg = C->getSExtValue();
      I.setMetadata(IsAGPR ? "amdgpu.pin.agpr" : "amdgpu.pin.vgpr", nullptr);
      Instruction *After = isa<PHINode>(I) ? &*I.getParent()->getFirstNonPHIIt()
                                           : I.getNextNode();
      lowerToNamedReg(&I, &I, After, Reg, IsAGPR);
      Changed = true;
    }
  }
  return Changed;
}

char AMDGPULowerPinToNamedRegLegacy::ID = 0;
char &llvm::AMDGPULowerPinToNamedRegLegacyID =
    AMDGPULowerPinToNamedRegLegacy::ID;

INITIALIZE_PASS(AMDGPULowerPinToNamedRegLegacy, DEBUG_TYPE,
                "AMDGPU Lower Pin To Named Register", false, false)

FunctionPass *llvm::createAMDGPULowerPinToNamedRegLegacyPass() {
  return new AMDGPULowerPinToNamedRegLegacy();
}

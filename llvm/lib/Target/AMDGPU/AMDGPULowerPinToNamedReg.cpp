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
// llvm.amdgcn.pin.lds.addr(V, First, Count) asks for the LDS loads producing V
// to take their address register from [First, First + Count). An LDS address is
// @lds + root + constant, and instruction selection folds up to 64 KB of the
// constant into the DS offset field. So each distinct root, split into
// clusters under 64 KB, gets one base register, pinned through the request
// above, and every such load becomes base + (constant - cluster start).
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Operator.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-lower-pin-to-named-reg"

static cl::opt<bool> ExpPinLocal("exp-pin-local", cl::Hidden);
static cl::opt<bool> ExpJoinAll("exp-pin-join-all", cl::Hidden);

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
  if (ExpPinLocal) {
    IRBuilder<> B(WriteAt);
    B.CreateIntrinsic(Intrinsic::write_register, {IntTy},
                      {MD, B.CreateBitCast(V, IntTy)});
    Value *R = B.CreateBitCast(
        B.CreateIntrinsic(Intrinsic::read_register, {IntTy}, {MD}), Ty);
    Old->replaceAllUsesWith(R);
    return;
  }
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
        if (!ExpJoinAll &&
            PN->getIncomingBlock(K)->getSingleSuccessor() != PN->getParent())
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


namespace {
// An LDS address split as Base + Root + Off: Base a constant pointer, Root the
// one non-constant term.
struct LDSAddr {
  Constant *Base = nullptr;
  Value *Root = nullptr;
  int64_t Off = 0;
};
} // end anonymous namespace

// Split an i32 offset as Root + constant. A PHI whose incoming values all split
// to the same root and constant (GVN merges recomputed addresses this way)
// splits like them.
static Value *splitOffset(Value *Idx, int64_t &Off, unsigned Depth = 0) {
  while (auto *BO = dyn_cast<BinaryOperator>(Idx)) {
    auto *C = dyn_cast<ConstantInt>(BO->getOperand(1));
    bool IsAdd = BO->getOpcode() == Instruction::Add ||
                 (BO->getOpcode() == Instruction::Or &&
                  cast<PossiblyDisjointInst>(BO)->isDisjoint());
    if (!C || !IsAdd)
      break;
    Off += C->getSExtValue();
    Idx = BO->getOperand(0);
  }
  auto *PN = dyn_cast<PHINode>(Idx);
  if (!PN || Depth > 2)
    return Idx;
  Value *Root = nullptr;
  int64_t PhiOff = 0;
  for (Value *In : PN->incoming_values()) {
    int64_t InOff = 0;
    Value *InRoot = splitOffset(In, InOff, Depth + 1);
    if (Root && (InRoot != Root || InOff != PhiOff))
      return Idx;
    Root = InRoot;
    PhiOff = InOff;
  }
  Off += PhiOff;
  return Root;
}

static bool splitLDSAddr(Value *P, LDSAddr &A) {
  while (auto *GEP = dyn_cast<GEPOperator>(P)) {
    if (!GEP->getSourceElementType()->isIntegerTy(8) ||
        GEP->getNumIndices() != 1)
      return false;
    Value *Idx = GEP->getOperand(1);
    if (auto *C = dyn_cast<ConstantInt>(Idx)) {
      A.Off += C->getSExtValue();
    } else {
      if (A.Root)
        return false;
      A.Root = splitOffset(Idx, A.Off);
    }
    P = GEP->getPointerOperand();
  }
  A.Base = dyn_cast<Constant>(P);
  return A.Base && A.Root && A.Root->getType()->isIntegerTy(32);
}

// Rewrite every llvm.amdgcn.pin.lds.addr into base registers pinned with
// llvm.amdgcn.pin.vgpr, which the rest of the pass then lowers.
static bool lowerLDSAddrPins(Function &F) {
  struct Request {
    LoadInst *Load;
    int64_t First, Count;
    LDSAddr A;
  };
  SmallVector<Request> Reqs;
  SmallVector<IntrinsicInst *> Marks;
  SmallPtrSet<LoadInst *, 32> Seen;
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II || II->getIntrinsicID() != Intrinsic::amdgcn_pin_lds_addr)
      continue;
    Marks.push_back(II);
    int64_t First = cast<ConstantInt>(II->getArgOperand(1))->getSExtValue();
    int64_t Count = cast<ConstantInt>(II->getArgOperand(2))->getSExtValue();
    // The value is assembled from the loads by shuffles, inserts, extracts
    // and bitcasts.
    SmallVector<Value *> Work{II->getArgOperand(0)};
    SmallPtrSet<Value *, 32> Visited;
    while (!Work.empty()) {
      Value *V = Work.pop_back_val();
      if (!Visited.insert(V).second)
        continue;
      if (auto *LI = dyn_cast<LoadInst>(V)) {
        if (LI->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS &&
            Seen.insert(LI).second) {
          Request R{LI, First, Count, {}};
          if (!splitLDSAddr(LI->getPointerOperand(), R.A))
            R.A.Root = nullptr;
          Reqs.push_back(R);
        }
        continue;
      }
      if (isa<ShuffleVectorInst, InsertElementInst, ExtractElementInst,
              BitCastInst, ExtractValueInst, InsertValueInst>(V))
        for (Value *Op : cast<Instruction>(V)->operands())
          if (isa<Instruction>(Op))
            Work.push_back(Op);
    }
  }
  for (IntrinsicInst *II : Marks) {
    II->replaceAllUsesWith(II->getArgOperand(0));
    II->eraseFromParent();
  }
  if (Marks.empty())
    return false;

  // Cluster the offsets of each (range, base, root) so that every load's
  // offset from its cluster's start fits the 16-bit DS offset.
  using Key = std::tuple<int64_t, int64_t, Constant *, Value *>;
  auto keyOf = [](const Request &R) {
    return Key(R.First, R.Count, R.A.Base, R.A.Root);
  };
  MapVector<Key, SmallVector<int64_t>> Offsets;
  for (const Request &R : Reqs)
    if (R.A.Root)
      Offsets[keyOf(R)].push_back(R.A.Off);
  constexpr int64_t MaxDSOffset = 65535;
  struct Cluster {
    int64_t Start;
    int64_t Reg;
    Value *RawBase = nullptr;
  };
  MapVector<Key, SmallVector<Cluster>> Clusters;
  DenseMap<std::pair<int64_t, int64_t>, int64_t> NextReg;
  for (auto &[K, Offs] : Offsets) {
    llvm::sort(Offs);
    auto [First, Count, Base, Root] = K;
    for (int64_t Off : Offs) {
      SmallVector<Cluster> &Cs = Clusters[K];
      if (!Cs.empty() && Off - Cs.back().Start <= MaxDSOffset)
        continue;
      int64_t &N = NextReg[{First, Count}];
      if (N >= Count) {
        if (N++ == Count)
          F.getContext().diagnose(DiagnosticInfoGeneric(
              formatv("{0}: LDS address registers v[{1}:{2}] are too few for "
                      "the loads asking for them; the rest keep their "
                      "addresses",
                      F.getName(), First, First + Count - 1)
                  .str(),
              DS_Warning));
        Cs.push_back({Off, -1});
        continue;
      }
      Cs.push_back({Off, First + N++});
    }
  }

  // Each cluster's base is computed once, in the nearest block dominating all
  // its loads, so a base that only code after a loop needs is not live across
  // the loop.
  DominatorTree DT(F);
  auto clusterOf = [&](Request &R) -> Cluster * {
    SmallVector<Cluster> &Cs = Clusters[keyOf(R)];
    Cluster *C = &Cs.front();
    for (Cluster &Cand : Cs)
      if (Cand.Start <= R.A.Off)
        C = &Cand;
    return C;
  };
  DenseMap<Cluster *, BasicBlock *> Home;
  for (Request &R : Reqs) {
    if (!R.A.Root)
      continue;
    BasicBlock *&H = Home[clusterOf(R)];
    H = H ? DT.findNearestCommonDominator(H, R.Load->getParent())
          : R.Load->getParent();
  }
  LLVMContext &Ctx = F.getContext();
  Type *I8 = Type::getInt8Ty(Ctx), *I32 = Type::getInt32Ty(Ctx);
  auto rawBase = [&](Cluster &C, Constant *Base, Value *Root) -> Value * {
    if (C.RawBase)
      return C.RawBase;
    BasicBlock *H = Home.lookup(&C);
    BasicBlock::iterator IP = H->getFirstInsertionPt();
    if (auto *RI = dyn_cast<Instruction>(Root);
        RI && RI->getParent() == H && !isa<PHINode>(RI))
      IP = std::next(RI->getIterator());
    IRBuilder<> B(H, IP);
    Value *Sum = C.Start ? B.CreateAdd(Root, B.getInt32(C.Start)) : Root;
    Value *Ptr = B.CreateGEP(I8, Base, Sum);
    C.RawBase = B.CreatePtrToInt(Ptr, I32, "lds.base");
    return C.RawBase;
  };

  // One pin per base and block, so the named register stays local to it.
  DenseMap<std::pair<Value *, BasicBlock *>, Value *> Pinned;
  for (Request &R : Reqs) {
    if (!R.A.Root)
      continue;
    Cluster *C = clusterOf(R);
    if (C->Reg < 0)
      continue;
    Value *Raw = rawBase(*C, R.A.Base, R.A.Root);
    BasicBlock *BB = R.Load->getParent();
    Value *&Pin = Pinned[{Raw, BB}];
    if (!Pin) {
      auto *RawI = cast<Instruction>(Raw);
      BasicBlock::iterator IP = RawI->getParent() == BB
                                    ? std::next(RawI->getIterator())
                                    : BB->getFirstInsertionPt();
      IRBuilder<> B(BB, IP);
      Pin = B.CreateIntrinsic(Intrinsic::amdgcn_pin_vgpr, {I32},
                              {Raw, B.getInt32(C->Reg)});
    }
    IRBuilder<> B(R.Load);
    Value *Addr = B.CreateAdd(Pin, B.getInt32(R.A.Off - C->Start));
    R.Load->setOperand(R.Load->getPointerOperandIndex(),
                       B.CreateIntToPtr(Addr, R.Load->getPointerOperandType()));
  }
  return true;
}

bool AMDGPULowerPinToNamedRegLegacy::runOnFunction(Function &F) {
  bool Changed = lowerLDSAddrPins(F);
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

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
// @lds + the sum of some variable terms (each times a constant scale) + a
// constant, and instruction selection folds up to 64 KB of the constant into
// the DS offset field. Loads with the same variable terms, split into clusters
// under 64 KB, share one base register, pinned through the request above, and
// every such load becomes base + (constant - cluster start). The terms need
// not be loop invariant: the slot of a double buffer changes every iteration,
// and the base is then recomputed, into the same register, where its loads are.
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
// An LDS address split as Base + sum(Terms[i].first * Terms[i].second) + Off:
// Base a constant pointer, each term an i32 instruction or argument times a
// byte scale. Addresses are 32-bit, so all of it is mod 2^32.
struct LDSAddr {
  Constant *Base = nullptr;
  SmallVector<std::pair<Value *, int64_t>, 4> Terms;
  int64_t Off = 0;
  // The terms in a canonical order, to tell whether two loads share them.
  SmallVector<std::pair<Value *, int64_t>, 4> key() const {
    auto K = Terms;
    llvm::sort(K, [](auto &L, auto &R) {
      return std::less<Value *>()(L.first, R.first) ||
             (L.first == R.first && L.second < R.second);
    });
    return K;
  }
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

// A * B in 32-bit address arithmetic, sign-extended: products of huge
// constant indices and scales wrap instead of overflowing int64_t.
static int64_t mulAddr(int64_t A, int64_t B) {
  return static_cast<int32_t>(static_cast<uint32_t>(
      static_cast<uint64_t>(A) * static_cast<uint64_t>(B)));
}

// Walk the GEP chain of an LDS pointer. Each single-index GEP adds its index
// times the source element size: a constant index to Off, any other to Terms.
static bool splitLDSAddr(Value *P, const DataLayout &DL, LDSAddr &A) {
  while (auto *GEP = dyn_cast<GEPOperator>(P)) {
    if (GEP->getNumIndices() != 1)
      return false;
    TypeSize Size = DL.getTypeAllocSize(GEP->getSourceElementType());
    if (Size.isScalable())
      return false;
    // Only the low 32 bits of the scale matter for a 32-bit address.
    int64_t Scale = Size.getFixedValue() & 0xffffffff;
    Value *Idx = GEP->getOperand(1);
    if (auto *C = dyn_cast<ConstantInt>(Idx)) {
      A.Off += mulAddr(C->getSExtValue(), Scale);
    } else {
      if (!Idx->getType()->isIntegerTy(32))
        return false;
      int64_t Off = 0;
      Value *Root = splitOffset(Idx, Off);
      A.Off += mulAddr(Off, Scale);
      // A PHI with no incoming values (unreachable code) has no root.
      if (!Root)
        return false;
      if (auto *C = dyn_cast<ConstantInt>(Root))
        A.Off += mulAddr(C->getSExtValue(), Scale); // a PHI of equal constants
      else if (!isa<Instruction, Argument>(Root))
        return false;
      else if (Scale)
        A.Terms.push_back({Root, Scale});
    }
    P = GEP->getPointerOperand();
  }
  A.Base = dyn_cast<Constant>(P);
  return A.Base && !A.Terms.empty();
}

// The operand index of an LDS load's address: a load from addrspace(3) or one
// of the transposing DS load intrinsics. -1 for anything else.
static int ldsAddrOperand(Instruction *I) {
  if (auto *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS
               ? LI->getPointerOperandIndex()
               : -1;
  if (auto *II = dyn_cast<IntrinsicInst>(I)) {
    switch (II->getIntrinsicID()) {
    case Intrinsic::amdgcn_ds_load_tr4_b64:
    case Intrinsic::amdgcn_ds_load_tr6_b96:
    case Intrinsic::amdgcn_ds_load_tr8_b64:
    case Intrinsic::amdgcn_ds_load_tr16_b128:
    case Intrinsic::amdgcn_ds_read_tr4_b64:
    case Intrinsic::amdgcn_ds_read_tr6_b96:
    case Intrinsic::amdgcn_ds_read_tr8_b64:
    case Intrinsic::amdgcn_ds_read_tr16_b64:
      return 0;
    default:
      break;
    }
  }
  return -1;
}

// Rewrite every llvm.amdgcn.pin.lds.addr into base registers pinned with
// llvm.amdgcn.pin.vgpr, which the rest of the pass then lowers.
static bool lowerLDSAddrPins(Function &F) {
  struct Request {
    Instruction *Load;
    int AddrOp;
    int64_t First, Count;
    LDSAddr A;
    bool Ok;
  };
  const DataLayout &DL = F.getDataLayout();
  SmallVector<Request> Reqs;
  SmallVector<IntrinsicInst *> Marks;
  SmallPtrSet<Instruction *, 32> Seen;
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
      if (auto *I = dyn_cast<Instruction>(V); I && ldsAddrOperand(I) >= 0) {
        if (Seen.insert(I).second) {
          int Op = ldsAddrOperand(I);
          Request R{I, Op, First, Count, {}, false};
          R.Ok = splitLDSAddr(I->getOperand(Op), DL, R.A);
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

  // Cluster the offsets of each (range, base, terms) so that every load's
  // offset from its cluster's start fits the 16-bit DS offset. Term lists are
  // numbered so that they can be part of a key.
  SmallVector<SmallVector<std::pair<Value *, int64_t>, 4>> TermLists;
  auto termsId = [&](const LDSAddr &A) -> int64_t {
    auto K = A.key();
    for (auto [I, L] : enumerate(TermLists))
      if (L == K)
        return I;
    TermLists.push_back(K);
    return TermLists.size() - 1;
  };
  using Key = std::tuple<int64_t, int64_t, Constant *, int64_t>;
  auto keyOf = [&](const Request &R) {
    return Key(R.First, R.Count, R.A.Base, termsId(R.A));
  };
  // Loads in unreachable blocks keep their addresses.
  DominatorTree DT(F);
  for (Request &R : Reqs)
    if (!DT.isReachableFromEntry(R.Load->getParent()))
      R.Ok = false;
  MapVector<Key, SmallVector<int64_t>> Offsets;
  for (const Request &R : Reqs)
    if (R.Ok)
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
    auto [First, Count, Base, Terms] = K;
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
    if (!R.Ok)
      continue;
    BasicBlock *&H = Home[clusterOf(R)];
    H = H ? DT.findNearestCommonDominator(H, R.Load->getParent())
          : R.Load->getParent();
  }
  LLVMContext &Ctx = F.getContext();
  Type *I8 = Type::getInt8Ty(Ctx), *I32 = Type::getInt32Ty(Ctx);
  auto rawBase = [&](Cluster &C, const LDSAddr &A) -> Value * {
    if (C.RawBase)
      return C.RawBase;
    // Right after the last term computed in the home block, if any.
    BasicBlock *H = Home.lookup(&C);
    BasicBlock::iterator IP = H->getFirstInsertionPt();
    for (Instruction &I : *H)
      if (!isa<PHINode>(I) &&
          any_of(A.Terms, [&](auto &T) { return T.first == &I; }))
        IP = std::next(I.getIterator());
    IRBuilder<> B(H, IP);
    Value *Sum = nullptr;
    for (auto [Term, Scale] : A.Terms) {
      Value *V = Scale == 1 ? Term
                 : isPowerOf2_64(Scale)
                     ? B.CreateShl(Term, Log2_64(Scale))
                     : B.CreateMul(Term, B.getInt32(Scale));
      Sum = Sum ? B.CreateAdd(Sum, V) : V;
    }
    if (C.Start)
      Sum = B.CreateAdd(Sum, B.getInt32(C.Start));
    Value *Ptr = B.CreateGEP(I8, A.Base, Sum);
    C.RawBase = B.CreatePtrToInt(Ptr, I32, "lds.base");
    return C.RawBase;
  };

  // One pin per base and block, so the named register stays local to it.
  DenseMap<std::pair<Value *, BasicBlock *>, Value *> Pinned;
  for (Request &R : Reqs) {
    if (!R.Ok)
      continue;
    Cluster *C = clusterOf(R);
    if (C->Reg < 0)
      continue;
    Value *Raw = rawBase(*C, R.A);
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
    R.Load->setOperand(
        R.AddrOp,
        B.CreateIntToPtr(Addr, R.Load->getOperand(R.AddrOp)->getType()));
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

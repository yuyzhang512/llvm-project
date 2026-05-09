# CP5: Fold the Coupled i64 Phi Pair and Bake VALID into the Phi

## Goal

Reduce per-iteration SALU cost of TDM descriptor address advancement from
6 SALU/iter (after cp4) to 4 SALU/iter by:

1. **Baking VALID** (bit 63) into the i64 address phi so the per-iter
   `s_or_b32 ..., 0x80000000` re-OR is unnecessary.
2. **Co-locating** the i64 phi with the descriptor's sub2:sub3 lanes so the
   `s_mov_b32` copy and the K-bump fuse into a single in-place
   `s_add_nc_u64`.

Target codegen (per descriptor, per iter):

```asm
s_add_nc_u64 s[sub2:sub3], s[sub2:sub3], 0x80   ; K-bump, in-place on descriptor
```

---

## The Problem: InstCombine Splits the i64 Phi into a Coupled Pair

When VALID is baked into the i64 phi at the Triton level (no strip/re-OR in
`fillTDMDescriptor`), the resulting LLVM IR regresses from 9 SALU to 15 SALU,
and the kernel can hit a hardware assert ("Tensor Load/Store invalid type 0").

The root cause is **InstCombine**, not IndVarSimplify or LSR. Two transforms,
applied in sequence, create a redundant coupled phi pair.

### Step 1: Triton Emits a Struct Phi

The Triton frontend produces a descriptor struct phi in the loop:

```llvm
%desc = phi { i32, i32, i32, i32 } [ %init_desc, %preheader ], [ %next_desc, %latch ]
%sub2 = extractvalue %desc, 2          ; addr_lo
%sub3 = extractvalue %desc, 3          ; addr_hi | VALID (bit 31 of i32 = bit 63 of i64)
; ... i64 add to bump address by K bytes ...
%next_desc = insertvalue ..., %new_lo, 2
%next_desc2 = insertvalue %next_desc, %new_hi, 3
```

### Step 2: `foldPHIArgInsertValueInstructionIntoPHI` Splits the Struct

**Location**: `InstCombinePHI.cpp:366`

InstCombine sees `phi [insertvalue(a, x, idx), insertvalue(b, y, idx)]` and
splits the struct phi into per-element scalar phis. Applied recursively for
each index:

```llvm
%sub0 = phi i32 [ init_sub0, %preheader ], [ next_sub0, %latch ]
%sub1 = phi i32 [ init_sub1, %preheader ], [ next_sub1, %latch ]
%sub2 = phi i32 [ init_sub2, %preheader ], [ next_sub2, %latch ]   ; addr_lo
%sub3 = phi i32 [ init_sub3, %preheader ], [ next_sub3, %latch ]   ; addr_hi | VALID
```

This is a generally beneficial transform: it replaces an aggregate phi with
scalar phis that unlock further scalar optimizations.

### Step 3: Trunc-Folding Recombines Into Coupled i64 Phis

**Location**: `InstCombineCasts.cpp:240` → `foldOpIntoPhi`

Later in the InstCombine worklist, the pass sees patterns like
`trunc i64 %val to i32` where `%val` feeds from a phi. It folds the trunc
into the phi operands. When two adjacent i32 phis (sub2, sub3) are combined
back into i64 via `zext`/`shl`/`or` patterns, the trunc-folding creates
**two separate i64 phis** rather than one:

```llvm
P1 = phi i64 [ %init_full,            %preheader ], [ %next_full,          %latch ]
P2 = phi i64 [ %init_high_with_VALID,  %preheader ], [ %next_high,         %latch ]
```

Where:
- `P1` = the "full" i64 value (raw address without VALID at iter 0, with VALID
  at iter 1+)
- `P2` = the "high half" tracker (always has VALID, only upper 32 bits matter)

### Step 4: The Reconstruction Pattern

Inside the loop body, the two phis are recombined every iteration:

```llvm
%low  = and i64 P1, 0x00000000FFFFFFFF       ; extract low half of P1
%addr = or [disjoint] i64 %low, P2           ; recombine with P2's high half
; ... uses of %addr ...
%next_full = add i64 %addr, 0x80             ; K-bump
%next_high = and i64 %next_full, 0xFFFFFFFF00000000  ; extract high half for P2
```

This `and` + `or` pair is **redundant at iter 1+**: since `P2_latch = P1_latch &
hi_mask`, at iteration 1+ we have `P2 = P1 & hi_mask`, so `(P1 & lo_mask) | P2
= (P1 & lo_mask) | (P1 & hi_mask) = P1`. Only at iteration 0 do P1 and P2
disagree (P1 lacks VALID, P2 has it).

The `and` + `or` show up as `s_and_b64` + `s_or_b64` in the final assembly.
Worse, the LSR/recombine chain occasionally drops bit 63 from the
reconstruction, causing the hardware assert.

---

## Correctness Argument: Why P1 = recombined at All Iterations

Let `latch(n)` denote the K-bumped value computed at iteration `n`.

**Before transform:**

| Iteration | P1              | P2                      | recombined = (P1 & lo) \| P2           |
|-----------|-----------------|-------------------------|----------------------------------------|
| 0         | init_full       | init_high_with_VALID    | (init_full & lo) \| init_high          |
| 1         | latch(0)        | latch(0) & hi           | (latch(0) & lo) \| (latch(0) & hi) = latch(0) |
| n >= 1    | latch(n-1)      | latch(n-1) & hi         | latch(n-1) = P1                        |

At iter 0, `recombined != P1` (P1 lacks VALID).
At iter 1+, `recombined == P1` exactly.

**The fix**: replace P1's preheader value with `(init_full & lo_mask) | init_high`,
so P1 = recombined at iter 0 too. Then `recombined == P1` at all iterations,
and the AND + OR + P2 are dead.

**Inductive proof**:
- Base (iter 0): `P1_new = (init_full & lo) | init_high = recombined_0`. ✓
- Step: `latch(n)` is computed from `recombined(n)`, now replaced by `P1(n)`.
  Since `P1(n) = recombined(n)` by hypothesis, `latch(n)` is unchanged, so
  `P1(n+1) = latch(n)` is unchanged, and `recombined(n+1) = P1(n+1)` by the
  iter-1+ identity. ✓

**Caveat — trunc users of P1**: If P1 has `trunc i64 to i32` users, they read
only the low 32 bits. The fusion changes P1's iter-0 value by OR'ing in
`init_high`'s bits. This is safe **only if `init_high[31:0] == 0`** (i.e., the
high-half initializer has no low bits set). The implementation must verify this
via known-bits analysis or restrict to the case where P1's only user is the
AND-mask.

**Domain invariant**: The K-bump (`add i64 %addr, K`) never carries into bit 63.
This holds for AMDGPU's 57-bit GPU virtual addresses and practical K values
(K << 2^32). The optimization relies on this — if carry reached bit 63, P2
would not equal `P1 & hi_mask` at iter 1+.

---

## Analysis of Fix Approaches

### Approach A: Cleanup Peephole in InstCombine (the original patch)

Add `foldHighHalfPhiPair` to `visitPHINode` in `InstCombinePHI.cpp`. When it
sees the coupled pair pattern (P1 + P2 + AND/OR reconstruction + P2's
high-half-mask loop-back), it fuses them.

| Aspect     | Assessment |
|------------|------------|
| Correctness | Sound (with the `init_high` low-bits fix) |
| Generality  | Very narrow pattern; only fires for AMDGPU TDM descriptors in practice |
| Upstreamability | **Poor** — target-specific peephole in a target-independent pass; InstCombine reviewers will push back |
| Alive2 | Cannot prove it (inductive across loop iterations; Alive2 uses bounded unrolling) |
| Complexity | Moderate — one function, ~130 lines |

### Approach B: Prevent Struct Phi Split (`shouldSplitStructPhi` Hook)

Teach `foldPHIArgInsertValueInstructionIntoPHI` to not split when the result
would create coupled i64 pairs.

| Aspect     | Assessment |
|------------|------------|
| Root cause | Addresses step 2, but the actual problem is in step 3 (trunc re-folding) |
| Feasibility | **Poor** — would need to predict that step 3 will later create a bad coupling. Step 2 splits one insertvalue index at a time; it has no visibility into which field pairs will later be recombined into i64, or whether their preheader values disagree in the high half |
| Side effects | Blocking the struct split hurts all other scalar optimizations for those fields |
| Hook design | A per-index `shouldSplitStructPhi(StructType, index)` can't express "don't split index 2 AND 3 together" — the decision is per-index |
| Verdict | **Not recommended** — the information needed to make the decision isn't available at split time |

### Approach C: Gate the Trunc Folding (Step 3)

Prevent `foldOpIntoPhi` from folding trunc into the phi when it would create a
coupled pair.

| Aspect     | Assessment |
|------------|------------|
| Root cause | Closer — this is where the coupling is born |
| Feasibility | **Poor** — `foldOpIntoPhi` is a generic transform for any operation; gating it for this specific case requires recognizing the full coupled-pair pattern at trunc-folding time, before the coupling exists |
| Side effects | Trunc folding is broadly beneficial; false positives would regress other code |
| Verdict | **Not recommended** — the coupling hasn't materialized yet when the decision is made |

### Approach D: AMDGPU-Specific IR Pass (Recommended)

Place the `foldHighHalfPhiPair` logic in an AMDGPU target-specific IR pass,
such as `AMDGPUCodeGenPrepare` or a dedicated pass. Same algorithm, right
abstraction layer.

| Aspect     | Assessment |
|------------|------------|
| Correctness | Same as Approach A |
| Generality  | Explicitly target-specific — no pretense of generality |
| Upstreamability | **Good** — target passes get less scrutiny; AMDGPU team controls it |
| Complexity | Same as A, plus pass boilerplate (~30 extra lines) |
| Timing | Runs after InstCombine, so the coupled pair pattern is fully materialized and easy to match |
| Verdict | **Recommended** |

### Approach E: MachineIR Backend Pass

Recognize the coupled-phi pattern after instruction selection in MachineIR,
e.g., in `SITDMDescHoist` or a new pass.

| Aspect     | Assessment |
|------------|------------|
| Correctness | Easier to verify — register-level, no abstract IR reasoning |
| Register coalescing | Can directly influence allocation of phi into descriptor sub2:sub3 |
| Upstreamability | Good |
| Complexity | Higher — MachineIR pattern matching is more verbose |
| Verdict | Viable alternative if IR-level approach proves insufficient |

---

## Recommended Implementation Plan

### Phase 1: AMDGPU IR Pass (Approach D)

1. Add a function `foldCoupledHighHalfPhiPair(PHINode &PN)` to
   `AMDGPUCodeGenPrepare` (or a new `AMDGPUInstCombine` pass).

2. The pattern matcher looks for:
   - `P1 = phi i64 [init_full, preheader], [next_full, latch]`
   - `AndInst = and i64 P1, 0xFFFFFFFF` (single use)
   - `OrInst = or i64 AndInst, P2` where P2 is a phi in the same block
   - `P2_latch = and i64 P1_latch, 0xFFFFFFFF00000000`
   - All other users of P1 are `trunc i64 to i32`

3. Safety checks:
   - Verify `P2_preheader[31:0] == 0` via `KnownBits` (for trunc-user safety)
   - Or: require P1 has no users other than `AndInst` (conservative)

4. Transform:
   - Compute `merged_init = (init_full & 0xFFFFFFFF) | init_high` in preheader
   - Set P1's preheader incoming to `merged_init`
   - RAUW `OrInst` → `P1`
   - Dead-code-eliminate AND, OR, P2

### Phase 2: Triton Side

In `fillTDMDescriptor` (`TDMUtility.cpp`):
- Remove the `and_(addrHi, 0x7FFFFFFF)` strip before the i64 add
- Remove the `or_(newHi, 1 << 31)` re-OR after the i64 add
- `createTDMDescriptor` still sets VALID once at make-time; advances preserve it

### Phase 3: Lit Tests

Required tests for the LLVM patch:
- `test/CodeGen/AMDGPU/tdm-desc-phi-fusion.ll` — basic pattern match
- Positive: two coupled phis with AND/OR reconstruction → fused to single phi
- Negative: wrong mask values → no transform
- Negative: P2 preheader has non-zero low bits → no transform
- Negative: P1 has unsafe users (not trunc-to-i32 or the AND) → no transform
- Edge: P2 has additional uses beyond the OR → P2 kept alive, uses still valid

---

## Assembly Before/After (K=1024 Advance Kernel)

### After cp4 (6 SALU/iter, 2 descriptors):

```asm
s_or_b32   sub3_A, phi_A.hi, 0x80000000     ; A: re-OR VALID
s_mov_b32  sub2_A, phi_A.lo                  ; A: copy addr_lo into desc
s_or_b32   sub3_B, phi_B.hi, 0x80000000     ; B: re-OR VALID
s_mov_b32  sub2_B, phi_B.lo                  ; B: copy addr_lo into desc
s_add_nc_u64 phi_A, phi_A, 0x80              ; A: K-bump
s_add_nc_u64 phi_B, phi_B, 0x80              ; B: K-bump
```

### After cp5 (4 SALU/iter, 2 descriptors):

```asm
s_mov_b32  sub2_B, phi_B.lo                  ; B: copy (not yet coalesced)
s_mov_b32  sub3_B, phi_B.hi                  ; B: copy (VALID already baked in)
s_add_nc_u64 phi_B, phi_B, 0x80              ; B: K-bump
s_add_nc_u64 s[sub2_A:sub3_A], s[sub2_A:sub3_A], 0x80  ; A: K-bump in-place
```

### Ideal (2 SALU/iter, if both coalesce):

```asm
s_add_nc_u64 s[sub2_A:sub3_A], s[sub2_A:sub3_A], 0x80  ; A: in-place
s_add_nc_u64 s[sub2_B:sub3_B], s[sub2_B:sub3_B], 0x80  ; B: in-place
```

Reaching the ideal requires the register coalescer to place each i64 phi
directly into the descriptor's sub2:sub3 slots (a backend coalescing problem,
not an IR problem).

---

## Key Invariants and Assumptions

1. **57-bit address space**: AMDGPU GPU virtual addresses are at most 57 bits.
   K-bumps (typically 64–8192 bytes) never carry into bit 63. This guarantees
   `P2 = P1 & hi_mask` holds at iter 1+.

2. **VALID is bit 63**: The TDM descriptor's VALID flag is bit 31 of sub3
   (= bit 63 of the i64 formed by sub2:sub3). Hardware requires this bit set
   for valid tensor loads/stores.

3. **Disjoint OR**: The `or` in the reconstruction is `disjoint` — `P1 & lo_mask`
   and `P2` have no overlapping set bits. This follows from P2 tracking only
   the high 32 bits.

4. **Alive2 cannot prove this**: The correctness requires an inductive argument
   across loop iterations. Alive2 uses bounded unrolling and cannot verify
   inter-iteration invariants. A manual proof (see §Correctness Argument
   above) and comprehensive lit tests are the verification strategy.

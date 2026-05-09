# RegisterCoalescer: COPY Narrowing for IMPLICIT_DEF Lanes

---

## Problem

When `%dst = COPY %src` and `%src` has lanes that are entirely IMPLICIT_DEF (or
have no sub-range at all), the full COPY creates false interference on those
undefined lanes, blocking coalescing.

---

## Example

A and B descriptors share a CSE'd template
`%tpl = REG_SEQUENCE 1, sub0, IMPLICIT_DEF, sub1, IMPLICIT_DEF, sub2, IMPLICIT_DEF, sub3`.
After A's COPY coalesces and prunes the IMPLICIT_DEF lanes:

```mir
preheader:
  ; A's COPY already merged — %A now has only sub0+sub1 sub-ranges.
  ; B's template COPY was renamed to use A's coalesced vreg:
  %B = COPY %A              ; fails on sub1 (A's lds_offset ≠ B's) — kept
                             ; %B inherits sub2+sub3 IMPLICIT_DEF lanes

loop.body:
  %B.sub2 = ...              ; real def (addr_lo)
  %B.sub3 = ...              ; real def (addr_hi|VALID)
  ; The downstream COPY into %B FAILS because %B's fossil IMPLICIT_DEF
  ; sub2/sub3 from the preheader COPY are live-through and interfere.
```

### Fix: COPY Narrowing

Narrow the COPY to only copy the defined lanes:

```
%B = COPY %A                         ; FAILS — false interference on sub2/sub3
→ %B.sub0_sub1<def,undef> = COPY %A.sub0_sub1  ; only copies defined lanes
```

The narrowed COPY avoids interference on sub2/sub3 entirely. The coalescer
retries and succeeds.

---

## Bugs in the Original Implementation and Fixes

The original implementation had six bugs. This section documents each one,
why it causes incorrect behavior, and what the fix is.

### Bug 1: `CopyIdx` missing `.getRegSlot()`

**Original:**
```cpp
SlotIndex CopyIdx = LIS->getInstructionIndex(*CopyMI);
```

**Problem:** `getInstructionIndex` returns the base slot. VNInfo defs live at
the *register* slot (`getRegSlot()`). So `VNI->def == CopyIdx` never matches,
making the entire sub-range cleanup loop a dead no-op.

**Fix:**
```cpp
SlotIndex CopyIdx = LIS->getInstructionIndex(*CopyMI).getRegSlot();
```

### Bug 2: `CopyIdx` captured after mutating `CopyMI`

**Original:**
```cpp
CopyMI->getOperand(0).setSubReg(SubIdx);   // mutate
CopyMI->getOperand(0).setIsUndef(true);     // mutate
CopyMI->getOperand(1).setSubReg(SubIdx);    // mutate
SlotIndex CopyIdx = LIS->getInstructionIndex(*CopyMI);  // query after mutate
```

**Problem:** Querying the instruction index after modifying operands is fragile.
The slot index should be captured before any mutation.

**Fix:** Move `getInstructionIndex` before the operand mutations.

### Bug 3: Flipped register mapping (CP.getSrcReg/getDstReg vs COPY operands)

**Original:**
```cpp
Register SrcReg = CP.getSrcReg();
Register DstReg = CP.getDstReg();
```

**Problem:** `CP.getSrcReg()` is the "register being coalesced away" and
`CP.getDstReg()` is the "register that survives." These are coalescer-level
concepts. When `CP.isFlipped()`, they are **swapped** relative to the actual
COPY instruction's operands:

```
COPY: %a = COPY %b
If CP.isFlipped():
  CP.getSrcReg() = %a  (COPY's def,  operand 0)
  CP.getDstReg() = %b  (COPY's use,  operand 1)
```

The code would analyze the wrong register's live interval for IMPLICIT_DEF lanes
and apply the narrowing to the wrong operands.

**Fix:** Read registers directly from the COPY instruction:
```cpp
Register CopySrcReg = CopyMI->getOperand(1).getReg();  // always the use
Register CopyDstReg = CopyMI->getOperand(0).getReg();  // always the def
```

This matches the pattern at line 2210 where existing code handles the flip:
```cpp
Register DstReg = CP.isFlipped() ? CP.getSrcReg() : CP.getDstReg();
```

### Bug 4: SubIdx not validated for the destination register class

**Original:**
```cpp
const TargetRegisterClass *RC = MRI->getRegClass(SrcReg);
// ... uses RC for getCoveringSubRegIndexes ...
// SubIdx is applied to BOTH src and dst operands without checking dst
```

**Problem:** The sub-register index was found using only the source's register
class. If the destination has a different register class, the index might not be
valid for it.

**Fix:** Get both register classes and verify:
```cpp
const TargetRegisterClass *CopySrcRC = MRI->getRegClass(CopySrcReg);
const TargetRegisterClass *CopyDstRC = MRI->getRegClass(CopyDstReg);
// ...
if (TRI->getSubClassWithSubReg(CopyDstRC, SubIdx)) {
  // SubIdx is valid for the destination — proceed
}
```

### Bug 5: `implicit-def` wrong for virtual registers

**Original:**
```cpp
CopyMI->addOperand(MachineOperand::CreateReg(
    DstReg, true /*IsDef*/, true /*IsImp*/));
```

**Problem:** Adding `implicit-def %dst` creates a full-register def, telling
LiveIntervals that ALL lanes of `%dst` are defined at this point. But we just
removed VNInfos for the IMPLICIT_DEF lanes from the destination's sub-ranges.
This inconsistency (instruction says "all lanes defined" but sub-ranges have no
VNInfos for some lanes) causes MachineVerifier failures.

For **virtual** registers (which is always the case here since `!CP.isPhys()`),
the `undef` flag on the sub-register def is sufficient to signal that the other
lanes are undefined and a new lifetime begins. The `implicit-def` pattern is
only needed for **physical** registers.

**Fix:** Remove the `implicit-def` — the `undef` flag is sufficient:
```cpp
CopyMI->getOperand(0).setSubReg(SubIdx);
CopyMI->getOperand(0).setIsUndef(true);   // signals new lifetime
CopyMI->getOperand(1).setSubReg(SubIdx);
// No implicit-def needed for virtual registers.
```

### Bug 6: Sub-range VNInfo removal breaks straddling sub-ranges

**Original:**
```cpp
for (auto &SR : DstLI.subranges()) {
  if ((SR.LaneMask & ImplicitDefLanes).any()) {
    if (VNInfo *VNI = SR.getVNInfoAt(CopyIdx))
      if (VNI->def == CopyIdx)
        SR.removeValNo(VNI);
  }
}
DstLI.removeEmptySubRanges();
```

**Problem:** This removes VNInfos from any sub-range that *overlaps*
`ImplicitDefLanes`. But a single sub-range can cover BOTH defined lanes AND
IMPLICIT_DEF lanes. For example:

```
DefinedLanes     = 0x0F   (sub0_sub1 for sgpr_128)
ImplicitDefLanes = 0xF0   (sub2_sub3)

Destination sub-ranges:
  L000000000000000C  — sub1 area, within DefinedLanes ✓
  L00000000000000F3  — straddles: 0x03 (defined) + 0xF0 (IMPLICIT_DEF)
```

The sub-range `L00000000000000F3` has `(0xF3 & 0xF0) = 0xF0 != 0`, so the
original code removes its VNInfo. But that VNInfo also covers the **real**
defined lanes `0x03`! Removing it destroys the liveness tracking for lanes that
the narrowed COPY still defines.

This manifests as a MachineVerifier error:
```
*** Bad machine code: Defining instruction does not modify register ***
- instruction: %X.sub0_sub1:sgpr_128 = COPY %Y.sub0_sub1:sgpr_128
- lanemask:    0000000000000030
- ValNo:       1 (def 10944r)
```

The instruction defines `sub0_sub1` lanes but the sub-range for lane `0x30`
(which was part of the straddling `L00000000000000F3`) still has a VNInfo
claiming a def at that point.

**Why surgical removal is impossible:** The sub-range `L00000000000000F3` can't
be split into `L0000000000000003` + `L00000000000000F0` at this point in the
coalescer — LiveIntervals doesn't provide a sub-range splitting API, and
manually doing it would require duplicating all VNInfos, segments, and PHI
structure.

**Fix:** Don't manually edit sub-ranges at all. Instead, recompute the live
intervals from scratch:
```cpp
LIS->removeInterval(CopyDstReg);
LIS->createAndComputeVirtRegInterval(CopyDstReg);
LIS->removeInterval(CopySrcReg);
LIS->createAndComputeVirtRegInterval(CopySrcReg);
```

This is safe and correct — `computeVirtRegInterval` re-derives all sub-ranges
and VNInfos from the instructions, which now include the narrowed COPY with the
correct sub-register indices. It also handles the source interval, which no
longer has uses from the dropped lanes.

---

## Final Corrected Code

```cpp
// Try narrowing a full COPY when the source has IMPLICIT_DEF-only lanes.
if (!CP.isPartial() && !CP.isPhys()) {
  // Bug 3 fix: use COPY instruction's actual operand registers,
  // not CP's coalescer-level Src/Dst which may be flipped.
  Register CopySrcReg = CopyMI->getOperand(1).getReg();
  Register CopyDstReg = CopyMI->getOperand(0).getReg();
  LiveInterval &CopySrcLI = LIS->getInterval(CopySrcReg);
  if (CopySrcLI.hasSubRanges()) {
    // Bug 4 fix: get both register classes.
    const TargetRegisterClass *CopySrcRC = MRI->getRegClass(CopySrcReg);
    const TargetRegisterClass *CopyDstRC = MRI->getRegClass(CopyDstReg);
    LaneBitmask FullMask = CopySrcRC->getLaneMask();

    LaneBitmask ImplicitDefLanes = LaneBitmask::getNone();
    LaneBitmask DefinedLanes = LaneBitmask::getNone();
    for (auto &SR : CopySrcLI.subranges()) {
      bool AllImplicitDef = true;
      for (VNInfo *VNI : SR.valnos) {
        if (VNI->isUnused()) continue;
        MachineInstr *DefMI = LIS->getInstructionFromIndex(VNI->def);
        if (!DefMI || !DefMI->isImplicitDef()) {
          AllImplicitDef = false;
          break;
        }
      }
      if (AllImplicitDef) ImplicitDefLanes |= SR.LaneMask;
      else                DefinedLanes |= SR.LaneMask;
    }
    LaneBitmask UncoveredLanes = FullMask & ~(ImplicitDefLanes | DefinedLanes);
    ImplicitDefLanes |= UncoveredLanes;

    if (ImplicitDefLanes.any() && DefinedLanes.any()) {
      SmallVector<unsigned, 4> SubRegIdxs;
      if (TRI->getCoveringSubRegIndexes(CopySrcRC, DefinedLanes,
                                        SubRegIdxs) &&
          SubRegIdxs.size() == 1) {
        unsigned SubIdx = SubRegIdxs[0];
        // Bug 4 fix: verify SubIdx is valid for the destination class.
        if (TRI->getSubClassWithSubReg(CopyDstRC, SubIdx)) {
          // Bug 2 fix: capture slot index BEFORE mutating CopyMI.
          // Bug 1 fix: use getRegSlot() — VNInfo defs are at the reg slot.
          SlotIndex CopyIdx =
              LIS->getInstructionIndex(*CopyMI).getRegSlot();

          CopyMI->getOperand(0).setSubReg(SubIdx);
          CopyMI->getOperand(0).setIsUndef(true);
          CopyMI->getOperand(1).setSubReg(SubIdx);
          // Bug 5 fix: no implicit-def for virtual registers.

          // Bug 6 fix: recompute intervals from scratch instead of
          // surgically editing sub-ranges (straddling sub-ranges make
          // surgical edits unsafe).
          LIS->removeInterval(CopyDstReg);
          LIS->createAndComputeVirtRegInterval(CopyDstReg);
          LIS->removeInterval(CopySrcReg);
          LIS->createAndComputeVirtRegInterval(CopySrcReg);

          Again = true;
          return false;
        }
      }
    }
  }
}
```

---

## Summary of Bugs

| # | Bug | Symptom | Root Cause |
|---|-----|---------|------------|
| 1 | Missing `.getRegSlot()` | Sub-range cleanup is silently a no-op | VNI defs are at register slot, not base slot |
| 2 | `CopyIdx` after mutation | Fragile/wrong slot index | Query before mutate, not after |
| 3 | Flipped registers | Wrong register analyzed/modified | CP.getSrc/DstReg ≠ COPY operand 0/1 when flipped |
| 4 | SubIdx not checked for dst | Invalid sub-reg index on destination | Only validated against source register class |
| 5 | `implicit-def` on virtual reg | Verifier: "all lanes defined" vs missing VNInfos | `undef` flag is sufficient for virtual regs |
| 6 | Straddling sub-range removal | Verifier: "instruction does not modify register" | Sub-range covers both defined + IMPLICIT_DEF lanes |

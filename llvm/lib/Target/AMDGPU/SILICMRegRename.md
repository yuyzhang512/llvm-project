# SILICMRegRename — Register Rename Pass to Enable Post-RA MachineLICM

## Problem

After register allocation, AMDGPU loops often contain loop-invariant instructions whose destination physical register is also defined by another (non-invariant) instruction in the same loop. Post-RA MachineLICM tracks `RUDefs`/`RUClobbers` per register unit — when it sees a register unit defined twice, it marks the unit as "clobbered" and refuses to hoist either def.

```
loop:
  $vgpr0 = V_MOV_B32 $sgpr0     ; loop-invariant, but can't hoist
  ... = use $vgpr0
  $vgpr0 = V_ADD_F32 ...        ; non-invariant, clobbers $vgpr0
```

## Solution

A new AMDGPU-specific `MachineFunctionPass` (`si-licm-reg-rename`) runs just before post-RA MachineLICM. It renames the invariant instruction's output to a free register and rewrites its users:

```
preheader:                           ; MachineLICM can now hoist here
loop:
  $vgpr42 = V_MOV_B32 $sgpr0    ; renamed from $vgpr0 to free $vgpr42
  ... = use $vgpr42              ; user rewritten
  $vgpr0 = V_ADD_F32 ...        ; untouched, no longer clobbers
```

## Inspiration

Triton's `amdgcnas.py` tool's `can_hoist()` function (line 2439) does the same thing at the assembly text level — it picks a free register from `bb.free_regs`, renames the hoisted instruction's output, and updates its users.

## Pipeline Position

```
... → MachineCopyPropagation → SI LICM Register Rename → MachineLICM (post-RA) → ...
```

Inserted via `insertPass(&MachineCopyPropagationID, &SILICMRegRenameLegacyID)` in `GCNPassConfig::addOptimizedRegAlloc()`.

## Files Created

### 1. `SILICMRegRename.h`
New-PM pass struct following the `SIPostRABundler.h` pattern.

### 2. `SILICMRegRename.cpp`
Full implementation containing:

| Component | Description |
|---|---|
| Legacy PM wrapper | `SILICMRegRenameLegacy` class with `INITIALIZE_PASS` macros |
| New PM entry point | `SILICMRegRenamePass::run()` |
| `computeLoopDefs()` | Walks all instructions in the loop block, builds `RUDefs` (register units defined at least once) and `RUClobbers` (defined more than once) BitVectors |
| `isRenameCandidate()` | Checks if an instruction is: loop-invariant (all sources defined outside loop), has exactly one non-dead explicit VGPR def, def is clobbered, safe to move, exactly 2 total defs of the register in the loop |
| `computeFreeRegUnits()` | Finds register units not touched anywhere in the loop — not in live-ins, live-outs, any instruction's defs/uses, or reserved registers |
| `findFreeReg()` | Picks a free VGPR from the same `TargetRegisterClass` as the original def |
| `renameDefAndUsers()` | Rewrites the def operand and all users reached by this def (handles cyclic single-block loop ordering) |

## Files Modified

### 3. `AMDGPU.h`
Added declarations:
```cpp
void initializeSILICMRegRenameLegacyPass(PassRegistry &);
extern char &SILICMRegRenameLegacyID;
```

### 4. `AMDGPUPassRegistry.def`
Added:
```cpp
MACHINE_FUNCTION_PASS("si-licm-reg-rename", SILICMRegRenamePass())
```

### 5. `AMDGPUTargetMachine.cpp`
- `#include "SILICMRegRename.h"`
- `initializeSILICMRegRenameLegacyPass(*PR)` in the initializer block
- Legacy: `insertPass(&MachineCopyPropagationID, &SILICMRegRenameLegacyID)` in `GCNPassConfig::addOptimizedRegAlloc()`
- New-PM: `insertPass<MachineCopyPropagationPass>(SILICMRegRenamePass())` in `AMDGPUCodeGenPassBuilder::addOptimizedRegAlloc()`

### 6. `CMakeLists.txt`
Added `SILICMRegRename.cpp` to the `add_llvm_target(AMDGPUCodeGen ...)` source list.

## Algorithm

### Per loop (innermost first, single-block only):

1. **Compute defs/clobbers** — Walk all instructions. For each physical reg def, set its register units in `RUDefs`. If already set, also set in `RUClobbers`. Handle regmasks conservatively.

2. **Identify rename candidates** — For each instruction:
   - Has exactly one non-dead explicit physical def (VGPR only)
   - All source operand register units are NOT in `LoopRUDefs` (loop-invariant)
   - Def's register units ARE in `RUClobbers` (would block MachineLICM)
   - `isSafeToMove()`, not a call/inline-asm/convergent
   - Exactly 2 defs of the same register in the loop (this one + one other)

3. **Compute free registers** — Build a BitVector of all register units used anywhere (live-ins, live-outs, defs, uses, reserved). Invert to get free units.

4. **Rename** — For each candidate:
   - Find a free register in the same class
   - Rewrite the def operand
   - Walk the cyclic instruction order, rewrite uses between this def and the other def

### User rewriting in cyclic single-block loops

In a single-block loop, instruction order is cyclic. Given two defs of the same register (the invariant one `MI` and the other `OtherDef`), users reached by `MI` are those between `MI` and `OtherDef` going forward with wrap-around:

```
  OtherDef: $vgpr0 = ...         ; idx=2
  use $vgpr0                     ; idx=3 → reached by OtherDef (NOT renamed)
  MI:       $vgpr0 = invariant   ; idx=5 → renamed to $vgpr42
  use $vgpr0                     ; idx=6 → reached by MI (RENAMED to $vgpr42)
  use $vgpr0                     ; idx=7 → reached by MI (RENAMED to $vgpr42)
  ; wraps around to idx=0,1 → reached by MI (RENAMED) until OtherDef at idx=2
```

## CLI Flag

```
-amdgpu-licm-reg-rename    (default: true)
```

Enable/disable the pass. Hidden flag for debugging.

## V1 Limitations

| Limitation | Reason |
|---|---|
| Single-block loops only | Avoids cross-block reaching-def analysis complexity |
| VGPR register class only | Most common case; AMDGPU has up to 256 VGPRs, making free registers likely |
| Exactly 2 defs of clobbered register | 3+ defs require iterative renaming |

## How to Test

```bash
# Run the pass in isolation on a .mir file
./bin/llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
  -run-pass=si-licm-reg-rename test.mir -o -

# Run combined with MachineLICM to verify hoisting
./bin/llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
  -run-pass=si-licm-reg-rename,machinelicm test.mir -o -

# Run full pipeline with debug output
./bin/llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
  -debug-only=si-licm-reg-rename test.ll -o /dev/null

# Disable the pass
./bin/llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx942 \
  -amdgpu-licm-reg-rename=false test.ll -o /dev/null
```

## Future Work

- Multi-block loop support (requires reaching-def analysis)
- SGPR and AGPR register classes
- Handle 3+ defs via iterative renaming
- Register pressure awareness (avoid increasing occupancy pressure)
- Integration with MachineLICM profitability heuristics

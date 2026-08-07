# AMDGPU L1 cache-capacity hazard: gfx950 evaluation

Evaluation of the four MISched commits that add and enable the L1 data cache
capacity hazard recognizer, measured on gfx950 (MI350X) with the AITER Triton
benchmark suite.

Commits under test, applied on top of LLVM `850a2b1b`:

```
4a5ad7bd6 [AMDGPU][MISched] Add critical-resource scheduling strategies
8d6faf439 [AMDGPU][MISched] Add resource distance map for critical-resource scheduling
419d2af87 [AMDGPU][MISched] Add L1 data cache capacity hazard recognizer
7c2075cb8 [AMDGPU] Enable L1 cache-capacity hazard by default with gfx1250 params
```

Hardware: AMD Instinct MI350X (gfx950), 256 CU, 4 SIMD/CU, wave64, 32 KB L1/CU,
max 32 waves/CU. ROCm 7.2.

## Summary of results

Median of 3 runs, 795 data points across 41 benchmarks. Higher-is-better and
lower-is-better metrics are direction-corrected.

| Category | improved >=2% | improved >=10% | regressed <=-2% | regressed <=-10% | flat |
|---|---|---|---|---|---|
| Batched GEMM | 0 | 0 | 7 | 0 | 68 |
| GEMM | 7 | 0 | 21 | 1 | 138 |
| MoE | 19 | 3 | 25 | 0 | 204 |
| Attention | 9 | 0 | 5 | 0 | 231 |
| Normalization | 0 | 0 | 13 | 0 | 32 |
| Total | 35 | 3 | 71 | 1 | 676 |

Net regression on this target, roughly 2:1 against, with 85% of points unchanged.

Largest real gains, consistent across all three metrics of the same case:

```
+15.71%  bench_moe_gemm_a4w4  batch=8   TFLOPS
+13.59%  bench_moe_gemm_a4w4  batch=8   kernel_latency_us
+11.89%  bench_moe_gemm_a4w4  batch=8   total_latency_us
 +6.00%  bench_moe_gemm_a8w8  batch=16  TFLOPS
```

Largest real losses:

```
-11.96%  bench_mla_decode_rope             1 x 48 x 163 x 512
-10.06%  bench_fused_gemm_afp4wfp4_a16w16  256 x 512 x 256
 -9.16%  bench_gemm_afp4wfp4               1 x 1280 x 8192
```

## Findings that affect how the results should be read

### 1. The default parameters never reach the pass when LLVM is used as a library

`amdgpu-l1-speed` is a `cl::opt<std::pair>` with a custom parser. Its
`cl::init({1,1})` is applied under `llc`, but not when LLVM is driven as a
library, which is how Triton compiles. The recognizer then receives a `{0,0}`
drain rate and aborts:

```
AMDGPUCacheCapacityHazardRecognizer.cpp:67:
Assertion `DrainRate.Dividend > 0 && DrainRate.Divisor > 0' failed.
```

This killed 38 of 50 benchmarks before any measurement was possible. `llc` on
the identical IR compiles fine, including with post-RA MISched forced, so this
is an invocation-path problem rather than anything target specific.

`patches/0001-occupancy-scaled-l1-params.patch` removes the dependency on the
`cl::init` defaults by computing the values, and keeps explicit command-line
values winning through `getNumOccurrences()`.

### 2. The feature is not gated to gfx1250

`maybeCombineL1Hazard()` tests only the `L1Hazard` flag, and the recognizer uses
`IsaVersion` only to decode vmcnt. Nothing restricts it to gfx1250, so the
gfx1250-tuned defaults (128 B/thread, 1 B/cycle, 256 cycles) are applied to
gfx950 as well.

### 3. Parameters scaled with occupancy

The numbers above were measured with the geometry derived per function instead
of fixed:

```
occ     = SIMachineFunctionInfo::getOccupancy()
bytes   = 128 * occ
speed   = occ / 1
latency = 256          (1024 hw cycles / 4 cycles per compiler cycle)
```

With the fixed gfx1250 constants the same suite gave 6 regressions of 10% or
worse and a worst case of -18.5%. Scaling with occupancy reduced that to 1 and
-12.0%. Script `05-check-occupancy.sh` confirms `getOccupancy()` matches the
`; Occupancy:` the AsmPrinter records, so the input is correct.

### 4. Most kernels are untouched, so single-run deltas are mostly noise

For `bench_gemm_a16w8_blockscale`, only 4 of 17 kernels differ between the two
builds, and all 4 have occupancy 1. At occupancy 3 the budget is 3x larger, the
hazard never fires, and the generated code is byte-identical.

A single run of that benchmark reported -11.39% at M=128. The two binaries for
that shape are identical apart from a `.file` directive, and three repeats of
the base build alone spanned 69.37 to 77.88 TFLOPS, a 12.3% spread. With medians
the same point is -0.94%.

Use `--repeat 3` or more. Script `04-diff-kernels.py` shows which kernels the
pass actually changes; deltas on the others are variance.

## Reproduction

Runs end to end from a clean machine. `$WORK` needs about 200 GB, and the whole
sequence takes roughly 4 hours, most of it in the two LLVM builds and the
benchmark repeats.

```bash
export WORK=/raid/yuyzhang_llvm
export AITER=$HOME/aiter
export GPU=7                            # a gfx950 device

bash scripts/00-setup-aiter.sh  $AITER  # aiter 8e1308d3 + native ops, ~20 min
bash scripts/01-build-llvm.sh   $WORK   # base and base+pass LLVM,    ~40 min
bash scripts/02-build-triton.sh $WORK   # one Triton commit per LLVM, ~30 min
python3 scripts/03-run-perf-ab.py --repeat 3 \
        --output $WORK/l1_hazard_perf.xlsx      # 50 benchmarks x 2 x 3, ~2 h
```

The workbook lands at `$WORK/l1_hazard_perf.xlsx`, with per-category sheets and
a Summary sheet whose final table is the improvement/regression count quoted
above. Per-run logs are under `$WORK/bench_logs_l1ab/rerun_{l1off,l1on}/`.

Supporting checks, both on a single benchmark:

```bash
python3 scripts/04-diff-kernels.py bench_gemm_a16w8_blockscale
bash    scripts/05-check-occupancy.sh bench_gemm_a16w8_blockscale
```

`04` prints which kernels the pass actually changes, which is what shows that
deltas on the untouched ones are noise. `05` cross-checks the occupancy the
recognizer uses against the occupancy the emitted code achieves.

Notes:

- The benchmark scripts that get run are listed in `CASES.md`. 50 are launched,
  each sweeping its own shapes, which is where the 795 data points come from.
- `run_all_benchmarks.py` supplies the benchmark list, the runner and the
  workbook. It is not part of aiter, so a copy is kept in `vendor/` and
  `00-setup-aiter.sh` installs it into the aiter root, which is where it expects
  to live (it locates the benchmarks relative to its own path).
- `03-run-perf-ab.py` swaps between the two Triton builds with editable installs
  rather than wheels: both builds are the same Triton commit, so their wheels
  would carry the same version label and their logs would collide.
- Use `ROCR_VISIBLE_DEVICES` to pick the GPU. `HIP_VISIBLE_DEVICES` breaks
  Proton on AMD.
- `--repeat 3` matters. With single runs the same suite reported 52 improvements
  against 53 regressions; medians give 35 against 71.
- To rerun only one side after a rebuild, pass `--only l1on`, which keeps the
  existing logs for the other configuration.
- Pinned versions: aiter `8e1308d3`, Triton `bf64a5db1b`, LLVM base `850a2b1b`.
  `nanobind==2.10.2` is required to build this Triton and is installed by
  `02-build-triton.sh`.

## Suggested follow-ups for the pass

1. Do not rely on `cl::init` for values read during library invocation, or the
   recognizer is unusable outside `llc`.
2. Gate the feature on the target, or give each subtarget its own cache
   geometry, since the gfx1250 constants are applied unconditionally today.
3. On gfx950 the pass only changes occupancy-1 kernels. It helps small-batch MoE
   GEMM by up to 15.7% and hurts afp4wfp4 fused GEMM and MLA decode-rope by
   around 10%, so a heuristic narrower than "on by default" looks preferable.

Eight benchmarks fail with the pass enabled that pass on base LLVM, and are not
explained by the parameter problem above:

```
bench_gemm_afp4wfp4_pre_quant_atomic   bench_moe_gemm_a8w8_blockscale
bench_moe_gemm_int8_smoothquant        bench_batch_prefill
bench_hstu_attn                        bench_la_paged_decode
bench_mla_decode                       bench_sage
```

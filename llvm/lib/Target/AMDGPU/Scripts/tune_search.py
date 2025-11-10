import argparse
import os
import random
import re
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict

ROOT = "/home/jeff/source/llvm_emu/llvm-project"
BASE_LL = "/home/jeff/work/tickets/566006/blockScaleTemp/xfer8/no_debug.ll"
OUTPUT_S = "/home/jeff/work/tickets/566006/blockScaleTemp/xfer8/res.s"

_DEFAULT_LLC = os.path.join(ROOT, "build", "bin", "llc")
LLC = os.environ.get("LLC", _DEFAULT_LLC if os.path.exists(_DEFAULT_LLC) else "llc")

ASM_CYCLES_RE = re.compile(r"^;+\s*Cycles:\s*(\d+)\s*$")
ASM_COLD_WARM_RE = re.compile(
    r"^;+\s*Cold:\s*(\d+)\s*cycles\s*\|\s*Warm:\s*(\d+)\s*cycles.*$"
)


def run_cmd(cmd):
    r = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return r.returncode, r.stdout, r.stderr


def parse_asm_report(asm_text: str):
    """Parse integrated static simulator report from an asm (.s) file.

    Returns:
      (raw_cycles, scaled_cycles, warm_loop_cycles)
    """
    raw_cycles = None
    scaled_cycles = None
    warm_loop_cycles = None

    in_raw = False
    in_scaled = False

    for line in asm_text.splitlines():
        if "=== Raw Metrics" in line:
            in_raw = True
            in_scaled = False
            continue
        if "=== Scaled Metrics" in line:
            in_scaled = True
            in_raw = False
            continue

        m = ASM_CYCLES_RE.match(line)
        if m:
            val = int(m.group(1))
            if in_scaled:
                scaled_cycles = val
            elif in_raw:
                raw_cycles = val
            continue

        m = ASM_COLD_WARM_RE.match(line)
        if m:
            warm_loop_cycles = int(m.group(2))

    return raw_cycles, scaled_cycles, warm_loop_cycles


def run_trial(params: Dict[str, Any], output_s: str = OUTPUT_S):
    llc_cmd = [
        LLC,
        "-mtriple=amdgcn-amd-amdhsa",
        "-mcpu=gfx1250",
        "-amdgpu-enable-static-simulator",
        "-amdgpu-static-sim-trip-count=64",
        f"-enable-post-misched={'false' if params['postra'] else 'false'}",
        "-o",
        output_s,
        BASE_LL,
        # AMDGPUMLSchedStrategy.cpp options
        f"-amdgpu-resource-balancing={params['resource_balancing']}",
        f"-amdgpu-ds-latency={params['ds_latency']}",
        f"-amdgpu-ds-latency-split={params['ds_latency_split']}",
        f"-amdgpu-ds-fifo-latency={params['ds_fifo_latency']}",
        f"-amdgpu-signal-latency={params['signal_latency']}",
        f"-amdgpu-ds-fence-latency={params['ds_fence_latency']}",
        f"-amdgpu-ds-fifo-size={params['ds_fifo_size']}",
        f"-amdgpu-ignore-valu-resource-balancing={'true' if params['ignore_valu'] else 'false'}",
        f"-amdgpu-avoid-exp-final-islot={'true' if params['avoid_exp_final_islot'] else 'false'}",
        f"-amdgpu-shadow-mix={'true' if params['shadow_mix'] else 'false'}",
        f"-amdgpu-shadow-mix-wmma-min-valu1c={params['shadow_mix_wmma_min_valu1c']}",
        f"-amdgpu-shadow-mix-wmma-min-ds={params['shadow_mix_wmma_min_ds']}",
        f"-amdgpu-shadow-mix-wmma-min-salu={params['shadow_mix_wmma_min_salu']}",
        f"-amdgpu-shadow-mix-lookahead-depth={params['shadow_mix_lookahead_depth']}",
        f"-amdgpu-shadow-mix-max-blocking-cost={params['shadow_mix_max_blocking_cost']}",
        f"-amdgpu-shadow-mix-max-candidates={params['shadow_mix_max_candidates']}",
        f"-amdgpu-shadow-priority-wmma-over-ds={'true' if params['shadow_priority_wmma_over_ds'] else 'false'}",
        f"-amdgpu-shadow-priority-wmma-over-salu={'true' if params['shadow_priority_wmma_over_salu'] else 'false'}",
        f"-amdgpu-shadow-priority-cvt-over-ds={'true' if params['shadow_priority_cvt_over_ds'] else 'false'}",
        f"-amdgpu-shadow-priority-cvt-over-salu={'true' if params['shadow_priority_cvt_over_salu'] else 'false'}",
        f"-amdgpu-shadow-priority-trans32-over-valu1c={'true' if params['shadow_priority_trans32_over_valu1c'] else 'false'}",
        f"-amdgpu-shadow-defer-trans32={'true' if params['shadow_defer_trans32'] else 'false'}",
        f"-amdgpu-shadow-mix-trans32-min-valu1c={params['shadow_mix_trans32_min_valu1c']}",
        f"-amdgpu-shadow-prefer-valu-over-salu-for-trans={'true' if params['shadow_prefer_valu_over_salu_for_trans'] else 'false'}",
        f"-amdgpu-resource-priority-coexec-producer={'true' if params['resource_priority_coexec_producer'] else 'false'}",
	f"-amdgpu-resource-priority-coexec-windows-size={'true' if params['resource_priority_coexec_windows_size'] else 'false'}",
	f"-amdgpu-resource-priority-coexec-exposed-cycles={'true' if params['resource_priority_coexec_exposed_cycles'] else 'false'}",
        f"-amdgpu-use-shadow-mix-rules={'true' if params['use_shadow_mix_rules'] else 'false'}",
    ]
    rc, out, err = run_cmd(llc_cmd)
    if rc != 0:
        return None, f"llc failed: {err.strip()}"

    try:
        with open(os.path.join(ROOT, output_s), "r", encoding="utf-8", errors="replace") as fh:
            asm_text = fh.read()
    except OSError as e:
        return None, f"failed to read {output_s}: {e}"

    raw_cycles, scaled_cycles, warm_loop_cycles = parse_asm_report(asm_text)
    if scaled_cycles is None:
        return None, "parse failed: missing scaled cycles in asm report"
    if warm_loop_cycles is None:
        warm_loop_cycles = 0

    score = scaled_cycles

    return {
        "raw_cycles": raw_cycles,
        "scaled_cycles": scaled_cycles,
        "warm_loop_cycles": warm_loop_cycles,
        "score": score,
        "params": params,
    }, None


def sample_params():
    return {
        "postra": random.choice([False, False]),
        # Latency/resource parameters
        "resource_balancing": random.choice([0, 50, 75, 100, 125, 150, 200]),
        "ds_latency": random.randint(20, 100),
        "ds_latency_split": random.randint(0, 10),
        "ds_fifo_latency": random.randint(20, 100),
        "signal_latency": random.randint(0, 70),
        "ds_fence_latency": random.randint(20, 100),
        "ds_fifo_size": random.randint(4, 16),
        "ignore_valu": random.choice([True, False]),
        "avoid_exp_final_islot": random.choice([True, False]),
        # Shadow mix parameters
        "shadow_mix": random.choice([True, False]),
        "shadow_mix_wmma_min_valu1c": random.randint(0, 6),
        "shadow_mix_wmma_min_ds": random.randint(0, 6),
        "shadow_mix_wmma_min_salu": random.randint(0, 4),
        "shadow_mix_lookahead_depth": random.randint(0, 16),
        "shadow_mix_max_blocking_cost": random.randint(4, 24),
        "shadow_mix_max_candidates": random.randint(4, 24),
        # Shadow priority toggles
        "shadow_priority_wmma_over_ds": random.choice([True, False]),
        "shadow_priority_wmma_over_salu": random.choice([True, False]),
        "shadow_priority_cvt_over_ds": random.choice([True, False]),
        "shadow_priority_cvt_over_salu": random.choice([True, False]),
        "shadow_priority_trans32_over_valu1c": random.choice([True, False]),
        "shadow_defer_trans32": random.choice([True, False]),
        "shadow_mix_trans32_min_valu1c": random.randint(0, 4),
        "shadow_prefer_valu_over_salu_for_trans": random.choice([True, False]),
	"resource_priority_coexec_exposed_cycles": random.choice([True, False]),
	"resource_priority_coexec_windows_size": random.choice([True, False]),
	"resource_priority_coexec_producer": random.choice([True, False]),
        "use_shadow_mix_rules": random.choice([True, False]),
    }


# Default values matching cl::init in AMDGPUMLSchedStrategy.cpp
BEST_SEED = {
    "postra": False,
    "resource_balancing": 100,
    "ds_latency": 53,
    "ds_latency_split": 0,
    "ds_fifo_latency": 50,
    "signal_latency": 43,
    "ds_fence_latency": 52,
    "ds_fifo_size": 10,
    "ignore_valu": True,
    "avoid_exp_final_islot": True,
    "shadow_mix": True,
    "shadow_mix_wmma_min_valu1c": 3,
    "shadow_mix_wmma_min_ds": 1,
    "shadow_mix_wmma_min_salu": 2,
    "shadow_mix_lookahead_depth": 6,
    "shadow_mix_max_blocking_cost": 10,
    "shadow_mix_max_candidates": 14,
    "shadow_priority_wmma_over_ds": False,
    "shadow_priority_wmma_over_salu": True,
    "shadow_priority_cvt_over_ds": True,
    "shadow_priority_cvt_over_salu": True,
    "shadow_priority_trans32_over_valu1c": True,
    "shadow_defer_trans32": True,
    "shadow_mix_trans32_min_valu1c": 0,
    "shadow_prefer_valu_over_salu_for_trans": True,
    "resource_priority_coexec_exposed_cycles": True,
    "resource_priority_coexec_windows_size": True,
    "resource_priority_coexec_producer": True,
    "use_shadow_mix_rules": False,
}


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def sample_local(base: Dict[str, Any]):
    return {
        "postra": base["postra"] if random.random() < 0.8 else (not base["postra"]),
        # Latency/resource parameters
        "resource_balancing": clamp(
            base["resource_balancing"] + random.choice([-25, -10, 0, 10, 25]), 0, 300
        ),
        "ds_latency": clamp(base["ds_latency"] + random.randint(-10, 10), 10, 120),
        "ds_latency_split": clamp(base["ds_latency_split"] + random.randint(-2, 2), 0, 20),
        "ds_fifo_latency": clamp(base["ds_fifo_latency"] + random.randint(-10, 10), 10, 120),
        "signal_latency": clamp(base["signal_latency"] + random.randint(-8, 8), 0, 80),
        "ds_fence_latency": clamp(base["ds_fence_latency"] + random.randint(-10, 10), 10, 120),
        "ds_fifo_size": clamp(base["ds_fifo_size"] + random.randint(-2, 2), 2, 24),
        "ignore_valu": base["ignore_valu"] if random.random() < 0.85 else (not base["ignore_valu"]),
        "avoid_exp_final_islot": base["avoid_exp_final_islot"] if random.random() < 0.85 else (not base["avoid_exp_final_islot"]),
        # Shadow mix parameters
        "shadow_mix": base["shadow_mix"] if random.random() < 0.9 else (not base["shadow_mix"]),
        "shadow_mix_wmma_min_valu1c": clamp(base["shadow_mix_wmma_min_valu1c"] + random.randint(-1, 1), 0, 8),
        "shadow_mix_wmma_min_ds": clamp(base["shadow_mix_wmma_min_ds"] + random.randint(-1, 1), 0, 8),
        "shadow_mix_wmma_min_salu": clamp(base["shadow_mix_wmma_min_salu"] + random.randint(-1, 1), 0, 6),
        "shadow_mix_lookahead_depth": clamp(base["shadow_mix_lookahead_depth"] + random.randint(-2, 2), 0, 20),
        "shadow_mix_max_blocking_cost": clamp(base["shadow_mix_max_blocking_cost"] + random.randint(-2, 2), 2, 30),
        "shadow_mix_max_candidates": clamp(base["shadow_mix_max_candidates"] + random.randint(-2, 2), 2, 30),
        # Shadow priority toggles
        "shadow_priority_wmma_over_ds": base["shadow_priority_wmma_over_ds"] if random.random() < 0.85 else (not base["shadow_priority_wmma_over_ds"]),
        "shadow_priority_wmma_over_salu": base["shadow_priority_wmma_over_salu"] if random.random() < 0.85 else (not base["shadow_priority_wmma_over_salu"]),
        "shadow_priority_cvt_over_ds": base["shadow_priority_cvt_over_ds"] if random.random() < 0.85 else (not base["shadow_priority_cvt_over_ds"]),
        "shadow_priority_cvt_over_salu": base["shadow_priority_cvt_over_salu"] if random.random() < 0.85 else (not base["shadow_priority_cvt_over_salu"]),
        "shadow_priority_trans32_over_valu1c": base["shadow_priority_trans32_over_valu1c"] if random.random() < 0.85 else (not base["shadow_priority_trans32_over_valu1c"]),
        "shadow_defer_trans32": base["shadow_defer_trans32"] if random.random() < 0.85 else (not base["shadow_defer_trans32"]),
        "shadow_mix_trans32_min_valu1c": clamp(base["shadow_mix_trans32_min_valu1c"] + random.randint(-1, 1), 0, 6),
        "shadow_prefer_valu_over_salu_for_trans": base["shadow_prefer_valu_over_salu_for_trans"] if random.random() < 0.85 else (not base["shadow_prefer_valu_over_salu_for_trans"]),
        "resource_priority_coexec_exposed_cycles": base["resource_priority_coexec_exposed_cycles"] if random.random() < 0.85 else (not base["resource_priority_coexec_exposed_cycles"]),
        "resource_priority_coexec_windows_size": base["resource_priority_coexec_windows_size"] if random.random() < 0.85 else (not base["resource_priority_coexec_windows_size"]),
        "resource_priority_coexec_producer": base["resource_priority_coexec_producer"] if random.random() < 0.85 else (not base["resource_priority_coexec_producer"]),
        "use_shadow_mix_rules": base["use_shadow_mix_rules"] if random.random() < 0.85 else (not base["use_shadow_mix_rules"]),

    }


def build_llc_cmd(p: Dict[str, Any], output_file: str = "res.s", input_file: str = BASE_LL) -> str:
    """Build llc command string from parameters."""
    return (
        f"{LLC} -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1250 -amdgpu-enable-static-simulator -amdgpu-static-sim-trip-count=64 "
        f"-enable-post-misched={'false' if p['postra'] else 'false'} -o {output_file} {input_file} "
        f"-amdgpu-resource-balancing={p['resource_balancing']} "
        f"-amdgpu-ds-latency={p['ds_latency']} "
        f"-amdgpu-ds-latency-split={p['ds_latency_split']} "
        f"-amdgpu-ds-fifo-latency={p['ds_fifo_latency']} "
        f"-amdgpu-signal-latency={p['signal_latency']} "
        f"-amdgpu-ds-fence-latency={p['ds_fence_latency']} "
        f"-amdgpu-ds-fifo-size={p['ds_fifo_size']} "
        f"-amdgpu-ignore-valu-resource-balancing={'true' if p['ignore_valu'] else 'false'} "
        f"-amdgpu-avoid-exp-final-islot={'true' if p['avoid_exp_final_islot'] else 'false'} "
        f"-amdgpu-shadow-mix={'true' if p['shadow_mix'] else 'false'} "
        f"-amdgpu-shadow-mix-wmma-min-valu1c={p['shadow_mix_wmma_min_valu1c']} "
        f"-amdgpu-shadow-mix-wmma-min-ds={p['shadow_mix_wmma_min_ds']} "
        f"-amdgpu-shadow-mix-wmma-min-salu={p['shadow_mix_wmma_min_salu']} "
        f"-amdgpu-shadow-mix-lookahead-depth={p['shadow_mix_lookahead_depth']} "
        f"-amdgpu-shadow-mix-max-blocking-cost={p['shadow_mix_max_blocking_cost']} "
        f"-amdgpu-shadow-mix-max-candidates={p['shadow_mix_max_candidates']} "
        f"-amdgpu-shadow-priority-wmma-over-ds={'true' if p['shadow_priority_wmma_over_ds'] else 'false'} "
        f"-amdgpu-shadow-priority-wmma-over-salu={'true' if p['shadow_priority_wmma_over_salu'] else 'false'} "
        f"-amdgpu-shadow-priority-cvt-over-ds={'true' if p['shadow_priority_cvt_over_ds'] else 'false'} "
        f"-amdgpu-shadow-priority-cvt-over-salu={'true' if p['shadow_priority_cvt_over_salu'] else 'false'} "
        f"-amdgpu-shadow-priority-trans32-over-valu1c={'true' if p['shadow_priority_trans32_over_valu1c'] else 'false'} "
        f"-amdgpu-shadow-defer-trans32={'true' if p['shadow_defer_trans32'] else 'false'} "
        f"-amdgpu-shadow-mix-trans32-min-valu1c={p['shadow_mix_trans32_min_valu1c']} "
        f"-amdgpu-shadow-prefer-valu-over-salu-for-trans={'true' if p['shadow_prefer_valu_over_salu_for_trans'] else 'false'}"
        f"-amdgpu-resource-priority-coexec-producer={'true' if p['resource_priority_coexec_producer'] else 'false'} "
        f"-amdgpu-resource-priority-coexec-windows-size={'true' if p['resource_priority_coexec_windows_size'] else 'false'} "
        f"-amdgpu-resource-priority-coexec-exposed-cycles={'true' if p['resource_priority_coexec_exposed_cycles'] else 'false'} "
        f"-amdgpu-use-shadow-mix-rules={'true' if p['use_shadow_mix_rules'] else 'false'}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Parameter sweep for llc (integrated AMDGPU static simulator)"
    )
    parser.add_argument("--iters", type=int, default=100, help="number of trials to run")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: random each run)")
    parser.add_argument("--explore-frac", type=float, default=0.3, help="fraction of global random samples (rest local around best)")
    parser.add_argument("--log-file", type=str, default=None, help="optional CSV log file (appends)")
    parser.add_argument("--jobs", type=int, default=1, help="parallel jobs (recommended <= number of cores)")
    parser.add_argument("--print-every", type=int, default=1, help="print progress every N results (0 disables per-iter printing)")
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0, 2**32 - 1)
    random.seed(seed)
    print(f"Using seed: 0x{seed:08X}", flush=True)

    best = None  # best overall by scaled cycles
    best_loop = None  # best by warm-loop cycles

    def iter_trials():
        if args.iters <= 0:
            return
        yield BEST_SEED
        for _ in range(args.iters - 1):
            if random.random() < args.explore_frac:
                yield sample_params()
            else:
                yield sample_local(BEST_SEED)

    log_fh = None
    if args.log_file:
        log_exists = os.path.exists(args.log_file)
        log_fh = open(args.log_file, "a", buffering=1)
        if not log_exists:
            log_fh.write(
                "iter,scaled_cycles,warm_loop_cycles,score,raw_cycles,"
                "postra,resource_balancing,ds_latency,ds_latency_split,ds_fifo_latency,signal_latency,ds_fence_latency,ds_fifo_size,"
                "ignore_valu,avoid_exp_final_islot,shadow_mix,"
                "shadow_mix_wmma_min_valu1c,shadow_mix_wmma_min_ds,shadow_mix_wmma_min_salu,"
                "shadow_mix_lookahead_depth,shadow_mix_max_blocking_cost,shadow_mix_max_candidates,"
                "shadow_priority_wmma_over_ds,shadow_priority_wmma_over_salu,shadow_priority_cvt_over_ds,shadow_priority_cvt_over_salu,"
                "shadow_priority_trans32_over_valu1c,shadow_defer_trans32,shadow_mix_trans32_min_valu1c,shadow_prefer_valu_over_salu_for_trans,"
                "resource_priority_coexec_exposed_cycles,resource_priority_coexec_windows_size,resource_priority_coexec_producer,use_shadow_mix_rules\n"
            )

    def log_row(i, res):
        if not log_fh:
            return
        p = res["params"]
        log_fh.write(
            f"{i},{res['scaled_cycles']},{res['warm_loop_cycles']},{res['score']},{res.get('raw_cycles','')},"
            f"{p['postra']},{p['resource_balancing']},{p['ds_latency']},{p['ds_latency_split']},{p['ds_fifo_latency']},"
            f"{p['signal_latency']},{p['ds_fence_latency']},{p['ds_fifo_size']},"
            f"{p['ignore_valu']},{p['avoid_exp_final_islot']},{p['shadow_mix']},"
            f"{p['shadow_mix_wmma_min_valu1c']},{p['shadow_mix_wmma_min_ds']},{p['shadow_mix_wmma_min_salu']},"
            f"{p['shadow_mix_lookahead_depth']},{p['shadow_mix_max_blocking_cost']},{p['shadow_mix_max_candidates']},"
            f"{p['shadow_priority_wmma_over_ds']},{p['shadow_priority_wmma_over_salu']},{p['shadow_priority_cvt_over_ds']},{p['shadow_priority_cvt_over_salu']},"
            f"{p['shadow_priority_trans32_over_valu1c']},{p['shadow_defer_trans32']},{p['shadow_mix_trans32_min_valu1c']},{p['shadow_prefer_valu_over_salu_for_trans']}"
            f"{p['resource_priority_coexec_exposed_cycles']},{p['resource_priority_coexec_windows_size']},{p['resource_priority_coexec_producer']},{p['use_shadow_mix_rules']}\n"
        )

    def process_result(idx: int, params: Dict[str, Any], result: Dict[str, Any]):
        nonlocal best, best_loop
        scaled_cycles = result["scaled_cycles"]
        warm_loop_cycles = result["warm_loop_cycles"]
        score = result["score"]

        if best is None or score < best.get("score", float("inf")):
            best = result

        if best_loop is None or warm_loop_cycles < best_loop.get("warm_loop_cycles", float("inf")):
            best_loop = result

        log_row(idx, result)

        if args.print_every and (idx == 1 or (idx % args.print_every) == 0):
            p = params
            print(
                f"iter {idx:3d}: scaled={scaled_cycles:7d} warm_loop={warm_loop_cycles:7d} score={score:7.0f} "
                f"(postRA={p['postra']} rb={p['resource_balancing']} ds_lat={p['ds_latency']} shadow_mix={p['shadow_mix']} "
                f"wmma_valu={p['shadow_mix_wmma_min_valu1c']} wmma_ds={p['shadow_mix_wmma_min_ds']})",
                flush=True,
            )

    if args.jobs <= 1:
        for idx, params in enumerate(iter_trials(), 1):
            output_s = f"res_{idx}.s"
            result, err = run_trial(params, output_s)
            if err:
                print(f"iter {idx}: error: {err}", flush=True)
                continue
            process_result(idx, params, result)
            try:
                os.remove(os.path.join(ROOT, output_s))
            except OSError:
                pass
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            in_flight = {}
            max_in_flight = max(args.jobs * 4, 1)

            def submit_one(idx: int, params: Dict[str, Any]):
                output_s = f"res_{idx}.s"
                fut = ex.submit(run_trial, params, output_s)
                in_flight[fut] = (idx, params, output_s)

            def drain_one():
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    idx, params, output_s = in_flight.pop(fut)
                    try:
                        result, err = fut.result()
                    except Exception as e:
                        print(f"iter {idx}: error: {e}", flush=True)
                        continue
                    if err:
                        print(f"iter {idx}: error: {err}", flush=True)
                        continue
                    process_result(idx, params, result)
                    try:
                        os.remove(os.path.join(ROOT, output_s))
                    except OSError:
                        pass

            for idx, params in enumerate(iter_trials(), 1):
                submit_one(idx, params)
                while len(in_flight) >= max_in_flight:
                    drain_one()

            while in_flight:
                drain_one()

    print("\nBest overall:")
    if best:
        p = best["params"]
        print(
            f"  scaled={best['scaled_cycles']} warm_loop={best['warm_loop_cycles']} raw={best.get('raw_cycles')} score={best['score']:.0f} params={p}"
        )
        print("  llc:")
        print(f"    {build_llc_cmd(p)}")
    else:
        print("  none")

    print("\nBest loop (lowest warm loop cycles):")
    if best_loop:
        p = best_loop["params"]
        print(
            f"  warm_loop={best_loop['warm_loop_cycles']} scaled={best_loop['scaled_cycles']} raw={best_loop.get('raw_cycles')} score={best_loop['score']:.0f} params={p}"
        )
        print("  llc:")
        print(f"    {build_llc_cmd(p)}")
    else:
        print("  none")

    if log_fh:
        if best:
            p = best["params"]
            log_fh.write(
                "# Best overall: "
                f"scaled={best['scaled_cycles']} warm_loop={best['warm_loop_cycles']} raw={best.get('raw_cycles')} score={best['score']:.0f} params={p}\n"
            )
            log_fh.write(f"# Best overall llc: {build_llc_cmd(p)}\n")

        if best_loop:
            p = best_loop["params"]
            log_fh.write(
                "# Best loop: "
                f"warm_loop={best_loop['warm_loop_cycles']} scaled={best_loop['scaled_cycles']} raw={best_loop.get('raw_cycles')} score={best_loop['score']:.0f} params={p}\n"
            )
            log_fh.write(f"# Best loop llc: {build_llc_cmd(p)}\n")

        log_fh.write("\n# Final seed: 0x%08X\n" % seed)
        log_fh.close()


if __name__ == "__main__":
    sys.exit(main())

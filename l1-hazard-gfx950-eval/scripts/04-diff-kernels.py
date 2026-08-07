#!/usr/bin/env python3
"""Show which kernels the pass actually changes.

Compiles one AITER benchmark under both Triton builds with TRITON_KERNEL_DUMP,
then compares the generated .amdgcn per kernel. The `.file` directive is ignored
because it only records the source worktree path.

This is the check that showed the pass leaves most kernels byte-identical: it
only alters code where the modelled cache budget is tight enough to throttle,
which on gfx950 means the occupancy-1 kernels.

Usage: python3 04-diff-kernels.py bench_gemm_a16w8_blockscale [<workdir>]
"""
import os
import re
import subprocess
import sys

BENCH = sys.argv[1] if len(sys.argv) > 1 else "bench_gemm_a16w8_blockscale"
WORK = sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
    "WORK", os.path.expanduser("~/l1-hazard-eval"))
AITER = os.environ.get("AITER", os.path.expanduser("~/aiter"))
GPU = os.environ.get("GPU", "7")


def dump(cfg):
    out = f"/tmp/l1ab_dump_{cfg}"
    subprocess.run(["rm", "-rf", out, f"/tmp/l1ab_cache_{cfg}"], check=False)
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ,
               PYTHONPATH=f"{WORK}/triton_l1{cfg}/python:{AITER}",
               ROCR_VISIBLE_DEVICES=GPU,
               TRITON_CACHE_DIR=f"/tmp/l1ab_cache_{cfg}",
               TRITON_KERNEL_DUMP="1", TRITON_DUMP_DIR=out)
    subprocess.run([sys.executable,
                    f"{AITER}/op_tests/op_benchmarks/triton/{BENCH}.py"],
                   env=env, cwd=AITER, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, timeout=1800)
    return out


def body(path):
    return [l for l in open(path, errors="ignore") if ".file" not in l]


def info(path):
    d = {}
    for ln in open(path, errors="ignore"):
        for k in ("Occupancy", "NumVgprs", "NumAgprs", "ScratchSize"):
            m = re.match(rf"^;\s*{k}:\s*(\d+)", ln)
            if m:
                d[k] = m.group(1)
        m = re.search(r"GRID_MN_(\d+)", ln)
        if m:
            d.setdefault("grid", m.group(1))
    return d


off, on = dump("off"), dump("on")
same = changed = 0
print(f"{'kernel':10} {'grid':>6} {'occ':>4} {'vgpr':>5}  status")
for root, _, files in sorted(os.walk(off)):
    for fn in files:
        if not fn.endswith(".amdgcn"):
            continue
        a = os.path.join(root, fn)
        b = os.path.join(on, os.path.relpath(a, off))
        if not os.path.exists(b):
            continue
        i = info(a)
        h = os.path.basename(os.path.dirname(a))[:8]
        if body(a) == body(b):
            same += 1
            status = "identical"
        else:
            changed += 1
            status = "CHANGED"
        print(f"{h:10} {i.get('grid','?'):>6} {i.get('Occupancy','?'):>4} "
              f"{i.get('NumVgprs','?'):>5}  {status}")
print(f"\nidentical={same}  changed={changed}")
print("Deltas measured on identical kernels are run-to-run noise, not pass effects.")

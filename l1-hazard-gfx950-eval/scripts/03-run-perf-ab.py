#!/usr/bin/env python3
"""A/B performance run for the AMDGPU L1 cache-capacity hazard MISched change.

Both configs use the SAME triton commit (bf64a5db1b, which pins LLVM e2a39f504);
only the LLVM differs:

  l1off -> LLVM e2a39f504            (base, none of the 4 MISched commits)
  l1on  -> LLVM 7c2075cb8 (branch)   (recognizer added AND enabled by default)

Perf only. Reuses aiter's run_all_benchmarks for the bench list, runner and the
xlsx report, but switches triton via editable installs instead of wheels (both
builds are the same commit, so wheel labels would collide).

Usage:
  python3 run_perf_l1_ab.py                      # full bench set
  python3 run_perf_l1_ab.py --tests bench_mha bench_gemm_a16w16
  python3 run_perf_l1_ab.py --sheet gemm
  python3 run_perf_l1_ab.py --skip-run           # regenerate xlsx from logs
"""
import argparse
import os
import subprocess
import sys

AITER = os.environ.get("AITER", os.path.expanduser("~/aiter"))
sys.path.insert(0, AITER)
import run_all_benchmarks as R  # noqa: E402

WORK = os.environ.get("WORK", os.path.expanduser("~/l1-hazard-eval"))
CONFIGS = [
    ("l1off", f"{WORK}/triton_l1off"),
    ("l1on", f"{WORK}/triton_l1on"),
]
BASE_LOG_DIR = os.environ.get("L1AB_LOG_DIR", f"{WORK}/bench_logs_l1ab")


def activate(triton_dir):
    """Make this triton build the active `import triton` (editable install)."""
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", triton_dir,
         "--no-deps", "--no-build-isolation", "-q"],
        check=True,
    )
    R.clear_triton_cache()
    R.print_triton_version()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tests", nargs="+", help="only these bench names")
    p.add_argument("--sheet", help="only this category (gemm, attention, ...)")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--gpu", default=os.environ.get("GPU", "7"))
    p.add_argument("--skip-run", action="store_true")
    p.add_argument("--only", help="run only this config label (l1on/l1off); reuse existing logs for the other")
    p.add_argument("--repeat", type=int, default=1, help="passes per config; >1 aggregates medians")
    p.add_argument("--output", default=f"{WORK}/l1_hazard_perf.xlsx")
    args = p.parse_args()

    tests = [(n, a) for (n, a, cat) in R.BENCH_TESTS
             if (not args.tests or n in args.tests)
             and (not args.sheet or cat == args.sheet)]
    if not tests:
        sys.exit("no tests selected")

    # Same GPU for both configs so the comparison is apples-to-apples.
    os.environ["ROCR_VISIBLE_DEVICES"] = args.gpu

    labels = [lbl for lbl, _ in CONFIGS]
    log_dirs = [os.path.join(BASE_LOG_DIR, f"rerun_{lbl}") for lbl in labels]
    all_results = []

    for (label, triton_dir), log_dir in zip(CONFIGS, log_dirs):
        print(f"\n{'=' * 60}\n{label}: {triton_dir}\n{'=' * 60}")
        if args.skip_run or (args.only and label != args.only):
            print(f"  (reusing existing logs for {label})")
            all_results.append({n: "PASS" for n, _ in tests})
            continue
        activate(triton_dir)
        if args.repeat > 1:
            all_results.append(
                R.run_tests_repeated(tests, log_dir, args.timeout, args.repeat))
        else:
            all_results.append(R.run_tests(tests, log_dir, args.timeout))

    R.generate_excel(labels, log_dirs, all_results, args.output)
    print(f"\nreport: {args.output}")
    print(f"logs:   {BASE_LOG_DIR}/rerun_<label>/")


if __name__ == "__main__":
    main()

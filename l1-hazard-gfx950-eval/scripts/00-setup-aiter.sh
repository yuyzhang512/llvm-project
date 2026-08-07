#!/bin/bash
# Check out and build the AITER tree the benchmarks come from.
#
# All numbers in ../README.md were taken at aiter 8e1308d3, which lives on
# release-test/triton-afp4wfp4 in the yuyzhang512 fork. A recent ROCm/aiter main
# also works, but the benchmark list and the per-shape sweeps differ slightly,
# so the tables will not line up point for point.
#
# The ops are built against whatever Triton is installed at the time
# (AITER_USE_SYSTEM_TRITON=1), and 03-run-perf-ab.py swaps the Triton build
# underneath per configuration. The native ops do not need rebuilding between
# the two configurations.
#
# Usage: bash 00-setup-aiter.sh [<aiter-dir>]
set -eu
AITER=${1:-${AITER:-$HOME/aiter}}
COMMIT=${AITER_COMMIT:-8e1308d30aa7de808367311411e4c19f33947324}
FORK=${AITER_FORK:-https://github.com/yuyzhang512/aiter.git}
BRANCH=${AITER_BRANCH:-release-test/triton-afp4wfp4}

if [ ! -d "$AITER/.git" ]; then
  git clone "$FORK" "$AITER"
fi
cd "$AITER"
git fetch origin "$BRANCH" || git fetch "$FORK" "$BRANCH"
git checkout -f "$COMMIT"
git submodule update --init --recursive

# Reporting and test deps used by run_all_benchmarks.py.
pip install -q pytest openpyxl psutil llnl-hatchet

# Builds the native/JIT ops in place. Keeps the currently installed Triton.
AITER_USE_SYSTEM_TRITON=1 python3 setup.py develop

python3 -c "import aiter; print('aiter ok:', aiter.__file__)"
echo "aiter ready at $AITER ($(git rev-parse --short HEAD))"
echo "export AITER=$AITER before running 03-run-perf-ab.py"

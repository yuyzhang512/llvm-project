#!/bin/bash
# Cross-check the occupancy the recognizer uses against the occupancy the
# generated code actually achieves.
#
# The patch prints one line per function under AMDGPU_L1_DEBUG:
#   L1Hazard <kernel>: occ=N bytes=128*N speed=N/1 latency=256
# The AsmPrinter records the achieved value as "; Occupancy: N".
# The two distributions should match.
#
# Usage: bash 05-check-occupancy.sh [<bench>] [<workdir>]
set -eu
BENCH=${1:-bench_gemm_a16w8_blockscale}
WORK=${2:-${WORK:-$HOME/l1-hazard-eval}}
AITER=${AITER:-$HOME/aiter}
GPU=${GPU:-7}

rm -rf /tmp/l1occ_cache /tmp/l1occ_dump
mkdir -p /tmp/l1occ_dump
cd "$AITER"
PYTHONPATH="$WORK/triton_l1on/python:$AITER" \
ROCR_VISIBLE_DEVICES=$GPU TRITON_CACHE_DIR=/tmp/l1occ_cache \
AMDGPU_L1_DEBUG=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/l1occ_dump \
  python3 "op_tests/op_benchmarks/triton/$BENCH.py" > /tmp/l1occ.log 2>&1

echo "occupancy used by the pass:"
grep -oE "occ=[0-9]+ bytes=[0-9]+ speed=[0-9]+/[0-9]+" /tmp/l1occ.log |
  sort | uniq -c | sort -rn

echo
echo "occupancy achieved by the generated code:"
grep -h "^; Occupancy:" $(find /tmp/l1occ_dump -name '*.amdgcn') | sort | uniq -c

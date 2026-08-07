#!/bin/bash
# Build the same Triton commit twice, once against each LLVM install.
#
# Triton bf64a5db1b is used because cmake/llvm-info.json (the file that decides
# which LLVM Triton links against) pins 850a2b1b, which is the base of the
# branch under test. Note that cmake/llvm-build-info.json names a different,
# newer commit: that is only a package being prepared, not the LLVM in use.
# Building against that one fails with MLIR API errors.
#
# Usage: bash 02-build-triton.sh [<workdir>]
set -eu
WORK=${1:-${WORK:-$HOME/l1-hazard-eval}}
TRITON_COMMIT=bf64a5db1b
JOBS=${JOBS:-128}

pip install "nanobind==2.10.2" -q   # build dep, not fetched under --no-build-isolation

for cfg in on off; do
  SRC=$WORK/triton_l1$cfg
  if [ ! -d "$SRC" ]; then
    git clone https://github.com/triton-lang/triton.git "$SRC"
    git -C "$SRC" checkout -q "$TRITON_COMMIT"
  fi
  cd "$SRC"
  rm -rf build ./*.egg-info
  # AMD backend only: the nvidia backend needs a CUDA-side clang++ we do not care about.
  TRITON_CODEGEN_BACKENDS="amd" \
  TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
  LLVM_INCLUDE_DIRS="$WORK/install_850_$cfg/include" \
  LLVM_LIBRARY_DIR="$WORK/install_850_$cfg/lib" \
  LLVM_SYSPATH="$WORK/install_850_$cfg" \
  MAX_JOBS=$JOBS \
    pip install -e . --no-deps --no-build-isolation
  echo "built triton_l1$cfg against install_850_$cfg"
done

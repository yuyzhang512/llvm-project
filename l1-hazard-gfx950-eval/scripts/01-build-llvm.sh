#!/bin/bash
# Build the two LLVMs the A/B compares:
#   install_850_off : pure 850a2b1b            (base open source, what Triton pins)
#   install_850_on  : 850a2b1b + the 4 MISched L1-hazard commits
#
# Both come from one source tree and one build dir: the ON build is done first,
# then the tree is reset to the base commit and ninja rebuilds only the AMDGPU
# objects, which takes a couple of minutes instead of a full build.
#
# The build uses its own clone ($WORK/llvm-src) so that checking out the base
# commit cannot delete this script from under the running shell.
#
# Usage: bash 01-build-llvm.sh [<workdir>]
set -eu
WORK=${1:-${WORK:-$HOME/l1-hazard-eval}}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$WORK/llvm-src
BUILD=$WORK/build_l1ab
BASE=850a2b1b975c061ae0fc982ba68064d305485cb2   # Triton bf64a5db1b pins this
JOBS=${JOBS:-128}
LINKJOBS=${LINKJOBS:-32}

# The 4 commits under test, oldest first.
PASS_COMMITS="4a5ad7bd6 8d6faf439 419d2af87 7c2075cb8"

# Copy out of the source tree: git checkout below rewrites the worktree.
PATCH=$(mktemp /tmp/l1params.XXXX.patch)
cp "$SCRIPT_DIR/../patches/0001-occupancy-scaled-l1-params.patch" "$PATCH"

mkdir -p "$WORK"
if [ ! -d "$SRC" ]; then
  git clone --branch pr184657-l1on --single-branch --depth 50 \
    https://github.com/yuyzhang512/llvm-project.git "$SRC"
fi
cd "$SRC"
git fetch --depth 60 https://github.com/llvm/llvm-project.git "$BASE"

# ---- ON: base + the 4 commits + the occupancy-scaled parameters ----
git checkout -f "$BASE" --quiet
for c in $PASS_COMMITS; do
  git cherry-pick --no-commit "$c"
  git commit -q -m "cherry-pick $c"
done
git apply "$PATCH"
ON_TIP=$(git rev-parse HEAD)

cmake -G Ninja -S "$SRC/llvm" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_OPTIMIZED_TABLEGEN=ON -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_PARALLEL_COMPILE_JOBS="$JOBS" -DLLVM_PARALLEL_LINK_JOBS="$LINKJOBS" \
  -DCMAKE_INSTALL_PREFIX="$WORK/install_850_on"
ninja -C "$BUILD" -j "$JOBS"
cmake --install "$BUILD" --prefix "$WORK/install_850_on"

# ---- OFF: pure base, incremental rebuild of the AMDGPU objects only ----
git checkout -f "$BASE" --quiet     # also drops the parameter patch
ninja -C "$BUILD" -j "$JOBS"
cmake --install "$BUILD" --prefix "$WORK/install_850_off"

# Triton's nvidia backend looks for clang++ inside the LLVM install even when
# only the AMD backend is built.
for p in "$WORK/install_850_on" "$WORK/install_850_off"; do
  for tool in clang clang++; do
    [ -e "$p/bin/$tool" ] || ln -sf "$(command -v "$tool")" "$p/bin/$tool" 2>/dev/null || true
  done
done
rm -f "$PATCH"

echo "ON  (base + pass): $WORK/install_850_on   ($ON_TIP)"
echo "OFF (base)       : $WORK/install_850_off  ($BASE)"

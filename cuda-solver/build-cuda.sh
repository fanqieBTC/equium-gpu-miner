#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

NVCC="${NVCC:-nvcc}"
if ! command -v "$NVCC" >/dev/null 2>&1; then
  for p in /usr/local/cuda/bin/nvcc /opt/cuda/bin/nvcc; do
    if [[ -x "$p" ]]; then
      NVCC="$p"
      break
    fi
  done
fi

if ! command -v "$NVCC" >/dev/null 2>&1 && [[ ! -x "$NVCC" ]]; then
  echo "nvcc not found. Install the CUDA compiler first."
  exit 1
fi

ARCH="${CUDA_ARCH:-sm_89}"

echo "building equium-cuda-solver for ${ARCH} with ${NVCC}"
"$NVCC" -std=c++17 -O3 -arch="$ARCH" \
  equium-cuda-solver.cu \
  -lssl -lcrypto \
  -o equium-cuda-solver

echo "built ./equium-cuda-solver"

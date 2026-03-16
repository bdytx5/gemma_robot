#!/usr/bin/env bash
# inst_flash_attn.sh — install flash-attn from source using --no-build-isolation
# Pre-built wheels have C++ ABI mismatches; source build avoids this.
set -euo pipefail

if ! command -v nvcc &>/dev/null; then
    export PATH="/usr/local/cuda/bin:$PATH"
fi
if [ -z "${CUDA_HOME:-}" ]; then
    export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
fi

echo "=== flash-attn installer ==="
echo "  CUDA_HOME : $CUDA_HOME"
echo "  MAX_JOBS  : ${MAX_JOBS:-4}"
echo "  (build takes 10-30 min)"
echo ""

MAX_JOBS="${MAX_JOBS:-4}"
export MAX_JOBS

pip install flash-attn --no-build-isolation

echo ""
python -c "import flash_attn; print('flash_attn version:', flash_attn.__version__)"
echo "Done."

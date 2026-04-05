#!/usr/bin/env bash
# Install the flash_attn stub shim into the active Python's site-packages.
# This is needed for RTX 5090 (sm_120) where real flash_attn CUDA kernels
# don't compile. The shim lets transformers import flash_attn without error
# while we use SDPA for all attention.
set -e

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
SHIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/flash_attn"

echo "Installing flash_attn shim to $SITE/flash_attn ..."
rm -rf "$SITE/flash_attn"
cp -r "$SHIM_DIR" "$SITE/flash_attn"

# Create dist-info so importlib.metadata.version("flash_attn") works
DIST="$SITE/flash_attn-2.7.4.post1.dist-info"
mkdir -p "$DIST"
cat > "$DIST/METADATA" <<EOF
Metadata-Version: 2.1
Name: flash-attn
Version: 2.7.4.post1
Summary: SDPA stub shim for sm_120 (RTX 5090)
EOF
cat > "$DIST/RECORD" <<EOF
flash_attn/__init__.py,,
EOF

echo "Done. flash_attn shim installed."

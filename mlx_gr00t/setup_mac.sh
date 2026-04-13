#!/usr/bin/env bash
# setup_mac.sh — dev environment setup (Mac, Apple Silicon)
# Use this when working on the code.  For end-user installs use install.sh instead.
set -euo pipefail

ENV_NAME="mlx_robot"
PYTHON_VER="3.10"

C='\033[0;36m'; G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
log()  { echo -e "${C}==> $*${N}"; }
ok()   { echo -e "${G}    ✓ $*${N}"; }
warn() { echo -e "${Y}    ! $*${N}"; }
die()  { echo -e "${R}\nERROR: $*${N}"; exit 1; }

# ── find conda ─────────────────────────────────────────────────────────────────
CONDA=""
for c in \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "$HOME/opt/miniconda3/bin/conda" \
    "/opt/homebrew/bin/conda"
do
    [ -f "$c" ] && CONDA="$c" && break
done
[ -z "$CONDA" ] && command -v conda &>/dev/null && CONDA=$(command -v conda)
[ -z "$CONDA" ] && die "conda not found — run install.sh to get Miniforge first"
ok "conda: $CONDA"

# ── create / reuse env ─────────────────────────────────────────────────────────
if "$CONDA" env list | grep -q "^$ENV_NAME "; then
    warn "Env '$ENV_NAME' already exists — updating packages"
else
    log "Creating env '$ENV_NAME' (Python $PYTHON_VER)..."
    "$CONDA" create -y -n "$ENV_NAME" python="$PYTHON_VER" -q
    ok "Env created"
fi

# resolve pip / python in the env
CONDA_BASE=$("$CONDA" info --base 2>/dev/null || echo "")
PIP=""
for p in \
    "$HOME/miniforge3/envs/$ENV_NAME/bin/pip" \
    "$HOME/miniconda3/envs/$ENV_NAME/bin/pip" \
    "$HOME/anaconda3/envs/$ENV_NAME/bin/pip" \
    "$HOME/opt/miniconda3/envs/$ENV_NAME/bin/pip" \
    "${CONDA_BASE}/envs/$ENV_NAME/bin/pip"
do
    [ -f "$p" ] && PIP="$p" && break
done
[ -z "$PIP" ] && die "Can't find pip in '$ENV_NAME' — try: $CONDA env list"

# ── install everything (app + dev/validation tools) ───────────────────────────
log "Installing packages from requirements.txt + dev extras..."
"$PIP" install --upgrade pip -q

# app runtime (uncommented lines from requirements.txt)
"$PIP" install \
    "mlx>=0.31.0" \
    "mlx-lm>=0.29.0" \
    "transformers>=4.57.0,<5.0" \
    "huggingface_hub" \
    "safetensors" \
    "numpy" \
    "Pillow" \
    "einops>=0.8.0" \
    "requests" \
    "msgpack" \
    "tokenizers" \
    "sentencepiece" \
    "wandb" \
    "weave" \
    "moviepy" \
    "pygame>=2.1.0" \
    || die "Package install failed"

# dev / validation extras (torch for validate_mlx_vs_torch.py, extract_llm.py, etc.)
log "Installing dev extras (torch — this can take a few minutes)..."
"$PIP" install "torch>=2.3.0" torchvision || warn "torch install failed — dev scripts won't work but the app is fine"

ok "All packages installed"

echo ""
echo "  Activate with:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Run the app:"
echo "    python app.py"
echo ""
echo "  Weight download (first time):"
echo "    python -c \"from huggingface_hub import snapshot_download; snapshot_download('youngbrett48/ntl_gemma_robot_mlx')\""

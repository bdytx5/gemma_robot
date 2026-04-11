#!/bin/bash
# install.sh — GemmaRobot one-line installer
# Works on a fresh macOS account: no Xcode, no conda, no dev tools required.
# Usage: bash <(curl -fsSL https://raw.githubusercontent.com/bdytx5/gemma_robot/main/mlx_gr00t/install.sh)
set -euo pipefail

INSTALL_DIR="$HOME/gemma_robot"
REPO_GIT="https://github.com/bdytx5/gemma_robot.git"
REPO_ZIP="https://github.com/bdytx5/gemma_robot/archive/refs/heads/main.zip"
ENV_NAME="mlx_robot"

# ── colors ─────────────────────────────────────────────────────────────────────
C='\033[0;36m'; G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
log()  { echo -e "${C}==> $*${N}"; }
ok()   { echo -e "${G}    ✓ $*${N}"; }
warn() { echo -e "${Y}    ! $*${N}"; }
die()  { echo -e "${R}\nERROR: $*${N}"; echo -e "${R}Install failed. See output above for details.${N}"; exit 1; }

# ── error trap — prints the line that failed ───────────────────────────────────
trap 'echo -e "${R}\nInstall failed at line $LINENO: $BASH_COMMAND${N}" >&2' ERR

echo ""
echo "  GemmaRobot Installer"
echo "  ====================="
echo ""

# ── 1. get the repo ────────────────────────────────────────────────────────────
log "Getting repo..."

if [ -d "$INSTALL_DIR/.git" ]; then
    warn "Repo already at $INSTALL_DIR"
    if command -v git &>/dev/null; then
        git -C "$INSTALL_DIR" pull --ff-only 2>/dev/null && ok "Updated" || warn "Pull failed, using existing code"
    fi
elif command -v git &>/dev/null; then
    git clone --depth 1 "$REPO_GIT" "$INSTALL_DIR" || die "git clone failed"
    ok "Cloned"
else
    # no git (no Xcode CLI tools) — use curl + unzip, both built into macOS
    log "git not found — downloading zip via curl..."
    TMP_ZIP=$(mktemp /tmp/gemma_robot_XXXX.zip)
    curl -fsSL "$REPO_ZIP" -o "$TMP_ZIP" || die "Download failed — check your internet connection"
    TMP_DIR=$(mktemp -d /tmp/gemma_robot_XXXX)
    unzip -q "$TMP_ZIP" -d "$TMP_DIR" || die "Unzip failed"
    mkdir -p "$INSTALL_DIR"
    mv "$TMP_DIR"/gemma_robot-main/* "$INSTALL_DIR/" || die "Could not move files to $INSTALL_DIR"
    rm -rf "$TMP_ZIP" "$TMP_DIR"
    ok "Downloaded (no git needed)"
fi

cd "$INSTALL_DIR/mlx_gr00t"

# ── 2. ensure conda ────────────────────────────────────────────────────────────
log "Checking for conda..."

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

if [ -z "$CONDA" ] && command -v conda &>/dev/null; then
    CONDA=$(command -v conda)
fi

if [ -z "$CONDA" ]; then
    warn "conda not found — installing Miniforge3 (this is ~90MB)..."

    ARCH=$(uname -m)
    if [ "$ARCH" = "arm64" ]; then
        MF_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
    else
        MF_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
    fi

    MF_INSTALLER=$(mktemp /tmp/miniforge_XXXX.sh)
    curl -fsSL "$MF_URL" -o "$MF_INSTALLER" || die "Failed to download Miniforge from $MF_URL"
    bash "$MF_INSTALLER" -b -p "$HOME/miniforge3" || die "Miniforge install failed"
    rm -f "$MF_INSTALLER"

    CONDA="$HOME/miniforge3/bin/conda"
    ok "Miniforge installed at ~/miniforge3"
else
    ok "Found: $CONDA"
fi

# ── 3. create / update env ─────────────────────────────────────────────────────
log "Setting up Python env '$ENV_NAME'..."

if "$CONDA" env list | grep -q "^$ENV_NAME "; then
    ok "Env '$ENV_NAME' already exists"
else
    "$CONDA" create -y -n "$ENV_NAME" python=3.10 -q || die "Failed to create conda env '$ENV_NAME'"
    ok "Created env '$ENV_NAME' (Python 3.10)"
fi

# find pip in env
PIP=""
CONDA_BASE=$("$CONDA" info --base 2>/dev/null || echo "")
for p in \
    "$HOME/miniforge3/envs/$ENV_NAME/bin/pip" \
    "$HOME/miniconda3/envs/$ENV_NAME/bin/pip" \
    "$HOME/anaconda3/envs/$ENV_NAME/bin/pip" \
    "$HOME/opt/miniconda3/envs/$ENV_NAME/bin/pip" \
    "${CONDA_BASE}/envs/$ENV_NAME/bin/pip"
do
    [ -f "$p" ] && PIP="$p" && break
done

[ -z "$PIP" ] && die "Can't find pip in '$ENV_NAME' env — try: $CONDA env list"
PYTHON="${PIP%pip}python"

# ── 4. install runtime packages (no torch — not needed to run the app) ─────────
log "Installing packages (this takes ~2 min)..."

"$PIP" install --upgrade pip -q || die "pip upgrade failed"

"$PIP" install \
    "mlx>=0.22.0" \
    "mlx-lm>=0.20.0" \
    "transformers>=4.51.0,<5.0" \
    "huggingface_hub" \
    "safetensors" \
    "numpy" \
    "Pillow" \
    "requests" \
    "msgpack" \
    "tokenizers" \
    "sentencepiece" \
    "wandb" \
    "weave" \
    "moviepy" \
    || die "Package install failed — see pip output above"

ok "Packages installed"

# ── 5. check weights ───────────────────────────────────────────────────────────
log "Checking model weights..."

WEIGHTS_OK=false
if [ -f "gr00t_weights_mlx/meta.json" ] && \
   [ -f "gr00t_weights_mlx/vision.safetensors" ] && \
   [ -f "gr00t_weights_mlx/dit.safetensors" ] && \
   [ -d "gr00t_llm_mlx" ] && [ -n "$(ls -A gr00t_llm_mlx 2>/dev/null)" ]; then
    ok "Weights found"
    WEIGHTS_OK=true
else
    warn "Model weights not found."
    echo ""
    echo "  Weights must be exported once from the full PyTorch checkpoint."
    echo "  If you have them on another machine, copy them over:"
    echo ""
    echo "    scp -r <other_mac>:~/gemma_robot/mlx_gr00t/gr00t_weights_mlx \\"
    echo "        $INSTALL_DIR/mlx_gr00t/"
    echo "    scp -r <other_mac>:~/gemma_robot/mlx_gr00t/gr00t_llm_mlx \\"
    echo "        $INSTALL_DIR/mlx_gr00t/"
    echo ""
    echo "  Then re-run this script."
fi

# ── 6. build and open app ──────────────────────────────────────────────────────
if [ "$WEIGHTS_OK" = true ]; then
    log "Building GemmaRobot.app..."
    bash build_app.sh || die "App build failed — see output above"

    echo ""
    ok "Done!"
    echo ""

    DMG="$INSTALL_DIR/dist/GemmaRobot.dmg"
    APP="$INSTALL_DIR/dist/GemmaRobot.app"

    if [ -f "$DMG" ]; then
        echo "  Opening installer..."
        open "$DMG"
    elif [ -d "$APP" ]; then
        echo "  Opening app..."
        open "$APP"
    fi
else
    echo ""
    warn "Skipping app build — add weights first, then re-run:"
    echo ""
    echo "    bash $INSTALL_DIR/mlx_gr00t/install.sh"
    echo ""
fi

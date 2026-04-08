#!/bin/bash
# install.sh — one-line installer for GemmaRobot.app
# Usage: bash <(curl -fsSL https://raw.githubusercontent.com/bdytx5/gemma_robot/main/mlx_gr00t/install.sh)
set -e

INSTALL_DIR="$HOME/gemma_robot"
REPO="https://github.com/bdytx5/gemma_robot.git"

echo "==> GemmaRobot installer"

# ── clone ──────────────────────────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "==> Repo already exists at $INSTALL_DIR, pulling latest..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    echo "==> Cloning into $INSTALL_DIR ..."
    git clone --depth 1 "$REPO" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR/mlx_gr00t"

# ── find conda ─────────────────────────────────────────────────────────────────
CONDA=""
for c in \
    "$HOME/miniconda3/bin/conda" \
    "$HOME/anaconda3/bin/conda" \
    "$HOME/miniforge3/bin/conda" \
    "$HOME/opt/miniconda3/bin/conda" \
    "/opt/homebrew/bin/conda"
do
    [ -f "$c" ] && CONDA="$c" && break
done

if [ -z "$CONDA" ]; then
    echo ""
    echo "ERROR: conda not found."
    echo "Install Miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "==> Using conda: $CONDA"

# ── create / update mlx_robot env ─────────────────────────────────────────────
ENV_NAME="mlx_robot"
if "$CONDA" env list | grep -q "^$ENV_NAME "; then
    echo "==> Env '$ENV_NAME' already exists, updating packages..."
else
    echo "==> Creating env '$ENV_NAME' (Python 3.10)..."
    "$CONDA" create -y -n "$ENV_NAME" python=3.10
fi

# find pip in env
PIP=""
for p in \
    "$HOME/miniconda3/envs/$ENV_NAME/bin/pip" \
    "$HOME/anaconda3/envs/$ENV_NAME/bin/pip" \
    "$HOME/miniforge3/envs/$ENV_NAME/bin/pip" \
    "$HOME/opt/miniconda3/envs/$ENV_NAME/bin/pip"
do
    [ -f "$p" ] && PIP="$p" && break
done

if [ -z "$PIP" ]; then
    echo "ERROR: Could not find pip in '$ENV_NAME' env"
    exit 1
fi

echo "==> Installing dependencies from requirements.txt..."
"$PIP" install --upgrade pip -q
"$PIP" install -r requirements.txt -q
echo "    done."

# ── export MLX weights (skip if already done) ──────────────────────────────────
PYTHON="${PIP%pip}python"
if [ ! -f "gr00t_weights_mlx/meta.json" ]; then
    echo "==> Exporting MLX weights (first time, ~5 min)..."
    "$PYTHON" export_weights.py
else
    echo "==> MLX weights already exported, skipping."
fi

# ── build app ──────────────────────────────────────────────────────────────────
echo "==> Building GemmaRobot.app..."
bash build_app.sh

echo ""
echo "Done! Opening installer..."
open "$INSTALL_DIR/dist/GemmaRobot.dmg"

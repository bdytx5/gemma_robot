#!/bin/bash
# build_app.sh — builds GemmaRobot.app and GemmaRobot.dmg
set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MLX_DIR="$REPO_ROOT/mlx_gr00t"
OUT_DIR="$REPO_ROOT/dist"
APP_NAME="GemmaRobot"
APP_DIR="$OUT_DIR/$APP_NAME.app"

echo "Building $APP_NAME.app ..."
rm -rf "$APP_DIR"
mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources"

# ── Info.plist ─────────────────────────────────────────────────────────────
cat > "$APP_DIR/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>GemmaRobot</string>
  <key>CFBundleDisplayName</key>       <string>GemmaRobot</string>
  <key>CFBundleIdentifier</key>        <string>com.gemmarobot.eval</string>
  <key>CFBundleVersion</key>           <string>1.0</string>
  <key>CFBundleExecutable</key>        <string>launcher</string>
  <key>CFBundleIconFile</key>          <string>AppIcon</string>
  <key>LSMinimumSystemVersion</key>    <string>13.0</string>
  <key>NSHighResolutionCapable</key>   <true/>
  <key>NSAppleEventsUsageDescription</key>
    <string>GemmaRobot uses AppleScript to show setup dialogs.</string>
</dict>
</plist>
PLIST

# ── copy Python sources into Resources ────────────────────────────────────
cp "$MLX_DIR/app.py"           "$APP_DIR/Contents/Resources/"
cp "$MLX_DIR/gemma_vla.py"     "$APP_DIR/Contents/Resources/"
cp "$MLX_DIR/inference.py"     "$APP_DIR/Contents/Resources/"
cp "$MLX_DIR/dit_mlx.py"       "$APP_DIR/Contents/Resources/"
cp "$MLX_DIR/vision_mlx.py"    "$APP_DIR/Contents/Resources/"
cp "$MLX_DIR/extract_llm.py"   "$APP_DIR/Contents/Resources/"

# ── bundle pre-exported MLX weights ────────────────────────────────────────
echo "Copying model weights (this may take a moment)..."

cp -r "$MLX_DIR/gr00t_weights_mlx" "$APP_DIR/Contents/Resources/"
cp -r "$MLX_DIR/gr00t_llm_mlx"     "$APP_DIR/Contents/Resources/"

echo "  gr00t_weights_mlx: $(du -sh "$MLX_DIR/gr00t_weights_mlx" | cut -f1)"
echo "  gr00t_llm_mlx:     $(du -sh "$MLX_DIR/gr00t_llm_mlx" | cut -f1)"

# ── launcher shell script ─────────────────────────────────────────────────
cat > "$APP_DIR/Contents/MacOS/launcher" << 'LAUNCHER'
#!/bin/bash
# Launcher for GemmaRobot.app
# Finds or creates the mlx_robot conda env, then runs app.py

APP_SUPPORT="$HOME/Library/Application Support/GemmaRobot"
ENV_DIR="$APP_SUPPORT/env"
RESOURCES="$(dirname "$0")/../Resources"
REPO_ROOT="$(dirname "$0")/../../../../"

# Resolve the repo root (parent of mlx_gr00t folder)
# Structure: GemmaRobot.app is in dist/, sibling of mlx_gr00t/
DIST_DIR="$(dirname "$0")/../../.."
REPO_ROOT="$(cd "$DIST_DIR/.." && pwd)"

# Prefer the actual repo's env if it already exists
REPO_ENV="$REPO_ROOT/mlx_gr00t/../.."   # not reliable, skip

log() { echo "[GemmaRobot] $*" >&2; }

# ── find Python with mlx available ────────────────────────────────────────
find_python() {
    # 1. explicit conda env name
    for candidate in \
        "$HOME/miniconda3/envs/mlx_robot/bin/python" \
        "$HOME/anaconda3/envs/mlx_robot/bin/python" \
        "$HOME/miniforge3/envs/mlx_robot/bin/python" \
        "$HOME/opt/miniconda3/envs/mlx_robot/bin/python"
    do
        if [ -f "$candidate" ]; then
            if "$candidate" -c "import mlx" 2>/dev/null; then
                echo "$candidate"; return 0
            fi
        fi
    done

    # 2. active conda env
    if [ -n "$CONDA_PREFIX" ]; then
        PY="$CONDA_PREFIX/bin/python"
        if [ -f "$PY" ] && "$PY" -c "import mlx" 2>/dev/null; then
            echo "$PY"; return 0
        fi
    fi

    # 3. app-support env (created by this script on first run)
    if [ -f "$ENV_DIR/bin/python" ]; then
        echo "$ENV_DIR/bin/python"; return 0
    fi

    return 1
}

PYTHON=$(find_python)

if [ -z "$PYTHON" ]; then
    # ── first-run setup ────────────────────────────────────────────────────
    osascript -e 'display dialog "GemmaRobot: First-run setup.\n\nInstalling Python environment with MLX (Apple Silicon).\nThis will take 3-5 minutes." buttons {"Continue"} default button "Continue" with icon note' 2>/dev/null || true

    # find conda
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
        osascript -e 'display dialog "GemmaRobot requires conda (Miniconda or Anaconda).\n\nPlease install Miniconda from:\nhttps://docs.conda.io/en/latest/miniconda.html\n\nThen relaunch GemmaRobot." buttons {"OK"} default button "OK" with icon stop' 2>/dev/null || true
        exit 1
    fi

    log "Creating env at $ENV_DIR ..."
    mkdir -p "$APP_SUPPORT"
    "$CONDA" create -y -p "$ENV_DIR" python=3.10

    log "Installing dependencies..."
    "$ENV_DIR/bin/pip" install --upgrade pip
    "$ENV_DIR/bin/pip" install mlx mlx-lm \
        "transformers>=4.51.0,<5.0" \
        safetensors huggingface_hub \
        Pillow numpy requests msgpack tokenizers \
        wandb weave moviepy

    PYTHON="$ENV_DIR/bin/python"
fi

log "Using Python: $PYTHON"

# ── add repo paths ─────────────────────────────────────────────────────────
# Try to find the repo from common locations
for candidate in \
    "$HOME/Desktop/dev26/tutorials/gemma_robot" \
    "$HOME/Documents/gemma_robot" \
    "$HOME/gemma_robot"
do
    if [ -d "$candidate/Eagle" ] && [ -d "$candidate/Isaac-GR00T" ]; then
        REPO_ROOT="$candidate"
        break
    fi
done

export PYTHONPATH="$RESOURCES:$REPO_ROOT/Eagle/Eagle2_5:$REPO_ROOT/Isaac-GR00T:$PYTHONPATH"

exec "$PYTHON" "$RESOURCES/app.py"
LAUNCHER

chmod +x "$APP_DIR/Contents/MacOS/launcher"

echo "Built: $APP_DIR"

# ── optional: create .dmg ─────────────────────────────────────────────────
if command -v hdiutil &>/dev/null; then
    DMG="$OUT_DIR/$APP_NAME.dmg"
    echo "Creating $DMG ..."
    rm -f "$DMG"
    hdiutil create \
        -volname "$APP_NAME" \
        -srcfolder "$APP_DIR" \
        -ov -format UDZO \
        "$DMG"
    echo "Done: $DMG"
else
    echo "hdiutil not found — skipping .dmg"
fi

echo ""
echo "To run directly:  open \"$APP_DIR\""

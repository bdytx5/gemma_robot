#!/bin/bash
set -e

echo "============================================"
echo "Downloading all available GR00T datasets"
echo "============================================"
echo ""

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Check if dataset is fully downloaded:
#   - has meta/info.json
#   - has videos dir with content
#   - video count matches expected from info.json
is_complete() {
    local ds_path="$1"
    [ -f "$ds_path/meta/info.json" ] || return 1
    [ -d "$ds_path/videos" ] || return 1
    local expected=$(python3 -c "import json; print(json.load(open('$ds_path/meta/info.json')).get('total_videos', 0))" 2>/dev/null || echo "0")
    local actual=$(find "$ds_path/videos" -name "*.mp4" -type f 2>/dev/null | wc -l)
    [ "$actual" -ge "$expected" ]
}

# Robust download via git clone + git lfs pull with retries
# Usage: robust_git_download HF_REPO LOCAL_DIR
robust_git_download() {
    local repo="$1"
    local local_dir="$2"
    local url="https://huggingface.co/datasets/$repo"
    local max_retries=1000
    local attempt=0

    # If dir doesn't exist or has no .git, do a fresh clone
    if [ ! -d "$local_dir/.git" ]; then
        echo "  Cloning $repo..."
        rm -rf "$local_dir"
        git clone "$url" "$local_dir" 2>&1 || true
    fi

    # Retry git lfs pull until all files are fetched
    while [ $attempt -lt $max_retries ]; do
        attempt=$((attempt + 1))
        echo "  LFS pull attempt $attempt/$max_retries..."
        cd "$local_dir"
        git lfs pull 2>&1 || true
        cd "$REPO_ROOT"

        if is_complete "$local_dir"; then
            echo "  Download verified complete."
            return 0
        fi

        local expected=$(python3 -c "import json; print(json.load(open('$local_dir/meta/info.json')).get('total_videos', 0))" 2>/dev/null || echo "?")
        local actual=$(find "$local_dir/videos" -name "*.mp4" -type f 2>/dev/null | wc -l)
        echo "  Incomplete: $actual/$expected videos. Retrying in 30s..."
        sleep 30
    done

    echo "  WARNING: $repo may still be incomplete after $max_retries attempts"
    return 1
}

# For DROID which uses huggingface-cli with --include (subset download)
robust_hf_download() {
    local repo="$1"
    local local_dir="$2"
    shift 2
    local max_retries=10
    local attempt=0

    while [ $attempt -lt $max_retries ]; do
        attempt=$((attempt + 1))
        echo "  HF download attempt $attempt/$max_retries..."

        HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type dataset "$repo" \
            "$@" --local-dir "$local_dir" 2>&1 || true

        if [ -f "$local_dir/meta/info.json" ]; then
            echo "  Download verified (info.json exists)."
            return 0
        fi

        echo "  Incomplete. Waiting 60s before retry..."
        sleep 60
    done

    echo "  WARNING: $repo may still be incomplete after $max_retries attempts"
    return 1
}

# =============================================
# 1. Fractal (Google Robot) — oxe_google
# =============================================
if is_complete "$REPO_ROOT/examples/SimplerEnv/fractal20220817_data_lerobot"; then
    echo "[1/8] Fractal: COMPLETE, skipping."
else
    echo "[1/8] Downloading Fractal (Google Robot)..."
    robust_git_download "IPEC-COMMUNITY/fractal20220817_data_lerobot" \
        "$REPO_ROOT/examples/SimplerEnv/fractal20220817_data_lerobot"
fi
# Always ensure modality.json
cp "$REPO_ROOT/examples/SimplerEnv/fractal_modality.json" \
   "$REPO_ROOT/examples/SimplerEnv/fractal20220817_data_lerobot/meta/modality.json" 2>/dev/null || true
echo ""

# =============================================
# 2. Bridge (WidowX) — oxe_widowx
# =============================================
if is_complete "$REPO_ROOT/examples/SimplerEnv/bridge_orig_lerobot"; then
    echo "[2/8] Bridge: COMPLETE, skipping."
else
    echo "[2/8] Downloading Bridge (WidowX)..."
    robust_git_download "IPEC-COMMUNITY/bridge_orig_lerobot" \
        "$REPO_ROOT/examples/SimplerEnv/bridge_orig_lerobot"
fi
cp "$REPO_ROOT/examples/SimplerEnv/bridge_modality.json" \
   "$REPO_ROOT/examples/SimplerEnv/bridge_orig_lerobot/meta/modality.json" 2>/dev/null || true
echo ""

# =============================================
# 3. LIBERO 10 (Long) — libero_panda
# =============================================
if is_complete "$REPO_ROOT/examples/LIBERO/libero_10_no_noops_1.0.0_lerobot"; then
    echo "[3/8] LIBERO 10: COMPLETE, skipping."
else
    echo "[3/8] Downloading LIBERO 10..."
    robust_git_download "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot" \
        "$REPO_ROOT/examples/LIBERO/libero_10_no_noops_1.0.0_lerobot"
fi
cp "$REPO_ROOT/examples/LIBERO/modality.json" \
   "$REPO_ROOT/examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/meta/modality.json" 2>/dev/null || true
echo ""

# =============================================
# 4. LIBERO Goal — libero_panda
# =============================================
if is_complete "$REPO_ROOT/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot"; then
    echo "[4/8] LIBERO Goal: COMPLETE, skipping."
else
    echo "[4/8] Downloading LIBERO Goal..."
    robust_git_download "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot" \
        "$REPO_ROOT/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot"
fi
cp "$REPO_ROOT/examples/LIBERO/modality.json" \
   "$REPO_ROOT/examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/meta/modality.json" 2>/dev/null || true
echo ""

# =============================================
# 5. LIBERO Object — libero_panda
# =============================================
if is_complete "$REPO_ROOT/examples/LIBERO/libero_object_no_noops_1.0.0_lerobot"; then
    echo "[5/8] LIBERO Object: COMPLETE, skipping."
else
    echo "[5/8] Downloading LIBERO Object..."
    robust_git_download "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot" \
        "$REPO_ROOT/examples/LIBERO/libero_object_no_noops_1.0.0_lerobot"
fi
cp "$REPO_ROOT/examples/LIBERO/modality.json" \
   "$REPO_ROOT/examples/LIBERO/libero_object_no_noops_1.0.0_lerobot/meta/modality.json" 2>/dev/null || true
echo ""

# =============================================
# 6. LIBERO Spatial — libero_panda
# =============================================
if is_complete "$REPO_ROOT/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot"; then
    echo "[6/8] LIBERO Spatial: COMPLETE, skipping."
else
    echo "[6/8] Downloading LIBERO Spatial..."
    robust_git_download "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot" \
        "$REPO_ROOT/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot"
fi
cp "$REPO_ROOT/examples/LIBERO/modality.json" \
   "$REPO_ROOT/examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/meta/modality.json" 2>/dev/null || true
echo ""

# =============================================
# 7. DROID (subset: 2 chunks) — oxe_droid
# =============================================
if is_complete "$REPO_ROOT/examples/DROID/droid_subset"; then
    echo "[7/8] DROID: COMPLETE, skipping download."
else
    echo "[7/8] Downloading DROID subset (2 chunks)..."
    robust_hf_download "cadene/droid_1.0.1" \
        "$REPO_ROOT/examples/DROID/droid_subset" \
        --include "meta/*" "data/chunk-000/*" "data/chunk-001/*" \
                  "videos/chunk-000/*" "videos/chunk-001/*"
fi
echo "[7/8] Preparing DROID for GR00T..."
PYTHONPATH="$REPO_ROOT" python3 "$REPO_ROOT/scripts/prep_droid_for_gr00t.py" \
    --dataset_path "$REPO_ROOT/examples/DROID/droid_subset/"
echo "[7/8] DROID done."
echo ""

# =============================================
# 8. RoboCasa — robocasa_panda_omron
# =============================================
if is_complete "$REPO_ROOT/examples/robocasa/robocasa_mg_gr00t_300"; then
    echo "[8/8] RoboCasa: COMPLETE, skipping."
else
    echo "[8/8] Downloading RoboCasa (~37GB)..."
    robust_git_download "kimtaey/robocasa_mg_gr00t_300" \
        "$REPO_ROOT/examples/robocasa/robocasa_mg_gr00t_300"
fi
echo ""

# =============================================
# Final verification
# =============================================
echo "============================================"
echo "Final status:"
echo "============================================"
for d in \
    "examples/SimplerEnv/fractal20220817_data_lerobot" \
    "examples/SimplerEnv/bridge_orig_lerobot" \
    "examples/LIBERO/libero_10_no_noops_1.0.0_lerobot" \
    "examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot" \
    "examples/LIBERO/libero_object_no_noops_1.0.0_lerobot" \
    "examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot" \
    "examples/DROID/droid_subset" \
    "examples/robocasa/robocasa_mg_gr00t_300"; do
    name=$(basename "$d")
    sz=$(du -sh "$REPO_ROOT/$d" 2>/dev/null | cut -f1)
    if is_complete "$REPO_ROOT/$d"; then
        status="COMPLETE"
    else
        status="INCOMPLETE"
    fi
    printf "  %-45s %8s  %s\n" "$name" "$sz" "$status"
done
echo ""

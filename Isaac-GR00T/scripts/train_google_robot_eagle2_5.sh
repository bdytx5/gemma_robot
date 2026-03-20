#!/usr/bin/env bash
# train_google_robot_eagle2_5.sh
#
# End-to-end: download Fractal dataset (if needed) + finetune GR00T with your
# custom Eagle2.5 + Gemma3-270m backbone for Google robot tasks.
#
# Prerequisites:
#   - Eagle2.5 repo at ../Eagle/Eagle2_5 (relative to Isaac-GR00T/)
#   - Stage 2 checkpoint at HF: youngbrett48/train_stage2_gemma3_270m.sh
#     (or set BASE_MODEL_PATH to a local path)
#
# Usage:
#   bash scripts/train_google_robot_eagle2_5.sh [--good]
#
#   --good  Skip download entirely, use whatever data is already on disk
#
# Overrides:
#   NUM_GPUS        Number of GPUs (default: 1)
#   OUTPUT_DIR      Checkpoint output dir (default: ./output/gr00t-eagle2_5-fractal)
#   BASE_MODEL_PATH Eagle2.5 checkpoint (default: HF repo above)
#   HF_REPO_ID      If set, push checkpoints here
#   MAX_STEPS       Training steps (default: 20000)
#   DATASET_PATH    Override dataset location (skip auto-download)

set -e

# Parse --good flag
GOOD_TO_GO=0
for arg in "$@"; do
    if [ "$arg" = "--good" ]; then
        GOOD_TO_GO=1
    fi
done

# ── Config ────────────────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/gr00t-eagle2_5-fractal"}
MAX_STEPS=${MAX_STEPS:-20000}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-"youngbrett48/train_stage2_gemma3_270m.sh"}
EMBODIMENT="OXE_GOOGLE"
BACKBONE_EMBEDDING_DIM=640  # Gemma3-270m hidden_size (text_config.hidden_size)

DATASET_PATH=${DATASET_PATH:-"examples/SimplerEnv/fractal20220817_data_lerobot"}
MODALITY_SRC="examples/SimplerEnv/fractal_modality.json"
MODALITY_DST="${DATASET_PATH}/meta/modality.json"

HF_DATASET="IPEC-COMMUNITY/fractal20220817_data_lerobot"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Deps: ensure uv venv is set up first, then use its Python ────────────────
if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
    echo "[deps] Running uv sync --extra gpu..."
    uv sync --extra gpu
fi
PYTHON=${PYTHON:-"$REPO_ROOT/.venv/bin/python"}
echo "[python] Using: $PYTHON ($("$PYTHON" --version 2>&1))"

# ── Step 1: Check Eagle2.5 source is accessible ───────────────────────────────
EAGLE_REPO="$(cd "$REPO_ROOT/../Eagle/Eagle2_5" 2>/dev/null && pwd || true)"
if [ ! -d "$EAGLE_REPO" ]; then
    echo "ERROR: Eagle2.5 repo not found at $REPO_ROOT/../Eagle/Eagle2_5"
    echo "       Clone it there or set PYTHONPATH to include the eaglevl package."
    exit 1
fi
export PYTHONPATH="$EAGLE_REPO:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[eagle] Eagle2.5 source: $EAGLE_REPO"

# ── Step 2: Download dataset if not present ───────────────────────────────────
if [ "$GOOD_TO_GO" = "1" ]; then
    echo "[data] --good flag set — skipping download, using existing data."
fi

echo "[data] Preparing dataset from $HF_DATASET..."
"$PYTHON" - <<PYEOF
import time, json
from pathlib import Path
from huggingface_hub import snapshot_download

dataset_path = Path("$DATASET_PATH")
repo_id = "$HF_DATASET"
good_to_go = "$GOOD_TO_GO" == "1"

parquet_files = sorted(dataset_path.glob("data/*/*.parquet"))
if not parquet_files and not good_to_go:
    print("[data] No data on disk yet — downloading everything...")
    while True:
        try:
            snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(dataset_path), max_workers=32)
            break
        except Exception as e:
            if "429" in str(e):
                print("[data] Rate limited. Waiting 300s...")
                time.sleep(300)
            else:
                raise
    parquet_files = sorted(dataset_path.glob("data/*/*.parquet"))
elif not parquet_files:
    raise RuntimeError(f"No parquet files found at {dataset_path} — download the data first.")

# Generate meta files from whatever parquet files exist on disk
import pandas as pd

print(f"[data] Found {len(parquet_files)} parquet files on disk — generating meta...")

# Read all parquet files to build episodes/tasks
episodes = {}
all_tasks = {}
total_frames = 0
for pf in parquet_files:
    df = pd.read_parquet(pf, columns=["episode_index", "task_index"])
    for _, row in df.iterrows():
        ep_idx = int(row["episode_index"])
        task_idx = int(row["task_index"])
        if ep_idx not in episodes:
            episodes[ep_idx] = {"tasks": set(), "length": 0}
        episodes[ep_idx]["tasks"].add(task_idx)
        episodes[ep_idx]["length"] += 1
        all_tasks[task_idx] = task_idx
    total_frames += len(df)

# Get feature schema from first parquet
df0 = pd.read_parquet(parquet_files[0])
features = {}
for col in df0.columns:
    val = df0[col].iloc[0]
    if hasattr(val, "__len__"):
        features[col] = {"dtype": "float32", "shape": [len(val)], "names": None}
    else:
        features[col] = {"dtype": str(df0[col].dtype), "shape": [1], "names": None}

# Filter episodes to only those with video files on disk
video_keys = [vd.name for vd in sorted((dataset_path / "videos" / "chunk-000").iterdir())] if (dataset_path / "videos" / "chunk-000").exists() else []
if video_keys:
    vkey = video_keys[0]
    episodes = {
        ep_idx: ep for ep_idx, ep in episodes.items()
        if (dataset_path / "videos" / f"chunk-{ep_idx // 1000:03d}" / vkey / f"episode_{ep_idx:06d}.mp4").exists()
    }
    total_frames = sum(ep["length"] for ep in episodes.values())

# Add video features from actual folders on disk
for video_dir in sorted((dataset_path / "videos" / "chunk-000").iterdir()):
    key = video_dir.name  # e.g. observation.images.image
    features[key] = {
        "dtype": "video",
        "shape": [256, 320, 3],
        "names": ["height", "width", "channels"],
        "info": {
            "video.height": 256, "video.width": 320,
            "video.codec": "h264", "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False, "video.fps": 3,
            "video.channels": 3, "has_audio": False,
        },
    }

# info.json
info = {
    "codebase_version": "v2.1",
    "robot_type": "google_robot",
    "total_episodes": len(episodes),
    "total_frames": total_frames,
    "total_tasks": len(all_tasks),
    "chunks_size": 1000,
    "fps": 3,
    "splits": {"train": f"0:{len(episodes)}"},
    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    "features": features,
    "total_chunks": len(set(pf.parent.name for pf in parquet_files)),
    "total_videos": 0,
}
(dataset_path / "meta").mkdir(exist_ok=True)
with open(dataset_path / "meta/info.json", "w") as f:
    json.dump(info, f, indent=2)

# episodes.jsonl
with open(dataset_path / "meta/episodes.jsonl", "w") as f:
    for ep_idx, ep in sorted(episodes.items()):
        f.write(json.dumps({"episode_index": ep_idx, "tasks": list(ep["tasks"]), "length": ep["length"]}) + "\n")

# tasks.jsonl
with open(dataset_path / "meta/tasks.jsonl", "w") as f:
    for task_idx in sorted(all_tasks.keys()):
        f.write(json.dumps({"task_index": task_idx, "task": f"task_{task_idx}"}) + "\n")

print(f"[data] Ready: {len(episodes)} episodes, {total_frames} frames from {len(parquet_files)} parquet files.")
PYEOF

# ── Step 3: Copy modality.json if missing ─────────────────────────────────────
if [ ! -f "$MODALITY_DST" ]; then
    echo "[data] Copying modality.json..."
    mkdir -p "$(dirname "$MODALITY_DST")"
    cp "$MODALITY_SRC" "$MODALITY_DST"
else
    echo "[data] modality.json already in place."
fi

# ── Step 4: Convert AV1 videos to H.264 if needed ────────────────────────────
FIRST_VIDEO=$(find "$DATASET_PATH/videos" -name "*.mp4" 2>/dev/null | head -1)
if [ -n "$FIRST_VIDEO" ]; then
    CODEC=$(ffprobe -v error -select_streams v:0 \
        -show_entries stream=codec_name \
        -of default=nw=1:nk=1 "$FIRST_VIDEO" 2>/dev/null || echo "unknown")
    if [[ "$CODEC" == "av01" || "$CODEC" == "av1" ]]; then
        echo "[data] AV1 videos detected — converting to H.264..."
        "$PYTHON" examples/SimplerEnv/convert_av1_to_h264.py \
            "$DATASET_PATH" --jobs "$(nproc)"
        echo "[data] Video conversion complete."
    else
        echo "[data] Videos already H.264 — skipping conversion."
    fi
fi

# ── Step 5: Create output dir + check HF repo ────────────────────────────────
mkdir -p "$OUTPUT_DIR"

if [ -n "$HF_REPO_ID" ]; then
    echo "[hub] Checking HuggingFace repo: $HF_REPO_ID"
    python3 - <<EOF
from huggingface_hub import HfApi
import sys
api = HfApi()
try:
    api.repo_info(repo_id="$HF_REPO_ID", repo_type="model")
    print(f"Repo '$HF_REPO_ID' exists and is accessible.")
except Exception:
    print(f"Repo '$HF_REPO_ID' not found — creating it...")
    try:
        api.create_repo(repo_id="$HF_REPO_ID", repo_type="model", private=True)
        print(f"Created repo '$HF_REPO_ID'.")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
EOF
    if [ $? -ne 0 ]; then
        echo "Aborting: HuggingFace repo check failed."
        exit 1
    fi
    TRAIN_SCRIPT="scripts/finetune_with_hub_push.py"
else
    TRAIN_SCRIPT="gr00t/experiment/launch_finetune.py"
fi

# ── Step 6: Train ─────────────────────────────────────────────────────────────
echo "[train] Starting finetuning with Eagle2.5+Gemma3 backbone"
echo "[train] Base model: $BASE_MODEL_PATH"
echo "[train] GPUs=$NUM_GPUS  Steps=$MAX_STEPS  Embodiment=$EMBODIMENT"

"$PYTHON" -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --master_port=${MASTER_PORT:-29501} \
    $TRAIN_SCRIPT \
    --base_model_path "$BASE_MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --embodiment_tag $EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir "$OUTPUT_DIR" \
    --save_steps 200 \
    --save_total_limit 5 \
    --max_steps $MAX_STEPS \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4 \
    --state_dropout_prob 0.5 \
    --backbone_model_type eagle2_5 \
    --backbone_embedding_dim $BACKBONE_EMBEDDING_DIM \
    --tune_llm

echo "[done] Checkpoint saved to $OUTPUT_DIR"

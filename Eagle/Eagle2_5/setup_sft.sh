#!/usr/bin/env bash
# setup_sft.sh
# Downloads stage1 checkpoint, SigLIP2, Eagle-1.8M SFT data, tokenizes via
# prepare.sh → parquet, then deletes raw images + tar parts.
# Only parquet + checkpoints remain on disk.
#
# Usage: bash setup_sft.sh [HF_TOKEN] [TOKENIZER] [GPUS] [TAR_PARTS]
#   TAR_PARTS : number of Eagle-1.8M tar parts to download (default 4, max 11)
#               each part ~20GB → 4 parts ≈ first ~36% of images
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

HF_TOKEN="${1:-${HF_TOKEN:-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}}"
TOKENIZER="${2:-google/gemma-3-270m}"
export GPUS="${3:-${GPUS:-1}}"
TAR_PARTS="${4:-4}"
export HF_TOKEN="$HF_TOKEN"

SFT_DIR="${SCRIPT_DIR}/local_playground/sft_data/Eagle-1.8M"
JSONL_DIR="${SCRIPT_DIR}/local_playground/sft_data/Eagle-1.8M-jsonl"
RECIPE_PATH="${SCRIPT_DIR}/local_playground/recipe/stage2.json"
PREPARED_RECIPE="${SCRIPT_DIR}/local_playground/recipe/stage2.prepared.json"

ALL_PARTS=(aa ab ac ad ae af ag ah ai aj ak)

echo "========================================="
echo " Stage 2 (SFT) data setup"
echo "  tokenizer : $TOKENIZER"
echo "  GPUs      : $GPUS"
echo "  tar parts : ${TAR_PARTS}/11 (~$((TAR_PARTS * 20))GB download)"
echo "  images    : $SFT_DIR  (deleted after prepare)"
echo "  parquet   : local_playground/prepared_data/  (kept)"
echo "========================================="

mkdir -p "$SFT_DIR" "$JSONL_DIR" "$(dirname "$RECIPE_PATH")"

# ── STEP 1: stage1 checkpoint ─────────────────────────────────────────────────
if [ -d "./output/stage1_gemma3" ] && [ "$(ls -A ./output/stage1_gemma3 2>/dev/null)" ]; then
    echo "[1/5] Stage1 checkpoint already exists — skipping."
else
    echo "[1/5] Downloading stage1 checkpoint..."
    python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="youngbrett48/train_stage1_gemma3.sh", local_dir="./output/stage1_gemma3", token="$HF_TOKEN")
print("  Done.")
EOF
fi

# ── STEP 2: SigLIP2 vision encoder ───────────────────────────────────────────
if [ -d "./pretrained/siglip2-so400m-patch14-384" ] && [ "$(ls -A ./pretrained/siglip2-so400m-patch14-384 2>/dev/null)" ]; then
    echo "[2/5] SigLIP2 already exists — skipping."
else
    echo "[2/5] Downloading SigLIP2 vision encoder..."
    python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="google/siglip2-so400m-patch14-384", local_dir="./pretrained/siglip2-so400m-patch14-384", token="$HF_TOKEN")
print("  Done.")
EOF
fi

# ── STEP 3: Eagle-1.8M annotation JSON + tar parts → extract → delete tars ───
IMAGE_COUNT=$(find "${SFT_DIR}" -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" 2>/dev/null | wc -l || echo 0)
if [ "${IMAGE_COUNT}" -gt 1000 ]; then
    echo "[3/5] Images already extracted (${IMAGE_COUNT} files) — skipping download+extract."
else
echo "[3/5] Downloading Eagle-1.8M annotation JSON..."
python3 - <<EOF
from huggingface_hub import hf_hub_download
import os, subprocess, sys

token = "$HF_TOKEN"
repo_id = "shi-labs/Eagle-1.8M"
out_dir = "$SFT_DIR"
n_parts = $TAR_PARTS
all_suffixes = ["aa","ab","ac","ad","ae","af","ag","ah","ai","aj","ak"]
parts = [f"images.tar.part_{s}" for s in all_suffixes[:n_parts]]

# annotation JSON
ann_dest = os.path.join(out_dir, "eagle-1-sft-1_8M.json")
if not os.path.exists(ann_dest):
    print("  Downloading annotation JSON (~1.8 GB)...")
    hf_hub_download(repo_id=repo_id, filename="eagle-1-sft-1_8M.json",
                    repo_type="dataset", token=token, local_dir=out_dir)
else:
    print("  annotation JSON already exists, skipping.")

# tar parts — re-download if file is missing or suspiciously small (<1GB = corrupt/partial)
MIN_BYTES = 1 * 1024 ** 3  # 1 GB
downloaded = []
for part in parts:
    dest = os.path.join(out_dir, part)
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    if size >= MIN_BYTES:
        print(f"  already exists ({size/1024**3:.1f} GB): {part}")
    else:
        if size > 0:
            print(f"  corrupt/partial ({size/1024**3:.2f} GB) — re-downloading: {part}")
            os.remove(dest)
        else:
            print(f"  Downloading {part} (~20 GB)...")
        hf_hub_download(repo_id=repo_id, filename=part,
                        repo_type="dataset", token=token, local_dir=out_dir,
                        force_download=False)
    downloaded.append(dest)

# concatenate parts into temp file then extract (gz-compressed split archive)
tmp_tar = os.path.join(out_dir, "images_partial.tar.gz")
print("  Concatenating parts...", flush=True)
with open(tmp_tar, "wb") as out:
    for part in downloaded:
        with open(part, "rb") as fh:
            while True:
                chunk = fh.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

print("  Extracting (EOF at end of partial archive is expected)...", flush=True)
subprocess.run(
    ["tar", "-xzf", tmp_tar, "-C", out_dir],
    stderr=subprocess.DEVNULL,  # suppress gzip truncation noise
)  # don't check returncode — truncation exit is expected
os.remove(tmp_tar)
print("  Extracted.")

# delete tar parts now — images are extracted
for dest in downloaded:
    os.remove(dest)
    print(f"  Deleted {os.path.basename(dest)}")
print("  Extraction done.")
EOF
fi  # end image extraction skip

# ── STEP 4: build filtered JSONL + stage2.json recipe ────────────────────────
if [ -f "${RECIPE_PATH}" ]; then
    echo "[4/5] Recipe already exists — skipping."
else
echo "[4/5] Building stage2.json recipe (filtering to extracted images)..."
python3 - <<'PYEOF'
import json, os

data_root = "./local_playground/sft_data/Eagle-1.8M"
jsonl_path = "./local_playground/sft_data/Eagle-1.8M-jsonl/Eagle-1.8M-partial.jsonl"
recipe_path = "./local_playground/recipe/stage2.json"

print("  Loading annotation JSON...")
with open(os.path.join(data_root, "eagle-1-sft-1_8M.json")) as f:
    entries = json.load(f)
total = len(entries)

print(f"  Filtering {total:,} entries to extracted images...")
filtered = []
for entry in entries:
    imgs = entry.get("image", [])
    if isinstance(imgs, str):
        imgs = [imgs]
    if not imgs or all(os.path.exists(os.path.join(data_root, img)) for img in imgs):
        filtered.append(entry)

kept = len(filtered)
print(f"  Kept {kept:,} / {total:,} ({100*kept/total:.1f}%)")

os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
with open(jsonl_path, "w") as f:
    for e in filtered:
        f.write(json.dumps(e) + "\n")

recipe = {
    "Eagle-1.8M-partial": {
        "annotation": jsonl_path,
        "root": data_root,
        "repeat_time": 1,
    }
}
with open(recipe_path, "w") as f:
    json.dump(recipe, f, indent=2)
print(f"  Recipe: {recipe_path}")
PYEOF
fi  # end recipe skip

# ── STEP 5: tokenize + tile + pack → parquet (auto-writes stage2.prepared.json)
if [ -f "${PREPARED_RECIPE}" ]; then
    echo "[5/5] Prepared recipe already exists — skipping prepare."
else
    echo "[5/5] Running prepare (tokenize + image tiling → parquet)..."
    pip install . -q
    bash shell/prepare.sh "${RECIPE_PATH}" 1 work_dirs/prepare_stage2 "${TOKENIZER}"
fi

# ── CLEANUP ───────────────────────────────────────────────────────────────────
echo "Cleaning up raw images and jsonl..."
rm -rf "${SFT_DIR}/images" "${JSONL_DIR}"

echo ""
echo "========================================="
echo " Done."
echo " Prepared recipe : ${PREPARED_RECIPE}"
echo " Parquet data    : local_playground/prepared_data/"
du -sh "${SCRIPT_DIR}/local_playground/prepared_data/" 2>/dev/null || true
echo "========================================="
echo ""
echo " Next: bash shell/train_stage2_gemma3_270m.sh"
echo "========================================="

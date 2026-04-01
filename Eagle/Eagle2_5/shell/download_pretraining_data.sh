#!/usr/bin/env bash
# Download + prepare data for Eagle2.5 pretraining and/or Fractal robot dataset.
#
# Usage:
#   bash shell/download_pretraining_data.sh              # everything
#   bash shell/download_pretraining_data.sh stage1       # LLaVA-CC3M-Pretrain-595K + SigLIP2 + prepare
#   bash shell/download_pretraining_data.sh stage2       # Eagle-1.8M + SigLIP2 + prepare
#   bash shell/download_pretraining_data.sh robot        # Fractal robot dataset
#   bash shell/download_pretraining_data.sh stage1 robot # multiple
#
# Set HF_TOKEN env var for authenticated downloads.

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EAGLE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GROOT_ROOT="$(cd "$EAGLE_ROOT/../../Isaac-GR00T" 2>/dev/null && pwd || echo "")"
cd "$EAGLE_ROOT"

PYTHON=${PYTHON:-python3}
PRETRAINED_DIR="$EAGLE_ROOT/pretrained"
STAGE1_DATA_DIR="$EAGLE_ROOT/data/pretrain"
STAGE2_DATA_DIR="$EAGLE_ROOT/local_playground/sft_data/Eagle-1.8M"
RECIPE_DIR="$EAGLE_ROOT/local_playground/recipe"

STAGES=("$@")
if [ ${#STAGES[@]} -eq 0 ]; then
    STAGES=("stage1" "stage2" "robot")
fi

DO_STAGE1=0; DO_STAGE2=0; DO_ROBOT=0
for s in "${STAGES[@]}"; do
    case "$s" in
        stage1) DO_STAGE1=1 ;;
        stage2) DO_STAGE2=1 ;;
        robot)  DO_ROBOT=1 ;;
        *) echo "[ERROR] Unknown stage: $s (use stage1, stage2, robot)"; exit 1 ;;
    esac
done

banner() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    printf "║  %-58s  ║\n" "$1"
    echo "╚══════════════════════════════════════════════════════════════╝"
}

step() {
    echo "  ➤ $1"
}

ok() {
    echo "  ✓ $1"
}

skip() {
    echo "  - SKIP: $1"
}

fail() {
    echo "  ✗ FAIL: $1"
}

banner "Eagle2.5 Data Pipeline — ${STAGES[*]}"
echo "  Eagle root:  $EAGLE_ROOT"
echo "  GR00T root:  ${GROOT_ROOT:-NOT FOUND}"
echo "  Python:      $PYTHON"

if [ -z "$HF_TOKEN" ]; then
    read -rp "  HuggingFace token (blank to skip): " HF_TOKEN
fi
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN
    echo "  HF_TOKEN:    set (authenticated)"
else
    echo "  HF_TOKEN:    not set (anonymous — may hit rate limits)"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# SigLIP2-SO400M vision encoder (needed for stage1 + stage2)
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$DO_STAGE1" = "1" ] || [ "$DO_STAGE2" = "1" ]; then
    banner "SHARED — SigLIP2-SO400M Vision Encoder"
    SIGLIP_DIR="$PRETRAINED_DIR/siglip2-so400m-patch14-384"
    if [ -d "$SIGLIP_DIR" ] && [ "$(ls -A "$SIGLIP_DIR" 2>/dev/null)" ]; then
        skip "Already present at $SIGLIP_DIR ($(du -sh "$SIGLIP_DIR" | cut -f1))"
    else
        step "Downloading google/siglip2-so400m-patch14-384..."
        mkdir -p "$PRETRAINED_DIR"
        "$PYTHON" -u -c "
from huggingface_hub import snapshot_download; import os
snapshot_download(
    repo_id='google/siglip2-so400m-patch14-384',
    local_dir='$SIGLIP_DIR',
    token=os.environ.get('HF_TOKEN') or None,
)
" && ok "SigLIP2 downloaded ($(du -sh "$SIGLIP_DIR" | cut -f1))" \
  || fail "SigLIP2 download failed"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — LLaVA-CC3M-Pretrain-595K (558k image-caption pairs)
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$DO_STAGE1" = "1" ]; then
    banner "STAGE 1 — Download: LLaVA-CC3M-Pretrain-595K"
    if [ -f "$STAGE1_DATA_DIR/chat.jsonl" ]; then
        skip "Already present at $STAGE1_DATA_DIR ($(du -sh "$STAGE1_DATA_DIR" | cut -f1))"
    else
        step "Downloading liuhaotian/LLaVA-CC3M-Pretrain-595K..."
        mkdir -p "$STAGE1_DATA_DIR"
        "$PYTHON" -u -c "
from huggingface_hub import snapshot_download; import os
snapshot_download(
    repo_id='liuhaotian/LLaVA-CC3M-Pretrain-595K',
    repo_type='dataset',
    local_dir='$STAGE1_DATA_DIR',
    token=os.environ.get('HF_TOKEN') or None,
)
" && ok "LLaVA-CC3M downloaded ($(du -sh "$STAGE1_DATA_DIR" | cut -f1))" \
  || fail "LLaVA-CC3M download failed"
    fi

    # Extract images from zip if needed
    banner "STAGE 1 — Extract: images from images.zip"
    STAGE1_IMG_COUNT=$(ls "$STAGE1_DATA_DIR"/*.jpg 2>/dev/null | wc -l)
    if [ "$STAGE1_IMG_COUNT" -gt 100 ]; then
        skip "Images already extracted ($STAGE1_IMG_COUNT .jpg files)"
    elif [ -f "$STAGE1_DATA_DIR/images.zip" ]; then
        step "Extracting images.zip (595K images)..."
        "$PYTHON" -c "
import zipfile
z = zipfile.ZipFile('$STAGE1_DATA_DIR/images.zip')
z.extractall('$STAGE1_DATA_DIR/')
print(f'  Extracted {len(z.namelist())} files')
" && ok "Images extracted" \
  || fail "Image extraction failed"
    else
        fail "No images.zip found in $STAGE1_DATA_DIR"
    fi

    # Convert chat.json → chat.jsonl (prepare.sh requires JSONL)
    if [ ! -f "$STAGE1_DATA_DIR/chat.jsonl" ] && [ -f "$STAGE1_DATA_DIR/chat.json" ]; then
        step "Converting chat.json → chat.jsonl..."
        "$PYTHON" -c "
import json
with open('$STAGE1_DATA_DIR/chat.json') as f:
    data = json.load(f)
with open('$STAGE1_DATA_DIR/chat.jsonl', 'w') as f:
    for entry in data:
        f.write(json.dumps(entry) + '\n')
print(f'  Converted {len(data)} entries')
" && ok "chat.jsonl created" \
  || fail "JSON→JSONL conversion failed"
    fi

    banner "STAGE 1 — Prepare: tokenize + pack → stage1.prepared.json"
    STAGE1_PREPARED="$RECIPE_DIR/stage1.prepared.json"
    if [ -f "$STAGE1_PREPARED" ]; then
        skip "Already prepared at $STAGE1_PREPARED"
    else
        # Ensure recipe exists
        mkdir -p "$RECIPE_DIR"
        STAGE1_RECIPE="$RECIPE_DIR/stage1.json"
        if [ ! -f "$STAGE1_RECIPE" ]; then
            step "Creating stage1.json recipe..."
            cat > "$STAGE1_RECIPE" <<'EOF'
{
  "LLaVA-CC3M-Pretrain-595K": {
    "annotation": "./data/pretrain/chat.jsonl",
    "root": "./data/pretrain",
    "repeat_time": 1
  }
}
EOF
            ok "Created $STAGE1_RECIPE"
        fi
        step "Running prepare.sh on stage1.json (this may take a while)..."
        bash shell/prepare.sh "$STAGE1_RECIPE" \
            && ok "Stage 1 data prepared → $STAGE1_PREPARED" \
            || fail "Stage 1 prepare failed"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Eagle-1.8M (1.8M instruction-following samples)
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$DO_STAGE2" = "1" ]; then
    banner "STAGE 2 — Download: Eagle-1.8M"
    mkdir -p "$STAGE2_DATA_DIR"

    # Tar parts
    PARTS=("aa" "ab" "ac" "ad" "ae" "af")
    MISSING=()
    for part in "${PARTS[@]}"; do
        [ ! -f "$STAGE2_DATA_DIR/images.tar.part_${part}" ] && MISSING+=("$part")
    done

    if [ ${#MISSING[@]} -eq 0 ]; then
        skip "All 6 tar parts present."
    else
        step "Downloading ${#MISSING[@]} missing tar parts: ${MISSING[*]}"
        "$PYTHON" -u -c "
from huggingface_hub import hf_hub_download; import os
token = os.environ.get('HF_TOKEN') or None
parts = '${MISSING[*]}'.split()
for i, part in enumerate(parts):
    fname = f'images.tar.part_{part}'
    print(f'  [{i+1}/{len(parts)}] {fname}...')
    hf_hub_download(repo_id='shi-labs/Eagle-1.8M', repo_type='dataset',
                    filename=fname, local_dir='$STAGE2_DATA_DIR', token=token)
print('  All parts downloaded.')
" && ok "Tar parts downloaded" \
  || fail "Tar parts download failed"
    fi

    # Annotation JSON
    if [ ! -f "$STAGE2_DATA_DIR/eagle-1-sft-1_8M.json" ]; then
        step "Downloading annotation JSON..."
        "$PYTHON" -u -c "
from huggingface_hub import hf_hub_download; import os
hf_hub_download(repo_id='shi-labs/Eagle-1.8M', repo_type='dataset',
                filename='eagle-1-sft-1_8M.json', local_dir='$STAGE2_DATA_DIR',
                token=os.environ.get('HF_TOKEN') or None)
" && ok "Annotation JSON downloaded" \
  || fail "Annotation JSON download failed"
    else
        skip "Annotation JSON already present."
    fi

    # Extract images
    banner "STAGE 2 — Extract: images from tar parts"
    IMAGES_DIR="$STAGE2_DATA_DIR/images"
    if [ -d "$IMAGES_DIR" ] && [ "$(ls -A "$IMAGES_DIR" 2>/dev/null | head -1)" ]; then
        skip "Images already extracted ($(ls "$IMAGES_DIR" | wc -l) files)."
    else
        step "Concatenating 6 tar parts..."
        cd "$STAGE2_DATA_DIR"
        cat images.tar.part_aa images.tar.part_ab images.tar.part_ac \
            images.tar.part_ad images.tar.part_ae images.tar.part_af > images.tar.gz
        ok "Concatenated → images.tar.gz ($(du -sh images.tar.gz | cut -f1))"
        step "Extracting (this will take a while)..."
        tar -xf images.tar.gz
        rm -f images.tar.gz
        ok "Extracted $(ls "$IMAGES_DIR" | wc -l) images to $IMAGES_DIR"
        cd "$EAGLE_ROOT"
    fi

    # Prepare
    banner "STAGE 2 — Prepare: tokenize + pack → stage2.prepared.json"
    STAGE2_PREPARED="$RECIPE_DIR/stage2.prepared.json"
    if [ -f "$STAGE2_PREPARED" ]; then
        skip "Already prepared at $STAGE2_PREPARED"
    else
        mkdir -p "$RECIPE_DIR"
        STAGE2_RECIPE="$RECIPE_DIR/stage2.json"
        if [ ! -f "$STAGE2_RECIPE" ]; then
            step "Creating stage2.json recipe..."
            cat > "$STAGE2_RECIPE" <<'EOF'
{
  "Eagle-1.8M-partial": {
    "annotation": "./local_playground/sft_data/Eagle-1.8M-jsonl/Eagle-1.8M-partial.jsonl",
    "root": "./local_playground/sft_data/Eagle-1.8M",
    "repeat_time": 1
  }
}
EOF
            ok "Created $STAGE2_RECIPE"
        fi
        step "Running prepare.sh on stage2.json (this may take a while)..."
        bash shell/prepare.sh "$STAGE2_RECIPE" \
            && ok "Stage 2 data prepared → $STAGE2_PREPARED" \
            || fail "Stage 2 prepare failed"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# ROBOT — Fractal/Google robot dataset
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$DO_ROBOT" = "1" ]; then
    banner "ROBOT — Fractal dataset (Google robot)"
    if [ -z "$GROOT_ROOT" ]; then
        fail "Isaac-GR00T not found at ../../Isaac-GR00T"
    else
        step "Running download_fractal.sh..."
        bash "$GROOT_ROOT/scripts/download_fractal.sh" \
            && ok "Fractal dataset downloaded" \
            || fail "Fractal download failed"
    fi
fi

banner "COMPLETE"
[ "$DO_STAGE1" = "1" ] && echo "  Stage 1 data:  $STAGE1_DATA_DIR"
[ "$DO_STAGE2" = "1" ] && echo "  Stage 2 data:  $STAGE2_DATA_DIR"
[ "$DO_ROBOT" = "1" ]  && echo "  Robot data:    ${GROOT_ROOT:-?}/examples/SimplerEnv/fractal20220817_data_lerobot"
echo ""

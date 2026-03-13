#!/usr/bin/env bash
# Fine-tune GR00T with your Eagle2.5 + Gemma3-270m backbone.
#
# Prerequisites:
#   - Eagle2.5 repo cloned at ../Eagle/Eagle2_5 (relative to Isaac-GR00T/)
#   - Checkpoint at HF Hub: youngbrett48/train_stage2_gemma3_270m.sh
#     (or override BASE_MODEL_PATH to a local path)
#   - Dataset downloaded + modality.json in place
#
# Usage:
#   export HF_REPO_ID="your-org/gr00t-gemma3-widowx"  # where to push ckpts
#   bash scripts/finetune_eagle2_5_gemma3.sh

set -x -e

export NUM_GPUS=${NUM_GPUS:-8}
export EMBODIMENT=${EMBODIMENT:-OXE_WIDOWX}
export DATASET_PATH=${DATASET_PATH:-examples/SimplerEnv/bridge_orig_lerobot/}
export OUTPUT_DIR=${OUTPUT_DIR:-/tmp/gemma3_finetune}
export HF_REPO_ID=${HF_REPO_ID:-"youngbrett48/gr00t-gemma3-widowx"}
export BASE_MODEL_PATH=${BASE_MODEL_PATH:-"youngbrett48/train_stage2_gemma3_270m.sh"}

# Gemma3-270m hidden_size = 1152
BACKBONE_EMBEDDING_DIM=1152

SCRIPT="scripts/finetune_with_hub_push.py"

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    $SCRIPT \
    --base_model_path "$BASE_MODEL_PATH" \
    --dataset_path $DATASET_PATH \
    --embodiment_tag $EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir $OUTPUT_DIR \
    --save_steps 500 \
    --save_total_limit 5 \
    --max_steps 20000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 512 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4 \
    --state_dropout_prob 0.8 \
    --backbone_model_type eagle2_5 \
    --backbone_embedding_dim $BACKBONE_EMBEDDING_DIM

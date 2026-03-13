#!/usr/bin/env bash
# Fine-tune GR00T and push checkpoints to HuggingFace Hub.
# Set HF_REPO_ID before running, e.g.:
#   export HF_REPO_ID="your-org/gr00t-widowx-finetune"
#   bash scripts/finetune_with_hub_push.sh

set -x -e

export NUM_GPUS=${NUM_GPUS:-8}
export EMBODIMENT=${EMBODIMENT:-OXE_WIDOWX}
export DATASET_PATH=${DATASET_PATH:-examples/SimplerEnv/bridge_orig_lerobot/}
export OUTPUT_DIR=${OUTPUT_DIR:-/tmp/bridge_finetune}

if [ -z "$HF_REPO_ID" ]; then
    echo "ERROR: HF_REPO_ID is not set. Export it before running:"
    echo "  export HF_REPO_ID=your-org/your-model-repo"
    exit 1
fi

torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    scripts/finetune_with_hub_push.py \
    --base_model_path nvidia/GR00T-N1.6-3B \
    --dataset_path $DATASET_PATH \
    --embodiment_tag $EMBODIMENT \
    --num_gpus $NUM_GPUS \
    --output_dir $OUTPUT_DIR \
    --save_steps 1000 \
    --save_total_limit 5 \
    --max_steps 20000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 1e-4 \
    --use_wandb \
    --global_batch_size 1024 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --dataloader_num_workers 4 \
    --state_dropout_prob 0.8

#!/usr/bin/env bash
set -x

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export NCCL_TIMEOUT=1800
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT="eagle-gemma3"
export LAUNCHER=pytorch
export PYTHONPATH="/home/ubuntu/gemma_robot/Eagle/Eagle2_5:${PYTHONPATH:-}"

GPUS=${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
GPUS=${GPUS:-1}
NNODES=${1:-1}
OUTPUT_DIR=${2:-"./output/stage2_gemma3_1b"}
HF_TOKEN=${3:-""}
if [ -n "$HF_TOKEN" ]; then
  export HF_TOKEN="$HF_TOKEN"
fi
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

TOTAL_GPUS=$((GPUS * NNODES))

BATCH_SIZE=3
TOKEN_PER_GPU=4096
GRADIENT_ACC=${GRADIENT_ACC:-2}

LOSS_VERSION="efficient_v2_cp_head"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

script_path=${BASH_SOURCE[0]}
script_name=$(basename "$script_path")
HF_REPO_ID="youngbrett48/$script_name"

echo "Checking HuggingFace repo access: $HF_REPO_ID"
python3 - <<EOF
from huggingface_hub import HfApi
import sys
api = HfApi()
repo_id = "$HF_REPO_ID"
try:
    api.repo_info(repo_id=repo_id, repo_type="model")
    print(f"Repo '{repo_id}' exists and is accessible.")
except Exception:
    print(f"Repo '{repo_id}' not found — creating it...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True)
        print(f"Created repo '{repo_id}'.")
    except Exception as e:
        print(f"ERROR: Could not access or create repo '{repo_id}': {e}")
        sys.exit(1)
EOF
if [ $? -ne 0 ]; then
  echo "Aborting: HuggingFace repo check failed. Make sure HF_TOKEN is set or run 'huggingface-cli login'."
  exit 1
fi

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
  eaglevl/train/eagle_2_5_vl_finetune.py \
  --model_name_or_path "youngbrett48/train_stage1_gemma3_1b.sh" \
  --vision_path "./pretrained/siglip2-so400m-patch14-384" \
  --conv_style "gemma3-chat" \
  --normalize_type "siglip" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "local_playground/recipe/stage2.prepared.json" \
  --overwrite_output_dir False \
  --force_image_size 384 \
  --max_dynamic_tiles 12 \
  --down_sample_ratio 0.5 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 8 \
  --bf16 True \
  --use_online_packing True \
  --num_train_epochs 1 \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_steps 250 \
  --save_total_limit 5 \
  --learning_rate 1e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length ${TOKEN_PER_GPU} \
  --sequence_parallel_degree 1 \
  --sample_length_div 1 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --deepspeed "deepspeed_configs/zero_stage3_config.json" \
  --loss_version ${LOSS_VERSION} \
  --report_to "wandb" \
  --run_name $script_name \
  --save_every_n_hours 4 \
  --use_pixel_shuffle True \
  --mlp_connector_layers 2 \
  --hf_repo_id "$HF_REPO_ID" \
  --hf_private True \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

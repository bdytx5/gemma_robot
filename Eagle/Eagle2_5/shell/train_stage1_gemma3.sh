#!/usr/bin/env bash
set -x

export NCCL_TIMEOUT=1800
export WANDB_PROJECT="eagle-gemma3"

GPUS=${GPUS:-1}
NNODES=${1:-1}
OUTPUT_DIR=${2:-"./output/stage1_gemma3"}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

TOTAL_GPUS=$((GPUS * NNODES))

BATCH_SIZE=1
TOKEN_PER_GPU=32768
TOTAL_TOKENS_PER_ITER=$((TOKEN_PER_GPU * TOTAL_GPUS))
GRADIENT_ACC=$((TOTAL_TOKENS_PER_ITER / (TOKEN_PER_GPU * TOTAL_GPUS)))

LOSS_VERSION="efficient_v2_cp_head"

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

script_path=${BASH_SOURCE[0]}
script_name=$(basename "$script_path")

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
  eaglevl/train/eagle_2_5_vl_finetune.py \
  --llm_path "google/gemma-3-270m" \
  --vision_path "./pretrained/siglip2-so400m-patch14-448" \
  --conv_style "gemma3-chat" \
  --normalize_type "siglip" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "local_playground/recipe/stage1.prepared.json" \
  --overwrite_output_dir False \
  --force_image_size 448 \
  --max_dynamic_tiles 12 \
  --down_sample_ratio 0.5 \
  --pad2square False \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 4 \
  --bf16 True \
  --use_online_packing True \
  --num_train_epochs 1 \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --save_strategy "steps" \
  --save_steps 250 \
  --save_total_limit 5 \
  --learning_rate 2e-5 \
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
  --deepspeed "deepspeed_configs/zero_stage1_config.json" \
  --loss_version ${LOSS_VERSION} \
  --report_to "wandb" \
  --run_name $script_name \
  --save_every_n_hours 4 \
  --use_pixel_shuffle True \
  --mlp_connector_layers 2 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"

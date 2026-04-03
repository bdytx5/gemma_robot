#!/usr/bin/env python3
"""Fine-tune GR00T on balanced 8-dataset mixture from a pretrained checkpoint.

This script loads multi-embodiment datasets with equal mix_ratio and resumes
training from a given checkpoint (e.g. the 270m fractal-trained model).

Usage:
    python scripts/finetune_balanced.py \
        --checkpoint_path ./output/gr00t-eagle2_5-gemma3-1b-fractal/checkpoint-6200 \
        --output_dir ./output/balanced-from-270m \
        --max_steps 20000
"""

import argparse
import os
from pathlib import Path

from gr00t.configs.base_config import get_default_config
from gr00t.experiment import trainer as _trainer_mod
from gr00t.experiment.experiment import run

REPO_ROOT = Path(__file__).resolve().parent.parent

ALL_DATASETS = [
    {
        "dataset_paths": [str(REPO_ROOT / "examples/SimplerEnv/fractal20220817_data_lerobot")],
        "embodiment_tag": "oxe_google",
        "mix_ratio": 1.0,
    },
    {
        "dataset_paths": [str(REPO_ROOT / "examples/SimplerEnv/bridge_orig_lerobot")],
        "embodiment_tag": "oxe_widowx",
        "mix_ratio": 1.0,
    },
    {
        "dataset_paths": [str(REPO_ROOT / "examples/LIBERO/libero_10_no_noops_1.0.0_lerobot")],
        "embodiment_tag": "libero_panda",
        "mix_ratio": 1.0,
    },
    {
        "dataset_paths": [str(REPO_ROOT / "examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot")],
        "embodiment_tag": "libero_panda",
        "mix_ratio": 1.0,
    },
    {
        "dataset_paths": [str(REPO_ROOT / "examples/LIBERO/libero_object_no_noops_1.0.0_lerobot")],
        "embodiment_tag": "libero_panda",
        "mix_ratio": 1.0,
    },
    {
        "dataset_paths": [str(REPO_ROOT / "examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot")],
        "embodiment_tag": "libero_panda",
        "mix_ratio": 1.0,
    },
    {
        "dataset_paths": [str(REPO_ROOT / "examples/DROID/droid_subset")],
        "embodiment_tag": "oxe_droid",
        "mix_ratio": 1.0,
    },
    {
        "dataset_paths": [str(REPO_ROOT / "examples/robocasa/robocasa_mg_gr00t_300")],
        "embodiment_tag": "robocasa_panda_omron",
        "mix_ratio": 1.0,
    },
]


def get_ready_datasets():
    """Filter to only datasets that have meta/info.json."""
    ready = []
    for ds in ALL_DATASETS:
        ds_path = Path(ds["dataset_paths"][0])
        if (ds_path / "meta" / "info.json").exists():
            ready.append(ds)
            print(f"  READY: {ds_path.name} ({ds['embodiment_tag']})")
        else:
            print(f"  SKIP: {ds_path.name} (missing meta/info.json)")
    return ready


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to pretrained checkpoint to resume from")
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "output" / "balanced-from-270m"))
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--global_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=5)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--wandb_project", type=str, default="balanced-loss")
    parser.add_argument("--experiment_name", type=str, default="balanced-from-270m")
    parser.add_argument("--state_dropout_prob", type=float, default=0.5)
    parser.add_argument("--tune_llm", action="store_true", default=True)
    parser.add_argument("--no_tune_llm", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    # Optionally patch trainer for HF Hub push
    repo_id = os.environ.get("HF_REPO_ID")
    if repo_id:
        from scripts.hub_push_callback import HubPushCallback
        _orig_init = _trainer_mod.Gr00tTrainer.__init__

        def _patched_init(self, *a, **kw):
            _orig_init(self, *a, **kw)
            self.add_callback(HubPushCallback(repo_id=repo_id))

        _trainer_mod.Gr00tTrainer.__init__ = _patched_init
        print(f"[HubPush] Will push checkpoints to: {repo_id}")

    # The base model name (for architecture loading) is the original stage2 model
    BASE_MODEL = "youngbrett48/train_stage2_gemma3_1b.sh"

    print("Checking dataset readiness...")
    datasets = get_ready_datasets()
    if not datasets:
        print("ERROR: No datasets are ready.")
        return
    print(f"\nTraining with {len(datasets)} datasets from checkpoint: {args.checkpoint_path}")

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": datasets,
            }
        }
    )
    config.load_config_path = None

    # Model config — match the 270m fractal training setup
    tune_llm = args.tune_llm and not args.no_tune_llm
    config.model.tune_llm = tune_llm
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.state_dropout_prob = args.state_dropout_prob
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = False
    config.model.model_name = BASE_MODEL
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.model.backbone_model_type = "eagle2_5"
    config.model.backbone_embedding_dim = 1152

    # Training config — resume from the 270m checkpoint
    config.training.start_from_checkpoint = args.checkpoint_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = args.global_batch_size
    config.training.dataloader_num_workers = args.dataloader_num_workers
    config.training.learning_rate = args.learning_rate
    config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.training.output_dir = args.output_dir
    config.training.save_steps = args.save_steps
    config.training.save_total_limit = args.save_total_limit
    config.training.num_gpus = 1
    config.training.use_wandb = True
    config.training.max_steps = args.max_steps
    config.training.weight_decay = 1e-5
    config.training.warmup_ratio = 0.05
    config.training.wandb_project = args.wandb_project
    config.training.experiment_name = args.experiment_name

    config.data.shard_size = 1024
    config.data.episode_sampling_rate = 0.1
    config.data.num_shards_per_epoch = int(1e5)
    config.data.video_backend = "torchcodec"

    run(config)


if __name__ == "__main__":
    main()

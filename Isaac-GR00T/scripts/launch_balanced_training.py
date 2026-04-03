#!/usr/bin/env python3
"""Launch balanced multi-embodiment finetuning across all 8 datasets.

Each dataset gets mix_ratio=1.0 so they are sampled equally.
Loss is logged to W&B project "balanced-loss".
"""

import os
from pathlib import Path

from gr00t.configs.base_config import get_default_config
from gr00t.experiment.experiment import run

REPO_ROOT = Path(__file__).resolve().parent.parent

# All candidate datasets with their embodiment tags
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
    """Filter to only datasets that have finished downloading (have meta/info.json)."""
    ready = []
    for ds in ALL_DATASETS:
        ds_path = Path(ds["dataset_paths"][0])
        info_path = ds_path / "meta" / "info.json"
        if info_path.exists():
            ready.append(ds)
            print(f"  READY: {ds_path.name} ({ds['embodiment_tag']})")
        else:
            print(f"  SKIP (still downloading): {ds_path.name}")
    return ready

BASE_MODEL = "youngbrett48/train_stage2_gemma3_1b.sh"


def main():
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    print("Checking dataset readiness...")
    datasets = get_ready_datasets()
    if not datasets:
        print("ERROR: No datasets are ready. Wait for downloads to complete.")
        return
    print(f"\nTraining with {len(datasets)} datasets")

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": datasets,
            }
        }
    )
    config.load_config_path = None

    # Model config - match existing fractal training setup
    config.model.tune_llm = False
    config.model.tune_visual = False
    config.model.tune_projector = True
    config.model.tune_diffusion_model = True
    config.model.state_dropout_prob = 0.0
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = False
    config.model.model_name = BASE_MODEL
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True
    config.model.backbone_model_type = "eagle2_5"
    config.model.backbone_embedding_dim = 1152

    # Training config
    config.training.start_from_checkpoint = BASE_MODEL
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = 64
    config.training.dataloader_num_workers = 2
    config.training.learning_rate = 1e-4
    config.training.gradient_accumulation_steps = 1
    config.training.output_dir = str(REPO_ROOT / "output" / "balanced-8ds")
    config.training.save_steps = 500
    config.training.save_total_limit = 5
    config.training.num_gpus = 1
    config.training.use_wandb = True
    config.training.max_steps = 10000
    config.training.weight_decay = 1e-5
    config.training.warmup_ratio = 0.05
    config.training.wandb_project = "balanced-loss"
    config.training.experiment_name = "balanced-8ds"

    config.data.shard_size = 1024
    config.data.episode_sampling_rate = 0.1
    config.data.num_shards_per_epoch = int(1e5)

    run(config)


if __name__ == "__main__":
    main()

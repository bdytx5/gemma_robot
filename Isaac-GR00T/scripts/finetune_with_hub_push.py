"""
Drop-in replacement for gr00t/experiment/launch_finetune.py that additionally
pushes each checkpoint to a HuggingFace Hub repo.

Set the HF_REPO_ID environment variable before running:
    export HF_REPO_ID="your-org/gr00t-widowx-finetune"

Usage (via finetune_with_hub_push.sh):
    torchrun --nproc_per_node=8 scripts/finetune_with_hub_push.py [same args as launch_finetune.py]
"""
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment import trainer as _trainer_mod
from gr00t.experiment.experiment import run


def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


def _patch_trainer_with_hub_push(repo_id: str):
    from scripts.hub_push_callback import HubPushCallback

    _orig_init = _trainer_mod.Gr00tTrainer.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.add_callback(HubPushCallback(repo_id=repo_id))

    _trainer_mod.Gr00tTrainer.__init__ = _patched_init


if __name__ == "__main__":
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    repo_id = os.environ.get("HF_REPO_ID")
    if repo_id:
        _patch_trainer_with_hub_push(repo_id)
        print(f"[HubPush] Will push checkpoints to: {repo_id}")
    else:
        print("[HubPush] HF_REPO_ID not set — running without Hub push")

    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    embodiment_tag = ft_config.embodiment_tag.value

    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.eagle_collator = True
    config.model.model_name = ft_config.base_model_path
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    if ft_config.backbone_model_type is not None:
        config.model.backbone_model_type = ft_config.backbone_model_type
    if ft_config.backbone_embedding_dim is not None:
        config.model.backbone_embedding_dim = ft_config.backbone_embedding_dim
    if config.model.backbone_model_type == "eagle2_5":
        config.model.eagle_collator = False

    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d6"

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    run(config)

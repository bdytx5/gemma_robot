"""
TrainerCallback that pushes each saved checkpoint to a HuggingFace Hub repo.

Each checkpoint is uploaded to a subfolder (checkpoint-N) so all checkpoints
are preserved on the Hub. Uploads are non-blocking so training continues.
"""
import os
from transformers import TrainerCallback
from huggingface_hub import HfApi


class HubPushCallback(TrainerCallback):
    def __init__(self, repo_id: str, private: bool = True):
        self.repo_id = repo_id
        self.api = HfApi()
        self.api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"[HubPush] Initialized — pushing checkpoints to {repo_id}")

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        ckpt_name = f"checkpoint-{state.global_step}"
        ckpt_dir = os.path.join(args.output_dir, ckpt_name)
        print(f"[HubPush] Uploading {ckpt_dir} → {self.repo_id}/{ckpt_name} ...")
        self.api.upload_folder(
            folder_path=ckpt_dir,
            repo_id=self.repo_id,
            repo_type="model",
            commit_message=ckpt_name,
            path_in_repo=ckpt_name,
            run_as_future=True,  # non-blocking
        )

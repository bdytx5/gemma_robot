"""
Extract the Gemma3-1b language model from a GR00T checkpoint on HuggingFace
and save it as a standalone HF model so mlx-lm can convert it.

Usage:
    python extract_llm.py \
        --gr00t_repo youngbrett48/train_google_robot_eagle2_5 \
        --checkpoint checkpoint-200 \
        --out_dir ./gr00t_llm_hf
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download


def extract(gr00t_repo: str, checkpoint: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Downloading {gr00t_repo}/{checkpoint} from HuggingFace...")
    ckpt_dir = snapshot_download(
        repo_id=gr00t_repo,
        allow_patterns=[f"{checkpoint}/*"],
        token=True,
    )
    ckpt_path = Path(ckpt_dir) / checkpoint

    # GR00T saves a pytorch_model.bin or model.safetensors at checkpoint root
    bin_path = ckpt_path / "pytorch_model.bin"
    safetensors_path = ckpt_path / "model.safetensors"

    print(f"[2/4] Loading state dict...")
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
    elif bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
    else:
        # Try sharded
        import glob
        shards = sorted(glob.glob(str(ckpt_path / "model-*-of-*.safetensors")))
        if shards:
            from safetensors.torch import load_file
            state_dict = {}
            for s in shards:
                state_dict.update(load_file(s))
        else:
            raise FileNotFoundError(f"No model weights found in {ckpt_path}")

    print(f"  Loaded {len(state_dict)} keys")

    # GR00T stores backbone as "backbone.model.*"
    # The LLM is "backbone.model.language_model.*"
    prefix = "backbone.model.language_model."
    llm_keys = {k: v for k, v in state_dict.items() if k.startswith(prefix)}

    if not llm_keys:
        # Try alternate key prefix patterns
        for candidate in ["model.language_model.", "language_model.", "backbone.model.model."]:
            llm_keys = {k: v for k, v in state_dict.items() if k.startswith(candidate)}
            if llm_keys:
                prefix = candidate
                break

    if not llm_keys:
        print("Available top-level keys (first 20):")
        for k in list(state_dict.keys())[:20]:
            print(f"  {k}")
        raise ValueError("Could not find language_model keys — check the prefix above and update 'prefix'")

    print(f"[3/4] Extracting {len(llm_keys)} LLM keys (prefix: {prefix})...")
    stripped = {k[len(prefix):]: v for k, v in llm_keys.items()}

    # Save as safetensors
    from safetensors.torch import save_file
    save_file(stripped, str(out / "model.safetensors"))
    print(f"  Saved weights → {out}/model.safetensors")

    # Copy config from the base Eagle2.5 stage2 HF repo
    # The language model config lives inside the Eagle2.5 config as text_config
    print(f"[4/4] Fetching Gemma3-1b config from google/gemma-3-1b-it...")
    base_dir = snapshot_download("google/gemma-3-1b-it", allow_patterns=["*.json", "tokenizer*"])
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "tokenizer.model"]:
        src = Path(base_dir) / fname
        if src.exists():
            shutil.copy(src, out / fname)

    print(f"\nDone. Standalone Gemma3-1b LLM saved to: {out}")
    print("Next step — convert to MLX:")
    print(f"  python -m mlx_lm.convert --hf-path {out} --mlx-path ./gr00t_llm_mlx -q")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gr00t_repo", default="youngbrett48/train_google_robot_eagle2_5")
    p.add_argument("--checkpoint", default="checkpoint-200",
                   help="Checkpoint subfolder name in the HF repo")
    p.add_argument("--out_dir", default="./gr00t_llm_hf")
    args = p.parse_args()
    extract(args.gr00t_repo, args.checkpoint, args.out_dir)

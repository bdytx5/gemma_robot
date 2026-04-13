"""
Extract the Gemma3 language model from a GR00T checkpoint on HuggingFace
and save it as a standalone HF model so mlx-lm can convert it.

Usage:
    python extract_llm.py \
        --gr00t_repo youngbrett48/gr00t-post-train-fractal-270m \
        --out_dir ./gr00t_llm_hf
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download
from tqdm import tqdm


def _repo_relpath(path: str, checkpoint: str | None) -> bool:
    if checkpoint:
        if not path.startswith(f"{checkpoint}/"):
            return False
        name = Path(path).name
        return (
            name == "config.json"
            or name == "pytorch_model.bin"
            or name == "model.safetensors"
            or name == "model.safetensors.index.json"
            or name.endswith(".safetensors")
        )
    return "/" not in path


def _checkpoint_dir(root_dir: Path, checkpoint: str | None) -> Path:
    return root_dir / checkpoint if checkpoint else root_dir


def extract(
    gr00t_repo: str,
    checkpoint: str | None,
    out_dir: str,
    token: str | None = None,
    base_llm_repo: str = "google/gemma-3-270m-it",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = token or True
    repo_target = f"{gr00t_repo}/{checkpoint}" if checkpoint else gr00t_repo
    print(f"[1/4] Listing files in {repo_target}...")
    all_files = [f for f in list_repo_files(gr00t_repo, token=tok) if _repo_relpath(f, checkpoint)]
    print(f"  {len(all_files)} files to download:")
    for f in all_files:
        print(f"    {f}")

    print(f"\n  Downloading (this is ~10GB, will take a few minutes)...")
    local_files = []
    for f in tqdm(all_files, desc="  files", unit="file"):
        local_path = hf_hub_download(
            repo_id=gr00t_repo,
            filename=f,
            token=tok,
        )
        local_files.append(local_path)
        tqdm.write(f"  ✓ {f}")

    # All files land in the same HF cache dir.
    repo_root = Path(local_files[0]).parent if checkpoint is None else Path(local_files[0]).parent.parent
    ckpt_path = _checkpoint_dir(repo_root, checkpoint)

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

    print(f"[3/5] Extracting {len(llm_keys)} LLM keys (prefix: {prefix})...")
    stripped = {k[len(prefix):]: v for k, v in llm_keys.items()}

    # Download base model to fill in any missing weights (GR00T checkpoints
    # only save fine-tuned params, so unfrozen layers may be incomplete)
    print(f"[4/5] Fetching base {base_llm_repo} weights + config...")
    base_dir = snapshot_download(
        base_llm_repo, allow_patterns=["*.json", "tokenizer*", "*.safetensors", "*.model"]
    )

    # Load base weights and fill missing keys
    from safetensors.torch import save_file, load_file
    import glob as _glob
    base_shards = sorted(_glob.glob(str(Path(base_dir) / "*.safetensors")))
    base_weights = {}
    for s in base_shards:
        base_weights.update(load_file(s))
    missing = [k for k in base_weights if k not in stripped]
    if missing:
        print(f"  Filling {len(missing)} missing keys from base model")
        for k in missing:
            stripped[k] = base_weights[k]
    del base_weights

    save_file(stripped, str(out / "model.safetensors"))
    print(f"  Saved weights → {out}/model.safetensors")

    # Copy config + tokenizer files
    print(f"[5/5] Copying config and tokenizer...")
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "tokenizer.model"]:
        src = Path(base_dir) / fname
        if src.exists():
            shutil.copy(src, out / fname)

    # Patch vocab_size in config.json to match actual embed_tokens shape
    # (GR00T may add special tokens during fine-tuning)
    if "model.embed_tokens.weight" in stripped:
        actual_vocab = stripped["model.embed_tokens.weight"].shape[0]
        cfg_path = out / "config.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get("vocab_size") != actual_vocab:
            print(f"  Patching vocab_size: {cfg['vocab_size']} → {actual_vocab}")
            cfg["vocab_size"] = actual_vocab
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)

    print(f"\nDone. Standalone Gemma3 LLM saved to: {out}")
    print("Next step — convert to MLX:")
    print(f"  python -m mlx_lm.convert --hf-path {out} --mlx-path ./gr00t_llm_mlx -q")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gr00t_repo", default="youngbrett48/gr00t-post-train-fractal-270m")
    p.add_argument(
        "--checkpoint",
        default="checkpoint-2000",
        help="Checkpoint subfolder name in the HF repo. Leave unset for a repo-root model.",
    )
    p.add_argument("--out_dir", default="./gr00t_llm_hf")
    p.add_argument("--base_llm_repo", default="google/gemma-3-270m-it")
    p.add_argument("--hf_token", default=None, help="HuggingFace token (or run `huggingface-cli login` once)")
    args = p.parse_args()
    extract(
        args.gr00t_repo,
        args.checkpoint,
        args.out_dir,
        token=args.hf_token,
        base_llm_repo=args.base_llm_repo,
    )

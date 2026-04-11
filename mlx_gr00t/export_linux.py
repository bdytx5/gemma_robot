"""
export_linux.py — export GR00T MLX weights on Linux (no MLX required).

Converts the PyTorch GR00T checkpoint to MLX-compatible safetensors and
uploads them to HuggingFace. Runs entirely on PyTorch + numpy + safetensors.

Usage:
    python export_linux.py

Uploads to: bdytx5/ntl_gemma_robot_mlx
  gr00t_weights_mlx/  — vision.safetensors, dit.safetensors, config/meta/stats/tokenizer
  gr00t_llm_mlx/      — Gemma3 LLM in HF format (mlx-lm loads this directly)
"""

import sys
import json
import glob
import shutil
import getpass
import numpy as np
from pathlib import Path

# ── prompt for HF token ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  GemmaRobot — Linux MLX weight exporter")
print("=" * 60)
print()
print("This script converts the GR00T checkpoint to MLX safetensors")
print("and uploads them to HuggingFace.")
print()
HF_TOKEN = getpass.getpass("HuggingFace API token (input hidden): ").strip()
if not HF_TOKEN:
    print("ERROR: token required")
    sys.exit(1)

# ── config ─────────────────────────────────────────────────────────────────────
GR00T_REPO  = "youngbrett48/gr00t-post-train-fractal-270m"
CHECKPOINT  = "checkpoint-2000"
EAGLE_REPO  = "youngbrett48/train_stage2_gemma3_270m.sh"
BASE_LLM    = "google/gemma-3-270m-it"
UPLOAD_REPO = "bdytx5/ntl_gemma_robot_mlx"
OUT_DIR     = Path("./gr00t_export_tmp")
WEIGHTS_DIR = OUT_DIR / "gr00t_weights_mlx"
LLM_DIR     = OUT_DIR / "gr00t_llm_mlx"

WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
LLM_DIR.mkdir(parents=True, exist_ok=True)

import torch
from safetensors.torch import save_file as save_safetensors, load_file
from huggingface_hub import snapshot_download, hf_hub_download, HfApi


def to_bf16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().to(torch.bfloat16)


# ── 1. download checkpoint ─────────────────────────────────────────────────────
print(f"\n[1/6] Downloading GR00T checkpoint ({GR00T_REPO}/{CHECKPOINT})...")
allow = [
    f"{CHECKPOINT}/model.safetensors",
    f"{CHECKPOINT}/model.safetensors.index.json",
    f"{CHECKPOINT}/model-*-of-*.safetensors",
    f"{CHECKPOINT}/config.json",
    "statistics.json",
]
ckpt_root = snapshot_download(GR00T_REPO, allow_patterns=allow, token=HF_TOKEN)
ckpt_path = Path(ckpt_root) / CHECKPOINT

shards = sorted(glob.glob(str(ckpt_path / "*.safetensors")))
if not shards:
    single = ckpt_path / "model.safetensors"
    shards = [str(single)] if single.exists() else []
if not shards:
    raise FileNotFoundError(f"No safetensors found in {ckpt_path}")

print(f"  Loading {len(shards)} shard(s)...")
state_dict = {}
for s in shards:
    state_dict.update(load_file(s))
print(f"  {len(state_dict)} total keys loaded")

# copy config + statistics
shutil.copy(str(ckpt_path / "config.json"), str(WEIGHTS_DIR / "config.json"))
stats_src = Path(ckpt_root) / "statistics.json"
if stats_src.exists():
    shutil.copy(str(stats_src), str(WEIGHTS_DIR / "statistics.json"))


# ── 2. convert vision weights ──────────────────────────────────────────────────
print("\n[2/6] Converting vision encoder + MLP1 projector...")

prefix_vision = "backbone.model.vision_model.vision_model."
prefix_mlp1   = "backbone.model.mlp1."

vision_weights = {}

for pt_key, tensor in state_dict.items():
    arr = tensor.detach().cpu().float()

    if pt_key.startswith(prefix_vision):
        suffix = pt_key[len(prefix_vision):]
        if suffix.startswith("head."):
            continue
        suffix = suffix.replace("embeddings.patch_embedding",  "patch_embedding")
        suffix = suffix.replace("embeddings.position_embedding", "position_embedding")
        suffix = suffix.replace("encoder.layers.", "layers.")
        mlx_key = f"encoder.{suffix}"
        # Conv2d: PyTorch (Cout, Cin, kH, kW) → MLX (Cout, kH, kW, Cin)
        if "patch_embedding.weight" in suffix:
            arr = arr.permute(0, 2, 3, 1).contiguous()
        vision_weights[mlx_key] = to_bf16(arr)

    elif pt_key.startswith(prefix_mlp1):
        suffix = pt_key[len(prefix_mlp1):]
        if suffix.startswith("0."):
            mlx_key = f"mlp1.ln.{suffix[2:]}"
        elif suffix.startswith("1."):
            mlx_key = f"mlp1.linear1.{suffix[2:]}"
        elif suffix.startswith("3."):
            mlx_key = f"mlp1.linear2.{suffix[2:]}"
        else:
            continue
        vision_weights[mlx_key] = to_bf16(arr)

save_safetensors(vision_weights, str(WEIGHTS_DIR / "vision.safetensors"))
print(f"  {len(vision_weights)} vision keys → vision.safetensors")


# ── 3. convert DiT weights ─────────────────────────────────────────────────────
print("\n[3/6] Converting DiT action head...")

ah_sd = {
    k[len("action_head."):]: v
    for k, v in state_dict.items()
    if k.startswith("action_head.")
}
print(f"  {len(ah_sd)} action_head keys")

dit_weights = {}
skipped = []

for pt_key, tensor in ah_sd.items():
    mlx_key = pt_key
    mlx_key = mlx_key.replace(
        "model.timestep_encoder.timestep_embedder.",
        "model.timestep_encoder.",
    )
    mlx_key = mlx_key.replace(".to_out.0.", ".to_out.")
    mlx_key = mlx_key.replace(".ff.net.0.proj.", ".ff.gate_proj.")
    mlx_key = mlx_key.replace(".ff.net.2.", ".ff.out_proj.")
    if "time_proj" in mlx_key or "pos_embed.pe" in mlx_key:
        skipped.append(pt_key)
        continue
    dit_weights[mlx_key] = to_bf16(tensor)

save_safetensors(dit_weights, str(WEIGHTS_DIR / "dit.safetensors"))
print(f"  {len(dit_weights)} DiT keys → dit.safetensors  (skipped {len(skipped)} buffers)")


# ── 4. extract LLM weights ─────────────────────────────────────────────────────
print(f"\n[4/6] Extracting Gemma3 LLM weights...")

prefix = "backbone.model.language_model."
llm_keys = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

if not llm_keys:
    for candidate in ["model.language_model.", "language_model.", "backbone.model.model."]:
        llm_keys = {k[len(candidate):]: v for k, v in state_dict.items() if k.startswith(candidate)}
        if llm_keys:
            break

if not llm_keys:
    print("  ERROR: could not find LLM keys. Top-level keys:")
    for k in list(state_dict.keys())[:20]:
        print(f"    {k}")
    sys.exit(1)

print(f"  {len(llm_keys)} LLM keys extracted")

# fill missing keys from base Gemma3 model
print(f"  Downloading base {BASE_LLM} to fill any missing weights...")
base_dir = snapshot_download(
    BASE_LLM,
    allow_patterns=["*.json", "tokenizer*", "*.safetensors", "*.model"],
    token=HF_TOKEN,
)
base_shards = sorted(glob.glob(str(Path(base_dir) / "*.safetensors")))
base_weights = {}
for s in base_shards:
    base_weights.update(load_file(s))

missing = [k for k in base_weights if k not in llm_keys]
if missing:
    print(f"  Filling {len(missing)} missing keys from base model")
    for k in missing:
        llm_keys[k] = base_weights[k]
del base_weights

# save as bfloat16
llm_bf16 = {k: to_bf16(v) for k, v in llm_keys.items()}
save_safetensors(llm_bf16, str(LLM_DIR / "model.safetensors"))
print(f"  Saved {len(llm_bf16)} keys → gr00t_llm_mlx/model.safetensors")

# copy config + tokenizer from base model
for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "tokenizer.model", "generation_config.json"]:
    src = Path(base_dir) / fname
    if src.exists():
        shutil.copy(str(src), str(LLM_DIR / fname))

# patch vocab_size if GR00T added special tokens
if "model.embed_tokens.weight" in llm_bf16:
    actual_vocab = llm_bf16["model.embed_tokens.weight"].shape[0]
    cfg_path = LLM_DIR / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get("vocab_size") != actual_vocab:
        print(f"  Patching vocab_size: {cfg['vocab_size']} → {actual_vocab}")
        cfg["vocab_size"] = actual_vocab
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)


# ── 5. tokenizer + metadata ────────────────────────────────────────────────────
print("\n[5/6] Downloading Eagle tokenizer + writing metadata...")

eagle_dir = snapshot_download(
    EAGLE_REPO, token=HF_TOKEN,
    allow_patterns=["*.json", "*.model", "tokenizer*"],
)
tok_out = WEIGHTS_DIR / "eagle_tokenizer"
tok_out.mkdir(exist_ok=True)
for f in Path(eagle_dir).glob("*"):
    if f.is_file():
        shutil.copy(str(f), str(tok_out / f.name))
print(f"  Eagle tokenizer → eagle_tokenizer/")

# read image_size from eagle config
try:
    from transformers import AutoConfig
    eagle_cfg = AutoConfig.from_pretrained(eagle_dir, trust_remote_code=True)
    image_size = getattr(eagle_cfg, "force_image_size", None) or eagle_cfg.vision_config.image_size
    image_token_index = eagle_cfg.image_token_index
except Exception:
    image_size = 384
    image_token_index = 262144

with open(WEIGHTS_DIR / "config.json") as f:
    gr00t_cfg = json.load(f)

meta = {
    "image_size":        image_size,
    "image_token_index": image_token_index,
    "action_horizon":    gr00t_cfg.get("action_horizon", 16),
    "max_action_dim":    gr00t_cfg.get("max_action_dim", 128),
    "max_state_dim":     gr00t_cfg.get("max_state_dim", 128),
    "eagle_repo":        EAGLE_REPO,
    "gr00t_repo":        GR00T_REPO,
    "checkpoint":        CHECKPOINT,
}
with open(WEIGHTS_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"  meta.json written")


# ── 6. upload to HuggingFace ───────────────────────────────────────────────────
print(f"\n[6/6] Uploading to {UPLOAD_REPO}...")

api = HfApi(token=HF_TOKEN)
api.create_repo(UPLOAD_REPO, repo_type="model", private=False, exist_ok=True)

print("  Uploading gr00t_weights_mlx/ ...")
api.upload_folder(
    folder_path=str(WEIGHTS_DIR),
    repo_id=UPLOAD_REPO,
    path_in_repo="gr00t_weights_mlx",
    repo_type="model",
)
print("  ✓ gr00t_weights_mlx done")

print("  Uploading gr00t_llm_mlx/ ...")
api.upload_folder(
    folder_path=str(LLM_DIR),
    repo_id=UPLOAD_REPO,
    path_in_repo="gr00t_llm_mlx",
    repo_type="model",
)
print("  ✓ gr00t_llm_mlx done")

print(f"\nDone! Weights live at: https://huggingface.co/{UPLOAD_REPO}")

# cleanup
shutil.rmtree(str(OUT_DIR))
print("Cleaned up temp files.")

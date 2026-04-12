"""
export_weights.py — one-time export of vision + DiT weights to MLX safetensors.

Extracts only vision + DiT from the 7.9GB PyTorch checkpoint and saves
as MLX safetensors (~300-400MB).  The LLM is already in gr00t_llm_mlx/.

Usage:
    python export_weights.py
"""
import sys, json, glob, shutil
import mlx.core as mx
from mlx.utils import tree_flatten
from pathlib import Path

HERE  = Path(__file__).parent
ISAAC = HERE.parent / "Isaac-GR00T"
EAGLE = HERE.parent / "Eagle" / "Eagle2_5"
for p in [str(HERE), str(ISAAC), str(EAGLE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

GR00T_REPO = "youngbrett48/gr00t-post-train-fractal-270m"
CHECKPOINT = "checkpoint-2000"
HF_TOKEN   = True
OUT_DIR    = HERE / "gr00t_weights_mlx"


def save_mlx_model(model, path: Path):
    """Flatten MLX model params and save as safetensors."""
    flat = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(path), flat)
    print(f"  saved {len(flat)} tensors → {path.name}")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # ── 1. load PyTorch state dict ─────────────────────────────────────────
    print("[1/5] Downloading checkpoint (safetensors only)...")
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    allow = [
        f"{CHECKPOINT}/model.safetensors",
        f"{CHECKPOINT}/model.safetensors.index.json",
        f"{CHECKPOINT}/model-*-of-*.safetensors",
        f"{CHECKPOINT}/config.json",
        "statistics.json",
    ]
    ckpt_dir = snapshot_download(GR00T_REPO, allow_patterns=allow, token=HF_TOKEN)
    ckpt_path = Path(ckpt_dir) / CHECKPOINT

    shards = sorted(glob.glob(str(ckpt_path / "*.safetensors")))
    state_dict = {}
    for s in shards:
        state_dict.update(load_file(s))
    print(f"  {len(state_dict)} keys loaded")

    # copy config.json + statistics.json
    shutil.copy(str(ckpt_path / "config.json"), str(OUT_DIR / "config.json"))
    stats_src = Path(ckpt_dir) / "statistics.json"
    if stats_src.exists():
        shutil.copy(str(stats_src), str(OUT_DIR / "statistics.json"))

    # ── 2. Eagle2.5 config ────────────────────────────────────────────────
    print("[2/5] Loading Eagle2.5 config...")
    eagle_repo = "youngbrett48/train_stage2_gemma3_270m.sh"
    from eaglevl.model.eagle2_5.configuration_eagle2_5_vl import Eagle2_5_VLConfig
    from transformers import AutoConfig
    AutoConfig.register("eagle_2_5_vl", Eagle2_5_VLConfig)
    eagle_config = Eagle2_5_VLConfig.from_pretrained(
        eagle_repo, trust_remote_code=True, token=HF_TOKEN, local_files_only=True
    )

    # copy tokenizer files
    from huggingface_hub import snapshot_download as snap
    eagle_dir = snap(eagle_repo, token=HF_TOKEN,
                     allow_patterns=["*.json", "*.model", "tokenizer*"])
    eagle_out = OUT_DIR / "eagle_tokenizer"
    eagle_out.mkdir(exist_ok=True)
    for f in Path(eagle_dir).glob("*"):
        if f.is_file():
            shutil.copy(str(f), str(eagle_out / f.name))
    print(f"  Eagle2.5 tokenizer → eagle_tokenizer/")

    # ── 3. export vision model ────────────────────────────────────────────
    print("[3/5] Building + exporting vision model...")
    from vision_mlx import build_vision_mlx
    vision_model = build_vision_mlx(state_dict, eagle_config)
    mx.eval(vision_model.parameters())
    save_mlx_model(vision_model, OUT_DIR / "vision.safetensors")

    # ── 4. export DiT ─────────────────────────────────────────────────────
    print("[4/5] Building + exporting DiT action head...")
    from dit_mlx import build_dit_mlx
    dit, _ = build_dit_mlx(state_dict, str(OUT_DIR / "config.json"))
    mx.eval(dit.parameters())
    save_mlx_model(dit, OUT_DIR / "dit.safetensors")

    # free PyTorch memory
    del state_dict

    # ── 5. save metadata ──────────────────────────────────────────────────
    print("[5/5] Writing metadata...")
    with open(OUT_DIR / "config.json") as f:
        cfg = json.load(f)

    # SigLIP position embeddings are for 729 patches = 27x27, so native size = 27*14 = 378
    # eagle_config may report 384 but that's wrong — use 378 to match training
    import math as _math
    _num_patches = eagle_config.vision_config.num_positions if hasattr(eagle_config.vision_config, 'num_positions') else 729
    image_size = int(_math.sqrt(_num_patches)) * eagle_config.vision_config.patch_size  # 27*14=378
    meta = {
        "image_size":        image_size,
        "image_token_index": eagle_config.image_token_index,
        "action_horizon":    cfg.get("action_horizon", 16),
        "max_action_dim":    cfg.get("max_action_dim", 128),
        "max_state_dim":     cfg.get("max_state_dim", 128),
        "eagle_repo":        eagle_repo,
        "gr00t_repo":        GR00T_REPO,
        "checkpoint":        CHECKPOINT,
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    import subprocess
    sz = subprocess.run(["du", "-sh", str(OUT_DIR)],
                        capture_output=True, text=True).stdout.split()[0]
    print(f"\nDone. {OUT_DIR}/  ({sz})")
    print("Contents:")
    for f in sorted(OUT_DIR.iterdir()):
        fsize = subprocess.run(["du", "-sh", str(f)],
                               capture_output=True, text=True).stdout.split()[0]
        print(f"  {fsize:>6}  {f.name}")


if __name__ == "__main__":
    main()

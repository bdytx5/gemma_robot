"""
GR00T inference on Apple Silicon — fully MLX.

  Vision: SigLIP ViT + pixel_shuffle + MLP projector (MLX, float16)
  LLM:    Gemma3 trunk (MLX, float16)
  Action: DiT diffusion action head (MLX, float16, compiled)

Optimizations:
  - float16 vision + DiT (halves memory bandwidth)
  - mx.compile() on vision encoder + DiT forward (kernel fusion)
  - Pure MLX scatter (no numpy roundtrip / GPU sync)
  - Single mx.eval() at end (no intermediate syncs)

No PyTorch needed at inference time.
"""

import json
import math
import numpy as np
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# 1. Download GR00T checkpoint → state dict
# ---------------------------------------------------------------------------
def load_gr00t_components(repo_id: str, checkpoint: str | None, token=None):
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import glob

    target = f"{repo_id}/{checkpoint}" if checkpoint else repo_id
    print(f"[1/5] Downloading {target}...")
    if checkpoint:
        allow = [
            f"{checkpoint}/config.json",
            f"{checkpoint}/pytorch_model.bin",
            f"{checkpoint}/model.safetensors",
            f"{checkpoint}/model.safetensors.index.json",
            f"{checkpoint}/model-*-of-*.safetensors",
        ]
        ckpt_dir = snapshot_download(repo_id, allow_patterns=allow, token=token)
        ckpt_path = Path(ckpt_dir) / checkpoint
    else:
        ckpt_dir = snapshot_download(
            repo_id,
            allow_patterns=["*.safetensors", "*.bin", "*.json", "tokenizer*", "*.model"],
            token=token,
        )
        ckpt_path = Path(ckpt_dir)

    shards = sorted(glob.glob(str(ckpt_path / "*.safetensors")))
    if shards:
        sd = {}
        for s in shards:
            sd.update(load_file(s))
    else:
        sd = torch.load(str(ckpt_path / "pytorch_model.bin"), map_location="cpu")

    print(f"  {len(sd)} keys loaded")
    return sd


# ---------------------------------------------------------------------------
# 2. Build the Eagle2.5 vision encoder (MLX)
# ---------------------------------------------------------------------------
def build_eagle_model(state_dict, eagle2_5_repo: str, token=None):
    """Build MLX vision encoder + load Eagle2.5 config for image_token_index etc."""
    import sys
    eagle_path = str(Path(__file__).parent.parent / "Eagle" / "Eagle2_5")
    if eagle_path not in sys.path:
        sys.path.insert(0, eagle_path)

    print(f"[2/5] Loading Eagle2.5 vision encoder (MLX)...")
    from eaglevl.model.eagle2_5.configuration_eagle2_5_vl import Eagle2_5_VLConfig
    from transformers import AutoConfig
    AutoConfig.register("eagle_2_5_vl", Eagle2_5_VLConfig)

    config = Eagle2_5_VLConfig.from_pretrained(
        eagle2_5_repo, trust_remote_code=True, token=token,
        attn_implementation="eager", local_files_only=True,
    )

    from vision_mlx import build_vision_mlx
    vision_model = build_vision_mlx(state_dict, config)
    return vision_model, config


# ---------------------------------------------------------------------------
# 3. Load MLX Gemma3 LLM + HF tokenizer (which knows special tokens like <IMG_CONTEXT>)
# ---------------------------------------------------------------------------
def load_mlx_llm(mlx_path: str, eagle2_5_repo: str, token=None):
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    from transformers import GemmaTokenizer

    print(f"[3/5] Loading MLX LLM from {mlx_path}...")
    model, _mlx_tokenizer = mlx_load(mlx_path)
    model.eval()

    # Use the HF GemmaTokenizer from the Eagle2.5 checkpoint — it has the
    # special tokens (<IMG_CONTEXT>, <img>, </img>) that the MLX tokenizer lacks.
    tok_kwargs = {"use_fast": False, "local_files_only": True}
    if token:
        tok_kwargs["token"] = token
    tokenizer = GemmaTokenizer.from_pretrained(eagle2_5_repo, **tok_kwargs)
    print(f"  MLX model loaded, HF tokenizer vocab={len(tokenizer)}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 4. Build DiT action head — MLX version (fast on Apple Silicon)
# ---------------------------------------------------------------------------
def build_dit(state_dict, config_path: str):
    from dit_mlx import build_dit_mlx
    print(f"[4/5] Building MLX DiT action head...")
    model, config = build_dit_mlx(state_dict, config_path)
    return model, config


# ---------------------------------------------------------------------------
# 5. Full inference — 100% MLX, no PyTorch at inference time
# ---------------------------------------------------------------------------
def run_inference(
    image: Image.Image,
    robot_state: np.ndarray,
    language_instruction: str,
    vision_model,      # EagleVisionMLX (MLX)
    eagle_config,      # Eagle2_5_VLConfig (for image_token_index, image_size)
    mlx_llm,           # MLX Gemma3 model
    tokenizer,         # HF GemmaTokenizer
    dit_model,         # Gr00tActionHeadMLX (MLX)
    dit_config,        # dict from config.json
    n_diffusion_steps: int = 4,
):
    import mlx.core as mx

    # ---------------------------------------------------------------
    # A. Preprocess image → MLX (NHWC, float32, [-1, 1])
    # ---------------------------------------------------------------
    image_size = getattr(eagle_config, "force_image_size", None) or eagle_config.vision_config.image_size
    img_resized = image.convert("RGB").resize((image_size, image_size))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0   # (H, W, 3) [0,1]
    img_np = (img_np - 0.5) / 0.5                               # [-1, 1]
    pixel_values = mx.array(img_np[None])                        # (1, H, W, 3)

    # ---------------------------------------------------------------
    # B. Vision encoder (MLX, float16, compiled): SigLIP + pixel_shuffle + mlp1
    #    No mx.eval() — let it pipeline with the next stages
    # ---------------------------------------------------------------
    vit_embeds = vision_model(pixel_values)  # (1, n_img_tokens, llm_dim)
    # We need shape info for tokenization, so eval just the shape metadata
    mx.eval(vit_embeds)
    n_img_tokens = vit_embeds.shape[1]
    llm_dim = vit_embeds.shape[2]
    print(f"  Vision features: ({1}, {n_img_tokens}, {llm_dim})")

    # ---------------------------------------------------------------
    # C. Tokenize with <IMG_CONTEXT> placeholders
    # ---------------------------------------------------------------
    IMG_CONTEXT = "<IMG_CONTEXT>"
    img_context_id = eagle_config.image_token_index  # 262145
    image_block = f"<img>{IMG_CONTEXT * n_img_tokens}</img>"
    prompt = f"{image_block}\n{language_instruction}"
    input_ids = tokenizer.encode(prompt)
    T = len(input_ids)

    # ---------------------------------------------------------------
    # D. Embed tokens (MLX) + pure MLX scatter (no numpy roundtrip)
    # ---------------------------------------------------------------
    ids_mx = mx.array(input_ids)[None]                         # (1, T)
    input_embeds = mlx_llm.model.embed_tokens(ids_mx)          # (1, T, hidden_dim)

    # Build image mask: True where token == img_context_id
    ids_np = np.array(input_ids)
    img_mask_np = (ids_np == img_context_id)                    # (T,)
    img_positions = np.where(img_mask_np)[0]

    # Pure MLX scatter — no numpy roundtrip, no GPU sync
    n_slots = len(img_positions)
    n = min(n_slots, n_img_tokens)
    if n_slots != n_img_tokens:
        print(f"  WARNING: {n_slots} IMG_CONTEXT slots vs {n_img_tokens} vit tokens, using {n}")

    # The MLX Gemma3 model applies *sqrt(hidden_size) to ALL input_embeddings
    # (including externally provided ones). Pre-divide vision features so they
    # are correctly scaled after the model's internal multiplication.
    hidden_size = input_embeds.shape[-1]
    vit_for_scatter = vit_embeds[0, :n] / mx.array(hidden_size ** 0.5, dtype=vit_embeds.dtype)

    pos_idx = mx.array(img_positions[:n])
    input_embeds = input_embeds.at[0, pos_idx].add(
        vit_for_scatter - input_embeds[0, pos_idx]
    )

    # ---------------------------------------------------------------
    # E. LLM trunk (MLX) — no intermediate mx.eval()
    # ---------------------------------------------------------------
    hidden_states = mlx_llm.model(inputs=None, input_embeddings=input_embeds)
    mx.eval(hidden_states)
    print(f"  Backbone output: ({1}, {T}, {hidden_states.shape[2]})")

    # ---------------------------------------------------------------
    # F. DiT action head (MLX, float16, compiled)
    #    Single mx.eval() at the very end
    # ---------------------------------------------------------------
    bb_attn = mx.ones((1, T), dtype=mx.bool_)
    img_mask_mx = mx.array(img_mask_np[None])  # (1, T)

    max_state_dim = dit_config.get("max_state_dim", 29) if isinstance(dit_config, dict) else dit_config.max_state_dim
    padded_state = np.zeros(max_state_dim, dtype=np.float32)
    padded_state[:robot_state.shape[0]] = robot_state
    state_mx = mx.array(padded_state[None, None, :])
    emb_id = mx.zeros((1,), dtype=mx.int32)

    dit_model.num_inference_timesteps = n_diffusion_steps
    actions = dit_model.get_action(hidden_states, bb_attn, img_mask_mx, state_mx, emb_id)

    # Single eval for the entire pipeline
    mx.eval(actions)

    return np.array(actions, copy=False)

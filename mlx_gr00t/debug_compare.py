#!/usr/bin/env python3
"""
Diagnostic comparison: PyTorch (HF) vs MLX inference.

Loads the same checkpoint in both PyTorch and MLX, then compares:
  1. Weight values (DiT, vision)
  2. Vision encoder output
  3. LLM hidden-states
  4. DiT output / actions (deterministic: seeds fixed)

Usage:
    python debug_compare.py [--dtype bfloat16|float32|float16] [--stage all|weights|vision|llm|dit]
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path

# ── paths ───────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
EAGLE_PATH = str(HERE.parent / "Eagle" / "Eagle2_5")
ISAAC_PATH = str(HERE.parent / "Isaac-GR00T")
for p in [EAGLE_PATH, ISAAC_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

WEIGHTS_DIR  = HERE / "gr00t_weights_mlx"
MLX_LLM_PATH = HERE / "gr00t_llm_mlx"
HF_REPO      = "youngbrett48/gr00t-post-train-fractal-270m"
CHECKPOINT   = "checkpoint-2000"
EAGLE_REPO   = str(HERE.parent / "Eagle" / "Eagle2_5")  # local path

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32", "float16"],
                    help="dtype to use for MLX model")
parser.add_argument("--stage", default="all",
                    choices=["all", "weights", "vision", "llm", "dit"],
                    help="which comparison stage to run")
parser.add_argument("--no-pt", action="store_true",
                    help="skip PyTorch load (MLX-only checks)")
args = parser.parse_args()

print(f"\n{'='*60}")
print(f"  GR00T MLX vs PyTorch Diagnostic")
print(f"  dtype={args.dtype}  stage={args.stage}")
print(f"{'='*60}\n")

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def to_f32(arr):
    """Safely convert mlx or numpy array to float32 numpy."""
    try:
        import mlx.core as _mx
        if isinstance(arr, _mx.array):
            return np.array(arr.astype(_mx.float32)).flatten()
    except Exception:
        pass
    return np.array(arr, dtype=np.float32).flatten()


def arr_stats(arr, name=""):
    """Print min/max/mean/std of a numpy array."""
    arr = to_f32(arr)
    print(f"  {name:40s}  min={arr.min():.5f}  max={arr.max():.5f}  "
          f"mean={arr.mean():.5f}  std={arr.std():.5f}  shape={arr.shape}")


def compare(pt_arr, mlx_arr, name, rtol=1e-2, atol=1e-2):
    """Compare two arrays and print stats."""
    pt  = to_f32(pt_arr)
    mlx = to_f32(mlx_arr)
    if pt.shape != mlx.shape:
        print(f"  ❌ {name}: SHAPE MISMATCH  PT={pt.shape}  MLX={mlx.shape}")
        return False
    diff = np.abs(pt - mlx)
    max_diff = diff.max()
    mean_diff = diff.mean()
    rel_diff = (diff / (np.abs(pt) + 1e-8)).mean()
    ok = np.allclose(pt, mlx, rtol=rtol, atol=atol)
    sym = "✓" if ok else "❌"
    print(f"  {sym} {name:48s}  max_diff={max_diff:.5f}  mean_diff={mean_diff:.5f}  "
          f"rel_diff={rel_diff:.5f}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════
# 1. Load shared metadata / stats
# ═══════════════════════════════════════════════════════════════════════════
print("[1] Loading metadata...")
with open(WEIGHTS_DIR / "meta.json") as f:
    meta = json.load(f)
with open(WEIGHTS_DIR / "config.json") as f:
    dit_config = json.load(f)

IMAGE_SIZE        = meta["image_size"]         # 384
IMAGE_TOKEN_INDEX = meta["image_token_index"]  # 262145

print(f"  image_size={IMAGE_SIZE}  image_token_index={IMAGE_TOKEN_INDEX}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Build deterministic test inputs
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2] Building test inputs...")
rng = np.random.default_rng(42)

# Synthetic test image (H, W, 3) uint8
img_np_uint8 = (rng.random((IMAGE_SIZE, IMAGE_SIZE, 3)) * 255).astype(np.uint8)
# Normalize to [-1, 1]
img_norm = (img_np_uint8.astype(np.float32) / 255.0 - 0.5) / 0.5

robot_state = rng.random(8).astype(np.float32)
INSTRUCTION = "pick up the red block"
EMBODIMENT_ID = 0

print(f"  Image: {img_norm.shape}  range=[{img_norm.min():.2f}, {img_norm.max():.2f}]")
print(f"  State: {robot_state}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. Load PyTorch model
# ═══════════════════════════════════════════════════════════════════════════
pt_state = None
pt_backbone = None
pt_action_head = None

if not args.no_pt:
    print("\n[3] Loading PyTorch checkpoint...")
    import torch
    from safetensors.torch import load_file
    import glob

    from huggingface_hub import snapshot_download, hf_hub_download

    allow = [
        f"{CHECKPOINT}/config.json",
        f"{CHECKPOINT}/model.safetensors",
        f"{CHECKPOINT}/model.safetensors.index.json",
        f"{CHECKPOINT}/model-*-of-*.safetensors",
    ]
    print(f"  Downloading {HF_REPO}/{CHECKPOINT}...")
    ckpt_dir = snapshot_download(HF_REPO, allow_patterns=allow)
    ckpt_path = Path(ckpt_dir) / CHECKPOINT

    shards = sorted(glob.glob(str(ckpt_path / "*.safetensors")))
    pt_state = {}
    for s in shards:
        pt_state.update(load_file(s))
    print(f"  Loaded {len(pt_state)} keys from PyTorch checkpoint")

    # Print a few key prefixes
    prefixes = set()
    for k in pt_state:
        parts = k.split(".")
        if len(parts) >= 2:
            prefixes.add(".".join(parts[:2]))
    print(f"  Key prefixes: {sorted(prefixes)[:20]}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Weight comparison (DiT)
# ═══════════════════════════════════════════════════════════════════════════
if args.stage in ("all", "weights") and pt_state is not None:
    print("\n[4] Comparing DiT weights PT vs MLX...")

    import mlx.core as mx
    from dit_mlx import Gr00tActionHeadMLX, convert_torch_to_mlx

    dit_mlx_model = Gr00tActionHeadMLX(dit_config)

    # Load MLX weights from exported safetensors
    mlx_dit_weights = mx.load(str(WEIGHTS_DIR / "dit.safetensors"))
    dit_mlx_model.load_weights(list(mlx_dit_weights.items()))
    mx.eval(dit_mlx_model.parameters())

    # Extract PyTorch action_head weights
    ah_sd = {k[len("action_head."):]: v for k, v in pt_state.items()
             if k.startswith("action_head.")}
    print(f"  PT action_head keys: {len(ah_sd)}")

    # Convert PT → MLX
    mlx_from_pt = convert_torch_to_mlx(ah_sd)

    # Compare each mapped weight
    import mlx.utils
    mlx_flat = dict(mlx.utils.tree_flatten(dit_mlx_model.parameters()))

    n_ok = n_fail = 0
    for key, pt_tensor in sorted(mlx_from_pt.items()):
        if key not in mlx_flat:
            print(f"  ⚠ key not in MLX model: {key}")
            continue
        pt_np  = np.array(pt_tensor, dtype=np.float32)
        mlx_np = to_f32(mlx_flat[key])
        if pt_np.shape != mlx_np.shape:
            print(f"  ❌ shape mismatch {key}: PT={pt_np.shape} MLX={mlx_np.shape}")
            n_fail += 1
            continue
        if not np.allclose(pt_np, mlx_np, rtol=1e-3, atol=1e-3):
            diff = np.abs(pt_np - mlx_np).max()
            print(f"  ❌ value mismatch {key}: max_diff={diff:.5f}")
            n_fail += 1
        else:
            n_ok += 1
    print(f"  DiT weights: {n_ok} OK, {n_fail} FAILED out of {len(mlx_from_pt)}")

    # Check MLX keys not in PT
    pt_keyset = set(mlx_from_pt.keys())
    mlx_keyset = set(mlx_flat.keys())
    extra_mlx = mlx_keyset - pt_keyset
    extra_pt  = pt_keyset - mlx_keyset
    if extra_mlx:
        print(f"  ⚠ MLX keys with no PT match ({len(extra_mlx)}): {sorted(extra_mlx)[:10]}")
    if extra_pt:
        print(f"  ⚠ PT keys with no MLX match ({len(extra_pt)}): {sorted(extra_pt)[:10]}")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Vision encoder comparison
# ═══════════════════════════════════════════════════════════════════════════
if args.stage in ("all", "vision"):
    print("\n[5] Comparing vision encoder outputs...")

    import mlx.core as mx
    import mlx.nn as mxnn
    from vision_mlx import build_vision_mlx_from_exported

    # --- MLX vision ---
    dtype_map = {"bfloat16": mx.bfloat16, "float32": mx.float32, "float16": mx.float16}
    mlx_dtype = dtype_map[args.dtype]

    vision_mlx = build_vision_mlx_from_exported(str(WEIGHTS_DIR / "vision.safetensors"), meta)

    img_mlx = mx.array(img_norm[None])  # (1, H, W, 3)
    vit_out_mlx = vision_mlx(img_mlx)
    mx.eval(vit_out_mlx)
    vit_mlx_np = to_f32(vit_out_mlx).reshape(vit_out_mlx.shape)
    arr_stats(vit_mlx_np, "MLX vision output")

    if pt_state is not None:
        # --- PyTorch vision (Eagle2.5 SigLIP) ---
        print("  Running PyTorch vision encoder...")
        try:
            from eaglevl.model.eagle2_5.modeling_eagle2_5_vl import (
                Eagle2_5_VLForConditionalGeneration,
            )
            from transformers import AutoConfig
            from eaglevl.model.eagle2_5.configuration_eagle2_5_vl import Eagle2_5_VLConfig
            AutoConfig.register("eagle_2_5_vl", Eagle2_5_VLConfig, exist_ok=True)

            # Load just the vision parts
            import torch
            # Build a minimal model (load bf16 for speed)
            eagle_cfg_path = hf_hub_download(HF_REPO, f"{CHECKPOINT}/config.json")
            # Actually load the Eagle2.5 backbone tokenizer / model locally
            # The backbone is "youngbrett48/train_stage2_gemma3_270m.sh"
            eagle_backbone_repo = dit_config.get("model_name", "youngbrett48/train_stage2_gemma3_270m.sh")

            # Get vision encoder from state_dict directly
            from vision_mlx import convert_vision_weights, EagleVisionMLX
            from eaglevl.model.eagle2_5.configuration_eagle2_5_vl import Eagle2_5_VLConfig
            eagle_config = Eagle2_5_VLConfig.from_pretrained(
                eagle_backbone_repo, trust_remote_code=True, local_files_only=True
            )
            vc = eagle_config.vision_config
            # Build PT SigLIP manually and load weights from state_dict
            from transformers import SiglipVisionModel
            # Extract siglip weights
            siglip_prefix = "backbone.model.vision_model."
            siglip_sd = {k[len(siglip_prefix):]: v
                         for k, v in pt_state.items()
                         if k.startswith(siglip_prefix)}
            print(f"  SigLIP PT keys: {len(siglip_sd)}")

            # Use transformers SigLIP with exact same config
            from transformers import SiglipVisionConfig, SiglipVisionModel
            siglip_config = SiglipVisionConfig(
                hidden_size=vc.hidden_size,
                num_attention_heads=vc.num_attention_heads,
                intermediate_size=vc.intermediate_size,
                num_hidden_layers=vc.num_hidden_layers,
                image_size=vc.image_size,
                patch_size=vc.patch_size,
            )
            pt_siglip = SiglipVisionModel(siglip_config)
            missing, unexpected = pt_siglip.load_state_dict(
                {k: v for k, v in siglip_sd.items() if "vision_model." in k},
                strict=False
            )
            if missing:
                print(f"  SigLIP missing keys: {missing[:5]}")
            pt_siglip = pt_siglip.to(torch.bfloat16).eval()

            # Also load MLP1
            mlp1_prefix = "backbone.model.mlp1."
            mlp1_sd = {k[len(mlp1_prefix):]: v for k, v in pt_state.items()
                       if k.startswith(mlp1_prefix)}

            # Process image the same way as the backbone collator
            # SigLIP normalization: mean=0.5, std=0.5 (same as our [-1,1] norm) ✓
            img_pt = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16)  # (1,3,H,W)

            with torch.no_grad():
                vit_out_pt = pt_siglip.vision_model(
                    pixel_values=img_pt,
                    output_hidden_states=False,
                ).last_hidden_state  # (1, 729, 1152)

            # Apply pixel_shuffle + mlp1 using PT weights
            import torch.nn.functional as F

            # pixel_shuffle in PT
            def pt_pixel_shuffle(x, scale=0.5):
                B, N, C = x.shape
                import math
                h = w = int(math.sqrt(N))
                x = x.reshape(B, h, w, C)
                if h % 2 != 0:
                    x = F.pad(x.permute(0,3,1,2), (0,1,0,1)).permute(0,2,3,1)
                    h += 1; w += 1
                # Eagle2.5 pixel_shuffle
                x = x.reshape(B, w, int(h*scale), int(C/scale))
                x = x.permute(0,2,1,3)
                x = x.reshape(B, int(h*scale), int(w*scale), int(C/(scale*scale)))
                x = x.permute(0,2,1,3)
                return x.reshape(B, -1, x.shape[-1])

            vit_ps_pt = pt_pixel_shuffle(vit_out_pt)

            # MLP1: LayerNorm → Linear → GELU → Linear
            ln_w = mlp1_sd["0.weight"].float()
            ln_b = mlp1_sd["0.bias"].float()
            l1_w = mlp1_sd["1.weight"].float()
            l1_b = mlp1_sd["1.bias"].float()
            l2_w = mlp1_sd["3.weight"].float()
            l2_b = mlp1_sd["3.bias"].float()

            x = vit_ps_pt.float()
            # LayerNorm
            x = F.layer_norm(x, [x.shape[-1]], weight=ln_w, bias=ln_b)
            # Linear + GELU
            x = F.linear(x, l1_w, l1_b)
            x = F.gelu(x)  # exact GELU
            # Linear
            x = F.linear(x, l2_w, l2_b)

            vit_pt_np = x.numpy()  # (1, 196, 640)
            arr_stats(vit_pt_np, "PT vision output")

            compare(vit_pt_np, vit_mlx_np, "vision_encoder_output", rtol=5e-2, atol=5e-2)

        except Exception as e:
            print(f"  ⚠ PyTorch vision comparison failed: {e}")
            import traceback; traceback.print_exc()
    else:
        print("  (skipping PT vision — --no-pt set)")


# ═══════════════════════════════════════════════════════════════════════════
# 6. LLM hidden-states comparison
# ═══════════════════════════════════════════════════════════════════════════
if args.stage in ("all", "llm"):
    print("\n[6] Comparing LLM hidden-states (vision-injected)...")

    import mlx.core as mx
    from mlx_lm import load as mlx_load
    import mlx.utils as mlx_utils
    from transformers import GemmaTokenizer
    from vision_mlx import build_vision_mlx_from_exported

    dtype_map = {"bfloat16": mx.bfloat16, "float32": mx.float32, "float16": mx.float16}
    mlx_dtype = dtype_map[args.dtype]

    # Load MLX LLM
    print("  Loading MLX LLM...")
    llm_mlx, _ = mlx_load(str(MLX_LLM_PATH))
    llm_mlx.eval()
    llm_mlx.load_weights(list(mlx_utils.tree_flatten(
        mlx_utils.tree_map(
            lambda x: x.astype(mlx_dtype) if isinstance(x, mx.array) else x,
            llm_mlx.parameters()
        )
    )))

    tokenizer = GemmaTokenizer.from_pretrained(
        str(WEIGHTS_DIR / "eagle_tokenizer"), use_fast=False, local_files_only=True
    )
    print(f"  Tokenizer vocab={len(tokenizer)}")

    vision_mlx = build_vision_mlx_from_exported(str(WEIGHTS_DIR / "vision.safetensors"), meta)

    # Build prompt
    IMG_CONTEXT = "<IMG_CONTEXT>"
    img_mlx = mx.array(img_norm[None])
    vit_embeds_mlx = vision_mlx(img_mlx)
    mx.eval(vit_embeds_mlx)
    n_img_tokens = vit_embeds_mlx.shape[1]
    print(f"  n_img_tokens={n_img_tokens}")

    image_block = f"<img>{IMG_CONTEXT * n_img_tokens}</img>"
    prompt = (
        f"<start_of_turn>user\n"
        f"{image_block}\n"
        f"{INSTRUCTION}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    input_ids = tokenizer.encode(prompt)
    print(f"  Prompt tokens: {len(input_ids)}")

    ids_np = np.array(input_ids)
    img_mask_np = (ids_np == IMAGE_TOKEN_INDEX)
    img_positions = np.where(img_mask_np)[0]
    n = min(len(img_positions), n_img_tokens)
    print(f"  IMG_CONTEXT positions in prompt: {len(img_positions)}, using {n}")

    # Embed + inject vision
    ids_mx = mx.array(input_ids)[None]
    input_embeds = llm_mlx.model.embed_tokens(ids_mx)

    hidden_size = input_embeds.shape[-1]
    print(f"  LLM hidden_size={hidden_size}")

    vit_scaled = vit_embeds_mlx[0, :n] / mx.array(hidden_size ** 0.5, dtype=vit_embeds_mlx.dtype)
    pos_idx = mx.array(img_positions[:n])
    input_embeds = input_embeds.at[0, pos_idx].add(vit_scaled - input_embeds[0, pos_idx])

    arr_stats(input_embeds, "MLX input_embeds (post vision injection)")

    # Run LLM
    hidden_states_mlx = llm_mlx.model(inputs=None, input_embeddings=input_embeds)
    mx.eval(hidden_states_mlx)
    hs_mlx_np = to_f32(hidden_states_mlx).reshape(1, len(input_ids), -1)
    arr_stats(hs_mlx_np, "MLX LLM hidden_states output")

    # Check for NaN/Inf
    if np.isnan(hs_mlx_np).any():
        print("  ❌ NaN detected in MLX LLM output!")
    if np.isinf(hs_mlx_np).any():
        print("  ❌ Inf detected in MLX LLM output!")

    if pt_state is not None:
        # PyTorch LLM comparison via full Eagle2.5 backbone
        print("\n  Running PyTorch Eagle2.5 backbone...")
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from eaglevl.model.eagle2_5.modeling_eagle2_5_vl import (
                Eagle2_5_VLForConditionalGeneration,
            )
            from eaglevl.model.eagle2_5.configuration_eagle2_5_vl import Eagle2_5_VLConfig
            from transformers import AutoConfig

            AutoConfig.register("eagle_2_5_vl", Eagle2_5_VLConfig, exist_ok=True)

            eagle_backbone_repo = "youngbrett48/train_stage2_gemma3_270m.sh"
            print(f"  Loading Eagle2.5 backbone: {eagle_backbone_repo}")

            pt_eagle = Eagle2_5_VLForConditionalGeneration.from_pretrained(
                eagle_backbone_repo,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                trust_remote_code=True,
                local_files_only=True,
            )

            # Load the fine-tuned backbone weights from checkpoint
            eagle_sd = {
                k[len("backbone.model."):]: v
                for k, v in pt_state.items()
                if k.startswith("backbone.model.")
            }
            print(f"  Fine-tuned backbone keys: {len(eagle_sd)}")
            missing, unexp = pt_eagle.load_state_dict(eagle_sd, strict=False)
            if missing[:5]:
                print(f"  Missing: {missing[:5]}")
            if unexp[:5]:
                print(f"  Unexpected: {unexp[:5]}")
            pt_eagle = pt_eagle.eval().to(torch.bfloat16)

            # ---- PT tokenizer ----
            from transformers import GemmaTokenizer as PTGemmaTokenizer
            pt_tok = PTGemmaTokenizer.from_pretrained(
                eagle_backbone_repo, use_fast=False, local_files_only=True
            )
            pt_image_token_idx = pt_tok.convert_tokens_to_ids("<IMG_CONTEXT>")
            print(f"  PT image_token_index={pt_image_token_idx}")

            # ---- Build same prompt in PT ----
            pt_input_ids_raw = pt_tok.encode(prompt)
            pt_ids = torch.tensor([pt_input_ids_raw], dtype=torch.long)

            img_pt = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).to(torch.bfloat16)

            with torch.no_grad():
                # Run the full Eagle2.5 forward to get backbone_features
                # We need to replicate what Eagle2_5Backbone.forward() does
                pt_out = pt_eagle(
                    input_ids=pt_ids,
                    pixel_values=img_pt,
                    output_hidden_states=False,
                    return_dict=True,
                )
                # Eagle2.5 outputs hidden states from the last LLM layer
                # backbone_features = last hidden state
                hs_pt = pt_out.hidden_states if hasattr(pt_out, 'hidden_states') else None
                if hs_pt is None and hasattr(pt_out, 'last_hidden_state'):
                    hs_pt = pt_out.last_hidden_state

            if hs_pt is not None:
                hs_pt_np = hs_pt.to(torch.float32).numpy()
                arr_stats(hs_pt_np, "PT LLM hidden_states output")
                compare(hs_pt_np, hs_mlx_np, "LLM hidden_states", rtol=0.05, atol=0.05)
            else:
                print(f"  PT output keys: {list(pt_out.keys()) if hasattr(pt_out, 'keys') else dir(pt_out)}")

        except Exception as e:
            print(f"  ⚠ PyTorch LLM comparison failed: {e}")
            import traceback; traceback.print_exc()

    # ---- Check the vit scale factor ----
    print("\n  --- Checking vision scaling assumption ---")
    print(f"  MLX divides vit_embeds by sqrt({hidden_size}) = {hidden_size**0.5:.2f}")
    print(f"  (This assumes Gemma3 MLX embed_tokens multiplies by sqrt(hidden_size))")

    # Verify by checking embed_tokens scaling
    test_tok_id = 1  # some valid token
    test_tok_emb = llm_mlx.model.embed_tokens(mx.array([[test_tok_id]]))
    mx.eval(test_tok_emb)
    test_emb_np = to_f32(test_tok_emb)
    arr_stats(test_emb_np, f"embed_tokens(id={test_tok_id}) scale check")
    print(f"  If values are ~sqrt({hidden_size})={hidden_size**0.5:.2f}x raw embedding, scaling is applied")

    # Check if Gemma3 model has scale_embedding flag
    try:
        print(f"  llm_mlx model config: {llm_mlx.model.config if hasattr(llm_mlx.model, 'config') else 'N/A'}")
    except:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 7. DiT action head comparison (deterministic with fixed seed)
# ═══════════════════════════════════════════════════════════════════════════
if args.stage in ("all", "dit"):
    print("\n[7] Comparing DiT action head (deterministic)...")

    import mlx.core as mx
    from mlx_lm import load as mlx_load
    import mlx.utils as mlx_utils
    from transformers import GemmaTokenizer
    from vision_mlx import build_vision_mlx_from_exported
    from dit_mlx import Gr00tActionHeadMLX, build_dit_mlx_from_exported

    dtype_map = {"bfloat16": mx.bfloat16, "float32": mx.float32, "float16": mx.float16}
    mlx_dtype = dtype_map[args.dtype]

    # Load MLX components
    vision_mlx = build_vision_mlx_from_exported(str(WEIGHTS_DIR / "vision.safetensors"), meta)
    llm_mlx, _ = mlx_load(str(MLX_LLM_PATH))
    llm_mlx.eval()
    llm_mlx.load_weights(list(mlx_utils.tree_flatten(
        mlx_utils.tree_map(
            lambda x: x.astype(mlx_dtype) if isinstance(x, mx.array) else x,
            llm_mlx.parameters()
        )
    )))
    tokenizer = GemmaTokenizer.from_pretrained(
        str(WEIGHTS_DIR / "eagle_tokenizer"), use_fast=False, local_files_only=True
    )
    dit_mlx, _ = build_dit_mlx_from_exported(
        str(WEIGHTS_DIR / "dit.safetensors"),
        str(WEIGHTS_DIR / "config.json"),
    )

    # Build prompt and run LLM
    IMG_CONTEXT = "<IMG_CONTEXT>"
    img_mlx = mx.array(img_norm[None])
    vit_embeds_mlx = vision_mlx(img_mlx)
    mx.eval(vit_embeds_mlx)
    n_img_tokens = vit_embeds_mlx.shape[1]

    image_block = f"<img>{IMG_CONTEXT * n_img_tokens}</img>"
    prompt = (
        f"<start_of_turn>user\n"
        f"{image_block}\n"
        f"{INSTRUCTION}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    input_ids = tokenizer.encode(prompt)

    ids_np = np.array(input_ids)
    img_mask_np = (ids_np == IMAGE_TOKEN_INDEX)
    img_positions = np.where(img_mask_np)[0]
    n = min(len(img_positions), n_img_tokens)

    ids_mx = mx.array(input_ids)[None]
    input_embeds = llm_mlx.model.embed_tokens(ids_mx)
    hidden_size = input_embeds.shape[-1]
    vit_scaled = vit_embeds_mlx[0, :n] / mx.array(hidden_size ** 0.5, dtype=vit_embeds_mlx.dtype)
    pos_idx = mx.array(img_positions[:n])
    input_embeds = input_embeds.at[0, pos_idx].add(vit_scaled - input_embeds[0, pos_idx])

    hidden_states_mlx = llm_mlx.model(inputs=None, input_embeddings=input_embeds)
    mx.eval(hidden_states_mlx)

    T = len(input_ids)
    bb_attn = mx.ones((1, T), dtype=mx.bool_)
    img_mask_mx = mx.array(img_mask_np[None])

    # Robot state (padded to max_state_dim=128)
    padded_state = np.zeros(128, dtype=np.float32)
    padded_state[:len(robot_state)] = robot_state
    state_mx = mx.array(padded_state[None, None, :])
    emb_id_mx = mx.full((1,), EMBODIMENT_ID, dtype=mx.int32)

    # Run DiT with fixed random seed
    mx.random.seed(42)
    dit_mlx.num_inference_timesteps = 4
    actions_mlx = dit_mlx.get_action(hidden_states_mlx, bb_attn, img_mask_mx, state_mx, emb_id_mx)
    mx.eval(actions_mlx)
    actions_mlx_np = to_f32(actions_mlx).reshape(1, dit_mlx.action_horizon, -1)

    arr_stats(actions_mlx_np, "MLX DiT actions (raw, no denorm)")
    print(f"  MLX actions shape: {actions_mlx_np.shape}")
    print(f"  MLX actions [0,:5,:7]:\n{actions_mlx_np[0,:5,:7]}")

    if pt_state is not None:
        print("\n  Running PyTorch DiT...")
        try:
            import torch
            from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
            from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
            from transformers.feature_extraction_utils import BatchFeature

            # Load PT config
            from huggingface_hub import hf_hub_download
            pt_cfg_path = hf_hub_download(HF_REPO, f"{CHECKPOINT}/config.json")
            with open(pt_cfg_path) as f:
                pt_cfg_dict = json.load(f)

            pt_config = Gr00tN1d6Config(**{k: v for k, v in pt_cfg_dict.items()
                                           if hasattr(Gr00tN1d6Config, k)})

            pt_action_head = Gr00tN1d6ActionHead(pt_config).eval().to(torch.bfloat16)

            # Load weights
            ah_sd = {k[len("action_head."):]: v.to(torch.bfloat16)
                     for k, v in pt_state.items()
                     if k.startswith("action_head.")}
            missing, unexp = pt_action_head.load_state_dict(ah_sd, strict=False)
            if missing[:5]:
                print(f"  Missing PT action_head keys: {missing[:5]}")

            # Build synthetic backbone_features from MLX hidden states
            hs_pt = torch.from_numpy(to_f32(hidden_states_mlx).reshape(1, len(input_ids), -1)).to(torch.bfloat16)
            bb_attn_pt = torch.ones((1, T), dtype=torch.bool)
            img_mask_pt = torch.tensor(img_mask_np[None])

            backbone_output = BatchFeature(data={
                "backbone_features": hs_pt,
                "backbone_attention_mask": bb_attn_pt,
                "image_mask": img_mask_pt,
            })

            state_pt = torch.from_numpy(padded_state[None, None, :]).to(torch.bfloat16)
            emb_id_pt = torch.tensor([EMBODIMENT_ID], dtype=torch.long)

            action_input = BatchFeature(data={
                "state": state_pt,
                "embodiment_id": emb_id_pt,
            })

            # Fix seed
            torch.manual_seed(42)
            with torch.no_grad():
                pt_result = pt_action_head.get_action(backbone_output, action_input)

            actions_pt_np = pt_result["action_pred"].to(torch.float32).numpy()
            arr_stats(actions_pt_np, "PT DiT actions (raw, no denorm)")
            print(f"  PT actions [0,:5,:7]:\n{actions_pt_np[0,:5,:7]}")

            compare(actions_pt_np, actions_mlx_np, "DiT_actions_raw", rtol=0.1, atol=0.1)

            # Per-step comparison
            print("\n  Per action-dim comparison (first timestep):")
            for i in range(min(7, actions_pt_np.shape[-1])):
                compare(actions_pt_np[0, :, i], actions_mlx_np[0, :, i],
                        f"  action_dim_{i}", rtol=0.1, atol=0.1)

        except Exception as e:
            print(f"  ⚠ PyTorch DiT comparison failed: {e}")
            import traceback; traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════
# 8. Norm stats sanity check
# ═══════════════════════════════════════════════════════════════════════════
print("\n[8] Norm stats sanity check...")
with open(WEIGHTS_DIR / "statistics.json") as f:
    all_stats = json.load(f)

oxe_stats = all_stats.get("oxe_google", {})
if oxe_stats:
    s = oxe_stats.get("state", {})
    a = oxe_stats.get("action", {})

    state_keys = ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
    print("  State stats:")
    for k in state_keys:
        if k in s:
            print(f"    {k}: min={s[k].get('min')}, max={s[k].get('max')}")

    print("  Action stats (mean/std for x,y,z,roll,pitch,yaw; min/max for gripper):")
    ms_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    for k in ms_keys:
        if k in a:
            print(f"    {k}: mean={a[k].get('mean')}, std={a[k].get('std')}")
    if "gripper" in a:
        print(f"    gripper: min={a['gripper'].get('min')}, max={a['gripper'].get('max')}")
else:
    print("  ⚠ No oxe_google stats found!")
    print(f"  Available keys: {list(all_stats.keys())[:10]}")

print("\n[DONE]")

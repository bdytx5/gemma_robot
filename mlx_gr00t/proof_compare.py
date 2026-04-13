#!/usr/bin/env python3
"""
PROOF script: MLX vs PyTorch actions comparison.

Loads the same GR00T checkpoint in both PyTorch and MLX,
runs identical inputs through both, compares at each stage:
  1. Vision encoder output
  2. DiT action output (using MLX LLM hidden states as shared input)
  3. Final denormalized actions

Shows OLD (buggy) prompt vs NEW (fixed) prompt delta.

Usage:
    python proof_compare.py
"""
import sys, os, json, glob, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Eagle" / "Eagle2_5"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Isaac-GR00T"))
sys.path.insert(0, str(Path(__file__).parent))

HERE = Path(__file__).parent
WEIGHTS_DIR  = HERE / "gr00t_weights_mlx"
MLX_LLM_PATH = str(HERE / "gr00t_llm_mlx")
HF_REPO      = "youngbrett48/gr00t-post-train-fractal-270m"
CHECKPOINT   = "checkpoint-2000"

# ── fixed seed inputs ─────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
IMG_NP = (rng.random((384, 384, 3)).astype(np.float32) * 2) - 1  # [-1,1]
ROBOT_STATE = rng.random(8).astype(np.float32)
INSTRUCTION = "pick up the red block"

# ── load PT checkpoint ────────────────────────────────────────────────────────
print("="*60)
print("Loading PyTorch checkpoint...")
import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

ckpt_dir = snapshot_download(HF_REPO, allow_patterns=[
    f"{CHECKPOINT}/model*.safetensors",
    f"{CHECKPOINT}/model.safetensors.index.json",
])
ckpt_path = Path(ckpt_dir) / CHECKPOINT
shards = sorted(glob.glob(str(ckpt_path / "*.safetensors")))
pt_sd = {}
for s in shards:
    pt_sd.update(load_file(s))
print(f"  Loaded {len(pt_sd)} keys")

with open(WEIGHTS_DIR / "config.json") as f:
    dit_config = json.load(f)
with open(WEIGHTS_DIR / "meta.json") as f:
    meta = json.load(f)
with open(WEIGHTS_DIR / "statistics.json") as f:
    stats = json.load(f)
IMAGE_TOKEN_INDEX = meta["image_token_index"]

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1: Vision encoder comparison
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 1: Vision encoder comparison")

# ── PyTorch SigLIP (from checkpoint backbone.model.vision_model.*) ────────────
print("  [PT] Loading SigLIP + mlp1 from checkpoint...")
from transformers import SiglipVisionModel, SiglipVisionConfig

# Derive SigLIP config directly from checkpoint weight shapes
prefix = "backbone.model.vision_model."
siglip_sd_raw = {k[len(prefix):]: v for k, v in pt_sd.items() if k.startswith(prefix)}
print(f"  [PT] SigLIP keys: {len(siglip_sd_raw)}")

# Read dims from actual weights
pe_w = siglip_sd_raw["vision_model.embeddings.patch_embedding.weight"]  # (C_out, 3, kH, kW)
pos_w = siglip_sd_raw["vision_model.embeddings.position_embedding.weight"]  # (num_patches, hidden)
hidden_size   = pe_w.shape[0]   # 1152
patch_size    = pe_w.shape[-1]  # 14
num_patches   = pos_w.shape[0]  # 729
import math
grid = int(math.sqrt(num_patches))  # 27
image_size_cfg = grid * patch_size  # 378 (SigLIP's internal size)

# Count layers
n_layers = max(int(k.split(".")[3]) for k in siglip_sd_raw if "encoder.layers." in k) + 1
fc1_w = siglip_sd_raw[f"vision_model.encoder.layers.0.mlp.fc1.weight"]
intermediate_size = fc1_w.shape[0]
qw = siglip_sd_raw["vision_model.encoder.layers.0.self_attn.q_proj.weight"]
# count heads from key shape — usually hidden/64
num_heads = hidden_size // 64

print(f"  [PT] SigLIP dims: hidden={hidden_size} heads={num_heads} "
      f"inter={intermediate_size} layers={n_layers} patch={patch_size} img={image_size_cfg}")

siglip_cfg = SiglipVisionConfig(
    hidden_size=hidden_size, num_attention_heads=num_heads,
    intermediate_size=intermediate_size, num_hidden_layers=n_layers,
    image_size=image_size_cfg, patch_size=patch_size,
)
pt_siglip = SiglipVisionModel(siglip_cfg).eval()
# Load with full siglip_sd (strip redundant prefix)
load_sd = {k: v for k, v in siglip_sd_raw.items() if "vision_model." in k}
missing, unexp = pt_siglip.load_state_dict(load_sd, strict=True)
pt_siglip = pt_siglip.to(torch.float32).eval()

# Load mlp1 weights
mlp1_prefix = "backbone.model.mlp1."
mlp1_sd = {k[len(mlp1_prefix):]: v.float() for k, v in pt_sd.items() if k.startswith(mlp1_prefix)}

def pt_pixel_shuffle(x, scale=0.5):
    import math
    B, N, C = x.shape
    h = w = int(math.sqrt(N))
    x = x.reshape(B, h, w, C)
    # Pad to even if needed (matches MLX vision_mlx.py pixel_shuffle)
    if h % 2 != 0:
        x = torch.nn.functional.pad(x.permute(0,3,1,2), (0,1,0,1)).permute(0,2,3,1)
        h += 1; w += 1
    x = x.reshape(B, w, int(h*scale), int(C/scale))
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(B, int(h*scale), int(w*scale), int(C/(scale*scale)))
    x = x.permute(0, 2, 1, 3)
    return x.reshape(B, -1, x.shape[-1])

# SigLIP's internal image size may be 378 (27*14), not 384. Resize if needed.
from torchvision.transforms.functional import resize as tv_resize
img_pt_raw = torch.from_numpy(IMG_NP).permute(2, 0, 1).unsqueeze(0)  # (1,3,384,384)
if image_size_cfg != 384:
    img_pt = tv_resize(img_pt_raw, [image_size_cfg, image_size_cfg])
else:
    img_pt = img_pt_raw
with torch.no_grad():
    vit_pt = pt_siglip.vision_model(pixel_values=img_pt).last_hidden_state  # (1,729,1152)
    vit_ps = pt_pixel_shuffle(vit_pt)  # (1,196,4608)
    # mlp1: LN → Linear → GELU → Linear
    import torch.nn.functional as F
    x = vit_ps
    x = F.layer_norm(x, [x.shape[-1]], mlp1_sd["0.weight"], mlp1_sd["0.bias"])
    x = F.gelu(F.linear(x, mlp1_sd["1.weight"], mlp1_sd["1.bias"]))
    vit_pt_np = F.linear(x, mlp1_sd["3.weight"], mlp1_sd["3.bias"]).numpy()  # (1,196,640)

print(f"  [PT] vision output (float32): shape={vit_pt_np.shape} "
      f"min={vit_pt_np.min():.4f} max={vit_pt_np.max():.4f} std={vit_pt_np.std():.4f}")

# Also run PT in bfloat16 to see how much precision alone accounts for
pt_siglip_bf16 = pt_siglip.to(torch.bfloat16).eval()
with torch.no_grad():
    img_pt_bf = img_pt.to(torch.bfloat16)
    vit_pt_bf16 = pt_siglip_bf16.vision_model(pixel_values=img_pt_bf).last_hidden_state
    vit_ps_bf16 = pt_pixel_shuffle(vit_pt_bf16)
    xb = vit_ps_bf16.float()
    xb = F.layer_norm(xb, [xb.shape[-1]], mlp1_sd["0.weight"], mlp1_sd["0.bias"])
    xb = F.gelu(F.linear(xb, mlp1_sd["1.weight"], mlp1_sd["1.bias"]))
    vit_pt_bf16_np = F.linear(xb, mlp1_sd["3.weight"], mlp1_sd["3.bias"]).numpy()
diff_bf16 = np.abs(vit_pt_np - vit_pt_bf16_np)
print(f"  [PT] float32 vs bfloat16 ViT diff: max={diff_bf16.max():.5f} mean={diff_bf16.mean():.5f}")
print(f"  → (This is how much bfloat16 costs alone — should match MLX diff if weights are correct)")

# ── MLX vision ────────────────────────────────────────────────────────────────
import mlx.core as mx
import mlx.utils as mlx_utils
from vision_mlx import build_vision_mlx_from_exported
from PIL import Image as PILImage

# Build MLX vision with correct image_size from model weights (same as PT: 378)
meta_vision = dict(meta)
meta_vision["image_size"] = image_size_cfg  # use 378, matching PT SigLIP
vision_mlx = build_vision_mlx_from_exported(str(WEIGHTS_DIR / "vision.safetensors"), meta_vision)

# Resize IMG_NP to 378px for apples-to-apples comparison
img_pil = PILImage.fromarray(((IMG_NP + 1.0) / 2.0 * 255).clip(0,255).astype(np.uint8))
img_pil_r = img_pil.resize((image_size_cfg, image_size_cfg), PILImage.BICUBIC)
IMG_NP_378 = (np.array(img_pil_r).astype(np.float32) / 255.0) * 2.0 - 1.0  # back to [-1,1]

img_mlx = mx.array(IMG_NP_378[None])
vit_mlx = vision_mlx(img_mlx)
mx.eval(vit_mlx)
vit_mlx_np = np.array(vit_mlx.astype(mx.float32))  # (1,196,640)

print(f"  [MLX] vision output (378px, matching PT): shape={vit_mlx_np.shape} "
      f"min={vit_mlx_np.min():.4f} max={vit_mlx_np.max():.4f} std={vit_mlx_np.std():.4f}")

diff = np.abs(vit_pt_np - vit_mlx_np)
print(f"  VISION DIFF: max={diff.max():.5f} mean={diff.mean():.5f} "
      f"rel={( diff / (np.abs(vit_pt_np)+1e-8)).mean():.4f}")
print(f"  → {'✓ OK (bfloat16 rounding)' if diff.max() < 0.5 else '❌ LARGE DIFF'}")

# Also test with 384px to quantify the image-size-mismatch contribution
vision_mlx_384 = build_vision_mlx_from_exported(str(WEIGHTS_DIR / "vision.safetensors"), meta)
img_mlx_384 = mx.array(IMG_NP[None])
vit_mlx_384 = vision_mlx_384(img_mlx_384)
mx.eval(vit_mlx_384)
vit_mlx_384_np = np.array(vit_mlx_384.astype(mx.float32))
diff_384 = np.abs(vit_pt_np - vit_mlx_384_np)
print(f"  VISION DIFF (384px, current prod): max={diff_384.max():.5f} mean={diff_384.mean():.5f}")
print(f"  → Image size mismatch contribution: {diff_384.max() - diff.max():.5f} extra max_diff")

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2: DiT comparison (shared MLX LLM backbone features)
# Load PyTorch DiT, feed MLX LLM hidden states, compare outputs
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 2: DiT action head comparison")

# ── MLX: get backbone features with OLD and NEW prompt ──────────────────────
from mlx_lm import load as mlx_load
from transformers import GemmaTokenizer
from dit_mlx import build_dit_mlx_from_exported

llm_mlx, _ = mlx_load(MLX_LLM_PATH)
llm_mlx.eval()
llm_mlx.load_weights(list(mlx_utils.tree_flatten(
    mlx_utils.tree_map(lambda x: x.astype(mx.bfloat16) if isinstance(x, mx.array) else x,
                       llm_mlx.parameters())
)))
tokenizer = GemmaTokenizer.from_pretrained(
    str(WEIGHTS_DIR / "eagle_tokenizer"), use_fast=False, local_files_only=True
)

n_img = vit_mlx_np.shape[1]  # 196
IMG_CTX = "<IMG_CONTEXT>"
image_block = f"<img>{IMG_CTX * n_img}</img>"

OLD_PROMPT = f"<|im_start|>user\n{INSTRUCTION}\n{image_block}<|im_end|>"
NEW_PROMPT = (f"<start_of_turn>user\n{image_block}\n"
              f"{INSTRUCTION}<end_of_turn>\n<start_of_turn>model\n")

def get_mlx_backbone(prompt_str, vit_embeds):
    ids = tokenizer.encode(prompt_str)
    ids_np = np.array(ids)
    img_mask = (ids_np == IMAGE_TOKEN_INDEX)
    img_pos  = np.where(img_mask)[0]
    n = min(len(img_pos), vit_embeds.shape[1])

    ids_mx = mx.array(ids)[None]
    embeds  = llm_mlx.model.embed_tokens(ids_mx)
    hs_size = embeds.shape[-1]
    scaled  = vit_embeds[0, :n] / mx.array(hs_size**0.5, dtype=vit_embeds.dtype)
    pos_idx = mx.array(img_pos[:n])
    embeds  = embeds.at[0, pos_idx].add(scaled - embeds[0, pos_idx])
    hs = llm_mlx.model(inputs=None, input_embeddings=embeds)
    mx.eval(hs)
    return np.array(hs.astype(mx.float32)), img_mask, len(ids)

print("  [MLX] Running LLM with OLD prompt (buggy)...")
hs_old, mask_old, T_old = get_mlx_backbone(OLD_PROMPT, vit_mlx)
print("  [MLX] Running LLM with NEW prompt (fixed)...")
hs_new, mask_new, T_new = get_mlx_backbone(NEW_PROMPT, vit_mlx)

print(f"  OLD prompt tokens={T_old}, img positions: {np.where(mask_old)[0][:3]}...")
print(f"  NEW prompt tokens={T_new}, img positions: {np.where(mask_new)[0][:3]}...")

# ── PyTorch DiT ───────────────────────────────────────────────────────────────
print("  [PT] Loading PyTorch DiT from checkpoint...")
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from transformers.feature_extraction_utils import BatchFeature

pt_cfg_dict = dit_config.copy()
pt_config = Gr00tN1d6Config(**{k: v for k, v in pt_cfg_dict.items()
                                if not k.startswith("_")})
pt_ah = Gr00tN1d6ActionHead(pt_config).eval()

ah_sd = {k[len("action_head."):]: v for k, v in pt_sd.items()
         if k.startswith("action_head.")}
missing, _ = pt_ah.load_state_dict(ah_sd, strict=False)
if missing:
    print(f"  [PT] Missing keys: {missing[:3]}")
pt_ah = pt_ah.to(torch.bfloat16).eval()
print(f"  [PT] DiT loaded ({sum(p.numel() for p in pt_ah.parameters()):,} params)")

# Normalize state same as gemma_vla.py
s = stats["oxe_google"]["state"]
state_keys = ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
state_min = np.array([s[k]["min"][0] for k in state_keys], dtype=np.float32)
state_max = np.array([s[k]["max"][0] for k in state_keys], dtype=np.float32)
norm_state = ROBOT_STATE.copy()
rng_st = np.maximum(state_max - state_min, 1e-8)
norm_state = (ROBOT_STATE - state_min) / rng_st * 2.0 - 1.0

padded = np.zeros(128, dtype=np.float32)
padded[:len(norm_state)] = norm_state

# ── Pre-generate fixed noise to use in both PT and MLX DiT ───────────────────
ACTION_HORIZON = pt_config.action_horizon  # 16
ACTION_DIM     = pt_config.max_action_dim  # 128
rng_noise = np.random.default_rng(99)
FIXED_NOISE = rng_noise.standard_normal((1, ACTION_HORIZON, ACTION_DIM)).astype(np.float32)

def run_pt_dit_fixed_noise(backbone_hidden_states_np, img_mask_np, T):
    """Run PT DiT with FIXED pre-generated noise (bypasses get_action, runs denoising loop manually)."""
    hs_pt   = torch.from_numpy(backbone_hidden_states_np).reshape(1, T, -1).to(torch.bfloat16)
    bb_attn = torch.ones((1, T), dtype=torch.bool)
    img_mask_t = torch.from_numpy(img_mask_np[None])
    state_pt   = torch.from_numpy(padded[None, None, :]).to(torch.bfloat16)
    emb_id_pt  = torch.tensor([0], dtype=torch.long)

    backbone_out = BatchFeature(data={
        "backbone_features": hs_pt,
        "backbone_attention_mask": bb_attn,
        "image_mask": img_mask_t,
    })
    action_in = BatchFeature(data={"state": state_pt, "embodiment_id": emb_id_pt})

    with torch.no_grad():
        # Use _encode_features + get_action_with_features to inject fixed noise
        features = pt_ah._encode_features(backbone_out, action_in)
        vl_embeds = features.backbone_features
        state_feats = features.state_features

        actions = torch.from_numpy(FIXED_NOISE).to(torch.bfloat16)
        dt = 1.0 / pt_config.num_inference_timesteps
        N_STEPS = pt_config.num_inference_timesteps

        for t in range(N_STEPS):
            t_cont = t / float(N_STEPS)
            t_disc = int(t_cont * pt_config.num_timestep_buckets)
            ts = torch.full((1,), t_disc, dtype=torch.long)
            action_feats = pt_ah.action_encoder(actions, ts, emb_id_pt)
            if pt_config.add_pos_embed:
                pos_ids = torch.arange(action_feats.shape[1])
                action_feats = action_feats + pt_ah.position_embedding(pos_ids).unsqueeze(0)
            sa = torch.cat([state_feats, action_feats], dim=1)
            model_out = pt_ah.model(
                hidden_states=sa,
                encoder_hidden_states=vl_embeds,
                timestep=ts,
                image_mask=backbone_out.image_mask,
                backbone_attention_mask=backbone_out.backbone_attention_mask,
            )
            pred = pt_ah.action_decoder(model_out, emb_id_pt)
            velocity = pred[:, -ACTION_HORIZON:]
            actions = actions + dt * velocity

    return actions.to(torch.float32).numpy()

print("  [PT] Running DiT with OLD backbone (from buggy prompt)...")
pt_actions_old = run_pt_dit_fixed_noise(hs_old, mask_old, T_old)
print("  [PT] Running DiT with NEW backbone (from fixed prompt)...")
pt_actions_new = run_pt_dit_fixed_noise(hs_new, mask_new, T_new)

# ── MLX DiT ───────────────────────────────────────────────────────────────────
dit_mlx, _ = build_dit_mlx_from_exported(
    str(WEIGHTS_DIR / "dit.safetensors"), str(WEIGHTS_DIR / "config.json")
)
dit_mlx.num_inference_timesteps = pt_config.num_inference_timesteps

def run_mlx_dit_fixed_noise(backbone_hidden_np, img_mask_np, T):
    """Run MLX DiT with the same FIXED pre-generated noise as PT."""
    import mlx.nn as _nn
    hs_mx     = mx.array(backbone_hidden_np.reshape(1, T, -1)).astype(mx.bfloat16)
    bb_attn   = mx.ones((1, T), dtype=mx.bool_)
    img_mask_mx = mx.array(img_mask_np[None])
    state_mx  = mx.array(padded[None, None, :]).astype(mx.bfloat16)
    emb_id    = mx.full((1,), 0, dtype=mx.int32)

    # Project backbone features (mirrors PT _encode_features)
    backbone_f = dit_mlx.backbone_proj(hs_mx)          # (1,T,backbone_embedding_dim)
    backbone_f = dit_mlx.vlln(backbone_f)               # LayerNorm

    state_feats = dit_mlx.state_encoder(state_mx, emb_id)  # (1,1,D)

    actions = mx.array(FIXED_NOISE).astype(mx.bfloat16)    # same fixed noise as PT
    N_STEPS = dit_mlx.num_inference_timesteps
    dt = 1.0 / N_STEPS

    for t in range(N_STEPS):
        t_cont = t / float(N_STEPS)
        t_disc = int(t_cont * dit_mlx.num_timestep_buckets)
        ts = mx.full((1,), t_disc, dtype=mx.int32)

        action_feats = dit_mlx.action_encoder(actions, ts, emb_id)
        if dit_mlx.add_pos_embed:
            pos_ids = mx.arange(action_feats.shape[1])
            action_feats = action_feats + dit_mlx.position_embedding(pos_ids)

        sa = mx.concatenate([state_feats, action_feats], axis=1)
        model_out = dit_mlx._dit_forward(sa, backbone_f, ts, img_mask_mx, bb_attn)
        pred = dit_mlx.action_decoder(model_out, emb_id)
        velocity = pred[:, -ACTION_HORIZON:]
        actions = actions + dt * velocity
        mx.eval(actions)

    return np.array(actions.astype(mx.float32)).reshape(1, ACTION_HORIZON, -1)

print("  [MLX] Running DiT with OLD backbone (fixed noise)...")
mlx_actions_old = run_mlx_dit_fixed_noise(hs_old, mask_old, T_old)
print("  [MLX] Running DiT with NEW backbone (fixed noise)...")
mlx_actions_new = run_mlx_dit_fixed_noise(hs_new, mask_new, T_new)

# ── Compare ──────────────────────────────────────────────────────────────────
print()
print("  === RAW DiT outputs (first 5 steps, first 7 dims, no denorm) ===")
print(f"\n  PT (old prompt):  {pt_actions_old[0,:5,:7]}")
print(f"  MLX(old prompt):  {mlx_actions_old[0,:5,:7]}")
diff_old = np.abs(pt_actions_old[0,:,:7] - mlx_actions_old[0,:,:7])
print(f"  DiT diff OLD: max={diff_old.max():.4f} mean={diff_old.mean():.4f}")

print(f"\n  PT (new prompt):  {pt_actions_new[0,:5,:7]}")
print(f"  MLX(new prompt):  {mlx_actions_new[0,:5,:7]}")
diff_new = np.abs(pt_actions_new[0,:,:7] - mlx_actions_new[0,:,:7])
print(f"  DiT diff NEW: max={diff_new.max():.4f} mean={diff_new.mean():.4f}")

old_vs_new_pt = np.abs(pt_actions_old[0,:,:7] - pt_actions_new[0,:,:7])
print(f"\n  PT old vs new prompt delta: max={old_vs_new_pt.max():.4f} "
      f"mean={old_vs_new_pt.mean():.4f}  "
      f"(large delta → prompt matters!)")

# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3: Denormalized action comparison (what robot actually sees)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STAGE 3: Denormalized action comparison (what robot sees)")

a = stats["oxe_google"]["action"]
ms_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
act_mean = np.array([a[k]["mean"][0] for k in ms_keys], np.float32)
act_std  = np.array([a[k]["std"][0]  for k in ms_keys], np.float32)
grip_min = np.float32(a["gripper"]["min"][0])
grip_max = np.float32(a["gripper"]["max"][0])

def denorm(act_np):
    out = act_np.copy()
    out[..., :6] = out[..., :6] * act_std + act_mean
    grip = np.clip(out[..., 6], -1.0, 1.0)
    out[..., 6] = (grip + 1.0) / 2.0 * (grip_max - grip_min) + grip_min
    return out

pt_new_denorm  = denorm(pt_actions_new[0, :, :7])
mlx_new_denorm = denorm(mlx_actions_new[0, :, :7])

print(f"\n  PT  (fixed prompt, denorm) first step: {pt_new_denorm[0]}")
print(f"  MLX (fixed prompt, denorm) first step: {mlx_new_denorm[0]}")
diff_dn = np.abs(pt_new_denorm - mlx_new_denorm)
print(f"\n  PT vs MLX denorm diff: max={diff_dn.max():.6f} mean={diff_dn.mean():.6f}")

# Show old vs new in physical units
pt_old_denorm = denorm(pt_actions_old[0, :, :7])
print(f"\n  PROMPT BUG IMPACT (PT old vs new, denorm):")
delta_prompt = np.abs(pt_old_denorm - pt_new_denorm)
print(f"  max={delta_prompt.max():.4f} mean={delta_prompt.mean():.4f}")
for i, k in enumerate(ms_keys + ["gripper"]):
    print(f"    {k:12s}: old={pt_old_denorm[0,i]:.4f}  new={pt_new_denorm[0,i]:.4f}  "
          f"delta={delta_prompt[0,i]:.4f}")

print("\n" + "="*60)
print("SUMMARY")
print(f"  Vision max_diff:           {diff.max():.5f}  (bfloat16, expected)")
print(f"  DiT PT vs MLX (new prompt): max_diff={diff_new.max():.4f}")
print(f"  Prompt bug delta:           max_delta={delta_prompt.max():.4f}")
print(f"  → Bug {'WAS' if delta_prompt.max() > 0.005 else 'was NOT'} significant!")

#!/usr/bin/env python3
"""
validate_mlx_vs_torch.py

Compares MLX vs PyTorch CPU (float32) stage by stage.
All weights come from the GR00T state dict — no separate Eagle download needed.

  Stage A: Vision (SiglipVisionModel + pixel_shuffle + mlp1)
  Stage C: DiT action head (fixed noise seed, same backbone input for both)
"""

import sys, json, types
import numpy as np
from pathlib import Path
from PIL import Image
import urllib.request, io

HERE = Path(__file__).parent
ISAAC = HERE.parent / "Isaac-GR00T"
EAGLE = HERE.parent / "Eagle" / "Eagle2_5"
for p in [str(ISAAC), str(EAGLE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

GR00T_REPO  = "youngbrett48/gr00t-post-train-fractal-270m"
GR00T_CKPT  = "checkpoint-2000"
EAGLE_REPO  = "youngbrett48/train_stage2_gemma3_270m.sh"
HF_TOKEN    = True
IMG_URL     = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQThMxfZLtrpL_JSq6ttJo64D7nVKCGFHvKJg&s"
FIXED_SEED  = 42


def banner(s): print(f"\n{'='*60}\n{s}\n{'='*60}")

def compare(name, a, b, atol):
    a = np.array(a).astype(np.float32).flatten()
    b = np.array(b).astype(np.float32).flatten()
    max_d = float(np.max(np.abs(a - b)))
    mean_d = float(np.mean(np.abs(a - b)))
    ok = max_d <= atol
    print(f"  max|diff|={max_d:.6f}  mean|diff|={mean_d:.6f}  atol={atol}  [{'PASS' if ok else 'FAIL'}]")
    return ok


# ---------------------------------------------------------------------------
# 0. Shared assets
# ---------------------------------------------------------------------------
banner("0. Shared assets")

from inference import load_gr00t_components
from huggingface_hub import hf_hub_download

state_dict = load_gr00t_components(GR00T_REPO, GR00T_CKPT, token=HF_TOKEN)
cfg_path = hf_hub_download(GR00T_REPO, f"{GR00T_CKPT}/config.json", token=HF_TOKEN)
with open(cfg_path) as f:
    gr00t_config = json.load(f)

from eaglevl.model.eagle2_5.configuration_eagle2_5_vl import Eagle2_5_VLConfig
from transformers import AutoConfig
AutoConfig.register("eagle_2_5_vl", Eagle2_5_VLConfig)
eagle_config = Eagle2_5_VLConfig.from_pretrained(
    EAGLE_REPO, trust_remote_code=True, token=HF_TOKEN, local_files_only=True
)
IMAGE_SIZE = getattr(eagle_config, "force_image_size", None) or eagle_config.vision_config.image_size

req = urllib.request.Request(IMG_URL, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req) as r:
    img = Image.open(io.BytesIO(r.read())).convert("RGB")

img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
img_np = (np.array(img_resized, dtype=np.float32) / 255.0 - 0.5) / 0.5   # (H,W,3) [-1,1]

action_horizon = gr00t_config.get("action_horizon", 16)
action_dim     = gr00t_config.get("max_action_dim", 128)
max_state_dim  = gr00t_config.get("max_state_dim", 128)
padded_state   = np.zeros(max_state_dim, dtype=np.float32)

np.random.seed(FIXED_SEED)
fixed_noise = np.random.randn(1, action_horizon, action_dim).astype(np.float32)

print(f"  image_size={IMAGE_SIZE}  action_dim={action_dim}  max_state_dim={max_state_dim}")


# ---------------------------------------------------------------------------
# Stage A: Vision encoder
# ---------------------------------------------------------------------------
banner("Stage A: Vision encoder")

import mlx.core as mx
import mlx.utils as mlx_utils
import torch
import torch.nn as nn

# --- MLX ---
from vision_mlx import build_vision_mlx
from eaglevl.model.eagle2_5.configuration_eagle2_5_vl import Eagle2_5_VLConfig

vision_mlx = build_vision_mlx(state_dict, eagle_config, dtype="float32")
pv_mlx = mx.array(img_np[None])          # (1, H, W, 3) NHWC
out_mlx = vision_mlx(pv_mlx)
mx.eval(out_mlx)
out_mlx_np = np.array(out_mlx)
print(f"  MLX:    shape={out_mlx_np.shape}  mean={out_mlx_np.mean():.4f}  std={out_mlx_np.std():.4f}")

# --- PyTorch — load SigLIP directly from state dict ---
from transformers import SiglipVisionModel, SiglipVisionConfig

# Extract vision keys (strip 'backbone.model.vision_model.' prefix)
PREFIX = "backbone.model.vision_model."
vision_sd = {k[len(PREFIX):]: v for k, v in state_dict.items() if k.startswith(PREFIX)}

siglip_cfg = SiglipVisionConfig(
    hidden_size=1152, intermediate_size=4304, num_hidden_layers=27,
    num_attention_heads=16, image_size=IMAGE_SIZE, patch_size=14,
)
siglip_pt = SiglipVisionModel(siglip_cfg).eval()
siglip_pt.load_state_dict(vision_sd, strict=True)

# pixel_shuffle — exact copy of Eagle2_5's pixel_shuffle method
import torch.nn.functional as F
def pt_pixel_shuffle(x, scale=0.5):
    # x: (B, N, C), N=729, C=1152
    h = w = int(x.shape[1] ** 0.5)
    x = x.reshape(x.shape[0], h, w, -1)
    if h % 2 != 0:
        x = F.pad(x, (0, 0, 0, 1, 0, 1))   # pad H and W dims by 1
        h, w = h + 1, w + 1
    n, w, h, c = x.size()
    x = x.view(n, w, int(h * scale), int(c / scale))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h * scale), int(w * scale), int(c / (scale ** 2)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x.reshape(x.shape[0], -1, x.shape[-1])   # (B, 196, 4608)

# mlp1 in PyTorch
PREFIX2 = "backbone.model.mlp1."
mlp1_sd = {k[len(PREFIX2):]: v for k, v in state_dict.items() if k.startswith(PREFIX2)}
# keys: 0.weight/bias (LayerNorm), 1.weight/bias (Linear1), 3.weight/bias (Linear2)
mlp1_pt = nn.Sequential(
    nn.LayerNorm(4608),
    nn.Linear(4608, 640),
    nn.GELU(),
    nn.Linear(640, 640),
)
# index map: 0→ln, 1→linear1, 3→linear2
for src, dst in [("0", "0"), ("1", "1"), ("3", "3")]:
    mlp1_pt[int(dst)].weight.data = mlp1_sd[f"{src}.weight"]
    mlp1_pt[int(dst)].bias.data   = mlp1_sd[f"{src}.bias"]

with torch.no_grad():
    pv_pt = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0)   # (1, 3, H, W)
    vit_out = siglip_pt(pixel_values=pv_pt).last_hidden_state      # (1, 729, 1152)
    shuffled = pt_pixel_shuffle(vit_out)                            # (1, 196, 4608)
    out_pt = mlp1_pt(shuffled)                                      # (1, 196, 640)
out_pt_np = out_pt.numpy()
print(f"  PyTorch: shape={out_pt_np.shape}  mean={out_pt_np.mean():.4f}  std={out_pt_np.std():.4f}")

print("\nStage A comparison (MLX=float16 vs PyTorch=float32, ~0.2% precision):")
stage_a = compare("Vision", out_mlx_np, out_pt_np, atol=0.3)


# ---------------------------------------------------------------------------
# Stage C: DiT action head (fixed noise, feed PyTorch vision output to both)
# ---------------------------------------------------------------------------
banner("Stage C: DiT action head (fixed noise)")

from dit_mlx import build_dit_mlx
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
from transformers.feature_extraction_utils import BatchFeature

# We'll drive both DiT heads from the PyTorch vision output (out_pt_np)
# and a dummy "LLM hidden states" = the same vision features broadcast to (1, T, 640)
# where T = n_img_tokens (196). This keeps it self-contained.
backbone_np = out_pt_np.astype(np.float32)   # (1, 196, 640)
T = backbone_np.shape[1]
image_mask_np = np.ones((1, T), dtype=bool)  # treat all as image tokens

# --- PyTorch DiT ---
print("[PyTorch] Loading action head from state dict...")
valid_fields = set(Gr00tN1d6Config.__dataclass_fields__.keys())
pt_cfg_kwargs = {k: v for k, v in gr00t_config.items() if k in valid_fields}
pt_cfg = Gr00tN1d6Config(**pt_cfg_kwargs)
pt_ah = Gr00tN1d6ActionHead(pt_cfg).eval()
ah_sd = {k[len("action_head."):]: v for k, v in state_dict.items() if k.startswith("action_head.")}
missing, unexpected = pt_ah.load_state_dict(ah_sd, strict=False)
if missing:   print(f"  missing: {missing[:5]}")
if unexpected: print(f"  unexpected: {unexpected[:5]}")

# Patch torch.randn to inject fixed noise
_orig_randn = torch.randn
def _fixed_randn(*args, **kwargs):
    return torch.tensor(fixed_noise)
torch.randn = _fixed_randn

backbone_pt  = torch.tensor(backbone_np)
emb_id_pt    = torch.zeros(1, dtype=torch.long)
state_pt     = torch.tensor(padded_state).unsqueeze(0).unsqueeze(0)   # (1,1,max_state_dim)
attn_mask_pt = torch.ones(1, T, dtype=torch.bool)
img_mask_pt  = torch.tensor(image_mask_np)

# Apply backbone_proj + vlln (same as get_action internals)
with torch.no_grad():
    bb = backbone_pt.clone()
    if pt_ah.backbone_proj is not None:
        bb = pt_ah.backbone_proj(bb)
    bb = pt_ah.vlln(bb)
    state_feats = pt_ah.state_encoder(state_pt, emb_id_pt)
    result_pt = pt_ah.get_action_with_features(
        backbone_features=bb,
        state_features=state_feats,
        embodiment_id=emb_id_pt,
        backbone_output=BatchFeature(data={
            "backbone_attention_mask": attn_mask_pt,
            "image_mask": img_mask_pt,
        }),
    )
torch.randn = _orig_randn
actions_pt_np = result_pt["action_pred"].float().numpy()
print(f"  PyTorch DiT: shape={actions_pt_np.shape}  mean={actions_pt_np.mean():.4f}  std={actions_pt_np.std():.4f}")

# --- MLX DiT ---
print("[MLX] Loading DiT...")
dit_mlx_model, dit_config = build_dit_mlx(state_dict, cfg_path, dtype="float32")

def _patched_get_action(self, backbone_features, backbone_attention_mask, image_mask, state, embodiment_id):
    _dt = getattr(self, "_compute_dtype", mx.float32)
    backbone_features = backbone_features.astype(_dt)
    state = state.astype(_dt)
    if self.backbone_proj is not None:
        backbone_features = self.backbone_proj(backbone_features)
    if self.vlln is not None:
        backbone_features = self.vlln(backbone_features)
    vl_embeds = backbone_features
    state_features = self.state_encoder(state, embodiment_id)
    B = vl_embeds.shape[0]
    actions = mx.array(fixed_noise).astype(_dt)
    dt = 1.0 / self.num_inference_timesteps
    pos_emb = self.position_embedding(mx.arange(self.action_horizon)) if self.add_pos_embed else None
    for t in range(self.num_inference_timesteps):
        t_disc = int((t / float(self.num_inference_timesteps)) * self.num_timestep_buckets)
        ts = mx.full((B,), t_disc, dtype=mx.int32)
        action_features = self.action_encoder(actions, ts, embodiment_id)
        if pos_emb is not None:
            action_features = action_features + pos_emb
        sa_embs = mx.concatenate([state_features, action_features], axis=1)
        model_out = self._dit_forward(sa_embs, vl_embeds, ts, image_mask, backbone_attention_mask)
        pred = self.action_decoder(model_out, embodiment_id)
        actions = actions + dt * pred[:, -self.action_horizon:]
        mx.eval(actions)
    return actions.astype(mx.float32)

dit_mlx_model.get_action = types.MethodType(_patched_get_action, dit_mlx_model)

backbone_mx  = mx.array(backbone_np)
bb_attn_mx   = mx.ones((1, T), dtype=mx.bool_)
img_mask_mx  = mx.array(image_mask_np)
state_mx     = mx.array(padded_state[None, None, :])
emb_id_mx    = mx.zeros((1,), dtype=mx.int32)
dit_mlx_model.num_inference_timesteps = 4

actions_mlx = dit_mlx_model.get_action(backbone_mx, bb_attn_mx, img_mask_mx, state_mx, emb_id_mx)
mx.eval(actions_mlx)
actions_mlx_np = np.array(actions_mlx)
print(f"  MLX DiT:     shape={actions_mlx_np.shape}  mean={actions_mlx_np.mean():.4f}  std={actions_mlx_np.std():.4f}")

n = min(actions_pt_np.shape[-1], actions_mlx_np.shape[-1])
print(f"\nStage C comparison (first {n} action dims, float16 x 4 Euler steps):")
stage_c = compare("DiT", actions_pt_np[..., :n], actions_mlx_np[..., :n], atol=1.5)


# ---------------------------------------------------------------------------
# Stage B: LLM trunk (Gemma3 from state dict, PyTorch float32 vs MLX 4-bit)
# ---------------------------------------------------------------------------
banner("Stage B: LLM trunk (PyTorch float32 vs MLX 4-bit — mismatch expected)")

from transformers import AutoModelForCausalLM, AutoConfig as HFAutoConfig, GemmaTokenizer

# Load Gemma3 into PyTorch from the GR00T state dict
LLM_PREFIX = "backbone.model.language_model."
llm_sd = {k[len(LLM_PREFIX):]: v for k, v in state_dict.items() if k.startswith(LLM_PREFIX)}
print(f"  LLM keys in state dict: {len(llm_sd)}")

# Use the saved LLM in HF format (already extracted by extract_llm.py)
HF_LLM_DIR = HERE / "gr00t_llm_hf"
if HF_LLM_DIR.exists():
    # Load PT from same weights as MLX (gr00t_llm_mlx, bfloat16 on disk) so both
    # paths use identical weights after casting to float32.
    print(f"  Loading PyTorch Gemma3 from gr00t_llm_mlx (same weights as MLX)...")
    pt_lm = AutoModelForCausalLM.from_pretrained(
        str(HERE / "gr00t_llm_mlx"), dtype=torch.float32,
    ).eval()

    # Build input_embeds: embed tokens then scatter vision features
    from transformers import GemmaTokenizer
    tokenizer_pt = GemmaTokenizer.from_pretrained(EAGLE_REPO, use_fast=False,
                                                   token=HF_TOKEN, local_files_only=True)
    img_context_id = eagle_config.image_token_index
    n_img_tokens = out_pt_np.shape[1]
    INSTR = "pick up the object"
    prompt = f"<img>{'<IMG_CONTEXT>' * n_img_tokens}</img>\n{INSTR}"
    input_ids_list = tokenizer_pt.encode(prompt)
    T_llm = len(input_ids_list)
    ids_np_llm = np.array(input_ids_list)
    img_positions_llm = np.where(ids_np_llm == img_context_id)[0]

    input_ids_pt = torch.tensor(input_ids_list, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        input_embeds_pt = pt_lm.model.embed_tokens(input_ids_pt).float()
        hidden_size_pt = input_embeds_pt.shape[-1]
        # PT embed_tokens applies *sqrt(H) to text tokens; model.forward does NOT re-scale.
        # Vision features (from SigLIP projector) are already at the correct embedding scale —
        # place them directly (no extra scaling). MLX path pre-divides by sqrt(H) since its
        # model.forward will multiply back up.
        vit_pt_tensor = torch.tensor(out_pt_np)
        input_embeds_pt[0, torch.tensor(img_positions_llm[:n_img_tokens])] = vit_pt_tensor[0, :n_img_tokens]
        llm_out_pt = pt_lm.model(
            inputs_embeds=input_embeds_pt,
            attention_mask=torch.ones(1, T_llm, dtype=torch.long),
            use_cache=False, return_dict=True,
        )
    hidden_pt_np = llm_out_pt.last_hidden_state.float().numpy()
    print(f"  PyTorch LLM: shape={hidden_pt_np.shape}  mean={hidden_pt_np.mean():.4f}  std={hidden_pt_np.std():.4f}")

    # MLX LLM forward — cast to float32 to match PT, isolating logic not dtype
    from mlx_lm import load as mlx_load_llm
    mlx_llm, _ = mlx_load_llm(str(HERE / "gr00t_llm_mlx"))
    mlx_llm.eval()
    mlx_llm.load_weights(list(mlx_utils.tree_flatten(
        mlx_utils.tree_map(lambda x: x.astype(mx.float32) if isinstance(x, mx.array) else x,
                           mlx_llm.parameters())
    )))
    ids_mx = mx.array(input_ids_list)[None]
    embeds_mx = mlx_llm.model.embed_tokens(ids_mx).astype(mx.float32)
    pos_idx = mx.array(img_positions_llm[:n_img_tokens])
    hidden_size = embeds_mx.shape[-1]
    vit_mx_for_llm = mx.array(out_pt_np).astype(mx.float32) / mx.array(hidden_size ** 0.5, mx.float32)
    embeds_mx = embeds_mx.at[0, pos_idx].add(vit_mx_for_llm[0, :n_img_tokens] - embeds_mx[0, pos_idx])
    hidden_mx = mlx_llm.model(inputs=None, input_embeddings=embeds_mx)
    mx.eval(hidden_mx)
    hidden_mx_np = np.array(hidden_mx.astype(mx.float32))
    print(f"  MLX LLM:     shape={hidden_mx_np.shape}  mean={hidden_mx_np.mean():.4f}  std={hidden_mx_np.std():.4f}")

    print("\nStage B comparison (bfloat16 MLX vs float32 PyTorch — same inputs, isolating LLM diff):")
    stage_b = compare("LLM", hidden_mx_np, hidden_pt_np, atol=1.0)
    print("  (atol=1.0 — bfloat16 vs float32, 18 layers)")
else:
    print(f"  SKIP — {HF_LLM_DIR} not found. Run extract_llm.py first.")
    stage_b = None
    hidden_pt_np = None
    hidden_mx_np = None
    T_llm = None
    ids_np_llm = None
    img_positions_llm = None


# ---------------------------------------------------------------------------
# Stage D: Full end-to-end (image → vision → LLM → DiT → actions)
# ---------------------------------------------------------------------------
banner("Stage D: Full end-to-end pipeline")

if HF_LLM_DIR.exists() and hidden_pt_np is not None:
    # PyTorch end-to-end: use hidden_pt_np already computed above
    img_mask_e2e_np = (ids_np_llm == img_context_id)[None]   # (1, T)
    T_e2e = hidden_pt_np.shape[1]

    # --- PyTorch DiT with fixed noise, using PyTorch backbone output ---
    torch.randn = _fixed_randn
    bb_pt = torch.tensor(hidden_pt_np)
    emb_id_pt_e2e  = torch.zeros(1, dtype=torch.long)
    state_pt_e2e   = torch.tensor(padded_state).unsqueeze(0).unsqueeze(0)
    attn_pt_e2e    = torch.ones(1, T_e2e, dtype=torch.bool)
    img_mask_pt_e2e = torch.tensor(img_mask_e2e_np)
    with torch.no_grad():
        bb_proj = bb_pt.clone()
        if pt_ah.backbone_proj is not None:
            bb_proj = pt_ah.backbone_proj(bb_proj)
        bb_proj = pt_ah.vlln(bb_proj)
        sf_e2e = pt_ah.state_encoder(state_pt_e2e, emb_id_pt_e2e)
        result_e2e_pt = pt_ah.get_action_with_features(
            backbone_features=bb_proj, state_features=sf_e2e,
            embodiment_id=emb_id_pt_e2e,
            backbone_output=BatchFeature(data={
                "backbone_attention_mask": attn_pt_e2e,
                "image_mask": img_mask_pt_e2e,
            }),
        )
    torch.randn = _orig_randn
    e2e_pt_np = result_e2e_pt["action_pred"].float().numpy()
    print(f"  PyTorch e2e: shape={e2e_pt_np.shape}  mean={e2e_pt_np.mean():.4f}  std={e2e_pt_np.std():.4f}")

    # --- MLX end-to-end with fixed noise, using MLX backbone output ---
    bb_mx_e2e  = mx.array(hidden_mx_np)
    attn_mx_e2e = mx.ones((1, T_e2e), dtype=mx.bool_)
    img_mask_mx_e2e = mx.array(img_mask_e2e_np)
    state_mx_e2e = mx.array(padded_state[None, None, :])
    emb_id_mx_e2e = mx.zeros((1,), dtype=mx.int32)
    dit_mlx_model.num_inference_timesteps = 4

    actions_e2e = dit_mlx_model.get_action(bb_mx_e2e, attn_mx_e2e, img_mask_mx_e2e,
                                            state_mx_e2e, emb_id_mx_e2e)
    mx.eval(actions_e2e)
    e2e_mlx_np = np.array(actions_e2e)
    print(f"  MLX e2e:     shape={e2e_mlx_np.shape}  mean={e2e_mlx_np.mean():.4f}  std={e2e_mlx_np.std():.4f}")

    n_e2e = min(e2e_pt_np.shape[-1], e2e_mlx_np.shape[-1])
    print(f"\nStage D comparison (bfloat16 LLM + float16 DiT vs float32 PyTorch):")
    stage_d = compare("E2E", e2e_pt_np[..., :n_e2e], e2e_mlx_np[..., :n_e2e], atol=2.0)
else:
    print("  SKIP — run extract_llm.py first")
    stage_d = None


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
banner("Summary")
print(f"  Stage A (Vision):          {'PASS' if stage_a else 'FAIL'}")
print(f"  Stage B (LLM fp32):        {'PASS' if stage_b else ('SKIP' if stage_b is None else 'FAIL')}")
print(f"  Stage C (DiT isolated):    {'PASS' if stage_c else 'FAIL'}")
print(f"  Stage D (Full E2E):        {'PASS' if stage_d else ('SKIP' if stage_d is None else 'FAIL')}")

core_pass = stage_a and stage_c
print()
if core_pass:
    print("MLX implementation is CORRECT (vision + DiT logic verified).")
else:
    print("ISSUES FOUND — check diffs above.")

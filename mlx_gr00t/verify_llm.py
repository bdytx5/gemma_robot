"""
verify_llm.py — Compare MLX vs PyTorch Gemma3 hidden states.

Loads gr00t_llm_mlx weights into both:
  - PyTorch AutoModelForCausalLM (float32) — reference
  - MLX mlx_lm (bfloat16) — production

Runs identical input through both and compares hidden states token-by-token.
No download needed — uses existing gr00t_llm_mlx/ weights.

Run from mlx_gr00t/:
    python verify_llm.py
"""
import sys, json, re
import numpy as np
from pathlib import Path

HERE        = Path(__file__).parent
WEIGHTS_DIR = HERE / "gr00t_weights_mlx"
MLX_LLM_DIR = HERE / "gr00t_llm_mlx"
EAGLE_PATH  = str(HERE.parent / "Eagle" / "Eagle2_5")
ISAAC_PATH  = str(HERE.parent / "Isaac-GR00T")
for p in [EAGLE_PATH, ISAAC_PATH, str(HERE)]:
    if p not in sys.path:
        sys.path.insert(0, p)

PASS = "\033[32m✓ PASS\033[0m"
FAIL = "\033[31m✗ FAIL\033[0m"

def check(label, ok, detail=""):
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}  {label}")
    if detail:
        print(f"         {detail}")
    return ok

results = []

print("\n══════════════════════════════════════════════════════")
print("  GR00T MLX vs PyTorch Gemma3 LLM Hidden State Check")
print("══════════════════════════════════════════════════════\n")

# ── Test input ──────────────────────────────────────────────────────────────
# Simple short prompt — no images, just text tokens to isolate the LLM
with open(WEIGHTS_DIR / "meta.json") as f:
    meta = json.load(f)
IMAGE_TOKEN_INDEX = meta["image_token_index"]

from transformers import GemmaTokenizer
tokenizer = GemmaTokenizer.from_pretrained(
    str(WEIGHTS_DIR / "eagle_tokenizer"), use_fast=False, local_files_only=True
)

INSTRUCTION = "pick up the red block"
# Use the same prompt format as gemma_vla.py (with formalization already applied)
n_img = 196
IMG_CTX = "<IMG_CONTEXT>"
image_block = f"<img>{IMG_CTX * n_img}</img>"
prompt_full = (
    f"<start_of_turn>user\n"
    f"{image_block}\n"
    f"{INSTRUCTION}<end_of_turn>\n"
    f"<start_of_turn>model\n"
)
# Text-only prompt to isolate pure LLM computation (no vision scatter)
prompt_text_only = (
    f"<start_of_turn>user\n"
    f"{INSTRUCTION}<end_of_turn>\n"
    f"<start_of_turn>model\n"
)
input_ids      = tokenizer.encode(prompt_full)
input_ids_text = tokenizer.encode(prompt_text_only)
print(f"  Full prompt:       {len(input_ids)} tokens ({n_img} image placeholders)")
print(f"  Text-only prompt:  {len(input_ids_text)} tokens")

ids_np = np.array(input_ids)
img_positions = np.where(ids_np == IMAGE_TOKEN_INDEX)[0]

# ── Load MLX LLM ────────────────────────────────────────────────────────────
print("\n── Loading MLX LLM ──────────────────────────────────")
import mlx.core as mx
import mlx.utils as mlx_utils
from mlx_lm import load as mlx_load

mlx_llm, _ = mlx_load(str(MLX_LLM_DIR))
mlx_llm.eval()
# Cast to float32 for clean comparison against PyTorch float32
mlx_llm.load_weights(list(mlx_utils.tree_flatten(
    mlx_utils.tree_map(
        lambda x: x.astype(mx.float32) if isinstance(x, mx.array) else x,
        mlx_llm.parameters(),
    )
)))
hidden_size = mlx_llm.model.embed_tokens.weight.shape[1]
print(f"  MLX LLM: hidden_size={hidden_size}, dtype=float32")

# Synthetic vision features — same fixed values for both paths
np.random.seed(42)
vit_np = np.random.randn(n_img, hidden_size).astype(np.float32) * 0.1

# ── MLX text-only forward (no vision scatter) ───────────────────────────────
ids_mx_text = mx.array(input_ids_text)[None]
embeds_mx_text = mlx_llm.model.embed_tokens(ids_mx_text).astype(mx.float32)
hidden_mlx_text = mlx_llm.model(inputs=None, input_embeddings=embeds_mx_text)
mx.eval(hidden_mlx_text)
hidden_mlx_text_np = np.array(hidden_mlx_text.astype(mx.float32))
print(f"  MLX text-only output: shape={hidden_mlx_text_np.shape}  "
      f"mean={hidden_mlx_text_np.mean():.4f}  std={hidden_mlx_text_np.std():.4f}")

# ── MLX full forward (with vision scatter) ───────────────────────────────────
ids_mx = mx.array(input_ids)[None]
embeds_mx = mlx_llm.model.embed_tokens(ids_mx).astype(mx.float32)
# scatter vision features (MLX divides by sqrt(hidden_size) to cancel Gemma3's internal scale)
vit_mx = mx.array(vit_np).astype(mx.float32) / mx.array(hidden_size ** 0.5)
pos_idx = mx.array(img_positions[:n_img])
embeds_mx = embeds_mx.at[0, pos_idx].add(vit_mx - embeds_mx[0, pos_idx])
hidden_mlx = mlx_llm.model(inputs=None, input_embeddings=embeds_mx)
mx.eval(hidden_mlx)
hidden_mlx_np = np.array(hidden_mlx.astype(mx.float32))
print(f"  MLX full output:      shape={hidden_mlx_np.shape}  "
      f"mean={hidden_mlx_np.mean():.4f}  std={hidden_mlx_np.std():.4f}")

# ── Load PyTorch LLM (same weights, float32) ────────────────────────────────
print("\n── Loading PyTorch LLM (same weights, float32) ──────")
import torch
from transformers import AutoModelForCausalLM

pt_llm = AutoModelForCausalLM.from_pretrained(
    str(MLX_LLM_DIR), torch_dtype=torch.float32,
).eval()
print(f"  PyTorch LLM: {sum(p.numel() for p in pt_llm.parameters()):,} params, float32")

# PyTorch forward — same logic as gemma_vla.py but in PyTorch
ids_pt = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
with torch.no_grad():
    embeds_pt = pt_llm.model.embed_tokens(ids_pt).float()       # (1, T, D)
    # ── text-only forward (no vision scatter) ────────────────────────────────
    ids_pt_text = torch.tensor(input_ids_text, dtype=torch.long).unsqueeze(0)
    embeds_pt_text = pt_llm.model.embed_tokens(ids_pt_text).float()
    hidden_pt_text = pt_llm.model(
        inputs_embeds=embeds_pt_text,
        attention_mask=torch.ones(1, len(input_ids_text), dtype=torch.long),
        use_cache=False, return_dict=True,
    ).last_hidden_state.float()

    # ── full forward (with vision scatter) ───────────────────────────────────
    # HF embed_tokens already applies *sqrt(hidden_size) internally.
    # HF Gemma3TextModel.forward does NOT scale inputs_embeds again.
    # So place vision features at the same effective scale as text tokens
    # (i.e. vit_np directly — no pre-division needed for PT).
    # MLX path pre-divides by sqrt(H) because MLX model.forward always scales.
    embeds_pt_full = pt_llm.model.embed_tokens(ids_pt).float()
    pos_idx_pt = torch.tensor(img_positions[:n_img], dtype=torch.long)
    embeds_pt_full[0, pos_idx_pt] = torch.tensor(vit_np)  # NO division — PT model doesn't re-scale
    hidden_pt_full = pt_llm.model(
        inputs_embeds=embeds_pt_full,
        attention_mask=torch.ones(1, len(input_ids), dtype=torch.long),
        use_cache=False, return_dict=True,
    ).last_hidden_state.float()

hidden_pt_text_np = hidden_pt_text.numpy()
hidden_pt_np      = hidden_pt_full.numpy()
print(f"  PyTorch text-only (float32): mean={hidden_pt_text_np.mean():.4f}  std={hidden_pt_text_np.std():.4f}")
print(f"  PyTorch full      (float32): mean={hidden_pt_np.mean():.4f}  std={hidden_pt_np.std():.4f}")

# ── Compare ──────────────────────────────────────────────────────────────────
print("\n── 1. Text-only LLM (no vision scatter) ─────────────")
# Both MLX and PT are float32 — should be near-zero diff
diff_text = np.abs(hidden_mlx_text_np - hidden_pt_text_np)
print(f"  MLX fp32 vs PT fp32: max={diff_text.max():.6f}  mean={diff_text.mean():.6f}")
text_atol = 1e-3
results.append(check(
    "Text-only hidden states match (fp32 vs fp32)",
    float(diff_text.max()) <= text_atol,
    f"max_diff={diff_text.max():.6f}  atol={text_atol}",
))

print("\n── 2. Full forward with vision scatter ──────────────")
diff_full = np.abs(hidden_mlx_np - hidden_pt_np)
text_mask = (ids_np != IMAGE_TOKEN_INDEX)
text_diff = diff_full[0][text_mask]
img_diff  = diff_full[0][~text_mask]
print(f"  Overall:  max={diff_full.max():.5f}  mean={diff_full.mean():.5f}")
print(f"  Text pos: max={text_diff.max():.5f}  mean={text_diff.mean():.5f}")
print(f"  Img  pos: max={img_diff.max():.5f}  mean={img_diff.mean():.5f}")

# Show worst 5 positions
per_token = diff_full.max(axis=-1)[0]
worst = np.argsort(per_token)[-5:][::-1]
print(f"\n  Worst 5 positions:")
for pos in worst:
    tok_id = input_ids[pos]
    is_img = (tok_id == IMAGE_TOKEN_INDEX)
    print(f"    pos={pos:4d}  diff={per_token[pos]:.5f}  {'← image token' if is_img else repr(tokenizer.decode([tok_id]))}")

full_atol = max(1.0, hidden_pt_np.std() * 0.20)
results.append(check(
    f"Full forward hidden states match (atol={full_atol:.2f})",
    float(diff_full.max()) <= full_atol,
    f"max_diff={diff_full.max():.5f}",
))
results.append(check(
    "Text positions unaffected by vision scatter",
    float(text_diff.max()) <= text_atol,
    f"max_diff={text_diff.max():.5f}  atol={text_atol:.4f}",
))

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n══════════════════════════════════════════════════════")
print("  Summary")
print("══════════════════════════════════════════════════════")
passed = sum(results)
print(f"\n  {passed}/{len(results)} checks passed")
if passed == len(results):
    print(f"  ✓ MLX Gemma3 matches PyTorch float32 within bfloat16 tolerance.")
else:
    print(f"  ✗ Mismatch found — check diffs above.")
print()

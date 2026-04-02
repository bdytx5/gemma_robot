"""
GR00T inference on Apple Silicon using:
  - MLX for the Gemma3-1b language model trunk (fast)
  - PyTorch MPS for SigLIP vision encoder + MLP projector + DiT action head

Usage:
    python inference.py \
        --gr00t_ckpt youngbrett48/train_google_robot_eagle2_5 \
        --checkpoint checkpoint-200 \
        --mlx_llm_path ./gr00t_llm_mlx \
        --image path/to/image.jpg \
        --state "0.1,0.2,0.3,0.4,0.5,0.6,0.7"

Pipeline:
    image + language prompt
        → SigLIP vision encoder       (PyTorch MPS)
        → MLP projector               (PyTorch MPS)
        → scatter into token sequence
        → Gemma3-1b transformer       (MLX)
        → last hidden state           (back to MPS tensor)
        → VLLN layer norm             (PyTorch MPS)
        → DiT diffusion head          (PyTorch MPS)
        → robot actions
"""

import argparse
import json
import numpy as np
from pathlib import Path
from PIL import Image


# ---------------------------------------------------------------------------
# Lazy device setup
# ---------------------------------------------------------------------------
import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[init] PyTorch device: {DEVICE}")


# ---------------------------------------------------------------------------
# 1. Load full GR00T checkpoint (for vision encoder + DiT weights)
# ---------------------------------------------------------------------------
def load_gr00t_components(repo_id: str, checkpoint: str):
    """Download checkpoint and return state_dict split into component groups."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    import glob

    print(f"[1/5] Downloading {repo_id}/{checkpoint}...")
    ckpt_dir = snapshot_download(repo_id, allow_patterns=[f"{checkpoint}/*"])
    ckpt_path = Path(ckpt_dir) / checkpoint

    safetensors = sorted(glob.glob(str(ckpt_path / "*.safetensors")))
    if safetensors:
        sd = {}
        for s in safetensors:
            sd.update(load_file(s))
    else:
        sd = torch.load(str(ckpt_path / "pytorch_model.bin"), map_location="cpu")

    print(f"  {len(sd)} keys loaded")
    return sd


# ---------------------------------------------------------------------------
# 2. Build the Eagle2.5 vision encoder + MLP projector from state dict
# ---------------------------------------------------------------------------
def build_vision_components(state_dict, eagle2_5_repo: str):
    """
    Load the SigLIP vision encoder + MLP projector onto MPS.
    These live in state_dict under "backbone.model.vision_model.*"
    and "backbone.model.mlp1.*".
    """
    from transformers import AutoModel, AutoConfig
    import sys, os
    # Eagle2.5 model code lives in ../Eagle/Eagle2_5
    eagle_path = str(Path(__file__).parent.parent / "Eagle" / "Eagle2_5")
    if eagle_path not in sys.path:
        sys.path.insert(0, eagle_path)

    print(f"[2/5] Loading Eagle2.5 config from {eagle2_5_repo}...")
    from eaglevl.model.eagle2_5.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration

    config = AutoConfig.from_pretrained(eagle2_5_repo, trust_remote_code=True)
    # Build vision-only subset: just instantiate full model, we'll load weights selectively
    model = Eagle2_5_VLForConditionalGeneration(config)

    # Extract vision + projector weights
    vision_sd = {
        k[len("backbone.model."):]: v
        for k, v in state_dict.items()
        if k.startswith("backbone.model.vision_model.") or k.startswith("backbone.model.mlp1.")
    }
    missing, unexpected = model.load_state_dict(vision_sd, strict=False)
    print(f"  Vision/MLP weights loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")

    # We only need vision_model and mlp1
    vision_encoder = model.vision_model.to(DEVICE).eval()
    mlp_projector = model.mlp1.to(DEVICE).eval()
    return vision_encoder, mlp_projector, config


# ---------------------------------------------------------------------------
# 3. Load MLX Gemma3-1b LLM
# ---------------------------------------------------------------------------
def load_mlx_llm(mlx_path: str):
    """Load the converted MLX Gemma3-1b model."""
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    print(f"[3/5] Loading MLX LLM from {mlx_path}...")
    model, tokenizer = mlx_load(mlx_path)
    model.eval()
    print(f"  MLX model loaded")
    return model, tokenizer


# ---------------------------------------------------------------------------
# 4. Load DiT action head from state dict
# ---------------------------------------------------------------------------
def build_dit(state_dict, config_path: str):
    """Load the GR00T DiT diffusion head onto MPS."""
    import sys
    gr00t_path = str(Path(__file__).parent.parent / "Isaac-GR00T")
    if gr00t_path not in sys.path:
        sys.path.insert(0, gr00t_path)

    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
    from gr00t.model.gr00t_n1d6.setup import Gr00tN1d6Config

    print(f"[4/5] Building DiT action head...")
    with open(config_path) as f:
        cfg_dict = json.load(f)
    config = Gr00tN1d6Config(**cfg_dict)
    model = Gr00tN1d6(config)

    # Extract DiT + VLLN + projector weights (everything except backbone)
    dit_sd = {
        k[len("model."):] if k.startswith("model.") else k: v
        for k, v in state_dict.items()
        if not k.startswith("backbone.")
    }
    missing, unexpected = model.load_state_dict(dit_sd, strict=False)
    print(f"  DiT weights loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
    return model.to(DEVICE).eval()


# ---------------------------------------------------------------------------
# 5. Full inference pass
# ---------------------------------------------------------------------------
def run_inference(
    image: Image.Image,
    robot_state: np.ndarray,
    language_instruction: str,
    vision_encoder,
    mlp_projector,
    mlx_llm,
    tokenizer,
    dit_model,
    eagle_config,
    image_token_index: int = 151667,  # <image> token id for Eagle2.5
    n_diffusion_steps: int = 4,
):
    import mlx.core as mx
    from transformers import CLIPImageProcessor

    # --- Preprocess image ---
    processor = CLIPImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(DEVICE)

    # --- Vision encoder → MLP projector ---
    with torch.no_grad():
        vit_out = vision_encoder(pixel_values=pixel_values, output_hidden_states=True)
        vit_feats = vit_out.last_hidden_state  # (1, n_patches, vision_dim)
        vit_embeds = mlp_projector(vit_feats)   # (1, n_patches, llm_dim=1152)

    # --- Tokenize language instruction ---
    text_tokens = tokenizer.encode(language_instruction, return_tensors=None)
    input_ids = text_tokens  # list of ints

    # Build interleaved sequence: [<image> tokens] + [text tokens]
    # For Eagle2.5: image tokens replace <image> placeholder in the prompt
    n_img_tokens = vit_embeds.shape[1]  # e.g. 182 for 256px input
    img_placeholder = [image_token_index] * n_img_tokens
    full_ids = img_placeholder + input_ids

    # --- Get token embeddings from MLX model ---
    # Get embedding table from MLX model
    ids_mx = mx.array(full_ids)[None]  # (1, T)
    token_embeds_mx = mlx_llm.model.embed_tokens(ids_mx)  # (1, T, 1152) in MLX

    # Convert to torch, scatter in vision features
    token_embeds_np = np.array(token_embeds_mx.tolist(), dtype=np.float32)
    token_embeds = torch.from_numpy(token_embeds_np).to(DEVICE)  # (1, T, 1152)

    # Replace image placeholder positions with actual vit features
    token_embeds[0, :n_img_tokens] = vit_embeds[0].to(torch.float32)

    # --- Run Gemma3-1b transformer via MLX ---
    # Convert back to MLX for fast transformer pass
    input_embeds_mx = mx.array(token_embeds.cpu().numpy())  # (1, T, 1152)
    attention_mask_mx = mx.ones((1, len(full_ids)), dtype=mx.int32)

    # Forward through transformer layers
    lm_out = mlx_llm.model(
        inputs_embeds=input_embeds_mx,
        attention_mask=attention_mask_mx,
        output_hidden_states=False,
    )
    hidden_states_np = np.array(lm_out.last_hidden_state.tolist(), dtype=np.float32)
    backbone_features = torch.from_numpy(hidden_states_np).to(DEVICE)  # (1, T, 1152)
    backbone_attn_mask = torch.ones(1, len(full_ids), dtype=torch.bool, device=DEVICE)

    # --- Robot state encoding ---
    state_tensor = torch.tensor(robot_state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    # --- DiT action head ---
    from transformers.feature_extraction_utils import BatchFeature
    backbone_output = BatchFeature(data={
        "backbone_features": backbone_features,
        "backbone_attention_mask": backbone_attn_mask,
        "image_mask": torch.zeros(1, len(full_ids), dtype=torch.bool, device=DEVICE),
    })

    with torch.no_grad():
        actions = dit_model.get_action(
            backbone_output=backbone_output,
            state=state_tensor,
            n_diffusion_steps=n_diffusion_steps,
        )

    return actions.cpu().numpy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gr00t_ckpt", default="youngbrett48/train_google_robot_eagle2_5")
    p.add_argument("--checkpoint", default="checkpoint-200")
    p.add_argument("--mlx_llm_path", required=True, help="Path to converted MLX LLM (from extract_llm.py + mlx_lm.convert)")
    p.add_argument("--eagle2_5_repo", default="youngbrett48/train_stage2_gemma3_1b.sh")
    p.add_argument("--image", required=True, help="Path to robot camera image")
    p.add_argument("--state", default="0,0,0,0,0,0,0", help="Comma-separated robot joint states")
    p.add_argument("--instruction", default="pick up the object", help="Language instruction")
    p.add_argument("--config_json", default=None, help="Path to gr00t config.json (auto-downloaded if not set)")
    args = p.parse_args()

    # Load
    state_dict = load_gr00t_components(args.gr00t_ckpt, args.checkpoint)
    vision_encoder, mlp_projector, eagle_config = build_vision_components(state_dict, args.eagle2_5_repo)
    mlx_llm, tokenizer = load_mlx_llm(args.mlx_llm_path)

    # Get config.json for DiT
    if args.config_json is None:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(args.gr00t_ckpt, f"{args.checkpoint}/config.json")
    else:
        cfg_path = args.config_json
    dit_model = build_dit(state_dict, cfg_path)

    print(f"[5/5] Running inference...")
    image = Image.open(args.image).convert("RGB")
    robot_state = np.array([float(x) for x in args.state.split(",")])

    actions = run_inference(
        image=image,
        robot_state=robot_state,
        language_instruction=args.instruction,
        vision_encoder=vision_encoder,
        mlp_projector=mlp_projector,
        mlx_llm=mlx_llm,
        tokenizer=tokenizer,
        dit_model=dit_model,
        eagle_config=eagle_config,
    )

    print(f"\nPredicted actions (shape {actions.shape}):")
    print(actions)


if __name__ == "__main__":
    main()

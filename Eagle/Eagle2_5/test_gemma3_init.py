"""
Quick sanity test: initialize Eagle2.5 with a tiny Gemma3 text backbone from scratch
and run a forward pass with dummy data (no real images, no distributed).

Run from the Eagle2_5/ directory:
    python test_gemma3_init.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

from eaglevl.model.eagle2_5.configuration_eagle2_5_vl import Eagle2_5_VLConfig
from eaglevl.model.eagle2_5.modeling_eagle2_5_vl import Eagle2_5_VLForConditionalGeneration

# ------------------------------------------------------------------
# Tiny configs — just enough to instantiate without OOM
# ------------------------------------------------------------------

vision_cfg = {
    "model_type": "siglip_vision_model",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "image_size": 224,
    "patch_size": 14,
}

text_cfg = {
    "architectures": ["Gemma3ForCausalLM"],
    "model_type": "gemma3_text",
    "vocab_size": 512,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "sliding_window": 64,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_activation": "gelu_pytorch_tanh",
    "embedding_scaling": True,
    "initializer_range": 0.02,
    "pad_token_id": 0,
    "tie_word_embeddings": False,
}

config = Eagle2_5_VLConfig(
    vision_config=vision_cfg,
    text_config=text_cfg,
    downsample_ratio=0.5,
    use_pixel_shuffle=True,
    mlp_connector_layers=2,
    image_token_index=5,
    select_layer=-1,  # tiny vision model only has 2 layers
)

print("Config OK")
print(f"  text backbone: {config.text_config.model_type}")
print(f"  vision backbone: {config.vision_config.model_type}")

# ------------------------------------------------------------------
# Instantiate the model
# ------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Eagle2_5_VLForConditionalGeneration(config).to(device).to(torch.bfloat16)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel instantiated OK — {total_params:,} params")

# ------------------------------------------------------------------
# Dummy forward pass
# The forward() expects:
#   pixel_values: (num_images, C, H, W)
#   input_ids:    (batch, seq_len)  — some tokens are image_token_index
#   attention_mask, labels
# ------------------------------------------------------------------
BATCH = 1
NUM_IMAGES = 1
IMAGE_TOKEN_INDEX = config.image_token_index

# num_image_token tiles per image (set by config)
num_img_tok = model.num_image_token
print(f"  num_image_token per image: {num_img_tok}")

SEQ = num_img_tok + 32  # enough room for image tokens + some text

# Build input_ids: put image tokens at the start of each sequence
input_ids = torch.randint(1, config.text_config.vocab_size, (BATCH, SEQ)).to(device)
input_ids[:, :num_img_tok] = IMAGE_TOKEN_INDEX

attention_mask = torch.ones(BATCH, SEQ, dtype=torch.long).to(device)
labels = input_ids.clone()
labels[:, :num_img_tok] = -100

# pixel_values: (num_images, 3, H, W)
H = W = config.vision_config.image_size
pixel_values = torch.randn(NUM_IMAGES, 3, H, W).to(device).to(torch.bfloat16)

# image_flags: which pixel_value entries are real (all real here)
image_flags = torch.ones(NUM_IMAGES, 1, dtype=torch.long).to(device)

print("\nRunning forward pass...")
with torch.no_grad():
    out = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_flags=image_flags,
        labels=labels,
        use_cache=False,
    )

print(f"  loss: {out.loss.item():.4f}")
print(f"  logits shape: {out.logits.shape}")
print("\nAll checks passed.")

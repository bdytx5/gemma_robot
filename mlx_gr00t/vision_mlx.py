"""
MLX port of the SigLIP vision encoder + pixel shuffle + MLP projector.
Replaces the PyTorch MPS Eagle2.5 vision pipeline.

Architecture (428M + 3.4M params):
  SigLIP ViT: Conv2d patch embed → 27 transformer layers → post_layernorm
  pixel_shuffle: (B, 729, 1152) → (B, 196, 4608)
  mlp1: LayerNorm → Linear(4608→640) → GELU → Linear(640→640)

Optimizations:
  - float16 compute throughout (halves memory bandwidth)
  - mx.compile() on encoder forward pass (kernel fusion)
"""

import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# SigLIP Attention
# ---------------------------------------------------------------------------

class SiglipAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# SigLIP MLP
# ---------------------------------------------------------------------------

class SiglipMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, x):
        return self.fc2(nn.gelu_approx(self.fc1(x)))


# ---------------------------------------------------------------------------
# SigLIP Encoder Layer
# ---------------------------------------------------------------------------

class SiglipEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, eps=1e-6):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.self_attn = SiglipAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = SiglipMLP(hidden_size, intermediate_size)

    def __call__(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full SigLIP Vision Encoder
# ---------------------------------------------------------------------------

class SiglipVisionEncoder(nn.Module):
    def __init__(self, hidden_size=1152, num_heads=18, intermediate_size=4304,
                 num_layers=27, image_size=384, patch_size=14, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.image_size = image_size
        num_patches = (image_size // patch_size) ** 2  # 729

        # Patch embedding (Conv2d)
        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=hidden_size,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )
        self.position_embedding = nn.Embedding(num_patches, hidden_size)

        # Encoder layers
        self.layers = [
            SiglipEncoderLayer(hidden_size, num_heads, intermediate_size, eps)
            for _ in range(num_layers)
        ]
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=eps)

    def __call__(self, pixel_values):
        """
        Args:
            pixel_values: (B, H, W, 3) float32 in [-1, 1] (NHWC for MLX)
        Returns:
            (B, num_patches, hidden_size)
        """
        # Patch embed: (B, H, W, 3) → (B, H//P, W//P, hidden_size)
        x = self.patch_embedding(pixel_values)
        B = x.shape[0]
        x = x.reshape(B, -1, self.hidden_size)  # (B, num_patches, hidden_size)

        # Position embedding
        pos_ids = mx.arange(x.shape[1])
        x = x + self.position_embedding(pos_ids)

        # Encoder
        for layer in self.layers:
            x = layer(x)

        return self.post_layernorm(x)


# ---------------------------------------------------------------------------
# Pixel Shuffle — spatial downsampling
# ---------------------------------------------------------------------------

def pixel_shuffle(x, scale_factor=0.5):
    """
    Matches Eagle2_5's pixel_shuffle.
    Input: (B, num_patches, hidden_size) e.g. (B, 729, 1152)
    Output: (B, num_patches * scale^2, hidden_size / scale^2) e.g. (B, 196, 4608)
    """
    B = x.shape[0]
    h = w = int(math.sqrt(x.shape[1]))

    x = x.reshape(B, h, w, -1)

    # Pad to even if needed
    if h % 2 != 0:
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        h += 1
        w += 1

    c = x.shape[-1]
    # N, W, H, C → N, W, H*scale, C//scale
    x = x.reshape(B, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H*scale, C//scale → N, H*scale, W, C//scale
    x = x.transpose(0, 2, 1, 3)
    # N, H*scale, W, C//scale → N, H*scale, W*scale, C//(scale^2)
    x = x.reshape(B, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
    x = x.transpose(0, 2, 1, 3)

    # Flatten spatial dims
    return x.reshape(B, -1, x.shape[-1])


# ---------------------------------------------------------------------------
# MLP1 Projector — LayerNorm → Linear → GELU → Linear
# ---------------------------------------------------------------------------

class MLP1Projector(nn.Module):
    def __init__(self, in_dim=4608, out_dim=640):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def __call__(self, x):
        x = self.ln(x)
        x = nn.gelu(self.linear1(x))  # exact GELU (not approx) matching nn.GELU()
        return self.linear2(x)


# ---------------------------------------------------------------------------
# Full Vision Pipeline
# ---------------------------------------------------------------------------

class EagleVisionMLX(nn.Module):
    """Complete vision pipeline: SigLIP + pixel_shuffle + mlp1."""

    def __init__(self, hidden_size=1152, num_heads=18, intermediate_size=4304,
                 num_layers=27, image_size=384, patch_size=14,
                 downsample_ratio=0.5, mlp_out_dim=640):
        super().__init__()
        self.downsample_ratio = downsample_ratio
        ps_dim = hidden_size * int(1 / downsample_ratio) ** 2  # 4608

        self.encoder = SiglipVisionEncoder(
            hidden_size, num_heads, intermediate_size, num_layers, image_size, patch_size,
        )
        self.mlp1 = MLP1Projector(ps_dim, mlp_out_dim)

    def __call__(self, pixel_values):
        """
        Args:
            pixel_values: (B, H, W, 3) float32 in [-1, 1]
        Returns:
            (B, num_img_tokens, mlp_out_dim)
        """
        x = pixel_values.astype(getattr(self, "_compute_dtype", mx.float16))
        vit_embeds = self.encoder(x)
        vit_embeds = pixel_shuffle(vit_embeds, self.downsample_ratio)
        return self.mlp1(vit_embeds).astype(mx.float32)


# ---------------------------------------------------------------------------
# Weight conversion: PyTorch Eagle state_dict → MLX
# ---------------------------------------------------------------------------

def convert_vision_weights(state_dict):
    """Convert Eagle2.5 vision+mlp1 weights to MLX format.

    Expects full GR00T state dict with backbone.model.vision_model.* and
    backbone.model.mlp1.* keys.
    """
    import torch

    prefix_vision = "backbone.model.vision_model.vision_model."
    prefix_mlp1 = "backbone.model.mlp1."

    mlx_weights = {}

    for pt_key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().float().numpy()
        else:
            arr = np.array(tensor, dtype=np.float32)

        # --- Vision encoder ---
        if pt_key.startswith(prefix_vision):
            suffix = pt_key[len(prefix_vision):]

            # Skip pooling head (not needed for extract_feature)
            if suffix.startswith("head."):
                continue

            # Remap: embeddings.patch_embedding → encoder.patch_embedding
            suffix = suffix.replace("embeddings.patch_embedding", "patch_embedding")
            suffix = suffix.replace("embeddings.position_embedding", "position_embedding")

            # Remap: encoder.layers.N → layers.N
            suffix = suffix.replace("encoder.layers.", "layers.")

            # Remap: self_attn → attn (keep self_attn to match our class name)
            # Actually keep same names: self_attn.q_proj etc.

            mlx_key = f"encoder.{suffix}"

            # Conv2d weight: PyTorch (Cout, Cin, kH, kW) → MLX (Cout, kH, kW, Cin)
            if "patch_embedding.weight" in suffix:
                arr = np.transpose(arr, (0, 2, 3, 1))

            mlx_weights[mlx_key] = mx.array(arr)

        # --- MLP1 projector ---
        elif pt_key.startswith(prefix_mlp1):
            suffix = pt_key[len(prefix_mlp1):]
            # mlp1.0.weight/bias → mlp1.ln.weight/bias (LayerNorm)
            # mlp1.1.weight/bias → mlp1.linear1.weight/bias
            # mlp1.3.weight/bias → mlp1.linear2.weight/bias
            if suffix.startswith("0."):
                mlx_key = f"mlp1.ln.{suffix[2:]}"
            elif suffix.startswith("1."):
                mlx_key = f"mlp1.linear1.{suffix[2:]}"
            elif suffix.startswith("3."):
                mlx_key = f"mlp1.linear2.{suffix[2:]}"
            else:
                continue
            mlx_weights[mlx_key] = mx.array(arr)

    print(f"  Vision weight conversion: {len(mlx_weights)} parameters mapped")
    return mlx_weights


def build_vision_mlx(state_dict, eagle_config, dtype: str = "float16"):
    """Build MLX vision encoder from PyTorch state dict and Eagle config.

    Args:
        state_dict: Full GR00T state dict
        eagle_config: Eagle2_5_VLConfig (or dict with vision_config)

    Returns:
        EagleVisionMLX model with loaded weights
    """
    vc = eagle_config.vision_config if hasattr(eagle_config, 'vision_config') else eagle_config
    hidden_size = vc.hidden_size       # 1152
    # Eagle config may report 16 heads but SigLIP-So400M uses 18 (hidden_size/head_dim=1152/64=18)
    num_heads = vc.num_attention_heads
    if hidden_size % num_heads != 0 or (hidden_size // num_heads) not in (48, 64, 72, 96):
        num_heads = hidden_size // 64   # fallback: derive from standard head_dim=64
    intermediate = vc.intermediate_size  # 4304
    num_layers = vc.num_hidden_layers   # 27
    image_size = getattr(vc, 'image_size', 378)   # 378 = 27*14
    patch_size = vc.patch_size          # 14

    downsample_ratio = getattr(eagle_config, 'downsample_ratio', 0.5)
    llm_hidden = getattr(eagle_config, 'text_config', None)
    mlp_out = llm_hidden.hidden_size if llm_hidden else 640

    print(f"[vision_mlx] Building: SigLIP-{hidden_size}d x {num_layers}L, "
          f"{image_size}px/{patch_size}px, mlp1→{mlp_out}d")

    model = EagleVisionMLX(
        hidden_size=hidden_size, num_heads=num_heads,
        intermediate_size=intermediate, num_layers=num_layers,
        image_size=image_size, patch_size=patch_size,
        downsample_ratio=downsample_ratio, mlp_out_dim=mlp_out,
    )

    weights = convert_vision_weights(state_dict)
    model.load_weights(list(weights.items()))

    _dtype = getattr(mx, dtype)
    import mlx.utils
    params = mlx.utils.tree_map(lambda x: x.astype(_dtype) if x.dtype == mx.float32 else x,
                                 model.parameters())
    model.load_weights(list(mlx.utils.tree_flatten(params)))
    model._compute_dtype = _dtype

    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    bytes_per = {"float16": 2, "bfloat16": 2, "float32": 4}.get(dtype, 2)
    print(f"  MLX vision loaded: {n_params:,} parameters ({n_params * bytes_per / 1e9:.2f} GB {dtype})")

    return model


def build_vision_mlx_from_exported(safetensors_path: str, meta: dict, dtype: str = "float16"):
    """Load vision model from pre-exported MLX safetensors (no PyTorch needed)."""
    import mlx.utils

    hidden_size      = 1152
    num_heads        = 18   # 1152 / 64 = 18 heads (was wrong at 16)
    intermediate     = 4304
    num_layers       = 27
    image_size       = meta["image_size"]
    patch_size       = 14
    downsample_ratio = 0.5
    mlp_out          = 640

    model = EagleVisionMLX(
        hidden_size=hidden_size, num_heads=num_heads,
        intermediate_size=intermediate, num_layers=num_layers,
        image_size=image_size, patch_size=patch_size,
        downsample_ratio=downsample_ratio, mlp_out_dim=mlp_out,
    )

    _dtype = getattr(mx, dtype)
    weights = mx.load(safetensors_path)
    weights = {k: v.astype(_dtype) for k, v in weights.items()}
    model.load_weights(list(weights.items()))
    model._compute_dtype = _dtype

    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    bytes_per = {"float16": 2, "bfloat16": 2, "float32": 4}.get(dtype, 2)
    print(f"  Vision loaded from exported: {n_params:,} params ({n_params * bytes_per / 1e9:.2f} GB {dtype})")
    return model

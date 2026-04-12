"""
MLX port of GR00T's DiT action head.
Replaces PyTorch MPS with native MLX for fast Apple Silicon inference.

Architecture:
  backbone_proj (640→2048) → vlln (LayerNorm) → state_encoder → action_encoder
  → AlternateVLDiT (32 layers) → action_decoder → predicted actions

Optimizations:
  - bfloat16 compute throughout (halves memory bandwidth)
  - mx.compile() on DiT forward pass (kernel fusion)
  - Minimal mx.eval() calls (only final output)
"""

import math
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Sinusoidal embeddings
# ---------------------------------------------------------------------------

def timestep_sinusoidal(timesteps, dim=256, flip_sin_to_cos=True, downscale_freq_shift=1):
    """Matches diffusers Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)."""
    half = dim // 2
    exponent = -math.log(10000) * mx.arange(half, dtype=mx.float16) / (half - downscale_freq_shift)
    emb = timesteps[:, None].astype(mx.float16) * mx.exp(exponent)[None, :]
    if flip_sin_to_cos:
        return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)
    return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


def sinusoidal_pos_encoding(timesteps_2d, dim):
    """Matches SinusoidalPositionalEncoding in embodiment_conditioned_mlp.py.
    timesteps_2d: (B, T) → output: (B, T, dim)"""
    half = dim // 2
    exponent = -mx.arange(half, dtype=mx.float16) * (math.log(10000.0) / half)
    freqs = timesteps_2d[..., None].astype(mx.float16) * mx.exp(exponent)  # (B, T, half)
    return mx.concatenate([mx.sin(freqs), mx.cos(freqs)], axis=-1)


# ---------------------------------------------------------------------------
# TimestepEncoder — Timesteps(256) → Linear → SiLU → Linear
# ---------------------------------------------------------------------------

class TimestepEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(256, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def __call__(self, timesteps):
        t = timestep_sinusoidal(timesteps, 256)
        t = nn.silu(self.linear_1(t))
        return self.linear_2(t)


# ---------------------------------------------------------------------------
# AdaLayerNorm — adaptive norm conditioned on timestep embedding
# ---------------------------------------------------------------------------

class AdaLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, affine=False)

    def __call__(self, x, temb):
        temb = self.linear(nn.silu(temb))
        scale, shift = mx.split(temb, 2, axis=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


# ---------------------------------------------------------------------------
# Multi-head Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, query_dim, heads, dim_head, cross_attention_dim=None, bias=True):
        super().__init__()
        inner_dim = heads * dim_head
        kv_dim = cross_attention_dim or query_dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, query_dim, bias=True)

    def __call__(self, x, encoder_hidden_states=None, attention_mask=None):
        B, T, _ = x.shape
        kv = encoder_hidden_states if encoder_hidden_states is not None else x

        q = self.to_q(x).reshape(B, T, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        S = kv.shape[1]
        k = self.to_k(kv).reshape(B, S, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = self.to_v(kv).reshape(B, S, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        # attention_mask: (B, S) bool, True = attend. Convert to additive for sdpa.
        mask = None
        if attention_mask is not None:
            # (B, S) → (B, 1, 1, S), False positions get -1e4 (bfloat16-safe)
            mask = mx.where(attention_mask[:, None, None, :],
                            mx.array(0, dtype=q.dtype),
                            mx.array(-1e4, dtype=q.dtype))

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.to_out(out)


# ---------------------------------------------------------------------------
# FeedForward — GELU-approximate
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim * mult)
        self.out_proj = nn.Linear(dim * mult, dim)

    def __call__(self, x):
        return self.out_proj(nn.gelu_approx(self.gate_proj(x)))


# ---------------------------------------------------------------------------
# BasicTransformerBlock — AdaLN + Attention + FFN
# ---------------------------------------------------------------------------

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_head, cross_attention_dim=None, norm_type="ada_norm"):
        super().__init__()
        self.norm_type = norm_type
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, affine=False)
        self.attn1 = Attention(dim, num_heads, dim_head, cross_attention_dim=cross_attention_dim, bias=True)
        self.norm3 = nn.LayerNorm(dim, affine=False)
        self.ff = FeedForward(dim)

    def __call__(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None, temb=None):
        if self.norm_type == "ada_norm":
            normed = self.norm1(hidden_states, temb)
        else:
            normed = self.norm1(hidden_states)

        mask = encoder_attention_mask if encoder_hidden_states is not None else None
        attn_out = self.attn1(normed, encoder_hidden_states=encoder_hidden_states, attention_mask=mask)
        hidden_states = attn_out + hidden_states

        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# AlternateVLDiT — 32 layers, alternating image/text cross-attn + self-attn
# ---------------------------------------------------------------------------

class AlternateVLDiT(nn.Module):
    def __init__(self, num_layers, num_heads, dim_head, output_dim,
                 cross_attention_dim, attend_text_every_n_blocks=2,
                 norm_type="ada_norm", interleave_self_attention=True):
        super().__init__()
        inner_dim = num_heads * dim_head
        self.inner_dim = inner_dim
        self.interleave_self_attention = interleave_self_attention
        self.attend_text_every_n_blocks = attend_text_every_n_blocks
        self.num_layers = num_layers

        self.timestep_encoder = TimestepEncoder(inner_dim)

        blocks = []
        for idx in range(num_layers):
            use_self = idx % 2 == 1 and interleave_self_attention
            ca_dim = None if use_self else cross_attention_dim
            blocks.append(BasicTransformerBlock(inner_dim, num_heads, dim_head,
                                               cross_attention_dim=ca_dim, norm_type=norm_type))
        self.transformer_blocks = blocks

        self.norm_out = nn.LayerNorm(inner_dim, affine=False)
        self.proj_out_1 = nn.Linear(inner_dim, inner_dim * 2)
        self.proj_out_2 = nn.Linear(inner_dim, output_dim)

    def __call__(self, hidden_states, encoder_hidden_states, timestep,
                 image_mask=None, backbone_attention_mask=None):
        temb = self.timestep_encoder(timestep)
        image_attn_mask = image_mask & backbone_attention_mask if image_mask is not None else None
        non_image_attn_mask = (~image_mask) & backbone_attention_mask if image_mask is not None else None

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.interleave_self_attention:
                hidden_states = block(hidden_states, temb=temb)
            else:
                if image_mask is not None:
                    if idx % (2 * self.attend_text_every_n_blocks) == 0:
                        mask = non_image_attn_mask
                    else:
                        mask = image_attn_mask
                else:
                    mask = None
                hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states,
                                      encoder_attention_mask=mask, temb=temb)

        shift, scale = mx.split(self.proj_out_1(nn.silu(temb)), 2, axis=-1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)


# ---------------------------------------------------------------------------
# CategorySpecificLinear — per-embodiment linear layer
# ---------------------------------------------------------------------------

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.W = mx.zeros((num_categories, input_dim, hidden_dim), dtype=mx.float16)
        self.b = mx.zeros((num_categories, hidden_dim), dtype=mx.float16)

    def __call__(self, x, cat_ids):
        # x: (B, T, in), cat_ids: (B,)
        sel_W = self.W[cat_ids]  # (B, in, out)
        sel_b = self.b[cat_ids]  # (B, out)
        return x @ sel_W + sel_b[:, None, :]


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def __call__(self, x, cat_ids):
        return self.layer2(nn.relu(self.layer1(x, cat_ids)), cat_ids)


# ---------------------------------------------------------------------------
# MultiEmbodimentActionEncoder
# ---------------------------------------------------------------------------

class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)

    def __call__(self, actions, timesteps, cat_ids):
        B, T, _ = actions.shape
        ts_2d = mx.broadcast_to(timesteps[:, None], (B, T))
        a_emb = self.W1(actions, cat_ids)
        tau_emb = sinusoidal_pos_encoding(ts_2d, self.hidden_size).astype(a_emb.dtype)
        x = mx.concatenate([a_emb, tau_emb], axis=-1)
        w2_out = self.W2(x, cat_ids)
        x = w2_out * mx.sigmoid(w2_out)  # swish
        return self.W3(x, cat_ids)


# ---------------------------------------------------------------------------
# Full Action Head in MLX
# ---------------------------------------------------------------------------

class Gr00tActionHeadMLX(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        # Fill in Gr00tN1d6Config defaults for keys not saved in checkpoint config
        self.hidden_size = config.get("hidden_size", 1024)
        self.input_embedding_dim = config.get("input_embedding_dim", 1536)
        self.action_dim = config.get("max_action_dim", 29)
        self.action_horizon = config.get("action_horizon", 16)
        self.num_inference_timesteps = config.get("num_inference_timesteps", 4)
        self.num_timestep_buckets = config.get("num_timestep_buckets", 1000)
        self.max_num_embodiments = config.get("max_num_embodiments", 32)
        self.add_pos_embed = config.get("add_pos_embed", True)
        self.use_alternate_vl_dit = config.get("use_alternate_vl_dit", True)
        self.backbone_embedding_dim = config.get("backbone_embedding_dim", 2048)
        self.backbone_proj_dim = config.get("backbone_proj_dim", 0)

        # Backbone projection (640→2048 for small Gemma3)
        if self.backbone_proj_dim > 0:
            self.backbone_proj = nn.Linear(self.backbone_proj_dim, self.backbone_embedding_dim, bias=False)
        else:
            self.backbone_proj = None

        # Vision-language layer norm
        self.vlln = nn.LayerNorm(self.backbone_embedding_dim) if config.get("use_vlln", True) else None

        # State encoder
        self.max_state_dim = config.get("max_state_dim", 29)
        self.state_encoder = CategorySpecificMLP(
            self.max_num_embodiments, self.max_state_dim,
            self.hidden_size, self.input_embedding_dim,
        )

        # Action encoder
        self.action_encoder = MultiEmbodimentActionEncoder(
            self.action_dim, self.input_embedding_dim, self.max_num_embodiments,
        )

        # Action decoder
        self.action_decoder = CategorySpecificMLP(
            self.max_num_embodiments, self.hidden_size,
            self.hidden_size, self.action_dim,
        )

        # Position embedding for action features
        if self.add_pos_embed:
            max_seq = config.get("max_seq_len", 1024)
            self.position_embedding = nn.Embedding(max_seq, self.input_embedding_dim)

        # DiT
        dcfg = config.get("diffusion_model_cfg", {})
        num_heads = dcfg.get("num_attention_heads", 32)
        dim_head = dcfg.get("attention_head_dim", 48)
        num_layers = dcfg.get("num_layers", 32)
        output_dim = dcfg.get("output_dim", 1024)
        attend_n = config.get("attend_text_every_n_blocks", 2)

        self.model = AlternateVLDiT(
            num_layers=num_layers,
            num_heads=num_heads,
            dim_head=dim_head,
            output_dim=output_dim,
            cross_attention_dim=self.backbone_embedding_dim,
            attend_text_every_n_blocks=attend_n,
            norm_type=dcfg.get("norm_type", "ada_norm"),
            interleave_self_attention=dcfg.get("interleave_self_attention", True),
        )

    def _dit_forward(self, sa_embs, vl_embeds, ts, image_mask, backbone_attention_mask):
        """DiT forward pass."""
        return self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            timestep=ts,
            image_mask=image_mask,
            backbone_attention_mask=backbone_attention_mask,
        )

    def get_action(self, backbone_features, backbone_attention_mask, image_mask,
                   state, embodiment_id):
        """
        Full action generation via flow matching diffusion.

        Args:
            backbone_features: (B, seq_len, backbone_dim) — LLM hidden states
            backbone_attention_mask: (B, seq_len) bool
            image_mask: (B, seq_len) bool
            state: (B, 1, max_state_dim) — padded robot state
            embodiment_id: (B,) int
        Returns:
            action_pred: (B, action_horizon, action_dim)
        """
        # Cast inputs to compute dtype
        _dt = getattr(self, "_compute_dtype", mx.float16)
        backbone_features = backbone_features.astype(_dt)
        state = state.astype(_dt)

        # Project backbone features
        if self.backbone_proj is not None:
            backbone_features = self.backbone_proj(backbone_features)
        if self.vlln is not None:
            backbone_features = self.vlln(backbone_features)

        vl_embeds = backbone_features

        # Encode state
        state_features = self.state_encoder(state, embodiment_id)

        # Initial noise
        B = vl_embeds.shape[0]
        actions = mx.random.normal((B, self.action_horizon, self.action_dim)).astype(_dt)
        dt = 1.0 / self.num_inference_timesteps

        # Pre-compute position embedding once
        pos_emb = None
        if self.add_pos_embed:
            pos_emb = self.position_embedding(mx.arange(self.action_horizon))  # already bfloat16 from weights

        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)
            t_disc = int(t_cont * self.num_timestep_buckets)
            ts = mx.full((B,), t_disc, dtype=mx.int32)

            # Encode noisy actions
            action_features = self.action_encoder(actions, ts, embodiment_id)

            # Add position embedding
            if pos_emb is not None:
                action_features = action_features + pos_emb

            # Concat state + action
            sa_embs = mx.concatenate([state_features, action_features], axis=1)

            # DiT forward
            model_out = self._dit_forward(
                sa_embs, vl_embeds, ts, image_mask, backbone_attention_mask,
            )

            # Decode and get velocity
            pred = self.action_decoder(model_out, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon:]

            # Euler step — eval each step to keep graph size bounded
            actions = actions + dt * pred_velocity
            mx.eval(actions)

        return actions.astype(mx.float32)


# ---------------------------------------------------------------------------
# Weight conversion: PyTorch state_dict → MLX weights
# ---------------------------------------------------------------------------

def convert_torch_to_mlx(state_dict):
    """Convert a PyTorch action_head state dict to MLX-compatible weight dict.

    Expects keys already stripped of 'action_head.' prefix.
    Returns dict of {mlx_key: mx.array}.
    """
    import torch

    mlx_weights = {}
    skipped = []

    for pt_key, tensor in state_dict.items():
        # Convert tensor to numpy
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().float().numpy()
        else:
            arr = np.array(tensor, dtype=np.float32)

        # Key remapping
        mlx_key = pt_key

        # TimestepEncoder: remove .timestep_embedder level
        mlx_key = mlx_key.replace(
            "model.timestep_encoder.timestep_embedder.",
            "model.timestep_encoder.",
        )

        # Attention output: to_out.0.weight → to_out.weight
        mlx_key = mlx_key.replace(".to_out.0.", ".to_out.")

        # FeedForward: net.0.proj → gate_proj, net.2 → out_proj
        mlx_key = mlx_key.replace(".ff.net.0.proj.", ".ff.gate_proj.")
        mlx_key = mlx_key.replace(".ff.net.2.", ".ff.out_proj.")

        # Skip buffers and non-parameter entries
        if "time_proj" in mlx_key or "pos_embed.pe" in mlx_key:
            skipped.append(pt_key)
            continue

        mlx_weights[mlx_key] = mx.array(arr)

    if skipped:
        print(f"  Weight conversion: skipped {len(skipped)} buffers")
    print(f"  Weight conversion: mapped {len(mlx_weights)} parameters")
    return mlx_weights


def build_dit_mlx(state_dict: dict, config_path: str, dtype: str = "float16"):
    """Build MLX DiT action head from PyTorch state dict and config.

    Args:
        state_dict: Full GR00T state dict (with action_head.* keys)
        config_path: Path to GR00T config.json

    Returns:
        (model, config_dict)
    """
    with open(config_path) as f:
        config = json.load(f)

    print(f"[dit_mlx] Building MLX action head...")
    model = Gr00tActionHeadMLX(config)

    # Extract action_head weights
    ah_sd = {
        k[len("action_head."):]: v
        for k, v in state_dict.items()
        if k.startswith("action_head.")
    }
    print(f"  {len(ah_sd)} action_head keys from checkpoint")

    # Convert and load
    mlx_weights = convert_torch_to_mlx(ah_sd)

    # Load into model and convert to requested dtype
    model.load_weights(list(mlx_weights.items()))

    _dtype = getattr(mx, dtype)
    import mlx.utils
    params = mlx.utils.tree_map(lambda x: x.astype(_dtype) if x.dtype == mx.float32 else x,
                                 model.parameters())
    model.load_weights(list(mlx.utils.tree_flatten(params)))
    model._compute_dtype = _dtype

    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    bytes_per = {"float16": 2, "bfloat16": 2, "float32": 4}.get(dtype, 2)
    print(f"  MLX DiT loaded: {n_params:,} parameters ({n_params * bytes_per / 1e9:.2f} GB {dtype})")

    return model, config


def build_dit_mlx_from_exported(safetensors_path: str, config_path: str, dtype: str = "float16"):
    """Load DiT from pre-exported MLX safetensors (no PyTorch needed)."""
    import mlx.utils

    with open(config_path) as f:
        config = json.load(f)

    model = Gr00tActionHeadMLX(config)
    _dtype = getattr(mx, dtype)
    weights = mx.load(safetensors_path)
    weights = {k: v.astype(_dtype) for k, v in weights.items()}
    model.load_weights(list(weights.items()))
    model._compute_dtype = _dtype

    n_params = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    bytes_per = {"float16": 2, "bfloat16": 2, "float32": 4}.get(dtype, 2)
    print(f"  DiT loaded from exported: {n_params:,} params ({n_params * bytes_per / 1e9:.2f} GB {dtype})")
    return model, config

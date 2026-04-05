# Stub – flash_attn.layers.rotary (SDPA shim for sm_120)
# These are imported by transformers but only used in flash-attn code paths.
# We provide no-op stubs so the import succeeds; actual rotary is handled by
# the model's own RoPE implementation.

import torch


def apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False, conjugate=False):
    """Stub apply_rotary_emb — should never be called when using SDPA."""
    raise RuntimeError(
        "flash_attn.layers.rotary.apply_rotary_emb stub called — "
        "this should not happen when attn_implementation='sdpa'. "
        "Check that the model is not using flash_attention_2 code paths."
    )


class RotaryEmbedding(torch.nn.Module):
    """Stub RotaryEmbedding — only for import compatibility."""
    def __init__(self, dim, *args, **kwargs):
        super().__init__()
    def forward(self, x, seq_len=None):
        raise RuntimeError("flash_attn RotaryEmbedding stub called")

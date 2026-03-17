# coding=utf-8
# Adapted from transformers Gemma3 implementation for Eagle2.5.
# Modifications: sequence packing support (sub_sample_lengths), sequence parallel
# support, and ring attention — following the same pattern as eaglevl/model/qwen2/modeling_qwen2.py.
#
# Original copyright:
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0

import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, HybridCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from eaglevl.sp_utils import (
    pre_process_for_sequence_parallel_attn,
    post_process_for_sequence_parallel_attn,
    get_pg_manager,
    split_for_sequence_parallel,
    ring_split_for_sequence_parallel,
)
from eaglevl.sp_utils.ring import (
    zigzag_ring_flash_attn_varlen_func,
    zigzag_ring_flash_attn_func,
    ring_flash_attn_varlen_func,
    ring_flash_attn_func,
)

if is_flash_attn_2_available():
    try:
        from flash_attn_interface import flash_attn_func, flash_attn_varlen_func
        use_flash_attn3 = True
    except ImportError:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        use_flash_attn3 = False

    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Packing helpers (mirrored from qwen2/modeling_qwen2.py)
# ---------------------------------------------------------------------------

def _get_unpad_data(attention_mask):
    seqlens_in_batch = (attention_mask > 0).sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def _get_unpad_data_packing(attention_mask, sub_sample_lengths):
    seqlens_in_batch = []
    for i, per_sub_sample_lengths in enumerate(sub_sample_lengths):
        if (attention_mask[i] == 0).sum() == per_sub_sample_lengths[-1]:
            per_sub_sample_lengths = per_sub_sample_lengths[:-1]
        seqlens_in_batch.extend(per_sub_sample_lengths)
    seqlens_in_batch = torch.tensor(seqlens_in_batch, device=attention_mask.device, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


# ---------------------------------------------------------------------------
# Core building blocks (unchanged from stock Gemma3)
# ---------------------------------------------------------------------------

class Gemma3TextScaledWordEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx, embed_scale=1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_scale = embed_scale

    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Gemma3RotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma3TextConfig, device=None):
        super().__init__()
        # Gemma3 has separate RoPE configs per layer type
        self.rope_type_to_config = {
            "default": config.rope_scaling if hasattr(config, "rope_scaling") and config.rope_scaling else {},
        }
        # Build inv_freq inline — Gemma3 uses standard RoPE
        self.head_dim = config.head_dim
        self.base = config.rope_theta
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = config.max_position_embeddings

    def _online_get_cos_sin(self, seq_len, device):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, x, position_ids, layer_type="full_attention"):
        # position_ids: (batch, seq_len)
        seq_len = position_ids.shape[-1]
        max_pos = position_ids.max().item() + 1
        cos, sin = self._online_get_cos_sin(max(seq_len, max_pos), x.device)
        # index by position_ids: (batch, seq_len, head_dim)
        cos = cos[position_ids]
        sin = sin[position_ids]
        return cos.to(x.dtype), sin.to(x.dtype)


# ---------------------------------------------------------------------------
# Attention — with Eagle packing + sequence parallel support
# ---------------------------------------------------------------------------

class Gemma3Attention(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim ** -0.5
        # Gemma3 alternates full / sliding window attention per layer
        self.sliding_window = config.sliding_window if (layer_idx % 2 == 0) else None
        self.attn_logit_softcapping = getattr(config, "attn_logit_softcapping", None)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Gemma3-specific: Q and K get normed after projection
        self.q_norm = Gemma3RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.ring_attn_fn = ring_flash_attn_varlen_func

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        sub_sample_lengths=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Gemma3-specific: norm Q and K
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sliding_window": self.sliding_window}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        kv_seq_len = key_states.shape[-2]

        use_sliding_windows = (
            _flash_supports_window_size
            and self.sliding_window is not None
            and kv_seq_len > self.sliding_window
        )

        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # Cast to target dtype if needed
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Flash attn expects (bsz, seq_len, heads, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        softmax_scale = self.scaling
        # Gemma3 softcapping: not natively supported by flash_attn, apply pre/post manually
        # when softcapping is set. For now we skip it during training (flash path) and it
        # takes effect only in eager/SDPA fallback. TODO: implement custom kernel wrapper.
        if self.attn_logit_softcapping is not None:
            logger.warning_once(
                "attn_logit_softcapping is not supported in the flash attention path; "
                "it will be ignored. Use eager attention if you need it."
            )

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            softmax_scale=softmax_scale,
            use_sliding_windows=use_sliding_windows,
            sub_sample_lengths=sub_sample_lengths,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
        sub_sample_lengths=None,
    ):
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        enable_sequence_parallel = (
            dist.is_initialized()
            and get_pg_manager() is not None
            and get_pg_manager().sequence_parallel_world_size > 1
        )

        if enable_sequence_parallel:
            query_states, key_states, value_states = pre_process_for_sequence_parallel_attn(
                query_states, key_states, value_states
            )
            query_length = query_states.shape[1]

        if attention_mask is not None:
            batch_size = query_states.shape[0]

            if sub_sample_lengths is not None:
                query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = \
                    self._unpad_input_packing(query_states, key_states, value_states, attention_mask, query_length, sub_sample_lengths)
            else:
                query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = \
                    self._upad_input(query_states, key_states, value_states, attention_mask, query_length)

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            window_size = (self.sliding_window, self.sliding_window) if use_sliding_windows else (-1, -1)

            if get_pg_manager() is not None and get_pg_manager().ring_sequence_parallel_size > 1:
                assert torch.equal(cu_seqlens_q, cu_seqlens_k)
                attn_output_unpad = self.ring_attn_fn(
                    query_states, key_states, value_states,
                    cu_seqlens=cu_seqlens_q,
                    max_seqlen=max_seqlen_in_batch_q,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    group=get_pg_manager().ring_sequence_parallel_group,
                )
            else:
                if use_flash_attn3:
                    attn_output_unpad, _ = flash_attn_varlen_func(
                        query_states, key_states, value_states,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=window_size,
                    )
                else:
                    attn_output_unpad = flash_attn_varlen_func(
                        query_states, key_states, value_states,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_in_batch_q,
                        max_seqlen_k=max_seqlen_in_batch_k,
                        dropout_p=dropout,
                        softmax_scale=softmax_scale,
                        causal=causal,
                        window_size=window_size,
                    )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            window_size = (self.sliding_window, self.sliding_window) if use_sliding_windows else (-1, -1)

            if get_pg_manager() is not None and get_pg_manager().ring_sequence_parallel_size > 1:
                attn_output = self.ring_attn_fn(
                    query_states, key_states, value_states,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    group=get_pg_manager().ring_sequence_parallel_group,
                )
            else:
                attn_output = flash_attn_func(
                    query_states, key_states, value_states,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                )
                if isinstance(attn_output, tuple):
                    attn_output = attn_output[0]

        if enable_sequence_parallel:
            attn_output = post_process_for_sequence_parallel_attn(attn_output)

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
        _, _, num_heads_q, _ = query_layer.shape

        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask = attention_mask[:, attention_mask.shape[-1] - kv_seq_len:]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads_q, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return query_layer, key_layer, value_layer, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k)

    def _unpad_input_packing(self, query_layer, key_layer, value_layer, attention_mask, query_length, sub_sample_lengths):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
        _, _, num_heads_q, _ = query_layer.shape

        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask = attention_mask[:, attention_mask.shape[-1] - kv_seq_len:]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data_packing(attention_mask, sub_sample_lengths)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads_q, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(batch_size + 1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return query_layer, key_layer, value_layer, indices_q, (cu_seqlens_q, cu_seqlens_k), (max_seqlen_in_batch_q, max_seqlen_in_batch_k)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        sub_sample_lengths=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            sub_sample_lengths=sub_sample_lengths,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# ---------------------------------------------------------------------------
# PreTrainedModel base
# ---------------------------------------------------------------------------

class Gemma3TextPreTrainedModel(PreTrainedModel):
    config_class = Gemma3TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gemma3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Gemma3RMSNorm):
            module.weight.data.zero_()


# ---------------------------------------------------------------------------
# Text model (no vision — Eagle provides that)
# ---------------------------------------------------------------------------

class Gemma3TextModel(Gemma3TextPreTrainedModel):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        embed_scale = math.sqrt(config.hidden_size) if getattr(config, 'embedding_scaling', True) else 1.0
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=embed_scale
        )
        self.layers = nn.ModuleList(
            [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        sub_sample_lengths=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1],
                dtype=torch.long, device=inputs_embeds.device,
            ).unsqueeze(0)

        if get_pg_manager() is not None and get_pg_manager().sequence_parallel_world_size > 1:
            rotary_seq_len = inputs_embeds.shape[1] * get_pg_manager().sequence_parallel_world_size
        else:
            rotary_seq_len = inputs_embeds.shape[1]

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    sub_sample_lengths,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    sub_sample_lengths=sub_sample_lengths,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ---------------------------------------------------------------------------
# CausalLM head — used as the language_model inside Eagle2_5_VLForConditionalGeneration
# ---------------------------------------------------------------------------

class Gemma3ForCausalLM(Gemma3TextPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_logit_softcapping = getattr(config, "final_logit_softcapping", None)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        sub_sample_lengths=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            sub_sample_lengths=sub_sample_lengths,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

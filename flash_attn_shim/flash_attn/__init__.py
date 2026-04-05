"""Shim flash_attn — passes import checks, real attention uses SDPA."""
__version__ = "2.7.4.post1"
from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    _flash_attn_forward,
    _flash_attn_backward,
)
from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis

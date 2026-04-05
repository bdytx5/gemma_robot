"""Stub — should never be called if attn_implementation=sdpa is used."""
def flash_attn_func(*args, **kwargs):
    raise RuntimeError("flash_attn stub called — attn_implementation should be sdpa")
def flash_attn_varlen_func(*args, **kwargs):
    raise RuntimeError("flash_attn stub called — attn_implementation should be sdpa")
flash_attn_varlen_qkvpacked_func = flash_attn_varlen_func
_flash_attn_forward = flash_attn_func
_flash_attn_backward = flash_attn_func

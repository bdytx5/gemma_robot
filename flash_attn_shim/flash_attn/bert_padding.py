"""Stub — should never be called if attn_implementation=sdpa is used."""
def pad_input(*args, **kwargs):
    raise RuntimeError("flash_attn stub called")
def unpad_input(*args, **kwargs):
    raise RuntimeError("flash_attn stub called")
def index_first_axis(*args, **kwargs):
    raise RuntimeError("flash_attn stub called")

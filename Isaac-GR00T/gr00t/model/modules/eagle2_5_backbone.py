"""
Backbone wrapper for Eagle2.5 + Gemma3 VLM (your custom-trained checkpoint).

The checkpoint is an Eagle2_5_VLForConditionalGeneration model with:
  - vision: SiglipVisionModel (siglip2-so400m-patch14-384, force_image_size=384)
  - language: Gemma3ForCausalLM (gemma-3-270m, hidden_size=1152)
  - connector: 2-layer MLP with pixel_shuffle (downsample_ratio=0.5)
  - image_token_index: tokenizer IMG_CONTEXT token id

This class presents the same interface as EagleBackbone so the rest of GR00T
doesn't need to know which backbone is running:
  forward() -> {backbone_features, backbone_attention_mask, image_mask}

The backbone is loaded from a HuggingFace Hub repo (or local path) using
standard from_pretrained. The Eagle2.5 source code must be importable —
either install the eaglevl package or add the Eagle2_5 repo to sys.path.
"""

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
from transformers.feature_extraction_utils import BatchFeature

# Eagle2.5 lives in the sibling repo — add it to path if not already installed.
_EAGLE_REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "Eagle", "Eagle2_5")
)
if os.path.isdir(_EAGLE_REPO) and _EAGLE_REPO not in sys.path:
    sys.path.insert(0, _EAGLE_REPO)

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class Eagle2_5Backbone(nn.Module):
    """
    Wraps Eagle2_5_VLForConditionalGeneration as a GR00T backbone.

    Args:
        model_name: HF repo id or local path to the Eagle2.5+Gemma3 checkpoint.
        tune_llm: Whether to unfreeze the LLM layers.
        tune_visual: Whether to unfreeze the vision encoder + MLP connector.
        tune_top_llm_layers: Number of top LLM transformer layers to unfreeze
            even when tune_llm=False.
        load_bf16: Load weights in bfloat16.
        trainable_params_fp32: Cast trainable params to fp32 (mixed precision).
        transformers_loading_kwargs: Extra kwargs forwarded to from_pretrained.
    """

    def __init__(
        self,
        model_name: str,
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,        # kept for API compat; Eagle2.5 always uses last layer
        reproject_vision: bool = False, # kept for API compat; unused
        use_flash_attention: bool = True,
        projector_dim: int = -1,        # kept for API compat; unused
        load_bf16: bool = True,
        tune_top_llm_layers: int = 4,
        trainable_params_fp32: bool = True,
        transformers_loading_kwargs: dict = {},
    ):
        super().__init__()

        from eaglevl.model.eagle2_5.modeling_eagle2_5_vl import (
            Eagle2_5_VLForConditionalGeneration,
        )

        extra_kwargs = dict(transformers_loading_kwargs)
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16
        if use_flash_attention:
            extra_kwargs["attn_implementation"] = "flash_attention_2"

        print(f"[Eagle2_5Backbone] Loading checkpoint from: {model_name}")
        self.model = Eagle2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            **extra_kwargs,
        )

        # Resolve image token index from the tokenizer that was saved with the checkpoint
        self._resolve_image_token_index(model_name, extra_kwargs)

        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)

        if load_bf16 and trainable_params_fp32:
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    print(f"[Eagle2_5Backbone] Casting trainable param {n} to fp32")

    def _resolve_image_token_index(self, model_name: str, loading_kwargs: dict):
        """Load the tokenizer and resolve <IMG_CONTEXT> token id."""
        try:
            from transformers import GemmaTokenizer
            tok_kwargs = {k: v for k, v in loading_kwargs.items() if k not in ("torch_dtype", "attn_implementation", "trust_remote_code")}
            tokenizer = GemmaTokenizer.from_pretrained(model_name, use_fast=False, **tok_kwargs)
            idx = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            if idx is not None and idx != tokenizer.unk_token_id:
                self.model.image_token_index = idx
                print(f"[Eagle2_5Backbone] image_token_index={idx} ({IMG_CONTEXT_TOKEN})")
            else:
                print(f"[Eagle2_5Backbone] WARNING: could not resolve {IMG_CONTEXT_TOKEN}, "
                      f"using config value {self.model.image_token_index}")
        except Exception as e:
            print(f"[Eagle2_5Backbone] WARNING: tokenizer load failed ({e}), "
                  f"using config image_token_index={self.model.image_token_index}")

    def set_trainable_parameters(
        self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int
    ):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual

        # Start frozen
        for p in self.model.parameters():
            p.requires_grad = False

        if tune_llm:
            self.model.language_model.requires_grad_(True)
        elif tune_top_llm_layers > 0:
            layers = self.model.language_model.model.layers
            for layer in layers[-tune_top_llm_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True

        if tune_visual:
            self.model.vision_model.requires_grad_(True)
            self.model.mlp1.requires_grad_(True)

        print(f"[Eagle2_5Backbone] tune_llm={tune_llm}, tune_visual={tune_visual}, "
              f"tune_top_llm_layers={tune_top_llm_layers}")
        trainable = [n for n, p in self.named_parameters() if p.requires_grad]
        if trainable:
            for n in trainable:
                print(f"[Eagle2_5Backbone] trainable: {n}")
        else:
            print("[Eagle2_5Backbone] WARNING: no trainable parameters")

    def set_frozen_modules_to_eval_mode(self):
        """Keep frozen sub-modules in eval mode (dropout/BN behave correctly)."""
        if self.training:
            if not self.tune_llm:
                self.model.language_model.eval()
            if not self.tune_visual:
                self.model.vision_model.eval()
                self.model.mlp1.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(self, vl_input: BatchFeature) -> BatchFeature:
        """
        Args:
            vl_input: BatchFeature with keys:
                - input_ids:      (B, T) long
                - attention_mask: (B, T) long
                - pixel_values:   (num_tiles_total, 3, H, W) float  [tiles stacked across batch]
                - image_flags:    (num_tiles_total, 1) long — 1 for real tiles, 0 for padding

        Returns:
            BatchFeature with:
                - backbone_features:      (B, T, hidden_size)
                - backbone_attention_mask: (B, T) bool
                - image_mask:             (B, T) bool
        """
        self.set_frozen_modules_to_eval_mode()

        input_ids = vl_input["input_ids"]
        attention_mask = vl_input["attention_mask"]
        pixel_values = vl_input["pixel_values"]
        image_flags = vl_input.get("image_flags", None)

        if image_flags is None:
            # Assume all tiles are real
            image_flags = torch.ones(
                pixel_values.shape[0], 1, dtype=torch.long, device=pixel_values.device
            )

        # Run the Eagle2.5 language model trunk to get hidden states.
        # We bypass the lm_head (only need features, not logits).
        # Eagle2.5 forward embeds images then calls language_model.model.forward().
        # We replicate that here with output_hidden_states=True.

        # 1. Get input embeddings
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        # 2. Extract + project vision features
        vit_embeds = self.model.extract_feature(pixel_values)

        # 3. Filter by image_flags
        image_flags_squeezed = image_flags.squeeze(-1)
        vit_embeds = vit_embeds[image_flags_squeezed == 1]

        # 4. Scatter vision features into the sequence at image token positions.
        # Use clone() + masked_scatter to avoid in-place ops on autograd graph.
        B, N, C = input_embeds.shape
        input_ids_flat = input_ids.reshape(B * N)
        selected = (input_ids_flat == self.model.image_token_index)  # (B*N,)

        vit_flat = vit_embeds.reshape(-1, C)  # (num_img_tokens, C)
        n_selected = int(selected.sum().item())
        n_vit = vit_flat.shape[0]

        if n_selected != n_vit:
            # Mismatch — truncate to the shorter side and warn once
            n = min(n_selected, n_vit)
            print(
                f"[Eagle2_5Backbone] WARNING: image token count mismatch "
                f"({n_selected} slots vs {n_vit} vit tokens). Using first {n}."
            )
            # Rebuild selected mask to only cover the first n positions
            selected_indices = selected.nonzero(as_tuple=False).squeeze(1)[:n]
            selected = torch.zeros_like(selected)
            selected[selected_indices] = True
            vit_flat = vit_flat[:n]

        # Expand selected mask to (B*N, C) for masked_scatter
        input_embeds_flat = input_embeds.reshape(B * N, C).clone()
        mask_expanded = selected.unsqueeze(1).expand_as(input_embeds_flat)
        input_embeds_flat = input_embeds_flat.masked_scatter(mask_expanded, vit_flat.to(input_embeds_flat.dtype))
        input_embeds = input_embeds_flat.reshape(B, N, C)

        # 5. Run LM trunk
        if self.model.use_llm_lora:
            lm_forward = self.model.language_model.model.model.forward
        else:
            lm_forward = self.model.language_model.model.forward

        outputs = lm_forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )

        hidden_states = outputs.last_hidden_state  # (B, T, hidden_size)
        image_mask = input_ids == self.model.image_token_index  # (B, T)
        bool_attention_mask = attention_mask == 1                 # (B, T)

        return BatchFeature(
            data={
                "backbone_features": hidden_states,
                "backbone_attention_mask": bool_attention_mask,
                "image_mask": image_mask,
            }
        )

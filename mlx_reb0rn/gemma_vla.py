"""
GemmaVLA — reb0rn edition: uses official Gr00tN1d6Processor components for
image preprocessing, state normalization, and action denormalization.

This is a drop-in replacement for the original gemma_vla.py but wires in the
official eval pipeline from Isaac-GR00T so the preprocessing is guaranteed to
match the training distribution exactly.

Changes vs original:
  - Image preprocessing: uses eval_image_transform from Gr00tN1d6Processor
    (LetterBoxPad + SmallestMaxSize + FractionalCenterCrop, all albumentations)
    then resizes to model input size and normalizes to [-1, 1].
  - State normalization: uses StateActionProcessor.apply_state() instead of
    hand-rolled min/max code.
  - Action denormalization: uses StateActionProcessor.unapply_action() instead
    of hand-rolled mean/std + min/max code.

Everything else (vision encoder, LLM, DiT, tokenization) is unchanged.
"""

import sys
import json
import re
import numpy as np
from pathlib import Path
from PIL import Image

import mlx.core as mx

HERE = Path(__file__).parent
EAGLE_PATH = str(HERE.parent / "Eagle" / "Eagle2_5")
ISAAC_PATH = str(HERE.parent / "Isaac-GR00T")


def _build_proc_components(weights_dir: Path):
    """
    Build official preprocessing components from the saved processor_config.json
    and statistics.json that live next to the exported weights.

    Returns:
        eval_image_transform: albumentations Compose (LetterBoxPad + resize + crop)
        state_proc:           StateActionProcessor (state norm + action denorm)
        state_keys:           ordered list of state modality keys for oxe_google
        action_keys:          ordered list of action modality keys for oxe_google
    """
    for p in [EAGLE_PATH, ISAAC_PATH]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from gr00t.model.gr00t_n1d6.image_augmentations import (
        build_image_transformations_albumentations,
    )
    from gr00t.data.state_action.state_action_processor import StateActionProcessor

    # Load processor config
    with open(weights_dir / "processor_config.json") as f:
        proc_config = json.load(f)
    kwargs = proc_config["processor_kwargs"]

    # Load statistics
    with open(weights_dir / "statistics.json") as f:
        statistics = json.load(f)

    # Build eval image transform (deterministic center crop, no augmentation)
    _, eval_image_transform = build_image_transformations_albumentations(
        image_target_size=kwargs.get("image_target_size"),
        image_crop_size=kwargs.get("image_crop_size"),
        random_rotation_angle=None,
        color_jitter_params=None,
        shortest_image_edge=kwargs.get("shortest_image_edge", 512),
        crop_fraction=kwargs.get("crop_fraction", 0.95),
    )

    # Build StateActionProcessor
    modality_configs_raw = kwargs["modality_configs"]
    state_proc = StateActionProcessor(
        modality_configs=modality_configs_raw,
        statistics=statistics,
        use_percentiles=kwargs.get("use_percentiles", False),
        clip_outliers=kwargs.get("clip_outliers", True),
        apply_sincos_state_encoding=kwargs.get("apply_sincos_state_encoding", False),
        use_relative_action=kwargs.get("use_relative_action", False),
    )
    state_proc.eval()

    state_keys = modality_configs_raw["oxe_google"]["state"]["modality_keys"]
    action_keys = modality_configs_raw["oxe_google"]["action"]["modality_keys"]

    return eval_image_transform, state_proc, state_keys, action_keys


class GemmaVLA:
    """
    Full GR00T Vision-Language-Action model running natively on Apple Silicon via MLX.
    Uses official Gr00tN1d6Processor components for preprocessing (reb0rn edition).

    Components:
        vision_model  — SigLIP ViT + pixel_shuffle + MLP projector
        llm           — Gemma3 transformer trunk (bfloat16)
        dit           — AlternateVLDiT diffusion action head (bfloat16)
        tokenizer     — GemmaTokenizer with VLA special tokens
        eval_image_transform — official albumentations eval pipeline
        state_proc    — official StateActionProcessor
    """

    def __init__(self, vision_model, llm, tokenizer, dit, dit_config,
                 image_token_index: int, image_size: int, n_diffusion_steps: int = 4,
                 eval_image_transform=None, state_proc=None,
                 state_keys=None, action_keys=None):
        self.vision_model = vision_model
        self.llm = llm
        self.tokenizer = tokenizer
        self.dit = dit
        self.dit_config = dit_config
        self.image_token_index = image_token_index
        self.image_size = image_size
        self.n_diffusion_steps = n_diffusion_steps
        self._max_state_dim = (dit_config.get("max_state_dim", 128)
                               if isinstance(dit_config, dict) else dit_config.max_state_dim)

        # Official preprocessing components
        self.eval_image_transform = eval_image_transform
        self.state_proc = state_proc
        self.state_keys = state_keys or ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
        self.action_keys = action_keys or ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_exported(
        cls,
        weights_dir: str,
        mlx_llm_path: str,
        n_diffusion_steps: int = 4,
        dtype: str = "float16",
    ) -> "GemmaVLA":
        """
        Fast load from pre-exported MLX weights (output of export_weights.py).
        No PyTorch / HuggingFace download needed at runtime.

        Args:
            weights_dir:   Path to gr00t_weights_mlx/ directory
            mlx_llm_path:  Path to gr00t_llm_mlx/ directory
        """
        for p in [EAGLE_PATH, ISAAC_PATH]:
            if p not in sys.path:
                sys.path.insert(0, p)

        weights_dir = Path(weights_dir)

        # meta + model config
        with open(weights_dir / "meta.json") as f:
            meta = json.load(f)

        image_size        = meta["image_size"]
        image_token_index = meta["image_token_index"]
        dit_config        = meta

        # Build official preprocessing components
        print("[GemmaVLA] Building official preprocessing components from processor_config...")
        eval_image_transform, state_proc, state_keys, action_keys = _build_proc_components(weights_dir)
        print(f"  state_keys:  {state_keys}")
        print(f"  action_keys: {action_keys}")

        # Vision
        print(f"[GemmaVLA] Loading vision model from exported weights ({dtype})...")
        from vision_mlx import build_vision_mlx_from_exported
        vision_model = build_vision_mlx_from_exported(
            str(weights_dir / "vision.safetensors"), meta, dtype=dtype
        )

        # LLM
        print(f"[GemmaVLA] Loading Gemma3 LLM ({dtype})...")
        import mlx.core as mx
        import mlx.utils as mlx_utils
        from mlx_lm import load as mlx_load
        from transformers import GemmaTokenizer
        _llm_dtype = getattr(mx, dtype)
        llm, _ = mlx_load(mlx_llm_path)
        llm.eval()
        llm.load_weights(list(mlx_utils.tree_flatten(
            mlx_utils.tree_map(
                lambda x: x.astype(_llm_dtype) if isinstance(x, mx.array) else x,
                llm.parameters(),
            )
        )))
        print(f"  LLM forced to {dtype}")
        tokenizer = GemmaTokenizer.from_pretrained(
            str(weights_dir / "eagle_tokenizer"), use_fast=False, local_files_only=True
        )
        print(f"  Tokenizer vocab={len(tokenizer)}")

        # DiT
        print(f"[GemmaVLA] Loading DiT action head from exported weights ({dtype})...")
        from dit_mlx import build_dit_mlx_from_exported
        dit, _ = build_dit_mlx_from_exported(
            str(weights_dir / "dit.safetensors"),
            str(weights_dir / "config.json"),
            dtype=dtype,
        )
        dit.eval()
        vision_model.eval()

        obj = cls(
            vision_model=vision_model,
            llm=llm,
            tokenizer=tokenizer,
            dit=dit,
            dit_config=dit_config,
            image_token_index=image_token_index,
            image_size=image_size,
            n_diffusion_steps=n_diffusion_steps,
            eval_image_transform=eval_image_transform,
            state_proc=state_proc,
            state_keys=state_keys,
            action_keys=action_keys,
        )
        obj._dtype = dtype
        return obj

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def get_action(
        self,
        image: Image.Image,
        robot_state: np.ndarray,
        instruction: str,
        embodiment_id: int = 0,
        n_diffusion_steps: int | None = None,
    ) -> np.ndarray:
        """
        Run full VLA inference: image + state + text → action trajectory.

        Args:
            image:            PIL image (any size, will be resized internally)
            robot_state:      1-D numpy array of joint positions/velocities (8 dims)
            instruction:      Language instruction string
            embodiment_id:    Which embodiment to use (default 0)
            n_diffusion_steps: Override default diffusion steps

        Returns:
            actions: np.ndarray of shape (1, action_horizon, action_dim)
        """
        steps = n_diffusion_steps or self.n_diffusion_steps

        # --- A0. Formalize instruction (matches formalize_language=True) ---
        instruction = re.sub(r"[^\w\s]", "", instruction.lower())

        # --- A. Preprocess image using official eval pipeline ---
        # Step 1: albumentations eval transform: LetterBoxPad + SmallestMaxSize + CenterCrop
        img_np = np.array(image.convert("RGB"))    # HWC uint8
        result = self.eval_image_transform(image=img_np)
        img_cropped = result["image"]               # HWC uint8

        # Step 2: resize to model input size and normalize to [-1, 1]
        img_pil = Image.fromarray(img_cropped)
        img_pil = img_pil.resize((self.image_size, self.image_size), Image.BICUBIC)
        img_f32 = (np.array(img_pil, dtype=np.float32) / 255.0 - 0.5) / 0.5
        pixel_values = mx.array(img_f32[None])      # (1, H, W, 3)

        # --- B. Vision encoder ---
        vit_embeds = self.vision_model(pixel_values)    # (1, n_img_tokens, llm_dim)
        mx.eval(vit_embeds)
        n_img_tokens = vit_embeds.shape[1]

        # --- C. Tokenize ---
        IMG_CONTEXT = "<IMG_CONTEXT>"
        image_block = f"<img>{IMG_CONTEXT * n_img_tokens}</img>"
        prompt = (
            f"<start_of_turn>user\n"
            f"{image_block}\n"
            f"{instruction}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        input_ids = self.tokenizer.encode(prompt)
        T = len(input_ids)

        ids_np = np.array(input_ids)
        img_mask_np = (ids_np == self.image_token_index)
        img_positions = np.where(img_mask_np)[0]
        n = min(len(img_positions), n_img_tokens)

        # --- D. Embed tokens + scatter vision features ---
        ids_mx = mx.array(input_ids)[None]
        input_embeds = self.llm.model.embed_tokens(ids_mx)    # (1, T, hidden_dim)

        hidden_size = input_embeds.shape[-1]
        vit_scaled = vit_embeds[0, :n] / mx.array(hidden_size ** 0.5, dtype=vit_embeds.dtype)
        pos_idx = mx.array(img_positions[:n])
        input_embeds = input_embeds.at[0, pos_idx].add(vit_scaled - input_embeds[0, pos_idx])

        # --- E. LLM trunk ---
        hidden_states = self.llm.model(inputs=None, input_embeddings=input_embeds)
        mx.eval(hidden_states)

        # --- F. DiT action head ---
        bb_attn = mx.ones((1, T), dtype=mx.bool_)
        img_mask_mx = mx.array(img_mask_np[None])

        # Normalize state using official StateActionProcessor
        # Build per-key state dict: shape (1, 1) per key (1 timestep, 1 dim)
        n_state = min(len(self.state_keys), robot_state.shape[0])
        state_dict_raw = {
            k: np.array([[float(robot_state[i])]])
            for i, k in enumerate(self.state_keys[:n_state])
        }
        norm_dict = self.state_proc.apply_state(state_dict_raw, "oxe_google")
        norm_state = np.concatenate(
            [norm_dict[k].flatten() for k in self.state_keys[:n_state]]
        ).astype(np.float32)

        padded_state = np.zeros(self._max_state_dim, dtype=np.float32)
        padded_state[:norm_state.shape[0]] = norm_state
        state_mx = mx.array(padded_state[None, None, :])
        emb_id_mx = mx.full((1,), embodiment_id, dtype=mx.int32)

        self.dit.num_inference_timesteps = steps
        actions = self.dit.get_action(hidden_states, bb_attn, img_mask_mx, state_mx, emb_id_mx)
        mx.eval(actions)

        actions_np = np.array(actions, copy=False)    # (1, action_horizon, 128)

        # Denormalize actions using official StateActionProcessor
        # Build per-key action dict: shape (T_act, 1) per key
        T_act = actions_np.shape[1]
        n_act = len(self.action_keys)
        action_raw = actions_np[0, :, :n_act]         # (T_act, n_act)
        action_dict = {
            k: action_raw[:, i:i+1].astype(np.float64)   # (T_act, 1)
            for i, k in enumerate(self.action_keys)
        }

        denorm_dict = self.state_proc.unapply_action(action_dict, "oxe_google")

        # Reassemble (1, T_act, n_act) and slot back into full output
        denorm_action = np.concatenate(
            [denorm_dict[k] for k in self.action_keys], axis=-1
        )  # (T_act, n_act)

        out = actions_np.copy()
        out[0, :, :n_act] = denorm_action.astype(np.float32)
        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        dt = getattr(self, "_dtype", "float16")
        return (
            f"GemmaVLA-reb0rn(\n"
            f"  vision: SigLIP 27L {dt}\n"
            f"  llm:    Gemma3 18L {dt}\n"
            f"  dit:    AlternateVLDiT 32L {dt}  steps={self.n_diffusion_steps}\n"
            f"  image_size={self.image_size}  max_state_dim={self._max_state_dim}\n"
            f"  preprocessing: official Gr00tN1d6Processor (reb0rn)\n"
            f")"
        )

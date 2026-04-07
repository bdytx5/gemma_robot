"""
GemmaVLA — full Vision-Language-Action model for robot inference on Apple Silicon.

Wraps the complete GR00T pipeline as a single class:
  - SigLIP vision encoder   (MLX, float16)
  - Gemma3 LLM trunk        (MLX, bfloat16)
  - DiT action head         (MLX, float16, flow matching)

Usage:
    from gemma_vla import GemmaVLA

    vla = GemmaVLA.from_pretrained(
        gr00t_repo="youngbrett48/gr00t-post-train-fractal-270m",
        checkpoint="checkpoint-2000",
        eagle_repo="youngbrett48/train_stage2_gemma3_270m.sh",
        mlx_llm_path="./gr00t_llm_mlx",
        hf_token=True,
    )

    actions = vla.get_action(
        image=PIL.Image.open("frame.jpg"),
        robot_state=np.zeros(7),
        instruction="pick up the object",
    )
    # actions: np.ndarray (1, action_horizon, action_dim)
"""

import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

import mlx.core as mx

HERE = Path(__file__).parent
EAGLE_PATH = str(HERE.parent / "Eagle" / "Eagle2_5")
ISAAC_PATH = str(HERE.parent / "Isaac-GR00T")


class GemmaVLA:
    """
    Full GR00T Vision-Language-Action model running natively on Apple Silicon via MLX.

    Components:
        vision_model  — SigLIP ViT + pixel_shuffle + MLP projector
        llm           — Gemma3 transformer trunk (bfloat16)
        dit           — AlternateVLDiT diffusion action head (float16)
        tokenizer     — GemmaTokenizer with VLA special tokens

    All components share a unified forward pass in get_action().
    """

    def __init__(self, vision_model, llm, tokenizer, dit, dit_config,
                 image_token_index: int, image_size: int, n_diffusion_steps: int = 4):
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

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        gr00t_repo: str,
        checkpoint: str | None,
        eagle_repo: str,
        mlx_llm_path: str,
        hf_token=None,
        n_diffusion_steps: int = 4,
    ) -> "GemmaVLA":
        """
        Load all components from HuggingFace repos and return a ready GemmaVLA.

        Args:
            gr00t_repo:      HF repo containing the GR00T checkpoint (vision + DiT weights)
            checkpoint:      Subfolder within the repo (e.g. "checkpoint-2000"), or None
            eagle_repo:      HF repo containing Eagle2.5 config + tokenizer
            mlx_llm_path:    Local path to the MLX-converted Gemma3 LLM
            hf_token:        HuggingFace token for private repos
            n_diffusion_steps: Number of DiT denoising steps (default 4)
        """
        # Ensure Eagle2.5 and Isaac-GR00T are importable
        for p in [EAGLE_PATH, ISAAC_PATH]:
            if p not in sys.path:
                sys.path.insert(0, p)

        from huggingface_hub import snapshot_download, hf_hub_download
        from safetensors.torch import load_file
        import glob

        # 1. Download GR00T checkpoint weights
        print("[GemmaVLA] Loading GR00T checkpoint...")
        if checkpoint:
            allow = [
                f"{checkpoint}/config.json",
                f"{checkpoint}/model.safetensors",
                f"{checkpoint}/model.safetensors.index.json",
                f"{checkpoint}/model-*-of-*.safetensors",
            ]
            ckpt_dir = snapshot_download(gr00t_repo, allow_patterns=allow, token=hf_token)
            ckpt_path = Path(ckpt_dir) / checkpoint
        else:
            ckpt_dir = snapshot_download(gr00t_repo, token=hf_token)
            ckpt_path = Path(ckpt_dir)

        shards = sorted(glob.glob(str(ckpt_path / "*.safetensors")))
        state_dict = {}
        for s in shards:
            state_dict.update(load_file(s))
        print(f"  {len(state_dict)} keys loaded")

        cfg_name = f"{checkpoint}/config.json" if checkpoint else "config.json"
        cfg_path = hf_hub_download(gr00t_repo, cfg_name, token=hf_token)
        with open(cfg_path) as f:
            dit_config = json.load(f)

        # 2. Eagle2.5 config (for image_token_index, image_size)
        from eaglevl.model.eagle2_5.configuration_eagle2_5_vl import Eagle2_5_VLConfig
        from transformers import AutoConfig
        AutoConfig.register("eagle_2_5_vl", Eagle2_5_VLConfig)
        eagle_config = Eagle2_5_VLConfig.from_pretrained(
            eagle_repo, trust_remote_code=True, token=hf_token, local_files_only=True,
        )
        image_size = getattr(eagle_config, "force_image_size", None) or eagle_config.vision_config.image_size
        image_token_index = eagle_config.image_token_index

        # 3. Vision encoder (MLX, float16)
        print("[GemmaVLA] Building vision encoder...")
        from vision_mlx import build_vision_mlx
        vision_model = build_vision_mlx(state_dict, eagle_config)

        # 4. Gemma3 LLM (MLX, bfloat16)
        print("[GemmaVLA] Loading Gemma3 LLM...")
        from mlx_lm import load as mlx_load
        from transformers import GemmaTokenizer
        llm, _ = mlx_load(mlx_llm_path)
        llm.eval()
        tok_kwargs = {"use_fast": False, "local_files_only": True}
        if hf_token:
            tok_kwargs["token"] = hf_token
        tokenizer = GemmaTokenizer.from_pretrained(eagle_repo, **tok_kwargs)
        print(f"  Tokenizer vocab={len(tokenizer)}")

        # 5. DiT action head (MLX, float16)
        print("[GemmaVLA] Building DiT action head...")
        from dit_mlx import build_dit_mlx
        dit, _ = build_dit_mlx(state_dict, cfg_path)

        return cls(
            vision_model=vision_model,
            llm=llm,
            tokenizer=tokenizer,
            dit=dit,
            dit_config=dit_config,
            image_token_index=image_token_index,
            image_size=image_size,
            n_diffusion_steps=n_diffusion_steps,
        )

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
            robot_state:      1-D numpy array of joint positions/velocities
            instruction:      Language instruction string
            embodiment_id:    Which embodiment to use (default 0)
            n_diffusion_steps: Override default diffusion steps

        Returns:
            actions: np.ndarray of shape (1, action_horizon, action_dim)
        """
        steps = n_diffusion_steps or self.n_diffusion_steps

        # --- A. Preprocess image → MLX NHWC float32 [-1, 1] ---
        img = image.convert("RGB").resize((self.image_size, self.image_size))
        img_np = (np.array(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        pixel_values = mx.array(img_np[None])   # (1, H, W, 3)

        # --- B. Vision encoder ---
        vit_embeds = self.vision_model(pixel_values)   # (1, n_img_tokens, llm_dim)
        mx.eval(vit_embeds)
        n_img_tokens = vit_embeds.shape[1]

        # --- C. Tokenize ---
        IMG_CONTEXT = "<IMG_CONTEXT>"
        image_block = f"<img>{IMG_CONTEXT * n_img_tokens}</img>"
        prompt = f"{image_block}\n{instruction}"
        input_ids = self.tokenizer.encode(prompt)
        T = len(input_ids)

        ids_np = np.array(input_ids)
        img_mask_np = (ids_np == self.image_token_index)
        img_positions = np.where(img_mask_np)[0]
        n = min(len(img_positions), n_img_tokens)

        # --- D. Embed tokens + scatter vision features ---
        ids_mx = mx.array(input_ids)[None]
        input_embeds = self.llm.model.embed_tokens(ids_mx)   # (1, T, hidden_dim)

        # MLX Gemma3 applies *sqrt(hidden_size) internally to all input_embeddings.
        # Pre-divide vision features so they land at the correct scale after that multiplication.
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

        padded_state = np.zeros(self._max_state_dim, dtype=np.float32)
        padded_state[:robot_state.shape[0]] = robot_state
        state_mx = mx.array(padded_state[None, None, :])
        emb_id_mx = mx.full((1,), embodiment_id, dtype=mx.int32)

        self.dit.num_inference_timesteps = steps
        actions = self.dit.get_action(hidden_states, bb_attn, img_mask_mx, state_mx, emb_id_mx)
        mx.eval(actions)

        return np.array(actions, copy=False)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GemmaVLA(\n"
            f"  vision: SigLIP 27L float16\n"
            f"  llm:    Gemma3 18L bfloat16\n"
            f"  dit:    AlternateVLDiT 32L float16  steps={self.n_diffusion_steps}\n"
            f"  image_size={self.image_size}  max_state_dim={self._max_state_dim}\n"
            f")"
        )

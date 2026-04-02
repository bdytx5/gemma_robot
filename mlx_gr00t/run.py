"""
One-shot runner: extract weights → convert to MLX → run inference.
Skips extraction/conversion if already done.

Usage:
    # From a saved image file:
    python run.py --image /path/to/frame.jpg

    # Capture a single frame from webcam:
    python run.py --webcam

    # Provide robot state + instruction:
    python run.py --image frame.jpg --state "0.1,0.2,0.3,0.4,0.5,0.6,0.7" \
                  --instruction "pick up the can"

    # Force re-extract and re-convert even if cached:
    python run.py --image frame.jpg --force
"""

import argparse
import subprocess
import sys
import os
import numpy as np
from pathlib import Path

HERE = Path(__file__).parent

HF_LLM_DIR = HERE / "gr00t_llm_hf"
MLX_LLM_DIR = HERE / "gr00t_llm_mlx"


# ---------------------------------------------------------------------------
# Step 1: Extract LLM weights from GR00T HF checkpoint
# ---------------------------------------------------------------------------
def maybe_extract(gr00t_ckpt: str, checkpoint: str, force: bool):
    if HF_LLM_DIR.exists() and (HF_LLM_DIR / "model.safetensors").exists() and not force:
        print(f"[extract] Already done → {HF_LLM_DIR}  (use --force to redo)")
        return
    print("[extract] Extracting Gemma3-1b weights from GR00T checkpoint...")
    subprocess.run(
        [sys.executable, str(HERE / "extract_llm.py"),
         "--gr00t_repo", gr00t_ckpt,
         "--checkpoint", checkpoint,
         "--out_dir", str(HF_LLM_DIR)],
        check=True,
    )


# ---------------------------------------------------------------------------
# Step 2: Convert extracted HF weights → MLX (4-bit quantized)
# ---------------------------------------------------------------------------
def maybe_convert(force: bool):
    if MLX_LLM_DIR.exists() and any(MLX_LLM_DIR.iterdir()) and not force:
        print(f"[convert] Already done → {MLX_LLM_DIR}  (use --force to redo)")
        return
    print("[convert] Converting to MLX (4-bit quantized)...")
    subprocess.run(
        [sys.executable, "-m", "mlx_lm.convert",
         "--hf-path", str(HF_LLM_DIR),
         "--mlx-path", str(MLX_LLM_DIR),
         "-q"],
        check=True,
    )


# ---------------------------------------------------------------------------
# Step 3: Get an image — from file, URL, or webcam
# ---------------------------------------------------------------------------
def get_image(image_path: str | None, url: str | None, webcam: bool):
    from PIL import Image

    if url:
        import io, urllib.request
        print(f"[image] Downloading from URL...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        print(f"  Downloaded image: {img.size}")
        return img

    if image_path:
        print(f"[image] Loading from file: {image_path}")
        return Image.open(image_path).convert("RGB")

    if webcam:
        print("[image] Capturing from webcam...")
        try:
            import cv2
        except ImportError:
            print("  opencv-python not installed — run: pip install opencv-python")
            sys.exit(1)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (device 0)")
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture frame from webcam")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    raise ValueError("Provide --image <path> or --webcam")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()

    # Image source (mutually exclusive)
    img_group = p.add_mutually_exclusive_group(required=True)
    img_group.add_argument("--image", help="Path to image file (jpg/png)")
    img_group.add_argument("--url", help="Download image from a URL")
    img_group.add_argument("--webcam", action="store_true", help="Capture one frame from webcam")

    # Robot inputs
    p.add_argument("--state", default=None,
                   help="Comma-separated robot joint state values (leave blank for zeros)")
    p.add_argument("--state_dim", type=int, default=7,
                   help="State dimension (used if --state not provided, default 7)")
    p.add_argument("--instruction", default="pick up the object",
                   help="Language instruction for the robot")

    # Model config
    p.add_argument("--gr00t_ckpt", default="youngbrett48/train_google_robot_eagle2_5")
    p.add_argument("--checkpoint", default="checkpoint-200",
                   help="Which checkpoint subfolder to use from the HF repo")
    p.add_argument("--eagle2_5_repo", default="youngbrett48/train_stage2_gemma3_1b.sh")

    # Misc
    p.add_argument("--force", action="store_true",
                   help="Re-extract and re-convert even if cached artifacts exist")
    p.add_argument("--n_diffusion_steps", type=int, default=4)

    args = p.parse_args()

    # ---- Step 1: extract ----
    maybe_extract(args.gr00t_ckpt, args.checkpoint, args.force)

    # ---- Step 2: convert ----
    maybe_convert(args.force)

    # ---- Step 3: load models (deferred imports so MLX/torch only load once) ----
    from inference import (
        load_gr00t_components,
        build_vision_components,
        load_mlx_llm,
        build_dit,
        run_inference,
    )
    from huggingface_hub import hf_hub_download

    print("\n[load] Loading model components...")
    state_dict = load_gr00t_components(args.gr00t_ckpt, args.checkpoint)
    vision_encoder, mlp_projector, eagle_config = build_vision_components(
        state_dict, args.eagle2_5_repo
    )
    mlx_llm, tokenizer = load_mlx_llm(str(MLX_LLM_DIR))
    cfg_path = hf_hub_download(args.gr00t_ckpt, f"{args.checkpoint}/config.json")
    dit_model = build_dit(state_dict, cfg_path)

    # ---- Step 4: get image ----
    image = get_image(args.image, args.url, args.webcam)

    # ---- Step 5: parse robot state ----
    if args.state:
        robot_state = np.array([float(x) for x in args.state.split(",")], dtype=np.float32)
    else:
        robot_state = np.zeros(args.state_dim, dtype=np.float32)
        print(f"[state] No --state provided, using zeros (dim={args.state_dim})")

    # ---- Step 6: run inference ----
    print(f"\n[infer] Instruction: \"{args.instruction}\"")
    print(f"[infer] State: {robot_state}")
    actions = run_inference(
        image=image,
        robot_state=robot_state,
        language_instruction=args.instruction,
        vision_encoder=vision_encoder,
        mlp_projector=mlp_projector,
        mlx_llm=mlx_llm,
        tokenizer=tokenizer,
        dit_model=dit_model,
        eagle_config=eagle_config,
        n_diffusion_steps=args.n_diffusion_steps,
    )

    print(f"\n✓ Predicted actions  shape={actions.shape}")
    print(actions)


if __name__ == "__main__":
    main()

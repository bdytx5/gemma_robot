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
def maybe_extract(gr00t_ckpt: str, checkpoint: str | None, force: bool, hf_token: str | None = None):
    if HF_LLM_DIR.exists() and (HF_LLM_DIR / "model.safetensors").exists() and not force:
        print(f"[extract] Already done → {HF_LLM_DIR}  (use --force to redo)")
        return
    print("[extract] Extracting Gemma3-270m weights from GR00T checkpoint...")
    cmd = [sys.executable, str(HERE / "extract_llm.py"),
           "--gr00t_repo", gr00t_ckpt,
           "--out_dir", str(HF_LLM_DIR)]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if hf_token:
        cmd += ["--hf_token", hf_token]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Step 2: Convert extracted HF weights → MLX (4-bit quantized)
# ---------------------------------------------------------------------------
def maybe_convert(force: bool):
    if MLX_LLM_DIR.exists() and any(MLX_LLM_DIR.iterdir()) and not force:
        print(f"[convert] Already done → {MLX_LLM_DIR}  (use --force to redo)")
        return
    print("[convert] Converting to MLX (bfloat16)...")
    subprocess.run(
        [sys.executable, "-m", "mlx_lm", "convert",
         "--hf-path", str(HF_LLM_DIR),
         "--mlx-path", str(MLX_LLM_DIR),
         "--dtype", "bfloat16"],
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

    # Image source (mutually exclusive, defaults to test image URL)
    DEFAULT_TEST_IMAGE = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQThMxfZLtrpL_JSq6ttJo64D7nVKCGFHvKJg&s"
    img_group = p.add_mutually_exclusive_group(required=False)
    img_group.add_argument("--image", help="Path to image file (jpg/png)")
    img_group.add_argument("--url", default=None,
                           help="Download image from a URL")
    img_group.add_argument("--webcam", action="store_true", help="Capture one frame from webcam")

    # Robot inputs
    p.add_argument("--state", default=None,
                   help="Comma-separated robot joint state values (leave blank for zeros)")
    p.add_argument("--state_dim", type=int, default=7,
                   help="State dimension (used if --state not provided, default 7)")
    p.add_argument("--instruction", default="pick up the object",
                   help="Language instruction for the robot")

    # Model config
    p.add_argument("--gr00t_ckpt", default="youngbrett48/gr00t-post-train-fractal-270m")
    p.add_argument(
        "--checkpoint",
        default="checkpoint-2000",
        help="Checkpoint subfolder to use from the HF repo. Leave unset for repo-root models.",
    )
    p.add_argument("--eagle2_5_repo", default="youngbrett48/train_stage2_gemma3_270m.sh")

    # Misc
    p.add_argument("--hf_token", default=None, help="HuggingFace token for private repos")
    p.add_argument("--force", action="store_true",
                   help="Re-extract and re-convert even if cached artifacts exist")
    p.add_argument("--n_diffusion_steps", type=int, default=4)

    args = p.parse_args()
    if not args.image and not args.url and not args.webcam:
        args.url = DEFAULT_TEST_IMAGE

    # ---- Step 1: extract ----
    maybe_extract(args.gr00t_ckpt, args.checkpoint, args.force, args.hf_token)

    # ---- Step 2: convert ----
    maybe_convert(args.force)

    # ---- Step 3: load model ----
    from gemma_vla import GemmaVLA

    print("\n[load] Loading GemmaVLA...")
    tok = args.hf_token or True
    vla = GemmaVLA.from_pretrained(
        gr00t_repo=args.gr00t_ckpt,
        checkpoint=args.checkpoint,
        eagle_repo=args.eagle2_5_repo,
        mlx_llm_path=str(MLX_LLM_DIR),
        hf_token=tok,
        n_diffusion_steps=args.n_diffusion_steps,
    )
    print(vla)

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
    actions = vla.get_action(
        image=image,
        robot_state=robot_state,
        instruction=args.instruction,
    )

    print(f"\n✓ Predicted actions  shape={actions.shape}")
    print(actions)


if __name__ == "__main__":
    main()

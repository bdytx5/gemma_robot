#!/usr/bin/env bash
# Run this on your Mac to set up the MLX GR00T inference environment.
set -e

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

echo ""
echo "Setup complete. Workflow:"
echo ""
echo "Step 1 — Extract LLM weights from GR00T HF checkpoint:"
echo "  python extract_llm.py --gr00t_repo youngbrett48/gr00t-post-train-fractal-270m --out_dir ./gr00t_llm_hf"
echo ""
echo "Step 2 — Convert to MLX (with 4-bit quantization):"
echo "  python -m mlx_lm.convert --hf-path ./gr00t_llm_hf --mlx-path ./gr00t_llm_mlx -q"
echo ""
echo "Step 3 — Run inference:"
echo "  python inference.py --mlx_llm_path ./gr00t_llm_mlx --image /path/to/image.jpg"

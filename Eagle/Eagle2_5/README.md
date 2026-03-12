# 🦅 Eagle 2.5

Eagle 2.5 is a multimodal large model (image/video × text). This repository provides the end-to-end guidance and scripts for the environment setup, data preparation, training, and inference of the Eagle VLM.

---

## 📚 Quick Start (Onboarding)

Recommended order:

1) Set environment variables → 2) Install → 3) Prepare data → 4) Train → 5) Demo → 6) Inference

- Onboarding overview: see `./document/0.onboarding.md`

---

## ⚙️ Installation & Environment

- Detailed steps and dependencies: `./document/1.installing.md`
  - Conda environment (Python 3.10)
  - PyTorch and FlashAttention (match your CUDA)
  - Install this repo with `pip install -e .`
  - Troubleshooting notes (specific Transformers version, OpenCV dependencies, etc.)

---

## 📂 Data Preparation (Playground)

- Directory structure and JSONL/LMDB examples: `./document/2.preparing_playground.md`
  - `playground/sft_recipe` (data recipe)
  - `playground/sft_jsonl` and `playground/sft_data` (annotations and raw data)
  - Example parquet→LMDB conversion scripts are not included in this repo
  - Use `shell/prepare.sh` to normalize and generate `.prepare.json` (internal `submit_prepare_job.sh` is not included)
  - LMDB reading example and tips: `./document/how_to_use_lmdb_to_read_images.md`

---

## 💪 Training (Stage-2 / Finetuning)

- Full training entry points and multinode/multigpu options: `./document/3.training.md`
  - Single-node example: `GPUS=8 bash shell/train_stage2.sh 1 work_dirs/eagle2.5_debug`
  - Multi-node example (srun/internal submit_job): `PARTITION=xxx GPUS=16 bash shell/train_stage2.sh 2 work_dirs/eagle2.5_multinode`
  
---

## ✨ Launching Streamlit Demo

- Interactive testing of the VLM with UI. Refer to document for more details: `./document/4.streamlit_demo.md`

---

## 🔮 Inference

- End-to-end usage and multimodal examples (single/multiple images, single/multiple videos, streaming, batch): `./document/5.inference.md`
  - Load with `transformers` `AutoModel`/`AutoProcessor`: `"nvidia/Eagle-2.5-8B"`
  - Recommended `torch_dtype=torch.bfloat16`; run `model.generate(...)` on GPU

---

## License

- See `LICENSE` and `LICENSE_MODEL` at the repository root.

---

For detailed parameter explanations and launcher script notes, see: `./document/explain_script_arguments.md`.

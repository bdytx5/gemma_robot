# GemmaRobot

Run GR00T robot eval natively on Apple Silicon via MLX. Connects to a remote GPU sim server, runs inference locally, live video feed + action display.

## Install

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/bdytx5/gemma_robot/main/mlx_gr00t/install.sh)
```

No Xcode, no Python, no conda required — the script handles everything.  
**Requirements:** macOS 13+, Apple Silicon (M1/M2/M3/M4), internet connection.

## Usage

1. Start the sim server on your GPU machine:
   ```bash
   python Isaac-GR00T/remote_eval/sim_server.py
   ngrok http 8000
   ```
2. Open **GemmaRobot.app**
3. Paste the ngrok URL, pick a task, hit **Run Eval**

Episode videos are logged to [Weave](https://wandb.ai) if `WANDB_API_KEY` is set.

## Repo layout

```
mlx_gr00t/       macOS app + MLX inference pipeline
  app.py         Tkinter GUI
  gemma_vla.py   Full VLA model (vision + LLM + DiT)
  vision_mlx.py  SigLIP vision encoder in MLX
  dit_mlx.py     DiT action head in MLX
  inference.py   Inference utilities
  install.sh     One-line installer
  export_linux.py  Export weights from Linux GPU server
Isaac-GR00T/     GR00T sim environment + server
Eagle/           Eagle2.5 vision-language backbone
```

---

## Training

This project builds on NVIDIA's [Eagle2.5](https://github.com/NVlabs/Eagle) vision-language model repo and [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) VLA architecture, adapting Gemma3-270M as the language backbone in place of NVIDIA's internal Cosmos-Reason model. The DiT action head and training infrastructure come directly from GR00T; the VLM pretraining pipeline comes from Eagle2.5.

Training runs in four sequential stages. Each stage builds on frozen or partially-unfrozen weights from the previous one.

### Stage 1 — Connector Pretraining on LLaVA-CC3M-595K

Both SigLIP2 and Gemma3-270M are **frozen**. Only the 2-layer MLP connector between them is trained, on the full [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) dataset — 595k image-caption pairs from Conceptual Captions. The sole goal is to project vision tokens into the LLM's embedding space before any instruction tuning. LR 2e-5, ZeRO stage 1. Output → `youngbrett48/train_stage1_gemma3.sh`.

### Stage 2 — VLM Instruction Tuning on Eagle-1.8M (subset)

Starting from the Stage 1 checkpoint, the LLM is fully unfrozen while SigLIP stays frozen. The model is fine-tuned on roughly the first 30% of [Eagle-1.8M](https://huggingface.co/datasets/NVEagle/Eagle-1.8M) (~540k examples) — multimodal QA covering visual reasoning, OCR, charts, and spatial understanding. This is where the model learns to follow language instructions grounded in images. LR 1e-5, ZeRO stage 2, dynamic tiling up to 12 tiles at 384×384. Output → `youngbrett48/train_stage2_gemma3_270m.sh`.

### Stage 3 — Balanced Multi-Dataset Robot Training

The Stage 2 VLM checkpoint feeds into the GR00T training pipeline. The pretrained DiT action head and action projector are loaded directly from `nvidia/GR00T-N1.6-fractal`, with a learned projection layer bridging the embedding dimension gap between Gemma3's hidden size (1152) and the GR00T action head (2048). The **full VLA is unfrozen** (LLM + projector + DiT; only SigLIP stays frozen) and trained on 10 datasets (3 real, 7 sim), all sampled at equal weight (mix_ratio 1.0):

**Real robot**

| Dataset | Robot | Episodes | Frames | FPS | Resolution |
|---|---|---|---|---|---|
| fractal20220817 | Google Robot | 87,212 | 3.8M | 3 | 256×320 |
| bridge\_orig | WidowX | 53,192 | 1.9M | 5 | 256×256 (4 cams) |
| droid\_subset | Franka | 2,000 | 595K | 15 | 180×320 (wrist + 2 ext) |

**Sim**

| Dataset | Robot | Episodes | Frames |
|---|---|---|---|
| robocasa\_mg\_gr00t\_300 | Panda + Omron | 7,200 | 2.1M |
| nvidia-panda CloseDrawer | Panda | 3,000 | 619K |
| nvidia-panda OpenDrawer | Panda | 3,000 | 674K |
| nvidia-panda PnPCabToCounter | Panda | 3,011 | 1.0M |
| nvidia-panda PnPCounterToCab | Panda | 3,000 | 778K |
| nvidia-panda PnPCounterToMicrowave | Panda | 3,000 | 1.3M |
| nvidia-panda PnPMicrowaveToCounter | Panda | 2,998 | 944K |

20k steps, lr 1e-4. Output: `output/balanced-from-270m/`.

### Stage 4 — Post-Training on Fractal

The Stage 3 checkpoint is fine-tuned back on Fractal alone (lr 5e-5, 5k steps). Only the top 4 LLM transformer layers are unfrozen — enough to sharpen task-specific grounding without forgetting multi-embodiment knowledge from Stage 3. An eval watcher runs in parallel on a second GPU, scoring SimplerEnv checkpoints every 1k steps. Output → `youngbrett48/gr00t-post-train-fractal-270m`.

---

## Evaluation

Checkpoints are evaluated in [SimplerEnv](https://github.com/simpler-env/SimplerEnv), a GPU-accelerated physics sim that replicates the Google Robot tabletop setup. 50 episodes per task, randomized object positions and seeds. An episode is a success only if the robot completes the task within the step budget (504 steps). The eval server runs as a FastAPI process exposed via ngrok; the Mac client (this repo) drives the loop using MLX inference locally.

50 episodes per task, 300 total. Comparing NVIDIA's stock `GR00T-N1.6-fractal` checkpoint against this project's Gemma3-270M trained model:

| Task | GR00T-N1.6 (stock) | Ours (Gemma3) | Δ |
|---|---|---|---|
| close\_drawer | 60.0% | 84.0% | +24.0% |
| move\_near | 46.0% | 32.0% | −14.0% |
| open\_drawer | 8.0% | 46.0% | +38.0% |
| pick\_coke\_can | 48.0% | 66.0% | +18.0% |
| pick\_object | 28.0% | 52.0% | +24.0% |
| place\_in\_closed\_drawer | 0.0% | 2.0% | +2.0% |
| **Average** | **31.7%** | **47.0%** | **+15.3%** |

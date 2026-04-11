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

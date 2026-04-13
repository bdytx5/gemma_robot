# GemmaRobot — MLX Inference App (Apple Silicon)

Run GR00T robot eval locally on a Mac against a remote GPU sim server.  
Live video feed, per-dim action bars, Weave video logging.

---

## Install

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/bdytx5/gemma_robot/main/mlx_gr00t/install.sh)
```

No Xcode, no conda, no dev tools required — the script installs everything.  
**Requirements:** macOS 13+, Apple Silicon (M1/M2/M3/M4)

The script will:
1. Download the repo (via `git` if available, otherwise `curl`)
2. Install Miniforge3 if no conda is found
3. Create a `mlx_robot` conda env with all runtime deps
4. Build `GemmaRobot.app` + `GemmaRobot.dmg` in `~/gemma_robot/dist/`

> **Note on model weights:** The app needs pre-exported MLX weights in  
> `gr00t_weights_mlx/` and `gr00t_llm_mlx/`. No HuggingFace account needed  
> at runtime — weights load entirely from local files. If you don't have  
> them yet, see [Exporting weights](#exporting-weights) below.

---

## Usage

1. Start a sim server on your GPU instance (`Isaac-GR00T/remote_eval/sim_server.py`)
2. Expose it via ngrok: `ngrok http 8000`
3. Open **GemmaRobot.app**, paste the ngrok URL, pick a task, hit **Run Eval**

Episode videos are logged to [Weave](https://wandb.ai) automatically if `WANDB_API_KEY` is set.

---

## Exporting weights

Weights must be exported once from the PyTorch checkpoint. This requires the full dev setup (PyTorch, Eagle2.5, Isaac-GR00T repos). Once exported, copy them to any machine that just runs the app:

```bash
# on your dev machine
cd ~/gemma_robot/mlx_gr00t
python export_weights.py

# copy to another mac
scp -r gr00t_weights_mlx/ gr00t_llm_mlx/ other_mac:~/gemma_robot/mlx_gr00t/
```

Then re-run the install script on the target machine.

---

## Manual build (dev)

```bash
git clone https://github.com/bdytx5/gemma_robot.git
cd gemma_robot/mlx_gr00t
bash build_app.sh
open ../dist/GemmaRobot.app
```

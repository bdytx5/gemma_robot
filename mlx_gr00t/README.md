# GemmaRobot — MLX Inference App (Apple Silicon)

Run GR00T robot eval locally on a Mac against a remote GPU sim server.  
Live video feed, per-dim action bars, Weave video logging.

---

## One-line install (no cloning required)

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/bdytx5/gemma_robot/main/mlx_gr00t/install.sh)
```

This will:
1. Clone the repo into `~/gemma_robot`
2. Export MLX weights from the HF checkpoint
3. Build `GemmaRobot.app` + `GemmaRobot.dmg` in `~/gemma_robot/dist/`
4. Open the `.dmg` so you can drag the app to `/Applications`

**Requirements:** macOS 13+, Apple Silicon (M1/M2/M3), [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

---

## Manual install

```bash
git clone https://github.com/bdytx5/gemma_robot.git
cd gemma_robot/mlx_gr00t
bash build_app.sh
open ../dist/GemmaRobot.app
```

---

## Usage

1. Start a sim server on your GPU instance (see `Isaac-GR00T/remote_eval/sim_server.py`)
2. Expose it via ngrok: `ngrok http 8000`
3. Open **GemmaRobot.app**, paste the ngrok URL, pick a task, hit **Run Eval**

Videos from each episode are logged to [Weave](https://wandb.ai) automatically if `WANDB_API_KEY` is set in your environment.

---

## First-run / env setup

If the app can't find a Python env with MLX, click **Setup Env** in the toolbar.  
It will create a `mlx_robot` conda env and install everything from `requirements.txt`.

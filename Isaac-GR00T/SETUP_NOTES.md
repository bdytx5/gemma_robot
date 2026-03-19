# Setup Notes — Isaac-GR00T + Eagle2.5 + Gemma3

## What you need installed before running the train script

### System packages
```bash
sudo apt install ffmpeg   # for AV1->H264 video conversion
```

### Python environment
The project uses `uv` (not pip) to manage deps. Install uv if not present:
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Then from `Isaac-GR00T/`:
```bash
uv sync --extra gpu
```

This installs everything from `pyproject.toml` including:
- `torch==2.7.1`
- `torchvision==0.22.1`
- `transformers==4.51.3`
- `diffusers==0.35.1`
- `tyro==0.9.17`
- `omegaconf==2.3.0`
- `deepspeed==0.17.6`
- `flash-attn==2.7.4.post1`
- `wandb==0.23.0`
- `albumentations==1.4.18`
- `datasets==3.6.0`
- `huggingface_hub` (via transformers)
- plus everything else in pyproject.toml

The train script runs `uv sync --extra gpu` automatically if `.venv` doesn't exist yet.

### HuggingFace login
```bash
huggingface-cli login
```
Required to download the Fractal dataset and your checkpoint.

### Eagle2.5 source
Must be cloned at `../Eagle/Eagle2_5` relative to `Isaac-GR00T/`:
```
gemma_robot/
  Eagle/
    Eagle2_5/       <-- needs to be here
  Isaac-GR00T/
```

## Issues hit and how they were fixed

| Error | Fix |
|-------|-----|
| `torchrun` using wrong Python | Use `python -m torch.distributed.run` from `.venv` |
| `uv sync` not installing editable package | Use `uv sync` not `pip install -e .` (pyproject.toml backend doesn't support editable installs via pip) |
| `eagle_2_5_vl` not in transformers registry | Bypass `AutoModel`/`AutoProcessor` for `eagle2_5` backbone type, instantiate directly |
| Corrupted `tokenizer.model` in HF checkpoint | Load `GemmaTokenizerFast` from `google/gemma-3-270m-it` directly |
| `meta/info.json` missing (partial download) | Check `meta/info.json` not just `data/` dir to detect complete dataset |
| HF 429 rate limit during dataset download | Use `snapshot_download` with retry loop, wait 300s on 429 |
| `--tune_llm True` rejected by tyro | Tyro boolean flags take no value: `--tune_llm` not `--tune_llm True` |

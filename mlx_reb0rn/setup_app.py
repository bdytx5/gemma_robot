"""
py2app setup for GemmaRobot.app
Run: python setup_app.py py2app
"""
from setuptools import setup

APP = ["app.py"]

DATA_FILES = [
    # pre-exported MLX weights (vision + DiT + tokenizer + stats)
    ("gr00t_weights_mlx", [
        "gr00t_weights_mlx/vision.safetensors",
        "gr00t_weights_mlx/dit.safetensors",
        "gr00t_weights_mlx/config.json",
        "gr00t_weights_mlx/meta.json",
        "gr00t_weights_mlx/statistics.json",
    ]),
    ("gr00t_weights_mlx/eagle_tokenizer", [
        f for f in __import__("glob").glob("gr00t_weights_mlx/eagle_tokenizer/*")
    ]),
    # MLX LLM weights
    ("gr00t_llm_mlx", __import__("glob").glob("gr00t_llm_mlx/*")),
    # repo source modules needed at runtime
    ("", [
        "gemma_vla.py",
        "vision_mlx.py",
        "dit_mlx.py",
        "inference.py",
    ]),
]

OPTIONS = {
    "argv_emulation": False,
    "plist": {
        "CFBundleName":          "GemmaRobot",
        "CFBundleDisplayName":   "GemmaRobot",
        "CFBundleIdentifier":    "com.gemmarobot.eval",
        "CFBundleVersion":       "1.0",
        "LSMinimumSystemVersion":"13.0",
        "NSHighResolutionCapable": True,
    },
    "packages": [
        "mlx", "mlx_lm",
        "transformers", "tokenizers",
        "PIL", "numpy", "requests", "msgpack",
        "safetensors", "huggingface_hub",
        "tkinter",
    ],
    "includes": [
        "gemma_vla", "vision_mlx", "dit_mlx", "inference",
    ],
    "excludes": [
        "torch", "torchvision",   # not needed at runtime with from_exported
        "matplotlib", "scipy", "pandas",
        "IPython", "jupyter",
    ],
    "semi_standalone": False,
    "site_packages": True,
}

setup(
    app=APP,
    name="GemmaRobot",
    data_files=DATA_FILES,
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)

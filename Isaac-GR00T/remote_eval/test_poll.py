"""
test_poll.py — run on local machine

Connects to sim_server via ngrok, resets the env, and displays the first frame.

Usage:
    python remote_eval/test_poll.py --server_url https://abc123.ngrok-free.app
"""

import argparse
import io

import msgpack
import numpy as np
import requests
from PIL import Image


def _encode(obj):
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray__": True, "data": buf.getvalue()}
    raise TypeError(f"Cannot encode {type(obj)}")

def _decode(obj):
    if isinstance(obj, dict) and obj.get("__ndarray__"):
        return np.load(io.BytesIO(obj["data"]), allow_pickle=False)
    return obj

def pack(data) -> bytes:
    return msgpack.packb(data, default=_encode, use_bin_type=True)

def unpack(data: bytes):
    return msgpack.unpackb(data, object_hook=_decode, raw=False)

HEADERS = {"Content-Type": "application/msgpack"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_url", required=True, help="ngrok URL, e.g. https://abc123.ngrok-free.app")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="first_frame.png", help="Where to save the frame")
    args = parser.parse_args()

    base = args.server_url.rstrip("/")

    print(f"Pinging {base}/info ...")
    info = requests.get(f"{base}/info", timeout=10).json()
    print(f"Server: {info}")

    print("Resetting env ...")
    r = requests.post(f"{base}/reset", data=pack({"seed": args.seed}),
                      headers=HEADERS, timeout=120)
    r.raise_for_status()
    result = unpack(r.content)

    obs = result["obs"]
    print(f"done={result['done']}  success={result['success']}")
    print(f"obs keys: {list(obs.keys())}")

    # find the image key
    img_array = None
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim >= 3:
            # could be (T, H, W, C) from MultiStepWrapper — take last frame
            arr = v
            while arr.ndim > 3:
                arr = arr[-1]
            img_array = arr
            print(f"Using obs['{k}'] shape={v.shape} → frame shape={arr.shape}")
            break

    if img_array is None:
        print("No image found in obs. Keys:", list(obs.keys()))
    else:
        img = Image.fromarray(img_array.astype(np.uint8))
        img.save(args.save)
        print(f"Saved first frame to {args.save}")
        img.show()

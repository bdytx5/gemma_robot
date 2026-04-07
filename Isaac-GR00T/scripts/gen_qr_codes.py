#!/usr/bin/env python3
"""
Generate QR code PNGs for each demo task.
Usage: python scripts/gen_qr_codes.py --base-url https://byyoung3.github.io/gr00t-demo
Output: docs/qr/<task>.png
"""
import argparse
import os

TASKS = [
    "open_drawer",
    "close_drawer",
    "place_in_closed_drawer",
    "pick_coke_can",
    "pick_object",
    "move_near",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Base URL of the GitHub Pages frontend")
    parser.add_argument("--out-dir", default="docs/qr")
    args = parser.parse_args()

    try:
        import qrcode
    except ImportError:
        print("Install qrcode: pip install qrcode[pil]")
        raise SystemExit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    for task in TASKS:
        url = f"{args.base_url}/?task={task}"
        img = qrcode.make(url)
        out = os.path.join(args.out_dir, f"{task}.png")
        img.save(out)
        print(f"  {task} → {out}  ({url})")

    print(f"\nGenerated {len(TASKS)} QR codes in {args.out_dir}/")

if __name__ == "__main__":
    main()

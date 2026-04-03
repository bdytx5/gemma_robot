#!/usr/bin/env python3
"""Prepare ssh23/robocasa_mg_gr00t_300_refined_wiener_filter for GR00T N1 finetuning.

This dataset is already in LeRobot v2 format with modality.json, stats.json, etc.
We just need to:
1. Register a modality config for robocasa_panda_omron in embodiment_configs.py
   (done separately)
2. Verify the dataset structure is valid
3. Generate relative stats if missing
"""

import json
import sys
from pathlib import Path


def verify_dataset(dataset_path: Path):
    """Verify the dataset has all required files for GR00T."""
    required_files = [
        "meta/info.json",
        "meta/modality.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
    ]

    print(f"Verifying dataset at {dataset_path}...")
    all_ok = True
    for f in required_files:
        fp = dataset_path / f
        if fp.exists():
            print(f"  OK: {f}")
        else:
            print(f"  MISSING: {f}")
            all_ok = False

    # Check data dir
    data_chunks = list((dataset_path / "data").glob("chunk-*"))
    print(f"  Data chunks: {len(data_chunks)}")

    # Check video dir
    video_dir = dataset_path / "videos"
    if video_dir.exists():
        video_chunks = list(video_dir.glob("chunk-*"))
        print(f"  Video chunks: {len(video_chunks)}")
        # Check a sample video exists
        sample_vids = list(video_dir.glob("chunk-000/*/*.mp4"))
        print(f"  Sample videos in chunk-000: {len(sample_vids)}")
        if len(sample_vids) == 0:
            print("  WARNING: No videos found! Check VIDEO_NOTE.md")
            all_ok = False
    else:
        print("  MISSING: videos/ directory")
        all_ok = False

    # Read and display modality.json
    modality_path = dataset_path / "meta" / "modality.json"
    if modality_path.exists():
        with open(modality_path) as f:
            modality = json.load(f)
        print(f"\n  State keys: {list(modality.get('state', {}).keys())}")
        print(f"  Action keys: {list(modality.get('action', {}).keys())}")
        print(f"  Video keys: {list(modality.get('video', {}).keys())}")
        print(f"  Annotation keys: {list(modality.get('annotation', {}).keys())}")

    # Read info.json
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        print(f"\n  Episodes: {info.get('total_episodes', '?')}")
        print(f"  Frames: {info.get('total_frames', '?')}")
        print(f"  Tasks: {info.get('total_tasks', '?')}")
        print(f"  FPS: {info.get('fps', '?')}")
        print(f"  Codebase version: {info.get('codebase_version', '?')}")

    return all_ok


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify RoboCasa dataset for GR00T")
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist"

    ok = verify_dataset(dataset_path)
    if ok:
        print("\n Dataset is ready for GR00T finetuning!")
    else:
        print("\n Dataset has issues - see above")
        sys.exit(1)


if __name__ == "__main__":
    main()

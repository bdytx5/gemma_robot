#!/usr/bin/env python3
"""Prepare a subset of cadene/droid_1.0.1 for GR00T N1 finetuning.

This script:
1. Writes meta/modality.json for the oxe_droid embodiment
2. Builds meta/tasks.jsonl from language_instruction strings in parquet files
3. Rewrites parquet files to add annotation columns (task_description mapped to task index)
4. Updates meta/info.json with correct fields
5. Generates stats.json
6. Optionally converts AV1 videos to h264
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def write_modality_json(dataset_path: Path):
    """Write the modality.json that maps parquet columns to GR00T field names."""
    modality = {
        "state": {
            "joint_position": {"start": 0, "end": 7},
            "gripper_position": {"start": 7, "end": 8},
        },
        "action": {
            "joint_position": {"start": 0, "end": 7},
            "gripper_position": {"start": 7, "end": 8},
        },
        "video": {
            "exterior_image_1_left": {
                "original_key": "observation.images.exterior_1_left"
            },
            "wrist_image_left": {
                "original_key": "observation.images.wrist_left"
            },
        },
        "annotation": {
            "language.language_instruction": {
                "original_key": "task_index"
            },
        },
    }
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality, f, indent=4)
    print(f"[OK] Wrote {meta_dir / 'modality.json'}")


def build_tasks_and_rewrite_parquets(dataset_path: Path):
    """Extract unique language instructions, build tasks.jsonl, and add annotation columns."""
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        print(f"ERROR: No parquet files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(parquet_files)} parquet files")

    # First pass: collect all unique language instructions
    print("Pass 1: Collecting unique language instructions...")
    all_instructions = set()
    for pf in parquet_files:
        df = pd.read_parquet(pf, columns=["language_instruction"])
        for inst in df["language_instruction"].dropna().unique():
            if inst.strip():  # skip empty strings
                all_instructions.add(inst.strip())

    # Add a default task for episodes with no instruction
    default_task = "manipulate object"
    all_instructions.add(default_task)

    # Build task mapping
    task_list = sorted(all_instructions)
    task_to_idx = {task: idx for idx, task in enumerate(task_list)}
    default_idx = task_to_idx[default_task]
    print(f"Found {len(task_list)} unique tasks (including default)")

    # Write tasks.jsonl
    meta_dir = dataset_path / "meta"
    meta_dir.mkdir(exist_ok=True)
    tasks_path = meta_dir / "tasks.jsonl"
    with open(tasks_path, "w") as f:
        for idx, task in enumerate(task_list):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")
    print(f"[OK] Wrote {tasks_path}")

    # Second pass: rewrite parquets with annotation columns
    print("Pass 2: Rewriting parquet files with annotation columns...")
    episodes_info = []
    all_states = []
    all_actions = []

    for i, pf in enumerate(parquet_files):
        df = pd.read_parquet(pf)

        # Map language_instruction to task index
        def map_task(inst):
            if isinstance(inst, str) and inst.strip():
                return task_to_idx.get(inst.strip(), default_idx)
            return default_idx

        df["annotation.language.language_instruction"] = df["language_instruction"].apply(map_task)
        df["task_index"] = df["annotation.language.language_instruction"]

        # Add validity column
        df["annotation.human.validity"] = 1

        # Ensure next.reward and next.done exist
        if "next.reward" not in df.columns:
            df["next.reward"] = 0.0
        if "next.done" not in df.columns:
            # Mark last frame of episode as done
            df["next.done"] = False
            df.loc[df.index[-1], "next.done"] = True

        # Collect stats data
        states = np.stack(df["observation.state"].values)
        actions = np.stack(df["action"].values)
        all_states.append(states)
        all_actions.append(actions)

        # Collect episode info
        ep_idx = int(df["episode_index"].iloc[0])
        task_indices = df["task_index"].unique().tolist()
        episodes_info.append({
            "episode_index": ep_idx,
            "tasks": [task_list[t] for t in task_indices],
            "length": len(df),
        })

        df.to_parquet(pf, index=False)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(parquet_files)} episodes")

    print(f"  Processed {len(parquet_files)}/{len(parquet_files)} episodes")

    # Write episodes.jsonl
    episodes_path = meta_dir / "episodes.jsonl"
    with open(episodes_path, "w") as f:
        for ep in sorted(episodes_info, key=lambda x: x["episode_index"]):
            f.write(json.dumps(ep) + "\n")
    print(f"[OK] Wrote {episodes_path}")

    # Generate stats.json
    print("Generating stats.json...")
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    total_frames = len(all_states)

    stats = {
        "observation.state": {
            "mean": all_states.mean(axis=0).tolist(),
            "std": all_states.std(axis=0).tolist(),
            "min": all_states.min(axis=0).tolist(),
            "max": all_states.max(axis=0).tolist(),
        },
        "action": {
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
            "min": all_actions.min(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist(),
        },
    }
    stats_path = meta_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Wrote {stats_path}")

    return len(task_list), len(episodes_info), total_frames


def update_info_json(dataset_path: Path, n_tasks: int, n_episodes: int, total_frames: int):
    """Update info.json to be compatible with GR00T LeRobot loader."""
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
    else:
        # Minimal info.json if it doesn't exist
        info = {
            "codebase_version": "v2.1",
            "fps": 15,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        }

    # Update subset-specific fields without clobbering existing ones
    info["total_tasks"] = n_tasks
    info["total_episodes"] = n_episodes
    info["total_frames"] = total_frames
    # Fix splits for our subset
    info["splits"] = {"train": f"0:{n_episodes}"}
    # Fix total_chunks for subset
    data_chunks = list((dataset_path / "data").glob("chunk-*"))
    info["total_chunks"] = len(data_chunks)

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"[OK] Updated {info_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare DROID subset for GR00T")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to the downloaded droid subset")
    parser.add_argument("--convert_videos", action="store_true",
                        help="Convert AV1 videos to h264")
    parser.add_argument("--jobs", type=int, default=8,
                        help="Number of parallel jobs for video conversion")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist"

    # Step 1: Write modality.json
    write_modality_json(dataset_path)

    # Step 2: Build tasks.jsonl, rewrite parquets, generate stats
    n_tasks, n_episodes, total_frames = build_tasks_and_rewrite_parquets(dataset_path)

    # Step 3: Update info.json
    update_info_json(dataset_path, n_tasks, n_episodes, total_frames)

    # Step 4: Convert videos if requested
    if args.convert_videos:
        print("\nConverting AV1 videos to h264...")
        os.system(
            f"python examples/SimplerEnv/convert_av1_to_h264.py"
            f" --root {dataset_path} --jobs {args.jobs}"
        )

    print(f"\n[DONE] Dataset ready for GR00T finetuning with embodiment_tag=oxe_droid")
    print(f"  Tasks: {n_tasks}")
    print(f"  Episodes: {n_episodes}")


if __name__ == "__main__":
    main()

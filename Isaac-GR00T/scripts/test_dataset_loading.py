#!/usr/bin/env python3
"""Test that a dataset can be loaded through GR00T's data pipeline."""

import argparse
import json
import sys
from pathlib import Path


def test_load(dataset_path: str, embodiment_tag: str):
    """Try to load a dataset through the GR00T LeRobot loader."""
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    tag = EmbodimentTag(embodiment_tag)
    print(f"Embodiment tag: {tag}")
    print(f"Dataset path: {dataset_path}")

    if embodiment_tag not in MODALITY_CONFIGS:
        print(f"WARNING: {embodiment_tag} not in MODALITY_CONFIGS, it may need to be registered")
        print(f"Available configs: {list(MODALITY_CONFIGS.keys())}")
        return False

    modality_configs = MODALITY_CONFIGS[embodiment_tag]
    print(f"Modality config keys: {list(modality_configs.keys())}")

    try:
        loader = LeRobotEpisodeLoader(dataset_path, modality_configs)
        print(f"Loader created successfully")
        n_episodes = len(loader.episodes_metadata)
        print(f"Number of episodes: {n_episodes}")

        # Try loading first episode
        episode = loader[0]
        print(f"Loaded episode 0 successfully")
        print(f"Episode keys: {list(episode.keys())}")
        for k, v in episode.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif isinstance(v, list) and len(v) > 0:
                print(f"  {k}: len={len(v)}, type={type(v[0])}")
            else:
                print(f"  {k}: {type(v)}")

        print("\n Dataset loads correctly!")
        return True
    except Exception as e:
        print(f"\n FAILED to load: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--embodiment_tag", required=True)
    args = parser.parse_args()
    success = test_load(args.dataset_path, args.embodiment_tag)
    sys.exit(0 if success else 1)

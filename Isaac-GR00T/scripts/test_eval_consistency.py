#!/usr/bin/env python3
"""
Verify SimplerEnv produces identical initial observations for the same seed.
Run with the SimplerEnv venv:
    gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python scripts/test_eval_consistency.py
"""
import os
os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

import numpy as np
import simpler_env

ENVS = [
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_place_in_closed_drawer",
    "google_robot_pick_coke_can",
    "google_robot_pick_object",
    "google_robot_move_near",
]

SEED = 42
passed, failed = 0, 0

for env_name in ENVS:
    # Run A
    env_a = simpler_env.make(env_name)
    np.random.seed(SEED)
    obs_a, _ = env_a.reset(seed=SEED)
    env_a.close()

    # Run B — fresh instance, same seed
    env_b = simpler_env.make(env_name)
    np.random.seed(SEED)
    obs_b, _ = env_b.reset(seed=SEED)
    env_b.close()

    # Compare
    match = True
    for k in obs_a:
        if isinstance(obs_a[k], np.ndarray):
            if not np.array_equal(obs_a[k], obs_b[k]):
                print(f"FAIL {env_name}: key={k} max_diff={np.abs(obs_a[k].astype(float)-obs_b[k].astype(float)).max()}")
                match = False

    if match:
        print(f"PASS {env_name}")
        passed += 1
    else:
        failed += 1

print(f"\n{passed} passed, {failed} failed")

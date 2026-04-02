"""
Quick test: verify that SimplerEnv task variants are deterministic across resets.
Runs 2 rounds of 5 resets on open_drawer and prints the language instruction each time.
Both rounds should produce identical sequences.

Usage:
    python scripts/test_eval_determinism.py
"""
import sys
from pathlib import Path

# Ensure SimplerEnv is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs, GoogleFractalEnv
import gymnasium as gym

register_simpler_envs()

N_EPISODES = 5
BASE_SEED = 42

print("=== Testing deterministic eval on google_robot_open_drawer ===\n")

for round_num in range(2):
    print(f"--- Round {round_num + 1} ---")
    env = GoogleFractalEnv(env_name="google_robot_open_drawer", image_size=(256, 320))

    instructions = []
    env.reset(seed=BASE_SEED)  # set base seed on first reset
    instr = env.env.unwrapped.get_language_instruction()
    instructions.append(instr)
    print(f"  Episode 0: seed={BASE_SEED}, task=\"{instr}\"")

    for ep in range(1, N_EPISODES):
        env.reset()  # no seed — should use deterministic sequence
        instr = env.env.unwrapped.get_language_instruction()
        instructions.append(instr)
        print(f"  Episode {ep}: task=\"{instr}\"")

    env.env.close()
    if round_num == 0:
        round1 = instructions
    else:
        round2 = instructions
        match = round1 == round2
        print(f"\n{'✓ PASS' if match else '✗ FAIL'}: Round 1 == Round 2: {match}")
        if not match:
            for i, (a, b) in enumerate(zip(round1, round2)):
                if a != b:
                    print(f"  Mismatch at episode {i}: \"{a}\" vs \"{b}\"")
    print()

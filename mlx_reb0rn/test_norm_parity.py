"""
Direct numerical parity test: training normalization vs gemma_vla.py normalization.
Runs both on identical data and prints diffs. PASS = all zeros.
"""
import sys, json, numpy as np
from pathlib import Path

HERE = Path(__file__).parent
ISAAC = str(HERE.parent / "Isaac-GR00T")
if ISAAC not in sys.path:
    sys.path.insert(0, ISAAC)

# ---- load stats ----
with open(HERE / "gr00t_weights_mlx/statistics.json") as f:
    stats = json.load(f)["oxe_google"]

s = stats["state"]
a = stats["action"]

state_keys  = ["x", "y", "z", "rx", "ry", "rz", "rw", "gripper"]
action_ms   = ["x", "y", "z", "roll", "pitch", "yaw"]
action_grip = "gripper"

state_min = np.array([s[k]["min"][0] for k in state_keys], dtype=np.float64)
state_max = np.array([s[k]["max"][0] for k in state_keys], dtype=np.float64)

action_mean = np.array([a[k]["mean"][0] for k in action_ms], dtype=np.float64)
action_std  = np.array([a[k]["std"][0]  for k in action_ms], dtype=np.float64)
grip_min    = float(a[action_grip]["min"][0])
grip_max    = float(a[action_grip]["max"][0])

print("State min:", state_min)
print("State max:", state_max)
print()

# ---- training normalization (from gr00t/data/utils.py) ----

def training_normalize_minmax(values, mn, mx):
    """Exact copy of normalize_values_minmax + clip_outliers=True."""
    rng = mx - mn
    mask = rng != 0
    out = np.zeros_like(values, dtype=np.float64)
    out[mask] = (values[mask] - mn[mask]) / rng[mask]
    out[mask] = 2 * out[mask] - 1
    out[~mask] = values[~mask]
    return np.clip(out, -1.0, 1.0)   # clip_outliers=True

def training_normalize_meanstd(values, mean, std):
    """Exact copy of normalize_values_meanstd."""
    mask = std != 0
    out = np.zeros_like(values, dtype=np.float64)
    out[mask] = (values[mask] - mean[mask]) / std[mask]
    out[~mask] = values[~mask]
    return out   # no clip for mean/std in training

def training_unnorm_minmax(norm, mn, mx):
    """unnormalize_values_minmax — clips to [-1,1] before unnorm."""
    return (np.clip(norm, -1.0, 1.0) + 1.0) / 2.0 * (mx - mn) + mn

def training_unnorm_meanstd(norm, mean, std):
    """unnormalize_values_meanstd — x*std+mean."""
    mask = std != 0
    out = np.zeros_like(norm, dtype=np.float64)
    out[mask] = norm[mask] * std[mask] + mean[mask]
    out[~mask] = norm[~mask]
    return out

# ---- gemma_vla.py normalization (inlined) ----

def gemma_normalize_state(robot_state, s_min, s_max):
    out = robot_state.copy().astype(np.float64)
    d = len(s_min)
    rng = np.maximum(s_max - s_min, 1e-8)
    out[:d] = np.clip((robot_state[:d] - s_min) / rng * 2.0 - 1.0, -1.0, 1.0)
    return out

def gemma_denorm_action(model_out):
    """model_out shape: (16, 7) — first 6 are mean/std, last 1 is gripper min/max."""
    out = model_out.copy().astype(np.float64)
    out[:, :6] = out[:, :6] * action_std + action_mean
    grip = np.clip(out[:, 6], -1.0, 1.0)
    out[:, 6] = (grip + 1.0) / 2.0 * (grip_max - grip_min) + grip_min
    return out

# ============================================================
# TEST 1: State normalization
# ============================================================
print("=" * 60)
print("TEST 1: State normalization")
print("=" * 60)

# Test with values inside range, at boundary, and outside range
test_states = [
    state_min + (state_max - state_min) * 0.5,   # midpoint → should be 0.0
    state_min,                                      # min → should be -1.0
    state_max,                                      # max → should be +1.0
    state_min - 0.1 * (state_max - state_min),     # below min → clips to -1.0
    state_max + 0.1 * (state_max - state_min),     # above max → clips to +1.0
    np.random.RandomState(42).uniform(state_min * 0.9, state_max * 1.1, size=8),
]
labels = ["midpoint", "min", "max", "below_min", "above_max", "random"]

all_pass = True
for raw, label in zip(test_states, labels):
    train = training_normalize_minmax(raw, state_min, state_max)
    gemma = gemma_normalize_state(raw, state_min, state_max)
    diff  = np.abs(train - gemma).max()
    ok    = diff < 1e-10
    all_pass &= ok
    print(f"  {label:12s}  max_diff={diff:.2e}  {'PASS' if ok else 'FAIL'}")

print(f"  State norm: {'ALL PASS' if all_pass else 'FAILED'}\n")

# ============================================================
# TEST 2: Action denormalization
# ============================================================
print("=" * 60)
print("TEST 2: Action denormalization")
print("=" * 60)

rng = np.random.RandomState(7)
model_output = rng.randn(16, 7).astype(np.float64)   # DiT output (normalized space)

# Training path: use unnorm functions on each column
train_act = model_output.copy()
for t in range(16):
    train_act[t, :6] = training_unnorm_meanstd(train_act[t, :6], action_mean, action_std)
    train_act[t, 6]  = training_unnorm_minmax(
        np.array([train_act[t, 6]]), np.array([grip_min]), np.array([grip_max])
    )[0]

# gemma_vla path
gemma_act = gemma_denorm_action(model_output)

diff_ms   = np.abs(train_act[:, :6] - gemma_act[:, :6]).max()
diff_grip = np.abs(train_act[:, 6]  - gemma_act[:, 6]).max()
ok_ms     = diff_ms   < 1e-10
ok_grip   = diff_grip < 1e-10

print(f"  x/y/z/roll/pitch/yaw  max_diff={diff_ms:.2e}  {'PASS' if ok_ms else 'FAIL'}")
print(f"  gripper               max_diff={diff_grip:.2e}  {'PASS' if ok_grip else 'FAIL'}")

all_act_pass = ok_ms and ok_grip
print(f"  Action denorm: {'ALL PASS' if all_act_pass else 'FAILED'}\n")

# ============================================================
print("=" * 60)
if all_pass and all_act_pass:
    print("RESULT: BOTH PATHS IDENTICAL — normalization matches training exactly.")
else:
    print("RESULT: MISMATCH FOUND — fix gemma_vla.py")
print("=" * 60)

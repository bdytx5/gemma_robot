import argparse
import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
import time
from typing import Any
import uuid

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.sim.env_utils import get_embodiment_tag_from_env_name
from gr00t.eval.sim.wrapper.multistep_wrapper import MultiStepWrapper
from gr00t.policy import BasePolicy
import gymnasium as gym
import numpy as np
from tqdm import tqdm


class DeterministicResetWrapper(gym.Wrapper):
    """Intercepts reset() to inject a deterministic per-episode seed."""
    def __init__(self, env, base_seed):
        super().__init__(env)
        self._base_seed = base_seed
        self._episode_count = 0

    def reset(self, *, seed=None, options=None):
        if self._base_seed is not None:
            ep_seed = self._base_seed + self._episode_count * 1000
            self._episode_count += 1
            return self.env.reset(seed=ep_seed, options=options)
        self._episode_count += 1
        return self.env.reset(seed=seed, options=options)


def _make_deterministic_env(orig_fn, base_seed):
    """Picklable factory for DeterministicResetWrapper."""
    def wrapped():
        return DeterministicResetWrapper(orig_fn(), base_seed)
    return wrapped


@dataclass
class VideoConfig:
    """Configuration for video recording settings.

    Attributes:
        video_dir: Directory to save videos (if None, no videos are saved)
        steps_per_render: Number of steps between each call to env.render() while recording
            during rollout
        fps: Frames per second for the output video
        codec: Video codec to use for compression
        input_pix_fmt: Input pixel format
        crf: Constant Rate Factor for video compression (lower = better quality)
        thread_type: Threading strategy for video encoding
        thread_count: Number of threads to use for encoding
    """

    video_dir: str | None = None
    steps_per_render: int = 2
    max_episode_steps: int = 720
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1
    overlay_text: bool = True
    n_action_steps: int = 8


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings.

    Attributes:
        video_delta_indices: Indices of video observations to stack
        state_delta_indices: Indices of state observations to stack
        n_action_steps: Number of action steps to execute
        max_episode_steps: Maximum number of steps per episode
    """

    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 720
    terminate_on_success: bool = False


@dataclass
class WrapperConfigs:
    """Container for various environment wrapper configurations.

    Attributes:
        video: Configuration for video recording
        multistep: Configuration for multi-step processing
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


def get_robocasa_env_fn(
    env_name: str,
):
    def env_fn():
        import os

        import robocasa  # noqa: F401
        from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401
        import robosuite  # noqa: F401

        os.environ["MUJOCO_GL"] = "egl"
        return gym.make(env_name, enable_render=True)

    return env_fn


def get_groot_locomanip_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t_wbc.control.envs.robocasa.sync_env import SyncEnv  # noqa: F401
        from gr00t_wbc.control.main.teleop.configs.configs import BaseConfig
        from gr00t_wbc.control.utils.n1_utils import WholeBodyControlWrapper
        import robocasa  # noqa: F401

        gym_env = gym.make(
            env_name,
            onscreen=False,
            offscreen=True,
            enable_waist=True,
            randomize_cameras=False,
            camera_names=[
                "robot0_oak_egoview",
                "robot0_rs_tppview",
            ],
        )
        wbc_config = BaseConfig(wbc_version="gear_wbc", enable_waist=True).to_dict()
        gym_env = WholeBodyControlWrapper(gym_env, wbc_config)
        return gym_env

    return env_fn


def get_simpler_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

        register_simpler_envs()
        return gym.make(env_name)

    return env_fn


def get_libero_env_fn(
    env_name: str,
):
    def env_fn():
        from gr00t.eval.sim.LIBERO.libero_env import register_libero_envs

        register_libero_envs()
        return gym.make(env_name)

    return env_fn


def get_behavior_env_fn(
    env_name: str,
    env_idx: int,
    total_n_envs: int,
):
    def env_fn():
        from gr00t.eval.sim.BEHAVIOR.behavior_env import register_behavior_envs

        register_behavior_envs()
        return gym.make(env_name, env_idx=env_idx, total_n_envs=total_n_envs)

    return env_fn


def get_gym_env(env_name: str, env_idx: int, total_n_envs: int):
    """Create Ray environment factory function without wrappers."""

    env_embodiment = get_embodiment_tag_from_env_name(env_name)

    if env_embodiment in (
        EmbodimentTag.GR1,
        EmbodimentTag.ROBOCASA_PANDA_OMRON,
    ):
        env_fn = get_robocasa_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.UNITREE_G1,):
        env_fn = get_groot_locomanip_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.OXE_GOOGLE, EmbodimentTag.OXE_WIDOWX):
        env_fn = get_simpler_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.LIBERO_PANDA,):
        env_fn = get_libero_env_fn(env_name)

    elif env_embodiment in (EmbodimentTag.BEHAVIOR_R1_PRO,):
        env_fn = get_behavior_env_fn(env_name, env_idx, total_n_envs)
    else:
        raise ValueError(f"Invalid environment name: {env_name}")

    return env_fn()


def create_eval_env(
    env_name: str, env_idx: int, total_n_envs: int, wrapper_configs: WrapperConfigs
) -> gym.Env:
    """Create a single evaluation environment with wrappers.

    Args:
        env_name: Name of the gymnasium environment to use
        idx: Environment index (used to determine video recording)
        wrapper_configs: Configuration for environment wrappers
    Returns:
        Wrapped gymnasium environment
    """

    env = get_gym_env(env_name, env_idx, total_n_envs)
    if wrapper_configs.video.video_dir is not None:
        from gr00t.eval.sim.wrapper.video_recording_wrapper import (
            VideoRecorder,
            VideoRecordingWrapper,
        )

        video_recorder = VideoRecorder.create_h264(
            fps=wrapper_configs.video.fps,
            codec=wrapper_configs.video.codec,
            input_pix_fmt=wrapper_configs.video.input_pix_fmt,
            crf=wrapper_configs.video.crf,
            thread_type=wrapper_configs.video.thread_type,
            thread_count=wrapper_configs.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(wrapper_configs.video.video_dir),
            steps_per_render=wrapper_configs.video.steps_per_render,
            max_episode_steps=wrapper_configs.video.max_episode_steps,
            overlay_text=wrapper_configs.video.overlay_text,
            env_idx=env_idx,
        )

    env = MultiStepWrapper(
        env,
        video_delta_indices=wrapper_configs.multistep.video_delta_indices,
        state_delta_indices=wrapper_configs.multistep.state_delta_indices,
        n_action_steps=wrapper_configs.multistep.n_action_steps,
        max_episode_steps=wrapper_configs.multistep.max_episode_steps,
        terminate_on_success=wrapper_configs.multistep.terminate_on_success,
    )
    return env


def run_rollout_gymnasium_policy(
    env_name: str,
    policy: BasePolicy,
    wrapper_configs: WrapperConfigs,
    n_episodes: int = 10,
    n_envs: int = 1,
    seed: int | None = None,
    progress_file: str | None = None,
) -> Any:
    """Run policy rollouts in parallel environments.

    Args:
        env_name: Name of the gymnasium environment to use
        policy: Policy instance
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        wrapper_configs: Configuration for environment wrappers
        ray_env: Whether to use ray gym env to create each env.
    Returns:
        Collection results from running the episodes
    """
    start_time = time.time()
    n_episodes = max(n_episodes, n_envs)
    print(f"Running collecting {n_episodes} episodes for {env_name} with {n_envs} vec envs")

    # Seed BEFORE env construction so simpler_env.make() picks the same
    # object variants, textures, and camera angles across eval runs.
    if seed is not None:
        import random as _random
        np.random.seed(seed)
        _random.seed(seed)

    env_fns = [
        partial(
            create_eval_env,
            env_idx=idx,
            env_name=env_name,
            total_n_envs=n_envs,
            wrapper_configs=wrapper_configs,
        )
        for idx in range(n_envs)
    ]

    # Wrap each env_fn so that every reset() call uses a deterministic seed.
    # This ensures episodes are reproducible regardless of n_envs or autoreset behavior.
    # Uses module-level classes so they're picklable for AsyncVectorEnv (spawn context).
    # Each env gets a unique seed offset so parallel envs don't all see identical episodes.
    # env_idx=0 → seed, env_idx=1 → seed+100000, etc. The *1000 stride within each env
    # (see DeterministicResetWrapper) stays well within the 100k gap.
    wrapped_env_fns = [
        _make_deterministic_env(fn, seed + idx * 100000 if seed is not None else None)
        for idx, fn in enumerate(env_fns)
    ]

    # FORCE_SYNC=1 env var or n_envs==1 → SyncVectorEnv (safer for tasks with Vulkan spawn issues)
    force_sync = os.environ.get("FORCE_SYNC", "0") == "1"
    if n_envs == 1 or force_sync:
        env = gym.vector.SyncVectorEnv(wrapped_env_fns)
    else:
        env = gym.vector.AsyncVectorEnv(
            wrapped_env_fns,
            shared_memory=False,
            context="spawn",
        )

    # Storage for results
    episode_lengths = []
    current_rewards = [0] * n_envs
    current_lengths = [0] * n_envs
    completed_episodes = 0
    current_successes = [False] * n_envs
    episode_successes = []
    episode_infos = defaultdict(list)
    # Initial reset — DeterministicResetWrapper handles seeding automatically
    observations, _ = env.reset()
    policy.reset()
    i = 0

    MAX_CONSECUTIVE_FAILURES = 3  # give up on this env after this many crashes in a row
    consecutive_failures = 0
    max_steps = wrapper_configs.multistep.max_episode_steps

    def _write_progress():
        if not progress_file:
            return
        import json as _json
        status = {
            "env_name": env_name,
            "step": i,
            "completed_episodes": completed_episodes,
            "n_episodes": n_episodes,
            "n_envs": n_envs,
            "max_steps_per_episode": max_steps,
            "successes_so_far": sum(episode_successes) if episode_successes else 0,
            "elapsed": round(time.time() - start_time, 1),
            "current_ep_steps": [current_lengths[e] for e in range(n_envs)],
        }
        try:
            with open(progress_file + ".tmp", "w") as f:
                _json.dump(status, f)
            os.replace(progress_file + ".tmp", progress_file)
        except Exception:
            pass

    pbar = tqdm(total=n_episodes, desc="Episodes")
    while completed_episodes < n_episodes:
        try:
            actions, _ = policy.get_action(observations)
            next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        except Exception as e:
            # Env or policy crashed mid-episode — count current episodes as failures
            consecutive_failures += 1
            print(f"[rollout] env.step() crashed ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}")
            for env_idx in range(n_envs):
                if current_lengths[env_idx] > 0:
                    # Episode was in progress — record as failure
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(False)
                    completed_episodes += 1
                    pbar.update(1)
                    current_successes[env_idx] = False
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[rollout] {MAX_CONSECUTIVE_FAILURES} consecutive failures — returning partial results")
                break
            # Try to reset the env and keep going
            try:
                observations, _ = env.reset()
                policy.reset()
                print("[rollout] env reset OK — continuing")
                continue
            except Exception as reset_err:
                print(f"[rollout] env.reset() also failed — returning partial results: {reset_err}")
                break

        consecutive_failures = 0  # reset on successful step
        # NOTE (FY): Currently we don't properly handle policy reset. For now, our policy are stateless,
        # but in the future if we need policy to be stateful, we need to detect env reset and call policy.reset()
        i += 1
        _write_progress()
        # Update episode tracking
        for env_idx in range(n_envs):
            if "success" in env_infos:
                env_success = env_infos["success"][env_idx]
                if isinstance(env_success, list):
                    env_success = np.any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            else:
                current_successes[env_idx] = False

            if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                env_success = env_infos["final_info"][env_idx]["success"]
                if isinstance(env_success, list):
                    env_success = any(env_success)
                elif isinstance(env_success, np.ndarray):
                    env_success = np.any(env_success)
                elif isinstance(env_success, bool):
                    env_success = env_success
                elif isinstance(env_success, int):
                    env_success = bool(env_success)
                else:
                    raise ValueError(f"Unknown success dtype: {type(env_success)}")
                current_successes[env_idx] |= bool(env_success)
            current_rewards[env_idx] += rewards[env_idx]
            current_lengths[env_idx] += 1

            # If episode ended, store results
            if terminations[env_idx] or truncations[env_idx]:
                if "final_info" in env_infos and env_infos["final_info"][env_idx] is not None:
                    fi_success = env_infos["final_info"][env_idx]["success"]
                    fi_any = bool(np.any(fi_success))
                    if fi_any:
                        print(f"[rollout] Episode {completed_episodes}: SUCCESS detected via final_info (success={fi_success})")
                    current_successes[env_idx] |= fi_any
                if "task_progress" in env_infos:
                    episode_infos["task_progress"].append(env_infos["task_progress"][env_idx][-1])
                if "q_score" in env_infos:
                    episode_infos["q_score"].append(np.max(env_infos["q_score"][env_idx]))
                if "valid" in env_infos:
                    episode_infos["valid"].append(all(env_infos["valid"][env_idx]))
                # Accumulate results — cap at n_episodes to avoid overshoot
                if completed_episodes < n_episodes:
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(current_successes[env_idx])
                    # only update completed_episodes if valid
                    if "valid" in episode_infos:
                        if episode_infos["valid"][-1]:
                            completed_episodes += 1
                            pbar.update(1)
                    else:
                        completed_episodes += 1
                        pbar.update(1)
                # Reset trackers for this environment regardless
                current_successes[env_idx] = False
                current_rewards[env_idx] = 0
                current_lengths[env_idx] = 0

        # Autoreset handles the reset — DeterministicResetWrapper ensures
        # each sub-env gets a deterministic seed on every reset() call.
        observations = next_obs
    pbar.close()

    try:
        env.reset()
        env.close()
    except Exception:
        pass  # env may already be dead from a crash
    print(f"Collecting {completed_episodes}/{n_episodes} episodes took {time.time() - start_time} seconds")

    if len(episode_successes) < n_episodes:
        print(f"[rollout] WARNING: only {len(episode_successes)}/{n_episodes} episodes completed")

    episode_infos = dict(episode_infos)  # Convert defaultdict to dict
    for key, value in episode_infos.items():
        if len(value) != len(episode_successes):
            # Trim to match — crash recovery can cause mismatches
            min_len = min(len(value), len(episode_successes))
            episode_infos[key] = value[:min_len]
            episode_successes = episode_successes[:min_len]

    # process valid results
    if "valid" in episode_infos:
        valids = episode_infos["valid"]
        valid_idxs = np.where(valids)[0]
        episode_successes = [episode_successes[i] for i in valid_idxs]
        episode_infos = {k: [v[i] for i in valid_idxs] for k, v in episode_infos.items()}

    return env_name, episode_successes, episode_infos


def create_gr00t_sim_policy(
    model_path: str,
    embodiment_tag: EmbodimentTag,
    policy_client_host: str = "",
    policy_client_port: int | None = None,
) -> BasePolicy:
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper

    if policy_client_host and policy_client_port:
        from gr00t.policy.server_client import PolicyClient

        policy = PolicyClient(host=policy_client_host, port=policy_client_port)
    else:
        policy = Gr00tSimPolicyWrapper(
            Gr00tPolicy(
                embodiment_tag=embodiment_tag,
                model_path=model_path,
                device=0,
            )
        )
    return policy


def run_gr00t_sim_policy(
    env_name: str,
    n_episodes: int,
    max_episode_steps: int,
    model_path: str = "",
    policy_client_host: str = "",
    policy_client_port: int | None = None,
    n_envs: int = 8,
    n_action_steps: int = 8,
    video_dir: str | None = None,
    seed: int | None = None,
):
    embodiment_tag = get_embodiment_tag_from_env_name(env_name)

    if video_dir is None:
        if model_path:
            video_dir = (
                f"/tmp/sim_eval_videos_{model_path.split('/')[-3]}_ac{n_action_steps}_{uuid.uuid4()}"
            )
        else:
            video_dir = f"/tmp/sim_eval_videos_{env_name}_ac{n_action_steps}_{uuid.uuid4()}"
    wrapper_configs = WrapperConfigs(
        video=VideoConfig(
            video_dir=video_dir,
            max_episode_steps=max_episode_steps,
        ),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            terminate_on_success=True,
        ),
    )

    policy = create_gr00t_sim_policy(
        model_path, embodiment_tag, policy_client_host, policy_client_port
    )

    progress_file = os.path.join(video_dir, "progress.json") if video_dir else None

    results = run_rollout_gymnasium_policy(
        env_name=env_name,
        policy=policy,
        wrapper_configs=wrapper_configs,
        n_episodes=n_episodes,
        n_envs=n_envs,
        seed=seed,
        progress_file=progress_file,
    )
    print("Video saved to: ", wrapper_configs.video.video_dir)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episode_steps", type=int, default=504)
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
    )
    parser.add_argument("--policy_client_host", type=str, default="")
    parser.add_argument("--policy_client_port", type=int, default=None)
    parser.add_argument(
        "--env_name",
        type=str,
        default="gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env",
    )
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--output_json", type=str, default=None, help="Path to write JSON results")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory to save episode videos")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed for reproducible env resets")

    args = parser.parse_args()

    # validate policy configuration
    assert (args.model_path and not (args.policy_client_host or args.policy_client_port)) or (
        not args.model_path and args.policy_client_host and args.policy_client_port is not None
    ), (
        "Invalid policy configuration: You must provide EITHER model_path OR (policy_client_host & policy_client_port), not both.\n"
        "If all 3 arguments are provided, explicitly choose one:\n"
        '  - To use policy client: set --policy_client_host and --policy_client_port, and set --model_path ""\n'
        '  - To use model path: set --model_path, and set --policy_client_host "" (and leave --policy_client_port unset)'
    )

    results = run_gr00t_sim_policy(
        env_name=args.env_name,
        n_episodes=args.n_episodes,
        max_episode_steps=args.max_episode_steps,
        model_path=args.model_path,
        policy_client_host=args.policy_client_host,
        policy_client_port=args.policy_client_port,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        video_dir=args.video_dir,
        seed=args.seed,
    )
    env_name, episode_successes, episode_infos = results
    success_rate = float(np.mean(episode_successes)) if episode_successes else 0.0
    print("results: ", results)
    print("success rate: ", success_rate)
    if len(episode_successes) < args.n_episodes:
        print(f"WARNING: only {len(episode_successes)}/{args.n_episodes} episodes completed")

    if args.output_json is not None:
        import json
        output = {
            "env_name": env_name,
            "n_episodes": len(episode_successes),
            "n_episodes_requested": args.n_episodes,
            "success_rate": success_rate,
            "episode_successes": [bool(s) for s in episode_successes],
            "episode_infos": {k: [float(v) if hasattr(v, "item") else v for v in vals] for k, vals in episode_infos.items()},
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results written to {args.output_json}")

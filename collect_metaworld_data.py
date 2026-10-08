"""Collect AD training data from PPO checkpoints on ML1 reach-v3.

For each goal, loads checkpoints in order (early -> late) and rolls out
trajectories, producing learning histories for Algorithm Distillation.

Uses VecEnv to run all seeds in parallel per checkpoint (load model once,
predict for all envs simultaneously).

Supports two modes:
  - ad: standard AD data (obs, action, reward, done)
  - relabel_reward: replace reward with eval return of the checkpoint (auto_relabel)

Usage:
    python collect_metaworld_data.py --ckpt_dir ./logs/ml1_per_goal --output datasets/metaworld/reach-v3/train.pkl
    python collect_metaworld_data.py --ckpt_dir ./logs/ml1_per_goal --mode relabel_reward --output datasets/metaworld/reach-v3/train_relabel.pkl
"""

import argparse
import os
import pickle
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from loguru import logger
from tqdm import tqdm

from metaworld import ML1
from metaworld.wrappers import AutoTerminateOnSuccessWrapper


class FixedTaskWrapper(gym.Wrapper):
    def __init__(self, env, task):
        super().__init__(env)
        self.task = task
        self.unwrapped.set_task(task)

    def reset(self, **kwargs):
        self.unwrapped.set_task(self.task)
        return self.env.reset(**kwargs)


def make_fixed_task_env(env_cls, task, seed=None, terminate_on_success=True):
    env = env_cls()
    if seed is not None:
        env.seed(seed)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=env.max_path_length)
    env = AutoTerminateOnSuccessWrapper(env)
    env.toggle_terminate_on_success(terminate_on_success)
    env = FixedTaskWrapper(env, task)
    env = Monitor(env)
    return env


def get_sorted_checkpoints(ckpt_dir):
    """Return sorted list of (step, path) tuples for checkpoints in a goal dir."""
    ckpt_path = os.path.join(ckpt_dir, "checkpoints")
    if not os.path.exists(ckpt_path):
        return []
    files = [f for f in os.listdir(ckpt_path) if f.endswith(".zip")]
    result = []
    for f in files:
        step = int(f.split("_")[1])
        result.append((step, os.path.join(ckpt_path, f)))
    result.sort(key=lambda x: x[0])
    return result


def eval_checkpoint(model, env, n_episodes=5):
    """Evaluate a model and return (mean_reward, success_rate)."""
    rewards = []
    successes = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        success = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done_arr, info = env.step(action)
            total_reward += reward[0]
            done = done_arr[0]
            if info[0].get("success", False):
                success = True
        rewards.append(total_reward)
        successes.append(float(success))
    return np.mean(rewards), np.mean(successes)


def stored_icl_observations(env, policy_observations, raw):
    """Snapshot the pre-action state without changing PPO's input normalization."""
    if raw and isinstance(env, VecNormalize):
        return env.get_original_obs().copy()
    return np.asarray(policy_observations).copy()


def collect_goal_data(env_cls, task, goal_dir, steps_per_ckpt, horizon,
                      num_seeds, base_seed, mode="ad", vec_norm_path=None, eval_returns=None,
                      deterministic=False, raw_icl_observations=False):
    """Collect learning history for one goal using VecEnv for all seeds.

    Creates a DummyVecEnv with num_seeds envs (one per seed). Each checkpoint
    is loaded once, then rolled out across all envs simultaneously.
    """
    checkpoints = get_sorted_checkpoints(goal_dir)
    if not checkpoints:
        logger.warning(f"No checkpoints found in {goal_dir}")
        return None

    ckpts_per_traj = horizon // steps_per_ckpt
    num_ckpts = len(checkpoints)
    if num_ckpts < ckpts_per_traj:
        logger.warning(f"Only {num_ckpts} checkpoints, need {ckpts_per_traj}")
        return None

    # Create VecEnv with num_seeds parallel envs
    seeds = [base_seed + i for i in range(num_seeds)]
    env_fns = [
        (lambda s: lambda: make_fixed_task_env(env_cls, task, seed=s, terminate_on_success=True))(s)
        for s in seeds
    ]
    env = DummyVecEnv(env_fns)
    if vec_norm_path and os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

    # ckpt_data[seed_idx][ckpt_idx] = dict of arrays (steps_per_ckpt, dim)
    # Organize as: per-seed list of per-checkpoint data
    all_ckpt_data = [[] for _ in range(num_seeds)]

    # Stats across all checkpoints
    total_episodes = 0
    total_successes = 0
    total_ep_lengths = []
    ep_step_counter = np.zeros(num_seeds, dtype=int)

    obs = env.reset()  # (num_seeds, obs_dim)
    for ckpt_idx, (step, ckpt_path) in enumerate(checkpoints):
        model = PPO.load(ckpt_path, env=env, device="cpu")
        logger.info(f"  Checkpoint {ckpt_idx+1}/{num_ckpts}: step={step}")

        obs_buf = [[] for _ in range(num_seeds)]
        act_buf = [[] for _ in range(num_seeds)]
        rew_buf = [[] for _ in range(num_seeds)]
        done_buf = [[] for _ in range(num_seeds)]

        for _ in range(steps_per_ckpt):
            stored_obs = stored_icl_observations(env, obs, raw_icl_observations)
            actions, _ = model.predict(obs, deterministic=deterministic)
            new_obs, rewards, dones, infos = env.step(actions)

            for i in range(num_seeds):
                obs_buf[i].append(stored_obs[i])
                act_buf[i].append(actions[i])
                rew_buf[i].append(rewards[i])
                done_buf[i].append(dones[i])

                if dones[i]:
                    total_episodes += 1
                    total_ep_lengths.append(ep_step_counter[i] + 1)
                    if infos[i].get("success", False):
                        total_successes += 1
                    ep_step_counter[i] = 0
                else:
                    ep_step_counter[i] += 1

            obs = new_obs

        for i in range(num_seeds):
            entry = {
                "observations": np.array(obs_buf[i], dtype=np.float32),
                "actions": np.array(act_buf[i], dtype=np.float32),
                "rewards": np.array(rew_buf[i], dtype=np.float32),
                "dones": np.array(done_buf[i], dtype=np.float32),
            }
            if mode == "relabel_reward" and eval_returns is not None:
                entry["values"] = np.full(steps_per_ckpt, eval_returns[step], dtype=np.float32)
            all_ckpt_data[i].append(entry)

    env.close()

    # Print stats
    success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0
    avg_ep_len = np.mean(total_ep_lengths) if total_ep_lengths else 0.0
    logger.info(f"  Stats: episodes={total_episodes}, success={success_rate:.2%}, avg_ep_len={avg_ep_len:.1f}")

    # Sliding window with stride=1, per seed
    samples = []
    for i in range(num_seeds):
        for start in range(0, num_ckpts - ckpts_per_traj + 1, 1):
            window = all_ckpt_data[i][start:start + ckpts_per_traj]
            sample = {
                key: np.concatenate([w[key] for w in window], axis=0)
                for key in window[0].keys()
            }
            samples.append(sample)

    logger.info(f"  {len(samples)} samples ({num_seeds} seeds x {num_ckpts - ckpts_per_traj + 1} windows)")
    return samples


def compute_eval_returns(env_cls, task, goal_dir, seed, vec_norm_path):
    """Compute normalized eval returns for each checkpoint (for relabel_reward mode)."""
    checkpoints = get_sorted_checkpoints(goal_dir)
    if not checkpoints:
        return None

    eval_env = DummyVecEnv([lambda: make_fixed_task_env(env_cls, task, seed=seed, terminate_on_success=False)])
    if vec_norm_path and os.path.exists(vec_norm_path):
        eval_env = VecNormalize.load(vec_norm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False

    logger.info("Evaluating checkpoints for reward relabeling (fixed 500-step episodes)...")
    eval_returns = {}
    for step, ckpt_path in checkpoints:
        model = PPO.load(ckpt_path, env=eval_env, device="cpu")
        ret, succ = eval_checkpoint(model, eval_env, n_episodes=5)
        eval_returns[step] = ret
        logger.info(f"  Step {step}: return = {ret:.1f}, success = {succ:.2%}")
    eval_env.close()

    # Normalize values to [0, 1]
    vals = list(eval_returns.values())
    v_min, v_max = min(vals), max(vals)
    if v_max > v_min:
        eval_returns = {k: (v - v_min) / (v_max - v_min) for k, v in eval_returns.items()}
        logger.info(f"  Normalized values: min_ret={v_min:.1f}, max_ret={v_max:.1f}")
    else:
        eval_returns = {k: 0.0 for k in eval_returns}
        logger.warning(f"  All checkpoints have same return ({v_min:.1f}), values set to 0")

    return eval_returns


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", choices=["reach-v3", "push-v3"], default="reach-v3")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Root dir with goal_XX subdirs")
    parser.add_argument("--output", type=str, required=True, help="Output pkl path")
    parser.add_argument("--raw-icl-observations", action="store_true",
                        help="Store raw environment states while PPO still consumes its normalized inputs.")
    parser.add_argument("--mode", type=str, default="ad", choices=["ad", "relabel_reward"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_goals", type=int, default=50)
    parser.add_argument("--steps_per_ckpt", type=int, default=80,
                        help="Number of env steps to collect per checkpoint")
    parser.add_argument("--horizon", type=int, default=400,
                        help="Context length for each training sample (in steps)")
    parser.add_argument("--num_seeds", type=int, default=64,
                        help="Number of rollout seeds per goal (more seeds = more diverse data)")
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Use deterministic policy (higher success rate, diversity from env seed)")
    return parser.parse_args()


def main():
    args = parse_args()

    ml1 = ML1(args.env_name, seed=args.seed)
    env_cls = ml1.train_classes[args.env_name]
    train_tasks = ml1.train_tasks

    all_trajectories = []
    num_goals = min(args.num_goals, len(train_tasks))

    for goal_idx in tqdm(range(num_goals), desc="Goals", unit="goal"):
        goal_dir = os.path.join(args.ckpt_dir, f"goal_{goal_idx:02d}")
        if not os.path.exists(goal_dir):
            logger.warning(f"Goal dir {goal_dir} not found, skipping")
            continue

        task = train_tasks[goal_idx]
        vec_norm_path = os.path.join(goal_dir, "vec_normalize.pkl")
        logger.info(f"\n=== Goal {goal_idx}/{num_goals} ===")

        # Compute eval returns once per goal (for relabel_reward mode)
        eval_returns = None
        if args.mode == "relabel_reward":
            eval_returns = compute_eval_returns(env_cls, task, goal_dir, args.seed, vec_norm_path)

        base_seed = args.seed + goal_idx * 1000
        samples = collect_goal_data(
            env_cls, task, goal_dir,
            steps_per_ckpt=args.steps_per_ckpt,
            horizon=args.horizon,
            num_seeds=args.num_seeds,
            base_seed=base_seed,
            mode=args.mode,
            vec_norm_path=vec_norm_path,
            eval_returns=eval_returns,
            deterministic=args.deterministic,
            raw_icl_observations=args.raw_icl_observations,
        )

        if samples is None or len(samples) == 0:
            continue

        # Stack samples
        keys = samples[0].keys()
        stacked = {key: np.stack([s[key] for s in samples], axis=0) for key in keys}

        all_trajectories.append(stacked)
        logger.info(f"  Collected {len(samples)} samples of horizon {args.horizon}")

    if not all_trajectories: raise ValueError("No complete histories collected; check checkpoints and horizon.")
    # Concatenate all goals
    merged = {
        key: np.concatenate([t[key] for t in all_trajectories], axis=0)
        for key in all_trajectories[0].keys()
    }
    logger.info(f"\nTotal samples: {merged['observations'].shape[0]}")
    logger.info(f"Obs shape: {merged['observations'].shape}")
    logger.info(f"Action shape: {merged['actions'].shape}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(merged, f)
    Path(args.output + ".manifest.json").write_text(json.dumps({"env": args.env_name, "observation_mode": "raw" if args.raw_icl_observations else "legacy_normalized", "args": vars(args)}, indent=2))
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

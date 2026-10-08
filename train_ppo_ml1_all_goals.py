"""Train PPO separately on each of the 50 ML1 reach-v3 goals.

For each goal, trains PPO from scratch (goal hidden in obs) and saves:
  - Periodic checkpoints (for collecting AD learning histories later)
  - The final model

Usage:
    python train_ppo_ml1_all_goals.py
    python train_ppo_ml1_all_goals.py --num_goals 10 --timesteps 300000
    python train_ppo_ml1_all_goals.py --wandb_project my_project  # enable wandb
"""

import argparse
import os
import pickle

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import metaworld  # noqa: F401
from metaworld import ML1
from metaworld.wrappers import AutoTerminateOnSuccessWrapper


class FixedTaskWrapper(gym.Wrapper):
    """Wraps a SawyerXYZEnv to always use a single fixed task."""

    def __init__(self, env, task):
        super().__init__(env)
        self.task = task
        self.unwrapped.set_task(task)

    def reset(self, **kwargs):
        self.unwrapped.set_task(self.task)
        return self.env.reset(**kwargs)


def make_fixed_task_env(env_cls, task, seed=None, terminate_on_success=False):
    """Create a single ML1 env pinned to one specific task (goal)."""
    env = env_cls()
    if seed is not None:
        env.seed(seed)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=env.max_path_length)
    env = AutoTerminateOnSuccessWrapper(env)
    env.toggle_terminate_on_success(terminate_on_success)
    env = FixedTaskWrapper(env, task)
    env = Monitor(env)
    return env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", choices=["reach-v3", "push-v3"], default="reach-v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_goals", type=int, default=50, help="Number of goals to train on (max 50)")
    parser.add_argument("--timesteps", type=int, default=500_000, help="Training timesteps per goal")
    parser.add_argument("--checkpoint_freq", type=int, default=50_000, help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", type=str, default="./logs/ml1_per_goal")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (omit to disable wandb)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (team or username)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Create ML1 benchmark to get train tasks (50 goals)
    ml1 = ML1(args.env_name, seed=args.seed)
    env_cls = ml1.train_classes[args.env_name]
    train_tasks = ml1.train_tasks  # 50 tasks with different goal positions
    num_goals = min(args.num_goals, len(train_tasks))

    print(f"Training PPO on {num_goals} goals, {args.timesteps} steps each")
    print(f"Output: {args.output_dir}")

    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb

        class WandbTrainCallback(BaseCallback):
            """Log training episode rewards/lengths/success to wandb."""

            def __init__(self, verbose=0):
                super().__init__(verbose)

            def _on_step(self) -> bool:
                infos = self.locals.get("infos", [])
                for info in infos:
                    if "episode" in info:
                        log_dict = {
                            "timestep": self.num_timesteps,
                            "train/ep_reward": info["episode"]["r"],
                            "train/ep_length": info["episode"]["l"],
                        }
                        if "success" in info:
                            log_dict["train/success"] = info["success"]
                        wandb.log(log_dict)
                return True

        class WandbEvalCallback(BaseCallback):
            """Run eval manually to log reward + success rate to wandb."""

            def __init__(self, eval_env, eval_freq=10_000, n_eval_episodes=5, verbose=0):
                super().__init__(verbose)
                self.eval_env = eval_env
                self.eval_freq = eval_freq
                self.n_eval_episodes = n_eval_episodes

            def _on_step(self) -> bool:
                if self.num_timesteps % self.eval_freq != 0:
                    return True

                # Sync obs normalization stats from train env
                if hasattr(self.training_env, 'obs_rms'):
                    self.eval_env.obs_rms = self.training_env.obs_rms

                ep_rewards = []
                ep_successes = []
                for _ in range(self.n_eval_episodes):
                    obs = self.eval_env.reset()
                    done = False
                    total_reward = 0.0
                    success = 0.0
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, reward, done_arr, infos = self.eval_env.step(action)
                        total_reward += reward[0]
                        done = done_arr[0]
                        if infos[0].get("success", 0.0) == 1.0:
                            success = 1.0
                    ep_rewards.append(total_reward)
                    ep_successes.append(success)

                wandb.log({
                    "timestep": self.num_timesteps,
                    "eval/mean_reward": np.mean(ep_rewards),
                    "eval/success_rate": np.mean(ep_successes),
                })
                return True

    for goal_idx in range(num_goals):
        task = train_tasks[goal_idx]
        goal_dir = os.path.join(args.output_dir, f"goal_{goal_idx:02d}")
        os.makedirs(goal_dir, exist_ok=True)

        # Extract goal position for logging
        task_data = pickle.loads(task.data)
        rand_vec = task_data["rand_vec"][:3]
        print(f"\n=== Goal {goal_idx}/{num_goals} | rand_vec: {rand_vec} ===")

        # Init wandb run for this goal
        wandb_run = None
        if use_wandb:
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"goal_{goal_idx:02d}",
                group="ml1_reach_v3",
                config={
                    "goal_idx": goal_idx,
                    "rand_vec": rand_vec.tolist(),
                    "seed": args.seed,
                    "timesteps": args.timesteps,
                    "algorithm": "PPO",
                    "env": f"ML1/{args.env_name}",
                },
                reinit=True,
            )
            wandb.define_metric("timestep")
            wandb.define_metric("*", step_metric="timestep")

        # Create train and eval envs pinned to this single goal
        train_env = DummyVecEnv([
            lambda: make_fixed_task_env(
                env_cls, task, seed=args.seed + goal_idx, terminate_on_success=False
            )
        ])
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, gamma=0.99)

        eval_env = DummyVecEnv([
            lambda: make_fixed_task_env(
                env_cls, task, seed=args.seed + goal_idx + 1000, terminate_on_success=False
            )
        ])
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

        # Callbacks
        # Save 10 evenly-spaced checkpoints per goal for AD data collection
        checkpoint_freq = max(args.timesteps // 20, 1)
        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=os.path.join(goal_dir, "checkpoints"),
            name_prefix="ppo",
        )
        callbacks = [checkpoint_cb]
        if use_wandb:
            callbacks.append(WandbTrainCallback())
            callbacks.append(WandbEvalCallback(eval_env, eval_freq=10_000, n_eval_episodes=5))

        # Train with tuned hyperparameters for Metaworld
        model = PPO(
            "MlpPolicy",
            train_env,
            seed=args.seed + goal_idx,
            learning_rate=1e-4,
            n_steps=4000,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=0,
            tensorboard_log=os.path.join(goal_dir, "tb"),
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        model.save(os.path.join(goal_dir, "final_model"))
        train_env.save(os.path.join(goal_dir, "vec_normalize.pkl"))

        train_env.close()
        eval_env.close()

        if wandb_run is not None:
            wandb_run.finish()

    print(f"\nDone. All models saved to {args.output_dir}")


if __name__ == "__main__":
    main()

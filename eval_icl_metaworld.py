"""Evaluate AD / auto_relabel transformer on Metaworld ML1 reach-v3.

Tests on both seen (train) and unseen (test) goal positions.

Usage:
    # Eval on seen goals
    python eval_icl_metaworld.py --model_dir models/metaworld/reach-v3/ad --epoch 50 --split train

    # Eval on unseen goals
    python eval_icl_metaworld.py --model_dir models/metaworld/reach-v3/ad --epoch 50 --split test
"""

import argparse
import os
import json
from pathlib import Path
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import wandb
os.environ.setdefault("WANDB_MODE", "disabled")
from loguru import logger
from tqdm import tqdm

from metaworld import ML1
from metaworld.wrappers import AutoTerminateOnSuccessWrapper
from nets.metaworld_net import MetaworldTransformer, MetaworldMultiheadTransformer, MetaworldValueCondTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_DIM = 39
ACTION_DIM = 4


class FixedTaskWrapper(gym.Wrapper):
    def __init__(self, env, task):
        super().__init__(env)
        self.task = task
        self.unwrapped.set_task(task)

    def reset(self, **kwargs):
        self.unwrapped.set_task(self.task)
        return self.env.reset(**kwargs)


def make_eval_env(env_cls, task, seed=None):
    env = env_cls()
    if seed is not None:
        env.seed(seed)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=env.max_path_length)
    env = AutoTerminateOnSuccessWrapper(env)
    env.toggle_terminate_on_success(True)
    env = FixedTaskWrapper(env, task)
    return env


class ContextBuffer:
    """Rolling buffer storing (s, a, r, v) history for in-context inference."""

    def __init__(self, horizon, state_dim, action_dim, use_values=False):
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_values = use_values
        self.states = deque(maxlen=horizon)
        self.actions = deque(maxlen=horizon)
        self.rewards = deque(maxlen=horizon)
        if use_values:
            self.values = deque(maxlen=horizon)

    def reset(self, init_state):
        """Reset buffer and add initial state with dummy action/reward/value."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.states.append(init_state)
        self.actions.append(np.zeros(self.action_dim, dtype=np.float32))
        self.rewards.append(0.0)
        if self.use_values:
            self.values.clear()
            self.values.append(0.0)

    def add(self, state, action, reward, value=0.0):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if self.use_values:
            self.values.append(value)

    def get_context(self):
        """Return dict of tensors ready for the model. Shape: (1, T, dim)."""
        context_states = np.array(list(self.states), dtype=np.float32)
        context_actions = np.array(list(self.actions), dtype=np.float32)
        context_rewards = np.array(list(self.rewards), dtype=np.float32).reshape(-1, 1)

        context_states = torch.tensor(context_states).unsqueeze(0).to(device)
        context_actions = torch.tensor(context_actions).unsqueeze(0).to(device)
        context_rewards = torch.tensor(context_rewards).unsqueeze(0).to(device)

        result = {
            "context_states": context_states,
            "context_actions": context_actions,
            "context_rewards": context_rewards,
        }
        if self.use_values:
            context_values = np.array(list(self.values), dtype=np.float32).reshape(-1, 1)
            result["context_values"] = torch.tensor(context_values).unsqueeze(0).to(device)
        return result


def eval_on_tasks(model, env_cls, tasks, args, split_name="eval"):
    """Evaluate model on all tasks in parallel with real-time logging.

    Each task gets its own env and context buffer. All tasks run for
    num_steps steps, with episodes auto-resetting on done.
    """
    horizon = args.horizon
    num_steps = args.num_steps
    num_tasks = len(tasks)

    use_values = args.algorithm == "value_cond"

    # Create envs and buffers
    envs = []
    bufs = []
    for i, task in enumerate(tasks):
        env = make_eval_env(env_cls, task, seed=args.seed + i)
        envs.append(env)
        bufs.append(ContextBuffer(horizon, STATE_DIM, ACTION_DIM, use_values=use_values))

    # Per-task tracking
    cumulative_rewards = np.zeros(num_tasks)
    episode_counts = np.zeros(num_tasks, dtype=int)
    episode_rewards = [[] for _ in range(num_tasks)]
    episode_successes = [[] for _ in range(num_tasks)]
    current_ep_reward = np.zeros(num_tasks)
    current_ep_success = np.zeros(num_tasks, dtype=bool)

    # Initial reset
    for i in range(num_tasks):
        obs, _ = envs[i].reset()
        bufs[i].reset(obs)

    for step in tqdm(range(num_steps), desc=f"Eval {split_name}", unit="step"):
        for i in range(num_tasks):
            context = bufs[i].get_context()
            context_len = context["context_states"].shape[1]

            with torch.no_grad():
                output = model(context)
                if isinstance(output, tuple):
                    pred_actions, pred_values = output
                    value = pred_values[0, context_len - 1].item()
                else:
                    pred_actions = output
                    value = 0.0
                action = pred_actions[0, context_len - 1].cpu().numpy()

            obs, reward, terminated, truncated, info = envs[i].step(action)
            done = terminated or truncated
            # auto_relabel: use predicted value in place of reward as context input
            ctx_reward = value if args.algorithm == "auto_relabel" else reward
            bufs[i].add(obs, action, ctx_reward, value=value)

            current_ep_reward[i] += reward
            cumulative_rewards[i] += reward
            if info.get("success", False):
                current_ep_success[i] = True

            if done:
                episode_rewards[i].append(current_ep_reward[i])
                episode_successes[i].append(float(current_ep_success[i]))
                episode_counts[i] += 1

                logger.info(
                    f"  Task {i:02d} ep {episode_counts[i]}: "
                    f"reward={current_ep_reward[i]:.1f}, "
                    f"success={current_ep_success[i]}, "
                    f"cumulative={cumulative_rewards[i]:.1f}"
                )

                current_ep_reward[i] = 0.0
                current_ep_success[i] = False

                # Reset env but keep context buffer (cross-episode)
                obs, _ = envs[i].reset()
                bufs[i].add(obs, np.zeros(ACTION_DIM, dtype=np.float32), 0.0, value=0.0)

        # Log to wandb periodically
        if (step + 1) % 50 == 0:
            active = [i for i in range(num_tasks) if episode_counts[i] > 0]
            log_dict = {"step": step + 1}
            log_dict[f"{split_name}/mean_cumulative_reward"] = np.mean(cumulative_rewards)
            if active:
                log_dict[f"{split_name}/mean_episode_reward"] = np.mean(
                    [np.mean(episode_rewards[i]) for i in active]
                )
                log_dict[f"{split_name}/mean_success_rate"] = np.mean(
                    [np.mean(episode_successes[i]) for i in active]
                )
            log_dict[f"{split_name}/total_episodes"] = int(sum(episode_counts))
            wandb.log(log_dict)

    # Close envs
    for env in envs:
        env.close()

    # Compute final per-task metrics
    all_rewards = []
    all_success_rates = []
    for i in range(num_tasks):
        if len(episode_rewards[i]) > 0:
            mean_r = np.mean(episode_rewards[i])
            mean_s = np.mean(episode_successes[i])
        else:
            mean_r = 0.0
            mean_s = 0.0
        all_rewards.append(mean_r)
        all_success_rates.append(mean_s)
        logger.info(f"  Task {i:02d} final: reward={mean_r:.1f}, success={mean_s:.2f}")

    return all_rewards, all_success_rates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", choices=["reach-v3", "push-v3"], default="reach-v3")
    parser.add_argument("--output-json")
    parser.add_argument("--model_dir", type=str, required=True, help="Dir containing model checkpoints")
    parser.add_argument("--epoch", type=str, default="final", help="Epoch to load (e.g. '10', '50', 'final')")
    parser.add_argument("--algorithm", type=str, default="ad", choices=["ad", "auto_relabel", "value_cond"])
    parser.add_argument("--split", type=str, default="both", choices=["train", "test", "both"],
                        help="Evaluate on train (seen) / test (unseen) / both goals")
    parser.add_argument("--num_steps", type=int, default=5000, help="Steps per task")
    parser.add_argument("--num_tasks", type=int, default=10, help="Number of tasks to evaluate")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default="metaworld-icl-eval")
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = {
        "horizon": args.horizon,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "n_embd": args.n_embd,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "dropout": args.dropout,
    }

    # Load model
    if args.algorithm == "ad":
        model = MetaworldTransformer(config).to(device)
    elif args.algorithm == "auto_relabel":
        model = MetaworldMultiheadTransformer(config).to(device)
    elif args.algorithm == "value_cond":
        model = MetaworldValueCondTransformer(config).to(device)

    if args.epoch == "final":
        ckpt_path = os.path.join(args.model_dir, "final.pt")
    else:
        ckpt_path = os.path.join(args.model_dir, f"epoch_{args.epoch}.pt")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    logger.info(f"Loaded model from {ckpt_path}")

    # Setup ML1
    ml1 = ML1(args.env_name, seed=args.seed)
    env_cls = ml1.train_classes[args.env_name]
    train_tasks = ml1.train_tasks
    test_tasks = ml1.test_tasks

    wandb.init(
        project=args.wandb_project,
        name=f"eval-{args.algorithm}-epoch{args.epoch}-{args.split}",
        config=vars(args),
    )

    results = {}
    splits = []
    if args.split in ("train", "both"):
        splits.append(("train", train_tasks))
    if args.split in ("test", "both"):
        splits.append(("test", test_tasks))

    for split_name, tasks in splits:
        eval_tasks = tasks[:args.num_tasks]
        logger.info(f"\n=== Evaluating on {split_name} tasks ({len(eval_tasks)} tasks, {args.num_steps} steps each) ===")

        rewards, success_rates = eval_on_tasks(model, env_cls, eval_tasks, args, split_name=split_name)

        results[split_name] = {"legacy_aer_per_task": np.asarray(rewards).tolist(), "success_rate_per_task": np.asarray(success_rates).tolist()}
        mean_reward = np.mean(rewards)
        mean_success = np.mean(success_rates)

        logger.info(f"\n{split_name} results:")
        logger.info(f"  Mean reward: {mean_reward:.1f}")
        logger.info(f"  Mean success rate: {mean_success:.2%}")

        wandb.log({
            f"{split_name}/mean_reward": mean_reward,
            f"{split_name}/mean_success_rate": mean_success,
        })

        # Log per-task results
        for i, (r, s) in enumerate(zip(rewards, success_rates)):
            wandb.log({
                f"{split_name}/task_{i:02d}_reward": r,
                f"{split_name}/task_{i:02d}_success": s,
            })

    if args.output_json:
        path = Path(args.output_json)
        if path.exists(): raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"args": vars(args), "splits": results}, indent=2))
    wandb.finish()
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()

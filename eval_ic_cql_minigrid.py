"""Evaluate a trained MiniGrid IC-CQL policy with cross-episode context.

The evaluator keeps each task's interaction history across episode boundaries,
uses deterministic argmax-Q actions, and reports the paper's AER/LER/IF metrics
per unseen seed and in aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Sequence

import gymnasium as gym
import minigrid  # noqa: F401 - imports register MiniGrid environments
import numpy as np
import torch
from loguru import logger
from minigrid.wrappers import ImgObsWrapper
from tqdm import tqdm

from nets.ic_cql_minigrid import (
    IC_CQL_CHECKPOINT_FORMAT_VERSION,
    IC_CQL_CONTRACT_VERSION,
    IC_CQL_POLICY_Q_RULE,
    MinigridICCQLTransformer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a cross-episode IC-CQL checkpoint on unseen MiniGrid seeds."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=None,
        help="Optional lowercase SHA256 that the checkpoint must match before loading.",
    )
    parser.add_argument("--eval-env", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=8000,
        help="Synchronous environment steps per unseen task/seed.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=20000,
        help="First unseen MiniGrid task seed; the paper protocol uses 20000.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--wandb-project", default="eval-minigrid-ic-cql")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_observation(observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation)
    if observation.ndim != 3:
        raise ValueError(f"Expected a 3-D MiniGrid image, got {observation.shape}.")
    if observation.shape[-1] == 3:
        observation = np.transpose(observation, (2, 0, 1))
    if observation.shape != (3, 7, 7):
        raise ValueError(
            f"Expected a (3, 7, 7) MiniGrid observation, got {observation.shape}."
        )
    if not np.isfinite(observation).all():
        raise FloatingPointError("Observation contains NaN or Inf.")
    return np.ascontiguousarray(observation)


class OnlineHistory:
    """A fixed-window learning history that survives episode resets."""

    def __init__(self, horizon: int, initial_observation: np.ndarray):
        if horizon < 2:
            raise ValueError("IC-CQL context horizon must be at least two.")
        self.horizon = horizon
        self.states: Deque[np.ndarray] = deque(maxlen=horizon)
        self.actions: Deque[int] = deque(maxlen=horizon - 1)
        self.rewards: Deque[float] = deque(maxlen=horizon - 1)
        self.dones: Deque[bool] = deque(maxlen=horizon - 1)
        self.episode_steps: Deque[int] = deque(maxlen=horizon)
        self.initial_previous_action = 0
        self.initial_previous_reward = 0.0
        self.initial_previous_done = False
        self.initial_previous_valid = False
        self.states.append(preprocess_observation(initial_observation))
        self.episode_steps.append(0)

    def append(
        self,
        *,
        action: int,
        reward: float,
        done: bool,
        next_observation: np.ndarray,
    ) -> None:
        if not np.isfinite(reward):
            raise FloatingPointError(f"Environment returned non-finite reward {reward}.")
        previous_episode_step = self.episode_steps[-1]
        if len(self.states) == self.horizon:
            # The deques will drop their first state and transition.  Preserve
            # that transition as the predecessor of the new token zero.
            self.initial_previous_action = int(self.actions[0])
            self.initial_previous_reward = float(self.rewards[0])
            self.initial_previous_done = bool(self.dones[0])
            self.initial_previous_valid = True
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.states.append(preprocess_observation(next_observation))
        self.episode_steps.append(0 if done else previous_episode_step + 1)

    def as_arrays(self) -> Dict[str, np.ndarray]:
        sequence_length = len(self.states)
        if len(self.actions) != sequence_length - 1:
            raise AssertionError("Online history state/action alignment is broken.")
        actions = np.zeros(sequence_length, dtype=np.int64)
        rewards = np.zeros(sequence_length, dtype=np.float32)
        dones = np.zeros(sequence_length, dtype=np.bool_)
        if sequence_length > 1:
            actions[:-1] = np.asarray(self.actions, dtype=np.int64)
            rewards[:-1] = np.asarray(self.rewards, dtype=np.float32)
            dones[:-1] = np.asarray(self.dones, dtype=np.bool_)
        return {
            "context_states": np.stack(self.states),
            "context_actions": actions,
            "context_rewards": rewards,
            "context_dones": dones,
            "episode_steps": np.asarray(self.episode_steps, dtype=np.float32),
            "valid_mask": np.ones(sequence_length, dtype=np.bool_),
            "initial_previous_action": np.asarray(
                self.initial_previous_action,
                dtype=np.int64,
            ),
            "initial_previous_reward": np.asarray(
                self.initial_previous_reward,
                dtype=np.float32,
            ),
            "initial_previous_done": np.asarray(
                self.initial_previous_done,
                dtype=np.bool_,
            ),
            "initial_previous_valid": np.asarray(
                self.initial_previous_valid,
                dtype=np.bool_,
            ),
        }


def stack_histories(
    histories: Sequence[OnlineHistory],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    arrays = [history.as_arrays() for history in histories]
    lengths = {array["context_states"].shape[0] for array in arrays}
    if len(lengths) != 1:
        raise AssertionError(f"Synchronous histories have different lengths: {lengths}")
    batch: Dict[str, torch.Tensor] = {}
    for key in arrays[0]:
        stacked = np.stack([array[key] for array in arrays])
        tensor = torch.from_numpy(stacked)
        if key in (
            "context_states",
            "context_rewards",
            "episode_steps",
            "initial_previous_reward",
        ):
            tensor = tensor.float()
        elif key in ("context_actions", "initial_previous_action"):
            tensor = tensor.long()
        else:
            tensor = tensor.bool()
        batch[key] = tensor.to(device, non_blocking=True)
    return batch


def aggregate_seed_metrics(
    episode_returns: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    per_seed = []
    for returns in episode_returns:
        values = np.asarray(returns, dtype=np.float64)
        if values.size == 0:
            per_seed.append(
                {
                    "episodes": 0,
                    "aer": 0.0,
                    "ler": 0.0,
                    "if": None,
                    "success_rate": 0.0,
                    "episode_returns": [],
                }
            )
            continue
        instability_count = (
            int(np.sum(values[1:] <= 0.95 * values[:-1]))
            if values.size > 1
            else 0
        )
        per_seed.append(
            {
                "episodes": int(values.size),
                "aer": float(values.mean()),
                "ler": float(values[-1]),
                # The paper defines IF with N (the number of episodes) as denominator.
                "if": float(instability_count / values.size),
                "success_rate": float(np.mean(values > 0.0)),
                "episode_returns": values.tolist(),
            }
        )

    aggregate: Dict[str, Any] = {}
    for metric in ("aer", "ler", "success_rate", "episodes"):
        values = np.asarray([row[metric] for row in per_seed], dtype=np.float64)
        aggregate[f"{metric}_mean"] = float(values.mean())
        aggregate[f"{metric}_std"] = float(values.std(ddof=0))
    valid_if_values = np.asarray(
        [row["if"] for row in per_seed if row["if"] is not None],
        dtype=np.float64,
    )
    aggregate["if_valid_tasks"] = int(valid_if_values.size)
    aggregate["no_completed_episode_tasks"] = int(
        sum(row["episodes"] == 0 for row in per_seed)
    )
    if valid_if_values.size:
        aggregate["if_mean"] = float(valid_if_values.mean())
        aggregate["if_std"] = float(valid_if_values.std(ddof=0))
        aggregate["if_percent_mean"] = 100.0 * aggregate["if_mean"]
        aggregate["if_percent_std"] = 100.0 * aggregate["if_std"]
    else:
        aggregate["if_mean"] = None
        aggregate["if_std"] = None
        aggregate["if_percent_mean"] = None
        aggregate["if_percent_std"] = None
    aggregate["task_success_rate"] = float(
        np.mean(
            [
                bool(row["episode_returns"])
                and max(row["episode_returns"]) > 0.0
                for row in per_seed
            ]
        )
    )
    return {"aggregate": aggregate, "per_seed": per_seed}


class GreedyPolicyDiagnostics:
    """Accumulate action usage and Q-confidence without affecting rollout actions."""

    def __init__(self, *, num_envs: int, action_dim: int):
        if num_envs < 1:
            raise ValueError("Greedy diagnostics require at least one environment.")
        if action_dim < 2:
            raise ValueError("Top1-top2 Q margins require at least two actions.")
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.action_histograms = np.zeros(
            (self.num_envs, self.action_dim),
            dtype=np.int64,
        )
        self.q_margin_sums = np.zeros(self.num_envs, dtype=np.float64)
        self.num_steps = 0

    def record(self, q_values: torch.Tensor, actions: np.ndarray) -> None:
        """Record one synchronous greedy decision for every evaluation seed."""

        if q_values.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                "Expected final-token Q values with shape "
                f"{(self.num_envs, self.action_dim)}, got {tuple(q_values.shape)}."
            )
        if not torch.isfinite(q_values).all():
            raise FloatingPointError("Diagnostic Q values contain NaN or Inf.")

        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_envs,):
            raise ValueError(
                f"Expected {self.num_envs} greedy actions, got {actions.shape}."
            )
        expected_actions = q_values.argmax(dim=-1).detach().cpu().numpy()
        if not np.array_equal(actions, expected_actions):
            raise AssertionError("Recorded actions do not match deterministic argmax-Q.")

        top_two = torch.topk(q_values, k=2, dim=-1, sorted=True).values
        q_margins = (top_two[:, 0] - top_two[:, 1]).double().cpu().numpy()
        if not np.isfinite(q_margins).all():
            raise FloatingPointError("Top1-top2 Q margins contain NaN or Inf.")

        np.add.at(
            self.action_histograms,
            (np.arange(self.num_envs), actions),
            1,
        )
        self.q_margin_sums += q_margins
        self.num_steps += 1

    def as_dict(self, seeds: Sequence[int]) -> Dict[str, Any]:
        if len(seeds) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} seeds, got {len(seeds)}."
            )
        if self.num_steps < 1:
            raise ValueError("Cannot summarize empty greedy diagnostics.")

        per_seed_frequencies = (
            self.action_histograms.astype(np.float64) / self.num_steps
        )
        per_seed = []
        for index, seed in enumerate(seeds):
            per_seed.append(
                {
                    "seed": int(seed),
                    "num_action_decisions": int(self.num_steps),
                    "greedy_action_histogram": self.action_histograms[
                        index
                    ].tolist(),
                    "greedy_action_frequencies": per_seed_frequencies[
                        index
                    ].tolist(),
                    "mean_top1_top2_q_margin": float(
                        self.q_margin_sums[index] / self.num_steps
                    ),
                }
            )

        aggregate_histogram = self.action_histograms.sum(axis=0)
        total_decisions = self.num_steps * self.num_envs
        return {
            "action_dim": self.action_dim,
            "num_steps_per_seed": int(self.num_steps),
            "aggregate": {
                "num_action_decisions": int(total_decisions),
                "greedy_action_histogram": aggregate_histogram.tolist(),
                "greedy_action_frequencies": (
                    aggregate_histogram.astype(np.float64) / total_decisions
                ).tolist(),
                "mean_top1_top2_q_margin": float(
                    self.q_margin_sums.sum() / total_decisions
                ),
            },
            "per_seed": per_seed,
        }


def atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def load_trusted_checkpoint(path: Path) -> Dict[str, Any]:
    """Load a checkpoint produced by our trainer across PyTorch 2.x defaults."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def file_identity(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def checkpoint_result_contract(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and expose the official twin-Q checkpoint contract."""

    if checkpoint.get("algorithm") != "IC-CQL":
        raise ValueError("Checkpoint algorithm must be 'IC-CQL'.")
    format_version = checkpoint.get("format_version")
    if format_version != IC_CQL_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported IC-CQL checkpoint format_version "
            f"{format_version!r}; expected {IC_CQL_CHECKPOINT_FORMAT_VERSION}."
        )
    if checkpoint.get("twin_q") is not True:
        raise ValueError("IC-CQL checkpoint must declare twin_q=true.")
    if checkpoint.get("policy_q_rule") != IC_CQL_POLICY_Q_RULE:
        raise ValueError(
            "IC-CQL checkpoint policy_q_rule must be "
            f"{IC_CQL_POLICY_Q_RULE!r}."
        )
    contract = checkpoint.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("IC-CQL checkpoint must contain a mapping contract.")
    if contract.get("contract_version") != IC_CQL_CONTRACT_VERSION:
        raise ValueError(
            "IC-CQL checkpoint contract_version must be "
            f"{IC_CQL_CONTRACT_VERSION!r}."
        )
    if contract.get("checkpoint_format_version") != format_version:
        raise ValueError(
            "IC-CQL checkpoint contract has inconsistent checkpoint_format_version."
        )
    if contract.get("twin_q") is not True:
        raise ValueError("IC-CQL checkpoint contract must declare twin_q=true.")
    if contract.get("algorithm") != checkpoint.get("algorithm"):
        raise ValueError(
            "IC-CQL checkpoint and contract have inconsistent algorithms."
        )
    if contract.get("policy_q_rule") != IC_CQL_POLICY_Q_RULE:
        raise ValueError(
            "IC-CQL checkpoint contract has an inconsistent policy_q_rule."
        )
    model_config = checkpoint.get("model_config")
    contract_model_config = contract.get("model_config")
    if not isinstance(model_config, Mapping) or not isinstance(
        contract_model_config,
        Mapping,
    ):
        raise ValueError(
            "IC-CQL checkpoint and contract must both contain model_config mappings."
        )
    if dict(model_config) != dict(contract_model_config):
        raise ValueError(
            "IC-CQL checkpoint model_config differs from its declared contract."
        )
    train_args = checkpoint.get("args")
    if not isinstance(train_args, Mapping):
        raise ValueError("IC-CQL checkpoint args must be a mapping.")
    if train_args.get("tuple_mode") != contract.get("tuple_mode"):
        raise ValueError(
            "IC-CQL checkpoint tuple_mode differs from its declared contract."
        )
    provenance = checkpoint.get("provenance")
    provenance_git = (
        provenance.get("git") if isinstance(provenance, Mapping) else None
    )
    if not isinstance(provenance_git, Mapping):
        raise ValueError("IC-CQL checkpoint provenance.git must be a mapping.")
    if provenance_git.get("commit") != contract.get("git_commit"):
        raise ValueError(
            "IC-CQL checkpoint git commit differs from its declared contract."
        )
    if provenance_git.get("dirty") is not False:
        raise ValueError("IC-CQL checkpoint provenance must record a clean checkout.")
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("IC-CQL checkpoint epoch must be a non-negative integer.")
    return {
        "checkpoint_format_version": format_version,
        "checkpoint_contract": dict(contract),
        "checkpoint_completed_epochs": epoch + 1,
        "policy_q_rule": IC_CQL_POLICY_Q_RULE,
        "twin_q": True,
    }


def main() -> None:
    args = parse_args()
    if args.num_envs < 1 or args.num_steps < 1:
        raise ValueError("--num-envs and --num-steps must be positive.")
    seed_everything(args.torch_seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint_identity = file_identity(checkpoint_path)
    if (
        args.expected_checkpoint_sha256 is not None
        and checkpoint_identity["sha256"] != args.expected_checkpoint_sha256
    ):
        raise ValueError(
            "Checkpoint SHA256 mismatch: "
            f"{checkpoint_identity['sha256']} != {args.expected_checkpoint_sha256}."
        )
    checkpoint = load_trusted_checkpoint(checkpoint_path)
    if checkpoint.get("algorithm") != "IC-CQL":
        raise ValueError(
            f"Checkpoint algorithm is {checkpoint.get('algorithm')!r}, expected 'IC-CQL'."
        )
    result_contract = checkpoint_result_contract(checkpoint)

    train_args = checkpoint["args"]
    tuple_mode = train_args["tuple_mode"]
    model_config = checkpoint["model_config"]
    model = MinigridICCQLTransformer(
        model_config,
        tuple_mode=tuple_mode,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    horizon = int(model_config["horizon"])

    seeds = [args.seed_start + index for index in range(args.num_envs)]
    envs: List[gym.Env] = []
    histories: List[OnlineHistory] = []
    episode_returns: List[List[float]] = [[] for _ in seeds]
    current_returns = np.zeros(args.num_envs, dtype=np.float64)
    greedy_diagnostics = GreedyPolicyDiagnostics(
        num_envs=args.num_envs,
        action_dim=int(model_config["action_dim"]),
    )
    run = None
    try:
        for seed in seeds:
            env = ImgObsWrapper(gym.make(args.eval_env))
            observation, _ = env.reset(seed=seed)
            if env.action_space.n != int(model_config["action_dim"]):
                raise ValueError(
                    f"Environment action dim {env.action_space.n} != checkpoint "
                    f"{model_config['action_dim']}."
                )
            envs.append(env)
            histories.append(OnlineHistory(horizon, observation))

        import wandb

        short_env = args.eval_env.replace("MiniGrid-", "").replace("-v0", "")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name
            or f"eval-ic-cql-{short_env}-{tuple_mode}-{checkpoint_path.stem}",
            group=args.wandb_group or f"eval-ic-cql-{short_env}-{tuple_mode}",
            mode=args.wandb_mode,
            config={
                **vars(args),
                "checkpoint": checkpoint_identity,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_global_update": checkpoint.get("global_update"),
                "checkpoint_git": checkpoint.get("provenance", {}).get("git"),
                **result_contract,
                "model_config": model_config,
                "tuple_mode": tuple_mode,
                "seeds": seeds,
            },
            tags=["ic-cql", "twin-q", "evaluation", "cross-episode", tuple_mode],
        )

        for step in tqdm(range(args.num_steps), desc="IC-CQL evaluation"):
            batch = stack_histories(histories, device)
            with torch.no_grad():
                hidden = model.encode_context(batch)
                q1_values = model.q1_head(hidden)
                if not torch.isfinite(q1_values).all():
                    raise FloatingPointError("Evaluation Q values contain NaN or Inf.")
                final_q_values = q1_values[:, -1]
                actions = final_q_values.argmax(dim=-1).cpu().numpy()
                greedy_diagnostics.record(final_q_values, actions)

            for index, (env, action) in enumerate(zip(envs, actions)):
                next_observation, reward, terminated, truncated, _ = env.step(int(action))
                done = bool(terminated or truncated)
                current_returns[index] += float(reward)
                if done:
                    episode_returns[index].append(float(current_returns[index]))
                    current_returns[index] = 0.0
                    # Keep the seeded RNG stream, matching the existing evaluation
                    # protocol, while retaining the Transformer history across reset.
                    next_observation, _ = env.reset()
                histories[index].append(
                    action=int(action),
                    reward=float(reward),
                    done=done,
                    next_observation=next_observation,
                )

            if (step + 1) % args.log_every == 0:
                completed = [len(values) for values in episode_returns]
                observed_aer = [
                    float(np.mean(values)) for values in episode_returns if values
                ]
                run.log(
                    {
                        "rollout/step": step + 1,
                        "rollout/completed_episodes": int(sum(completed)),
                        "rollout/mean_completed_per_seed": float(np.mean(completed)),
                        "rollout/aer_so_far": (
                            float(np.mean(observed_aer)) if observed_aer else 0.0
                        ),
                    },
                    step=step + 1,
                )

        metrics = aggregate_seed_metrics(episode_returns)
        diagnostics = greedy_diagnostics.as_dict(seeds)
        result = {
            "algorithm": "IC-CQL",
            "eval_env": args.eval_env,
            "tuple_mode": tuple_mode,
            "checkpoint": checkpoint_identity,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_global_update": checkpoint.get("global_update"),
            "checkpoint_provenance": checkpoint.get("provenance"),
            **result_contract,
            "num_steps_per_seed": args.num_steps,
            "seeds": seeds,
            "greedy_action_selection": True,
            "context_reset_on_episode": False,
            "greedy_policy_diagnostics": diagnostics,
            "created_at_unix": time.time(),
            **metrics,
            "wandb_run_id": run.id,
        }
        output_path = Path(args.output_json).expanduser().resolve()
        atomic_json_dump(result, output_path)
        run.summary.update(
            {
                **metrics["aggregate"],
                "diagnostics/greedy_action_histogram": diagnostics["aggregate"][
                    "greedy_action_histogram"
                ],
                "diagnostics/greedy_action_frequencies": diagnostics["aggregate"][
                    "greedy_action_frequencies"
                ],
                "diagnostics/mean_top1_top2_q_margin": diagnostics["aggregate"][
                    "mean_top1_top2_q_margin"
                ],
                "status": "completed",
                "result_json": str(output_path),
            }
        )
        if metrics["aggregate"]["if_percent_mean"] is None:
            if_display = "n/a (0 valid tasks)"
        else:
            if_display = (
                f'{metrics["aggregate"]["if_percent_mean"]:.3f}'
                f'±{metrics["aggregate"]["if_percent_std"]:.3f}% '
                f'({metrics["aggregate"]["if_valid_tasks"]} valid tasks)'
            )
        logger.info(
            "AER {:.6f}±{:.6f}; LER {:.6f}±{:.6f}; IF {}",
            metrics["aggregate"]["aer_mean"],
            metrics["aggregate"]["aer_std"],
            metrics["aggregate"]["ler_mean"],
            metrics["aggregate"]["ler_std"],
            if_display,
        )
    finally:
        if run:
            run.finish()
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()

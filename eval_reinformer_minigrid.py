"""Evaluate the MiniGrid Reinformer adaptation with cross-episode context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

import gymnasium as gym
import minigrid  # noqa: F401 - registers MiniGrid environments
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper
from tqdm import tqdm

from nets.reinformer_minigrid import (
    REINFORMER_ACTION_HEAD_CONCAT_MLP_V1,
    REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3,
    REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2,
    REINFORMER_CHECKPOINT_FORMAT_VERSION,
    REINFORMER_CONTRACT_VERSION,
    REINFORMER_LEGACY_CHECKPOINT_FORMAT_VERSION,
    REINFORMER_LEGACY_CONTRACT_VERSION,
    REINFORMER_TWO_PASS_CHECKPOINT_FORMAT_VERSION,
    REINFORMER_TWO_PASS_CONTRACT_VERSION,
    SUPPORTED_REINFORMER_CHECKPOINT_CONTRACTS,
    MinigridReinformer,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-episode MiniGrid Reinformer on unseen seeds."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", default=None)
    parser.add_argument("--eval-env", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=8000)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--rtg-perturb-diagnostic-every",
        type=int,
        default=100,
        help=(
            "Measure the +0.25 RTG intervention every N rollout steps. This "
            "uses one additional action-head call without rerunning the v3 "
            "Transformer."
        ),
    )
    parser.add_argument(
        "--action-selection",
        choices=("sample", "argmax"),
        default="sample",
        help=(
            "Formal comparison uses sample to match the repository's AD/CV "
            "categorical policy evaluator. Argmax is available only as a "
            "predeclared diagnostic."
        ),
    )
    parser.add_argument("--wandb-project", default="eval-minigrid-reinformer")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    return parser.parse_args(argv)


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


class ReinformerOnlineHistory:
    """Sliding causal history retained when the environment resets."""

    def __init__(self, horizon: int, initial_observation: np.ndarray):
        if horizon < 1:
            raise ValueError("Reinformer horizon must be positive.")
        self.horizon = int(horizon)
        self.states: Deque[np.ndarray] = deque(maxlen=horizon)
        transition_capacity = max(1, horizon - 1)
        self.actions: Deque[int] = deque(maxlen=transition_capacity)
        self.rewards: Deque[float] = deque(maxlen=transition_capacity)
        self.dones: Deque[bool] = deque(maxlen=transition_capacity)
        self.predicted_rtgs: Deque[float] = deque(maxlen=transition_capacity)
        self.initial_previous_action = 0
        self.initial_previous_reward = 0.0
        self.initial_previous_done = False
        self.initial_previous_rtg = 0.0
        self.initial_previous_valid = False
        self.states.append(preprocess_observation(initial_observation))

    def append(
        self,
        *,
        action: int,
        reward: float,
        done: bool,
        predicted_rtg: float,
        next_observation: np.ndarray,
    ) -> None:
        if not np.isfinite(reward):
            raise FloatingPointError(f"Environment reward is non-finite: {reward}.")
        if not np.isfinite(predicted_rtg):
            raise FloatingPointError(f"Predicted RTG is non-finite: {predicted_rtg}.")
        if len(self.states) == self.horizon:
            self.initial_previous_action = int(self.actions[0])
            self.initial_previous_reward = float(self.rewards[0])
            self.initial_previous_done = bool(self.dones[0])
            self.initial_previous_rtg = float(self.predicted_rtgs[0])
            self.initial_previous_valid = True
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.predicted_rtgs.append(float(predicted_rtg))
        self.states.append(preprocess_observation(next_observation))

    def as_arrays(self) -> Dict[str, np.ndarray]:
        sequence_length = len(self.states)
        expected_transitions = sequence_length - 1
        if not all(
            len(values) == expected_transitions
            for values in (
                self.actions,
                self.rewards,
                self.dones,
                self.predicted_rtgs,
            )
        ):
            raise AssertionError("Reinformer online history alignment is broken.")
        actions = np.zeros(sequence_length, dtype=np.int64)
        rewards = np.zeros(sequence_length, dtype=np.float32)
        dones = np.zeros(sequence_length, dtype=np.bool_)
        rtgs = np.zeros(sequence_length, dtype=np.float32)
        if sequence_length > 1:
            actions[:-1] = np.asarray(self.actions, dtype=np.int64)
            rewards[:-1] = np.asarray(self.rewards, dtype=np.float32)
            dones[:-1] = np.asarray(self.dones, dtype=np.bool_)
            rtgs[:-1] = np.asarray(self.predicted_rtgs, dtype=np.float32)
        return {
            "context_states": np.stack(self.states),
            "context_actions": actions,
            "context_rewards": rewards,
            "context_dones": dones,
            "context_rtgs": rtgs,
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
            "initial_previous_rtg": np.asarray(
                self.initial_previous_rtg,
                dtype=np.float32,
            ),
        }


def stack_histories(
    histories: Sequence[ReinformerOnlineHistory],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    arrays = [history.as_arrays() for history in histories]
    lengths = {array["context_states"].shape[0] for array in arrays}
    if len(lengths) != 1:
        raise AssertionError(f"Synchronous histories have different lengths: {lengths}.")
    batch: Dict[str, torch.Tensor] = {}
    for key in arrays[0]:
        tensor = torch.from_numpy(np.stack([array[key] for array in arrays]))
        if key in (
            "context_states",
            "context_rewards",
            "context_rtgs",
            "initial_previous_reward",
            "initial_previous_rtg",
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
    valid_if = np.asarray(
        [row["if"] for row in per_seed if row["if"] is not None],
        dtype=np.float64,
    )
    aggregate["if_valid_tasks"] = int(valid_if.size)
    aggregate["no_completed_episode_tasks"] = int(
        sum(row["episodes"] == 0 for row in per_seed)
    )
    if valid_if.size:
        aggregate["if_mean"] = float(valid_if.mean())
        aggregate["if_std"] = float(valid_if.std(ddof=0))
        aggregate["if_percent_mean"] = 100.0 * aggregate["if_mean"]
        aggregate["if_percent_std"] = 100.0 * aggregate["if_std"]
    else:
        aggregate.update(
            {
                "if_mean": None,
                "if_std": None,
                "if_percent_mean": None,
                "if_percent_std": None,
            }
        )
    return {"aggregate": aggregate, "per_seed": per_seed}


def atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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


def load_trusted_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_result_contract(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    if checkpoint.get("algorithm") != "Reinformer-MiniGrid":
        raise ValueError("Checkpoint algorithm must be 'Reinformer-MiniGrid'.")
    checkpoint_identity = (
        checkpoint.get("format_version"),
        checkpoint.get("contract_version"),
    )
    if checkpoint_identity not in SUPPORTED_REINFORMER_CHECKPOINT_CONTRACTS:
        raise ValueError(
            "Unsupported Reinformer checkpoint format/contract pair: "
            f"{checkpoint_identity!r}."
        )
    contract = checkpoint.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Checkpoint must contain a contract mapping.")
    if contract.get("contract_version") != checkpoint.get("contract_version"):
        raise ValueError("Checkpoint and embedded contract versions differ.")
    if contract.get("checkpoint_format_version") != checkpoint.get("format_version"):
        raise ValueError("Checkpoint and contract format versions differ.")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("Checkpoint model_config must be a mapping.")
    if dict(model_config) != dict(contract.get("model_config", {})):
        raise ValueError("Checkpoint model_config differs from its contract.")
    action_head_type = model_config.get(
        "action_head_type",
        REINFORMER_ACTION_HEAD_CONCAT_MLP_V1,
    )
    if checkpoint_identity == (
        REINFORMER_LEGACY_CHECKPOINT_FORMAT_VERSION,
        REINFORMER_LEGACY_CONTRACT_VERSION,
    ):
        expected_action_head_type = REINFORMER_ACTION_HEAD_CONCAT_MLP_V1
    elif checkpoint_identity == (
        REINFORMER_TWO_PASS_CHECKPOINT_FORMAT_VERSION,
        REINFORMER_TWO_PASS_CONTRACT_VERSION,
    ):
        expected_action_head_type = REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2
    else:
        expected_action_head_type = REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3
    if action_head_type != expected_action_head_type:
        raise ValueError(
            "Checkpoint action-head type differs from its versioned contract."
        )
    if checkpoint_identity != (
        REINFORMER_LEGACY_CHECKPOINT_FORMAT_VERSION,
        REINFORMER_LEGACY_CONTRACT_VERSION,
    ):
        capacity_audit = contract.get("capacity_audit")
        if not isinstance(capacity_audit, Mapping):
            raise ValueError(
                "The matched Reinformer checkpoint contract lacks a capacity audit."
            )
        n_embd = int(model_config["n_embd"])
        action_dim = int(model_config["action_dim"])
        expected_head_parameters = n_embd * action_dim + action_dim
        expected_passes = (
            2
            if action_head_type == REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2
            else 1
        )
        recorded_passes = capacity_audit.get(
            "transformer_passes_per_action",
            2
            if action_head_type == REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2
            else None,
        )
        if (
            capacity_audit.get("action_output_head_parameters")
            != expected_head_parameters
            or capacity_audit.get("cv_icrl_action_head_parameters")
            != expected_head_parameters
            or capacity_audit.get(
                "parameter_delta_vs_cv_matched_architecture"
            )
            != n_embd
            or recorded_passes != expected_passes
        ):
            raise ValueError(
                "The checkpoint fails its CV-ICRL capacity/compute contract."
            )
    provenance = checkpoint.get("provenance")
    provenance_git = (
        provenance.get("git") if isinstance(provenance, Mapping) else None
    )
    if not isinstance(provenance_git, Mapping):
        raise ValueError("Checkpoint provenance.git must be a mapping.")
    if provenance_git.get("commit") != contract.get("git_commit"):
        raise ValueError("Checkpoint git commit differs from its contract.")
    if provenance_git.get("dirty") is not False:
        raise ValueError("Formal evaluation requires a clean training checkout.")
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Checkpoint epoch must be a non-negative integer.")
    return {
        "checkpoint_format_version": checkpoint["format_version"],
        "checkpoint_contract_version": checkpoint["contract_version"],
        "checkpoint_completed_epochs": epoch + 1,
        "checkpoint_contract": dict(contract),
    }


class PolicyDiagnostics:
    def __init__(self, *, num_envs: int, action_dim: int):
        self.num_envs = int(num_envs)
        self.action_dim = int(action_dim)
        self.action_histograms = np.zeros(
            (num_envs, action_dim),
            dtype=np.int64,
        )
        self.rtg_sum = np.zeros(num_envs, dtype=np.float64)
        self.rtg_square_sum = np.zeros(num_envs, dtype=np.float64)
        self.rtg_min = np.full(num_envs, np.inf, dtype=np.float64)
        self.rtg_max = np.full(num_envs, -np.inf, dtype=np.float64)
        self.entropy_sum = np.zeros(num_envs, dtype=np.float64)
        self.perturb_action_change = np.zeros(num_envs, dtype=np.int64)
        self.perturb_steps = 0
        self.steps = 0

    def record(
        self,
        *,
        actions: np.ndarray,
        probabilities: torch.Tensor,
        predicted_rtgs: torch.Tensor,
        perturbed_logits: Optional[torch.Tensor],
    ) -> None:
        actions = np.asarray(actions, dtype=np.int64)
        probabilities_np = probabilities.detach().float().cpu().numpy()
        rtgs = predicted_rtgs.detach().float().cpu().numpy()
        if actions.shape != (self.num_envs,):
            raise ValueError("Diagnostic action shape mismatch.")
        if probabilities_np.shape != (self.num_envs, self.action_dim):
            raise ValueError("Diagnostic probability shape mismatch.")
        if rtgs.shape != (self.num_envs,):
            raise ValueError("Diagnostic RTG shape mismatch.")
        if not np.isfinite(probabilities_np).all() or not np.isfinite(rtgs).all():
            raise FloatingPointError("Policy diagnostics contain NaN or Inf.")
        np.add.at(
            self.action_histograms,
            (np.arange(self.num_envs), actions),
            1,
        )
        entropy = -np.sum(
            probabilities_np * np.log(np.maximum(probabilities_np, 1e-12)),
            axis=-1,
        )
        self.entropy_sum += entropy
        self.rtg_sum += rtgs
        self.rtg_square_sum += rtgs**2
        self.rtg_min = np.minimum(self.rtg_min, rtgs)
        self.rtg_max = np.maximum(self.rtg_max, rtgs)
        if perturbed_logits is not None:
            perturbed_actions = (
                perturbed_logits.detach().argmax(dim=-1).cpu().numpy()
            )
            if perturbed_actions.shape != (self.num_envs,):
                raise ValueError("Perturbed-logit diagnostic shape mismatch.")
            original_modes = probabilities_np.argmax(axis=-1)
            self.perturb_action_change += perturbed_actions != original_modes
            self.perturb_steps += 1
        self.steps += 1

    def as_dict(self, seeds: Sequence[int]) -> Dict[str, Any]:
        if self.steps < 1:
            raise ValueError("Cannot summarize empty diagnostics.")
        per_seed = []
        for index, seed in enumerate(seeds):
            mean = self.rtg_sum[index] / self.steps
            variance = max(
                0.0,
                self.rtg_square_sum[index] / self.steps - mean**2,
            )
            per_seed.append(
                {
                    "seed": int(seed),
                    "action_histogram": self.action_histograms[index].tolist(),
                    "action_frequencies": (
                        self.action_histograms[index] / self.steps
                    ).tolist(),
                    "predicted_rtg_mean": float(mean),
                    "predicted_rtg_std": float(np.sqrt(variance)),
                    "predicted_rtg_min": float(self.rtg_min[index]),
                    "predicted_rtg_max": float(self.rtg_max[index]),
                    "action_entropy_mean": float(
                        self.entropy_sum[index] / self.steps
                    ),
                    "rtg_plus_025_mode_change_fraction": (
                        float(
                            self.perturb_action_change[index]
                            / self.perturb_steps
                        )
                        if self.perturb_steps
                        else None
                    ),
                }
            )
        histogram = self.action_histograms.sum(axis=0)
        total = self.steps * self.num_envs
        perturb_total = self.perturb_steps * self.num_envs
        return {
            "num_steps_per_seed": self.steps,
            "action_dim": self.action_dim,
            "aggregate": {
                "action_histogram": histogram.tolist(),
                "action_frequencies": (histogram / total).tolist(),
                "predicted_rtg_mean": float(
                    self.rtg_sum.sum() / total
                ),
                "action_entropy_mean": float(
                    self.entropy_sum.sum() / total
                ),
                "rtg_plus_025_mode_change_fraction": (
                    float(self.perturb_action_change.sum() / perturb_total)
                    if perturb_total
                    else None
                ),
                "rtg_plus_025_evaluated_steps_per_seed": self.perturb_steps,
            },
            "per_seed": per_seed,
        }


def main() -> None:
    args = parse_args()
    if args.num_envs < 1 or args.num_steps < 1:
        raise ValueError("--num-envs and --num-steps must be positive.")
    if args.rtg_perturb_diagnostic_every < 1:
        raise ValueError("--rtg-perturb-diagnostic-every must be positive.")
    seed_everything(args.torch_seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint_identity = file_identity(checkpoint_path)
    if (
        args.expected_checkpoint_sha256 is not None
        and checkpoint_identity["sha256"] != args.expected_checkpoint_sha256
    ):
        raise ValueError("Checkpoint SHA256 does not match the expected digest.")
    checkpoint = load_trusted_checkpoint(checkpoint_path)
    result_contract = checkpoint_result_contract(checkpoint)
    model_config = checkpoint["model_config"]
    model = MinigridReinformer(model_config).to(device)
    checkpoint_capacity_audit = checkpoint["contract"].get("capacity_audit")
    if checkpoint.get("format_version") in {
        REINFORMER_TWO_PASS_CHECKPOINT_FORMAT_VERSION,
        REINFORMER_CHECKPOINT_FORMAT_VERSION,
    }:
        recorded_capacity = dict(checkpoint_capacity_audit)
        if (
            checkpoint.get("format_version")
            == REINFORMER_TWO_PASS_CHECKPOINT_FORMAT_VERSION
        ):
            recorded_capacity.setdefault("transformer_passes_per_action", 2)
        if model.capacity_audit() != recorded_capacity:
            raise ValueError(
                "Instantiated matched model differs from the checkpoint capacity audit."
            )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    horizon = int(model_config["horizon"])
    action_dim = int(model_config["action_dim"])
    is_two_pass_v2 = (
        model.action_head_type
        == REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2
    )
    is_single_pass_v3 = (
        model.action_head_type
        == REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3
    )
    resets_predecessor_rtg = is_two_pass_v2 or is_single_pass_v3
    transformer_passes_per_action = 2 if is_two_pass_v2 else 1

    seeds = [args.seed_start + index for index in range(args.num_envs)]
    environments: List[gym.Env] = []
    histories: List[ReinformerOnlineHistory] = []
    episode_returns: List[List[float]] = [[] for _ in seeds]
    current_returns = np.zeros(args.num_envs, dtype=np.float64)
    diagnostics = PolicyDiagnostics(
        num_envs=args.num_envs,
        action_dim=action_dim,
    )
    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(args.torch_seed)
    run = None
    try:
        for seed in seeds:
            environment = ImgObsWrapper(gym.make(args.eval_env))
            observation, _ = environment.reset(seed=seed)
            if environment.action_space.n != action_dim:
                raise ValueError(
                    f"Environment action dim {environment.action_space.n} != {action_dim}."
                )
            environments.append(environment)
            histories.append(ReinformerOnlineHistory(horizon, observation))

        import wandb

        short_env = args.eval_env.replace("MiniGrid-", "").replace("-v0", "")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name
            or f"eval-reinformer-{short_env}-{checkpoint_path.stem}",
            group=args.wandb_group or f"eval-reinformer-{short_env}",
            mode=args.wandb_mode,
            config={
                **vars(args),
                "checkpoint": checkpoint_identity,
                "checkpoint_epoch": checkpoint.get("epoch"),
                "checkpoint_global_update": checkpoint.get("global_update"),
                "checkpoint_git": checkpoint.get("provenance", {}).get("git"),
                **result_contract,
                "model_config": model_config,
                "seeds": seeds,
                "context_reset_on_episode": False,
                "rtg_history_input_reset_on_episode": resets_predecessor_rtg,
                "rtg_repredicted_each_step": True,
                "transformer_passes_per_action": transformer_passes_per_action,
            },
            tags=["reinformer", "minigrid", "evaluation", "cross-episode"],
        )

        for step in tqdm(range(args.num_steps), desc="Reinformer evaluation"):
            batch = stack_histories(histories, device)
            with torch.no_grad():
                hidden = model.encode_context(batch)
                rtg_predictions = model.predict_rtg(hidden)
                final_rtgs = rtg_predictions[:, -1]
                current_action_rtgs = batch["context_rtgs"].clone()
                current_action_rtgs[:, -1] = final_rtgs[:, 0]
                final_logits = model.action_logits_for_rtgs(
                    batch,
                    current_action_rtgs,
                    prediction_hidden=hidden,
                )[:, -1]
                perturbed_logits = None
                if step % args.rtg_perturb_diagnostic_every == 0:
                    perturbed_action_rtgs = current_action_rtgs.clone()
                    perturbed_action_rtgs[:, -1] += 0.25
                    perturbed_logits = model.action_logits_for_rtgs(
                        batch,
                        perturbed_action_rtgs,
                        prediction_hidden=hidden,
                    )[:, -1]
                if not torch.isfinite(final_logits).all() or not torch.isfinite(
                    final_rtgs
                ).all():
                    raise FloatingPointError(
                        "Evaluation action logits or RTG predictions are non-finite."
                    )
                probabilities = torch.softmax(final_logits, dim=-1)
                if args.action_selection == "sample":
                    actions = torch.multinomial(
                        probabilities.detach().float().cpu(),
                        num_samples=1,
                        generator=sampling_generator,
                    ).squeeze(1).numpy()
                else:
                    actions = final_logits.argmax(dim=-1).cpu().numpy()
                diagnostics.record(
                    actions=actions,
                    probabilities=probabilities,
                    predicted_rtgs=final_rtgs[:, 0],
                    perturbed_logits=perturbed_logits,
                )

            for index, (environment, action) in enumerate(
                zip(environments, actions)
            ):
                next_observation, reward, terminated, truncated, _ = environment.step(
                    int(action)
                )
                done = bool(terminated or truncated)
                current_returns[index] += float(reward)
                if done:
                    episode_returns[index].append(float(current_returns[index]))
                    current_returns[index] = 0.0
                    next_observation, _ = environment.reset()
                histories[index].append(
                    action=int(action),
                    reward=float(reward),
                    done=done,
                    predicted_rtg=float(final_rtgs[index, 0].item()),
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
                        "rollout/predicted_rtg_mean": float(
                            final_rtgs.mean().item()
                        ),
                    },
                    step=step + 1,
                )

        metrics = aggregate_seed_metrics(episode_returns)
        diagnostic_result = diagnostics.as_dict(seeds)
        result = {
            "algorithm": "Reinformer-MiniGrid",
            "adaptation_status": (
                "independent_discrete_minigrid_two_pass_linear_adaptation"
                if is_two_pass_v2
                else (
                    "independent_discrete_minigrid_single_pass_linear_adaptation"
                    if is_single_pass_v3
                    else "legacy_independent_discrete_minigrid_adaptation"
                )
            ),
            "action_head_type": model.action_head_type,
            "capacity_audit": model.capacity_audit(),
            "eval_env": args.eval_env,
            "checkpoint": checkpoint_identity,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_global_update": checkpoint.get("global_update"),
            "checkpoint_provenance": checkpoint.get("provenance"),
            **result_contract,
            "num_steps_per_seed": args.num_steps,
            "seeds": seeds,
            "action_selection_mode": args.action_selection,
            "context_reset_on_episode": False,
            "rtg_history_input_reset_on_episode": resets_predecessor_rtg,
            "rtg_repredicted_each_step": True,
            "transformer_passes_per_action": transformer_passes_per_action,
            "policy_diagnostics": diagnostic_result,
            "created_at_unix": time.time(),
            **metrics,
            "wandb_run_id": run.id,
        }
        output_path = Path(args.output_json).expanduser().resolve()
        atomic_json_dump(result, output_path)
        run.summary.update(
            {
                **metrics["aggregate"],
                **{
                    f"diagnostics/{key}": value
                    for key, value in diagnostic_result["aggregate"].items()
                },
                "status": "completed",
                "result_json": str(output_path),
            }
        )
    finally:
        for environment in environments:
            environment.close()
        if run:
            run.finish()


if __name__ == "__main__":
    main()

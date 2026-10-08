"""Pure utilities for the MiniGrid CV-ICRL E1-lite evaluation suite.

This module intentionally has no torch, gym, or W&B dependency so that the
intervention and metric contracts can be unit-tested without the full training
environment.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


E1_CONTRACT_VERSION = "minigrid_cv_icrl_e1_lite_v1"
E1_RESULT_FORMAT_VERSION = 1
E1_ARCHIVED_EVALUATOR_COMMIT = "e5dbb9f44731627ef37488227bf9ecc8254284da"

E1_CONDITIONS = (
    "normal",
    "zero",
    "frozen",
    "shuffle",
    "no_history",
    "noise",
    "short_context",
)


def _splitmix64(values: np.ndarray) -> np.ndarray:
    mixed = np.asarray(values, dtype=np.uint64).copy()
    mixed ^= mixed >> np.uint64(30)
    mixed *= np.uint64(0xBF58476D1CE4E5B9)
    mixed ^= mixed >> np.uint64(27)
    mixed *= np.uint64(0x94D049BB133111EB)
    mixed ^= mixed >> np.uint64(31)
    return mixed


def validate_condition(condition: str) -> str:
    if condition not in E1_CONDITIONS:
        raise ValueError(
            f"Unsupported E1 condition {condition!r}; expected one of "
            f"{E1_CONDITIONS!r}."
        )
    return condition


@dataclass
class ScalarWritebackState:
    """Track the archived running-max write-back under one intervention.

    ``raw_running_max`` reproduces the archived evaluator's
    ``np.maximum(previous, predicted_value)`` state.  Zero and frozen modify
    the scalar written into the next context; shuffle/noise/window conditions
    modify how the realized stored history is read.
    """

    condition: str
    num_steps: int
    raw_running_max: float = 0.0
    frozen_value: Optional[float] = None

    def __post_init__(self) -> None:
        validate_condition(self.condition)
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive.")

    def update(self, predicted_value: float, step: int) -> Tuple[float, float]:
        if not math.isfinite(predicted_value):
            raise FloatingPointError("Predicted value is NaN or Inf.")
        if step < 0 or step >= self.num_steps:
            raise ValueError(
                f"step {step} is outside [0, {self.num_steps - 1}]."
            )
        self.raw_running_max = max(
            float(self.raw_running_max),
            float(predicted_value),
        )
        if self.frozen_value is None:
            self.frozen_value = float(self.raw_running_max)

        if self.condition == "zero":
            written = 0.0
        elif self.condition == "frozen":
            written = float(self.frozen_value)
        else:
            written = float(self.raw_running_max)

        if not math.isfinite(written):
            raise FloatingPointError("Written context scalar is NaN or Inf.")
        return float(self.raw_running_max), written


def _validate_history_arrays(
    states: np.ndarray,
    actions: np.ndarray,
    tokens: np.ndarray,
    transition_ids: np.ndarray,
) -> None:
    if states.ndim != 4:
        raise ValueError(
            f"states must have shape [T,C,H,W], got {states.shape}."
        )
    if (
        actions.ndim != 1
        or tokens.ndim != 1
        or transition_ids.ndim != 1
    ):
        raise ValueError(
            "actions, tokens, and transition_ids must be one-dimensional."
        )
    expected_transitions = states.shape[0] - 1
    if actions.shape[0] != expected_transitions:
        raise ValueError(
            "actions must have exactly one fewer entry than states."
        )
    if tokens.shape[0] != expected_transitions:
        raise ValueError(
            "tokens must have exactly one fewer entry than states."
        )
    if transition_ids.shape[0] != expected_transitions:
        raise ValueError(
            "transition_ids must have exactly one fewer entry than states."
        )
    if transition_ids.size:
        if np.any(transition_ids < 0):
            raise ValueError("transition_ids must be non-negative.")
        if np.any(np.diff(transition_ids) <= 0):
            raise ValueError("transition_ids must be strictly increasing.")
    if states.shape[0] < 1:
        raise ValueError("history must contain at least one state.")
    if not np.isfinite(states).all() or not np.isfinite(tokens).all():
        raise FloatingPointError("History contains NaN or Inf.")


def context_arrays(
    *,
    states: np.ndarray,
    actions: np.ndarray,
    tokens: np.ndarray,
    transition_ids: np.ndarray,
    condition: str,
    model_horizon: int,
    short_context_length: int,
    stream_index: int,
    global_step: int,
    shuffle_seed: int,
    noise_seed: int,
    noise_sigma: float,
    frozen_value: Optional[float],
) -> Dict[str, np.ndarray]:
    """Build the exact archived model input under an E1 intervention.

    The returned action/token arrays include the archived evaluator's leading
    zero.  ``MinigridMultiheadTransformer.forward`` applies its own additional
    causal shift, and this function deliberately preserves that behavior.

    The shuffle is causal: it permutes only scalar tokens already present in
    the current history.  Its seed is a deterministic function of the
    experiment seed, stream index, and current global step.
    """

    validate_condition(condition)
    _validate_history_arrays(states, actions, tokens, transition_ids)
    if model_horizon < 2:
        raise ValueError("model_horizon must be at least two.")
    if short_context_length < 2 or short_context_length > model_horizon:
        raise ValueError(
            "short_context_length must be in [2, model_horizon]."
        )
    if states.shape[0] > model_horizon:
        raise ValueError("history is longer than the model horizon.")
    if stream_index < 0 or global_step < 0:
        raise ValueError("stream_index and global_step must be non-negative.")
    if not math.isfinite(noise_sigma) or noise_sigma < 0.0:
        raise ValueError("noise_sigma must be finite and non-negative.")

    selected_states = np.asarray(states)
    selected_actions = np.asarray(actions)
    selected_tokens = np.asarray(tokens, dtype=np.float32)
    selected_transition_ids = np.asarray(transition_ids, dtype=np.int64)

    if condition == "no_history":
        selected_states = selected_states[-1:]
        selected_actions = selected_actions[:0]
        selected_tokens = selected_tokens[:0]
        selected_transition_ids = selected_transition_ids[:0]
    elif condition == "short_context":
        keep_states = min(short_context_length, selected_states.shape[0])
        selected_states = selected_states[-keep_states:]
        keep_transitions = keep_states - 1
        selected_actions = selected_actions[-keep_transitions:]
        selected_tokens = selected_tokens[-keep_transitions:]
        selected_transition_ids = selected_transition_ids[-keep_transitions:]

    identity_tokens = selected_tokens.astype(np.float32, copy=True)
    if condition == "zero":
        selected_tokens = np.zeros_like(selected_tokens, dtype=np.float32)
    elif condition == "frozen":
        replacement = 0.0 if frozen_value is None else float(frozen_value)
        if not math.isfinite(replacement):
            raise FloatingPointError("frozen_value is NaN or Inf.")
        selected_tokens = np.full_like(
            selected_tokens,
            replacement,
            dtype=np.float32,
        )
    elif condition == "shuffle" and selected_tokens.size > 1:
        # Assign every transition a deterministic priority that does not
        # change when the prefix grows.  This preserves the relative order of
        # previously observed shuffled tokens while remaining causal and
        # independent of the action-sampling RNG.
        stream_salt = np.uint64(
            ((stream_index + 1) * 0x9E3779B185EBCA87) & ((1 << 64) - 1)
        )
        priority_input = (
            selected_transition_ids.astype(np.uint64)
            + np.uint64(int(shuffle_seed) % (2**32))
            + stream_salt
        )
        priorities = _splitmix64(priority_input)
        permutation = np.argsort(priorities, kind="stable")
        selected_tokens = selected_tokens[permutation]
    elif condition == "noise" and selected_tokens.size:
        stream_salt = np.uint64(
            ((stream_index + 1) * 0xD1B54A32D192ED03)
            & ((1 << 64) - 1)
        )
        noise_key = (
            selected_transition_ids.astype(np.uint64)
            + np.uint64(int(noise_seed) % (2**32))
            + stream_salt
        )
        hash_one = _splitmix64(noise_key)
        hash_two = _splitmix64(
            noise_key + np.uint64(0x9E3779B97F4A7C15)
        )
        uniform_one = (
            (hash_one >> np.uint64(11)).astype(np.float64) + 0.5
        ) / float(1 << 53)
        uniform_two = (
            (hash_two >> np.uint64(11)).astype(np.float64) + 0.5
        ) / float(1 << 53)
        standard_normal = np.sqrt(-2.0 * np.log(uniform_one)) * np.cos(
            2.0 * np.pi * uniform_two
        )
        selected_tokens = (
            selected_tokens.astype(np.float32, copy=False)
            + (noise_sigma * standard_normal).astype(np.float32)
        )

    sequence_length = selected_states.shape[0]
    context_actions = np.zeros(sequence_length, dtype=np.int64)
    context_tokens = np.zeros(sequence_length, dtype=np.float32)
    if sequence_length > 1:
        context_actions[1:] = selected_actions.astype(np.int64, copy=False)
        context_tokens[1:] = selected_tokens.astype(np.float32, copy=False)

    if not np.isfinite(context_tokens).all():
        raise FloatingPointError("Transformed context token is NaN or Inf.")
    # The archived evaluator stages one leading-zero-aligned token sequence
    # and the model shifts it a second time.  Consequently, the newest staged
    # token is not consumed by the current action prediction.  Report
    # intervention strength over the actually consumed real-token subset.
    effective_tokens = selected_tokens[:-1]
    effective_identity_tokens = identity_tokens[:-1]
    if effective_tokens.size:
        token_delta = (
            effective_tokens.astype(np.float64)
            - effective_identity_tokens.astype(np.float64)
        )
        intervention_rms = float(np.sqrt(np.mean(token_delta**2)))
        intervention_changed_fraction = float(
            np.mean(np.abs(token_delta) > 1e-12)
        )
        fed_token_unique_count = int(np.unique(effective_tokens).size)
        fed_token_mean = float(np.mean(effective_tokens, dtype=np.float64))
        fed_token_std = float(np.std(effective_tokens, dtype=np.float64))
        fed_last_token = float(effective_tokens[-1])
    else:
        intervention_rms = 0.0
        intervention_changed_fraction = 0.0
        fed_token_unique_count = 0
        fed_token_mean = 0.0
        fed_token_std = 0.0
        fed_last_token = 0.0
    if selected_tokens.size:
        staged_token_mean = float(
            np.mean(selected_tokens, dtype=np.float64)
        )
        staged_token_std = float(np.std(selected_tokens, dtype=np.float64))
        staged_last_token = float(selected_tokens[-1])
    else:
        staged_token_mean = 0.0
        staged_token_std = 0.0
        staged_last_token = 0.0
    return {
        "context_states": np.ascontiguousarray(
            selected_states.astype(np.float32, copy=False)
        ),
        "context_actions": context_actions,
        "context_rewards": context_tokens[:, None],
        "intervention_token_rms": np.asarray(
            intervention_rms,
            dtype=np.float32,
        ),
        "intervention_changed_fraction": np.asarray(
            intervention_changed_fraction,
            dtype=np.float32,
        ),
        "fed_token_unique_count": np.asarray(
            fed_token_unique_count,
            dtype=np.int32,
        ),
        "fed_token_mean": np.asarray(fed_token_mean, dtype=np.float32),
        "fed_token_std": np.asarray(fed_token_std, dtype=np.float32),
        "fed_last_token": np.asarray(fed_last_token, dtype=np.float32),
        "staged_token_mean": np.asarray(
            staged_token_mean,
            dtype=np.float32,
        ),
        "staged_token_std": np.asarray(
            staged_token_std,
            dtype=np.float32,
        ),
        "staged_last_token": np.asarray(
            staged_last_token,
            dtype=np.float32,
        ),
        "sequence_length": np.asarray(sequence_length, dtype=np.int32),
    }


def stable_array_bundle_sha256(
    arrays: Mapping[str, np.ndarray],
    *,
    episode_records: Optional[
        Sequence[Sequence[Mapping[str, float]]]
    ] = None,
) -> str:
    """Hash core rollout arrays independently of optional probe artifacts."""

    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode(
                "ascii"
            )
        )
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        digest.update(b"\0")
    if episode_records is not None:
        canonical_records = json.dumps(
            episode_records,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(b"episode_records\0")
        digest.update(canonical_records)
    return digest.hexdigest()


def stream_episode_metrics(episode_returns: Sequence[float]) -> Dict[str, Any]:
    values = np.asarray(list(episode_returns), dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("episode_returns must be one-dimensional.")
    if not np.isfinite(values).all():
        raise FloatingPointError("episode_returns contains NaN or Inf.")
    if values.size == 0:
        return {
            "episodes": 0,
            "aer": None,
            "ler": None,
            "if": None,
            "success_rate": None,
            "episode_returns": [],
        }
    instability_count = (
        int(np.sum(values[1:] <= 0.95 * values[:-1]))
        if values.size > 1
        else 0
    )
    return {
        "episodes": int(values.size),
        "aer": float(values.mean()),
        "ler": float(values[-1]),
        "if": float(instability_count / values.size),
        "success_rate": float(np.mean(values > 0.0)),
        "episode_returns": values.tolist(),
    }


def _population_summary(
    values: Sequence[Optional[float]],
) -> Dict[str, Any]:
    valid = np.asarray(
        [float(value) for value in values if value is not None],
        dtype=np.float64,
    )
    if valid.size == 0:
        return {"mean": None, "std": None, "valid_streams": 0}
    if not np.isfinite(valid).all():
        raise FloatingPointError("Metric summary contains NaN or Inf.")
    return {
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=0)),
        "valid_streams": int(valid.size),
    }


def aggregate_episode_metrics(
    episode_returns: Sequence[Sequence[float]],
    *,
    seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    if seeds is not None and len(seeds) != len(episode_returns):
        raise ValueError("seeds and episode_returns must have equal length.")
    per_stream: List[Dict[str, Any]] = []
    for index, returns in enumerate(episode_returns):
        row = stream_episode_metrics(returns)
        row["stream_index"] = index
        if seeds is not None:
            row["seed"] = int(seeds[index])
        per_stream.append(row)

    aggregate: Dict[str, Any] = {}
    for metric in ("aer", "ler", "if", "success_rate"):
        summary = _population_summary([row[metric] for row in per_stream])
        aggregate[f"{metric}_mean"] = summary["mean"]
        aggregate[f"{metric}_std"] = summary["std"]
        aggregate[f"{metric}_valid_streams"] = summary["valid_streams"]
    episode_summary = _population_summary(
        [float(row["episodes"]) for row in per_stream]
    )
    aggregate["episodes_mean"] = episode_summary["mean"]
    aggregate["episodes_std"] = episode_summary["std"]
    aggregate["no_completed_episode_streams"] = int(
        sum(row["episodes"] == 0 for row in per_stream)
    )
    if aggregate["if_mean"] is not None:
        aggregate["if_percent_mean"] = 100.0 * aggregate["if_mean"]
        aggregate["if_percent_std"] = 100.0 * aggregate["if_std"]
    else:
        aggregate["if_percent_mean"] = None
        aggregate["if_percent_std"] = None
    return {"aggregate": aggregate, "per_stream": per_stream}


def _aggregate_stage_episode_metrics(
    episode_records: Sequence[Sequence[Dict[str, float]]],
    *,
    lower_exclusive: int,
    upper_inclusive: int,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    """Aggregate one completion-step stage with boundary-aware IF.

    An instability event is assigned to the completion stage of the later
    episode.  Thus a drop from the last episode of the previous stage to the
    first episode of this stage is retained rather than silently discarded.
    """

    per_stream: List[Dict[str, Any]] = []
    for stream_index, records in enumerate(episode_records):
        all_returns = [float(record["return"]) for record in records]
        selected_indices = [
            index
            for index, record in enumerate(records)
            if lower_exclusive
            < int(record["completion_step"])
            <= upper_inclusive
        ]
        selected_returns = np.asarray(
            [all_returns[index] for index in selected_indices],
            dtype=np.float64,
        )
        if not np.isfinite(selected_returns).all():
            raise FloatingPointError(
                "Stage episode returns contain NaN or Inf."
            )
        if selected_returns.size == 0:
            row: Dict[str, Any] = {
                "episodes": 0,
                "aer": None,
                "ler": None,
                "if": None,
                "success_rate": None,
                "episode_returns": [],
                "instability_events": 0,
            }
        else:
            instability_events = sum(
                index > 0
                and all_returns[index] <= 0.95 * all_returns[index - 1]
                for index in selected_indices
            )
            row = {
                "episodes": int(selected_returns.size),
                "aer": float(selected_returns.mean()),
                "ler": float(selected_returns[-1]),
                "if": float(
                    instability_events / selected_returns.size
                ),
                "success_rate": float(
                    np.mean(selected_returns > 0.0)
                ),
                "episode_returns": selected_returns.tolist(),
                "instability_events": int(instability_events),
            }
        row["stream_index"] = stream_index
        row["seed"] = int(seeds[stream_index])
        per_stream.append(row)

    aggregate: Dict[str, Any] = {}
    for metric in ("aer", "ler", "if", "success_rate"):
        summary = _population_summary([row[metric] for row in per_stream])
        aggregate[f"{metric}_mean"] = summary["mean"]
        aggregate[f"{metric}_std"] = summary["std"]
        aggregate[f"{metric}_valid_streams"] = summary["valid_streams"]
    episode_summary = _population_summary(
        [float(row["episodes"]) for row in per_stream]
    )
    aggregate["episodes_mean"] = episode_summary["mean"]
    aggregate["episodes_std"] = episode_summary["std"]
    aggregate["no_completed_episode_streams"] = int(
        sum(row["episodes"] == 0 for row in per_stream)
    )
    if aggregate["if_mean"] is not None:
        aggregate["if_percent_mean"] = 100.0 * aggregate["if_mean"]
        aggregate["if_percent_std"] = 100.0 * aggregate["if_std"]
    else:
        aggregate["if_percent_mean"] = None
        aggregate["if_percent_std"] = None
    return {"aggregate": aggregate, "per_stream": per_stream}


def horizon_metric_slices(
    episode_records: Sequence[Sequence[Dict[str, float]]],
    *,
    cutoffs: Sequence[int],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    """Compute cumulative-prefix and disjoint-stage metrics.

    Each episode record must contain ``return`` and one-based
    ``completion_step`` fields.
    """

    if len(episode_records) != len(seeds):
        raise ValueError("episode_records and seeds must have equal length.")
    normalized_cutoffs = [int(value) for value in cutoffs]
    if (
        not normalized_cutoffs
        or any(value < 1 for value in normalized_cutoffs)
        or normalized_cutoffs != sorted(set(normalized_cutoffs))
    ):
        raise ValueError("cutoffs must be unique, sorted positive integers.")

    prefix: Dict[str, Any] = {}
    stages: Dict[str, Any] = {}
    previous = 0
    for cutoff in normalized_cutoffs:
        prefix_returns: List[List[float]] = []
        for stream_records in episode_records:
            current_prefix = []
            for record in stream_records:
                completion_step = int(record["completion_step"])
                episode_return = float(record["return"])
                if not math.isfinite(episode_return):
                    raise FloatingPointError(
                        "episode record contains NaN or Inf."
                    )
                if completion_step <= cutoff:
                    current_prefix.append(episode_return)
            prefix_returns.append(current_prefix)
        prefix[str(cutoff)] = aggregate_episode_metrics(
            prefix_returns,
            seeds=seeds,
        )
        stages[f"{previous + 1}-{cutoff}"] = (
            _aggregate_stage_episode_metrics(
                episode_records,
                lower_exclusive=previous,
                upper_inclusive=cutoff,
                seeds=seeds,
            )
        )
        previous = cutoff
    return {
        "prefix": prefix,
        "stages": stages,
        "stage_if_assignment": (
            "instability drop is assigned to the completion stage of the "
            "later episode"
        ),
    }


def linear_drift_slope(values: np.ndarray) -> np.ndarray:
    """Return per-stream slope against normalized rollout time."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError("values must have shape [steps, streams], steps >= 2.")
    if not np.isfinite(array).all():
        raise FloatingPointError("drift values contain NaN or Inf.")
    x = np.linspace(0.0, 1.0, num=array.shape[0], dtype=np.float64)
    centered_x = x - x.mean()
    denominator = float(np.sum(centered_x**2))
    centered_y = array - array.mean(axis=0, keepdims=True)
    return np.sum(centered_x[:, None] * centered_y, axis=0) / denominator

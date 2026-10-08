"""Episode-aware MiniGrid histories for the Reinformer adaptation.

The legacy ``*-rtg.pkl`` files in this repository do not contain audited
return-to-go targets.  This loader authenticates the original AD trajectory
pickle and recomputes undiscounted (or explicitly discounted) return-to-go
inside each episode while preserving the collector's cross-row stream layout.
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from minigrid_ic_cql_dataset import (
    _all_finite_in_chunks,
    _as_bt_array,
    _file_sha256,
    _initial_stream_predecessors,
    _positive_int,
)


_REQUIRED_KEYS = ("observations", "actions", "rewards", "dones")


def episode_aware_return_to_go(
    rewards: np.ndarray,
    dones: np.ndarray,
    *,
    histories_per_stream: int,
    gamma: float = 1.0,
) -> np.ndarray:
    """Compute RTG without crossing episode or independent-stream boundaries.

    ``rewards`` and ``dones`` use the collector layout ``[rows, T]``.  Adjacent
    groups of ``histories_per_stream`` rows are contiguous chunks from one
    environment stream.  A terminal transition contributes its own reward and
    prevents all later rewards from entering its target.
    """

    rewards = np.asarray(rewards)
    dones = np.asarray(dones)
    histories_per_stream = _positive_int(
        histories_per_stream,
        name="histories_per_stream",
    )
    gamma = float(gamma)
    if rewards.ndim != 2 or dones.ndim != 2:
        raise ValueError("rewards and dones must both have shape [histories, sequence].")
    if rewards.shape != dones.shape:
        raise ValueError(
            f"rewards shape {rewards.shape} does not match dones shape {dones.shape}."
        )
    if rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError("rewards and dones must contain at least one token.")
    if rewards.shape[0] % histories_per_stream != 0:
        raise ValueError(
            f"{rewards.shape[0]} histories are not divisible by "
            f"histories_per_stream={histories_per_stream}."
        )
    if not np.issubdtype(rewards.dtype, np.number) or not np.isfinite(rewards).all():
        raise ValueError("rewards must contain only finite numeric values.")
    if not np.isin(dones, (0, 1)).all():
        raise ValueError("dones must contain only 0/1 values.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")

    stream_rewards = np.ascontiguousarray(rewards, dtype=np.float32).reshape(
        rewards.shape[0] // histories_per_stream,
        histories_per_stream * rewards.shape[1],
    )
    stream_dones = np.ascontiguousarray(dones, dtype=np.bool_).reshape(
        stream_rewards.shape
    )
    stream_rtgs = np.empty_like(stream_rewards, dtype=np.float32)
    running = np.zeros(stream_rewards.shape[0], dtype=np.float32)
    for time_index in range(stream_rewards.shape[1] - 1, -1, -1):
        running = stream_rewards[:, time_index] + (
            gamma * running * (~stream_dones[:, time_index])
        )
        stream_rtgs[:, time_index] = running
    if not np.isfinite(stream_rtgs).all():
        raise FloatingPointError("Episode-aware return-to-go contains NaN or Inf.")
    return np.ascontiguousarray(stream_rtgs.reshape(rewards.shape))


def complete_episode_mask(
    dones: np.ndarray,
    *,
    histories_per_stream: int,
) -> np.ndarray:
    """Mark tokens whose episode terminates inside the recorded stream.

    A stream can stop midway through its final episode.  Its last terminal
    transition and every earlier episode are valid, while the suffix after the
    final ``done`` has a right-truncated RTG target and must not supervise either
    the expectile head or the RTG-conditioned action head.
    """

    dones = np.asarray(dones)
    histories_per_stream = _positive_int(
        histories_per_stream,
        name="histories_per_stream",
    )
    if dones.ndim != 2 or dones.shape[0] % histories_per_stream != 0:
        raise ValueError(
            "dones must have shape [histories, sequence] with complete stream groups."
        )
    if not np.isin(dones, (0, 1)).all():
        raise ValueError("dones must contain only 0/1 values.")
    stream_dones = np.ascontiguousarray(dones, dtype=np.bool_).reshape(
        dones.shape[0] // histories_per_stream,
        histories_per_stream * dones.shape[1],
    )
    mask = np.zeros_like(stream_dones, dtype=np.bool_)
    for stream_index, terminal_flags in enumerate(stream_dones):
        terminal_indices = np.flatnonzero(terminal_flags)
        if terminal_indices.size:
            mask[stream_index, : terminal_indices[-1] + 1] = True
    return np.ascontiguousarray(mask.reshape(dones.shape))


def _initial_previous_rtgs(
    rtgs: np.ndarray,
    *,
    histories_per_stream: int,
) -> np.ndarray:
    histories_per_stream = _positive_int(
        histories_per_stream,
        name="histories_per_stream",
    )
    if rtgs.ndim != 2 or rtgs.shape[0] % histories_per_stream != 0:
        raise ValueError(
            "rtgs must have shape [histories, sequence] with complete stream groups."
        )
    initial = np.zeros(rtgs.shape[0], dtype=np.float32)
    valid = np.arange(rtgs.shape[0]) % histories_per_stream != 0
    rows = np.flatnonzero(valid)
    initial[rows] = rtgs[rows - 1, -1]
    return initial


class MinigridReinformerDataset(torch.utils.data.Dataset):
    """Strict original-corpus loader with freshly generated episode RTG."""

    def __init__(
        self,
        path: str | Path,
        *,
        histories_per_stream: int,
        rtg_gamma: float = 1.0,
        max_histories: Optional[int] = None,
        compute_sha256: bool = True,
        expected_sha256: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Reinformer dataset does not exist: {self.path}")
        histories_per_stream = _positive_int(
            histories_per_stream,
            name="histories_per_stream",
        )
        if not isinstance(compute_sha256, bool):
            raise TypeError("compute_sha256 must be a bool.")
        if expected_sha256 is not None:
            if (
                not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha256)
            ):
                raise ValueError("expected_sha256 must be a lowercase 64-character hex digest.")
            if not compute_sha256:
                raise ValueError("expected_sha256 requires compute_sha256=True.")
        if max_histories is not None:
            if isinstance(max_histories, bool) or not isinstance(max_histories, int):
                raise TypeError("max_histories must be an integer or None.")
            if max_histories <= 0:
                raise ValueError("max_histories must be positive when provided.")

        stat = self.path.stat()
        dataset_sha256 = _file_sha256(self.path) if compute_sha256 else None
        if expected_sha256 is not None and dataset_sha256 != expected_sha256:
            raise ValueError(
                f"Dataset SHA256 mismatch for {self.path}: "
                f"{dataset_sha256} != {expected_sha256}."
            )
        with self.path.open("rb") as file_handle:
            payload = pickle.load(file_handle)
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"Reinformer dataset must be a mapping, got {type(payload).__name__}."
            )
        missing_keys = [key for key in _REQUIRED_KEYS if key not in payload]
        if missing_keys:
            raise KeyError(
                "Reinformer dataset is missing required keys: "
                + ", ".join(missing_keys)
            )

        observations = np.asarray(payload["observations"])
        if observations.dtype == object or not np.issubdtype(
            observations.dtype,
            np.number,
        ):
            raise ValueError("observations must be a dense numeric array.")
        if observations.ndim != 5 or tuple(observations.shape[2:]) != (3, 7, 7):
            raise ValueError(
                "observations must have shape [histories, sequence, 3, 7, 7], "
                f"got {observations.shape}."
            )
        if observations.shape[0] == 0 or observations.shape[1] == 0:
            raise ValueError("observations must contain at least one history and token.")
        if np.issubdtype(observations.dtype, np.floating) and not _all_finite_in_chunks(
            observations
        ):
            raise ValueError("observations must contain only finite values.")

        actions = _as_bt_array(payload["actions"], name="actions")
        rewards = _as_bt_array(payload["rewards"], name="rewards")
        dones = _as_bt_array(payload["dones"], name="dones")
        expected_shape = observations.shape[:2]
        for name, array in (
            ("actions", actions),
            ("rewards", rewards),
            ("dones", dones),
        ):
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} shape {array.shape} does not match {expected_shape}."
                )
        if not np.issubdtype(actions.dtype, np.integer):
            if not (
                np.issubdtype(actions.dtype, np.floating)
                and np.isfinite(actions).all()
                and np.equal(actions, np.floor(actions)).all()
            ):
                raise ValueError("actions must contain finite integer indices.")
        if not np.issubdtype(rewards.dtype, np.number) or not np.isfinite(rewards).all():
            raise ValueError("rewards must contain only finite numeric values.")
        if not np.isin(dones, (0, 1)).all():
            raise ValueError("dones must contain only 0/1 values.")

        payload_layout = payload.get("histories_per_stream")
        if payload_layout is not None:
            payload_layout = _positive_int(
                payload_layout,
                name="payload histories_per_stream",
            )
            if payload_layout != histories_per_stream:
                raise ValueError(
                    "Configured histories_per_stream does not match the payload: "
                    f"{histories_per_stream} != {payload_layout}."
                )
            layout_source = "payload"
        else:
            layout_source = "explicit_legacy_contract"

        stored_histories = int(observations.shape[0])
        if stored_histories % histories_per_stream != 0:
            raise ValueError(
                f"Stored histories {stored_histories} are not divisible by "
                f"histories_per_stream={histories_per_stream}."
            )
        selected_histories = (
            stored_histories if max_histories is None else max_histories
        )
        if selected_histories > stored_histories:
            raise ValueError(
                f"max_histories={selected_histories} exceeds stored histories "
                f"{stored_histories}."
            )

        # A smoke prefix may end in the middle of a stream.  Load RTG through the
        # end of that stream before selecting the requested rows so the final
        # selected row is not treated as an artificial terminal boundary.
        complete_histories = min(
            stored_histories,
            math.ceil(selected_histories / histories_per_stream)
            * histories_per_stream,
        )
        actions_for_rtg = np.ascontiguousarray(
            actions[:complete_histories],
            dtype=np.int64,
        )
        rewards_for_rtg = np.ascontiguousarray(
            rewards[:complete_histories],
            dtype=np.float32,
        )
        dones_for_rtg = np.ascontiguousarray(
            dones[:complete_histories],
            dtype=np.bool_,
        )
        rtgs_for_rtg = episode_aware_return_to_go(
            rewards_for_rtg,
            dones_for_rtg,
            histories_per_stream=histories_per_stream,
            gamma=rtg_gamma,
        )
        valid_mask_for_rtg = complete_episode_mask(
            dones_for_rtg,
            histories_per_stream=histories_per_stream,
        )
        predecessors = _initial_stream_predecessors(
            actions_for_rtg,
            rewards_for_rtg,
            dones_for_rtg,
            histories_per_stream=histories_per_stream,
        )
        initial_previous_rtg = _initial_previous_rtgs(
            rtgs_for_rtg,
            histories_per_stream=histories_per_stream,
        )
        selection = slice(0, selected_histories)

        self.context_states = torch.from_numpy(
            np.ascontiguousarray(observations[selection])
        )
        self.context_actions = torch.from_numpy(
            np.ascontiguousarray(actions_for_rtg[selection], dtype=np.int64)
        )
        self.context_rewards = torch.from_numpy(
            np.ascontiguousarray(rewards_for_rtg[selection], dtype=np.float32)
        )
        self.context_dones = torch.from_numpy(
            np.ascontiguousarray(dones_for_rtg[selection], dtype=np.bool_)
        )
        self.context_rtgs = torch.from_numpy(
            np.ascontiguousarray(rtgs_for_rtg[selection], dtype=np.float32)
        )
        self.rtg_valid_mask = torch.from_numpy(
            np.ascontiguousarray(valid_mask_for_rtg[selection], dtype=np.bool_)
        )
        self.initial_previous_action = torch.from_numpy(
            np.ascontiguousarray(
                predecessors["initial_previous_action"][selection],
                dtype=np.int64,
            )
        )
        self.initial_previous_reward = torch.from_numpy(
            np.ascontiguousarray(
                predecessors["initial_previous_reward"][selection],
                dtype=np.float32,
            )
        )
        self.initial_previous_done = torch.from_numpy(
            np.ascontiguousarray(
                predecessors["initial_previous_done"][selection],
                dtype=np.bool_,
            )
        )
        self.initial_previous_valid = torch.from_numpy(
            np.ascontiguousarray(
                predecessors["initial_previous_valid"][selection],
                dtype=np.bool_,
            )
        )
        self.initial_previous_rtg = torch.from_numpy(
            np.ascontiguousarray(
                initial_previous_rtg[selection],
                dtype=np.float32,
            )
        )
        valid_rtg_count = int(self.rtg_valid_mask.sum().item())
        if valid_rtg_count < 1:
            raise ValueError("Reinformer dataset has no valid RTG supervision tokens.")
        valid_rtg_abs_sum = 0.0
        chunk_rows = 1024
        for row_start in range(0, selected_histories, chunk_rows):
            row_stop = min(selected_histories, row_start + chunk_rows)
            rtg_chunk = rtgs_for_rtg[row_start:row_stop]
            mask_chunk = valid_mask_for_rtg[row_start:row_stop]
            valid_rtg_abs_sum += float(
                np.abs(rtg_chunk[mask_chunk]).sum(dtype=np.float64)
            )
        valid_rtg_abs_mean = valid_rtg_abs_sum / valid_rtg_count
        if not math.isfinite(valid_rtg_abs_mean) or valid_rtg_abs_mean < 0.0:
            raise FloatingPointError(
                "Dataset-level valid RTG absolute mean is invalid."
            )
        self.sequence_length = int(observations.shape[1])
        self.fingerprint: Dict[str, Any] = {
            "path": str(self.path),
            "size_bytes": int(stat.st_size),
            "sha256": dataset_sha256,
            "expected_sha256": expected_sha256,
            "sha256_verified": bool(
                expected_sha256 is not None and dataset_sha256 == expected_sha256
            ),
            "stored_histories": stored_histories,
            "selected_histories": int(selected_histories),
            "complete_histories_used_for_rtg": int(complete_histories),
            "sequence_length": self.sequence_length,
            "observation_shape": [3, 7, 7],
            "observation_dtype": str(observations.dtype),
            "source_keys": sorted(str(key) for key in payload.keys()),
            "history_layout": "stream_major_contiguous_chunks",
            "history_layout_source": layout_source,
            "histories_per_stream": histories_per_stream,
            "stream_count": stored_histories // histories_per_stream,
            "rtg_source": "recomputed_from_original_rewards_and_dones",
            "rtg_gamma": float(rtg_gamma),
            "rtg_resets_on_done": True,
            "rtg_crosses_row_boundaries": True,
            "rtg_crosses_stream_boundaries": False,
            "rtg_min": float(self.context_rtgs.min().item()),
            "rtg_max": float(self.context_rtgs.max().item()),
            "rtg_mean": float(self.context_rtgs.mean().item()),
            "rtg_valid_token_count": valid_rtg_count,
            "rtg_valid_abs_mean": float(valid_rtg_abs_mean),
            "rtg_invalid_tail_token_count": int(
                self.rtg_valid_mask.numel() - valid_rtg_count
            ),
            "rtg_invalid_tail_fraction": float(
                1.0 - self.rtg_valid_mask.float().mean().item()
            ),
        }

    def __len__(self) -> int:
        return int(self.context_states.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "context_states": self.context_states[index],
            "context_actions": self.context_actions[index],
            "context_rewards": self.context_rewards[index],
            "context_dones": self.context_dones[index],
            "context_rtgs": self.context_rtgs[index],
            "rtg_valid_mask": self.rtg_valid_mask[index],
            "initial_previous_action": self.initial_previous_action[index],
            "initial_previous_reward": self.initial_previous_reward[index],
            "initial_previous_done": self.initial_previous_done[index],
            "initial_previous_valid": self.initial_previous_valid[index],
            "initial_previous_rtg": self.initial_previous_rtg[index],
        }

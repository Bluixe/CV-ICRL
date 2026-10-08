"""Strict, provenance-aware MiniGrid learning-history dataset for IC-CQL.

The collector stores one fixed-length cross-episode learning history per row.
Unlike the legacy :class:`dataset.MinigridDataset`, this loader never shuffles,
concatenates, crops, or pads histories.  Those properties are important for
temporal-difference targets conditioned on the complete preceding context.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


_REQUIRED_KEYS = ("observations", "actions", "rewards", "dones")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _all_finite_in_chunks(array: np.ndarray, chunk_size: int = 1024) -> bool:
    """Check large arrays without allocating a full-size boolean temporary."""

    for start in range(0, array.shape[0], chunk_size):
        if not np.isfinite(array[start : start + chunk_size]).all():
            return False
    return True


def _as_bt_array(value: Any, *, name: str) -> np.ndarray:
    """Return a non-ragged ``[B, T]`` array, accepting a final singleton dim."""

    array = np.asarray(value)
    if array.dtype == object:
        raise ValueError(f"{name} must be a dense fixed-size numeric array, not dtype=object.")
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(
            f"{name} must have shape [histories, sequence] or "
            f"[histories, sequence, 1], got {array.shape}."
        )
    return array


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _episode_steps_from_dones(
    dones: np.ndarray,
    *,
    histories_per_stream: int,
) -> np.ndarray:
    """Derive episode steps across contiguous fixed-length rows of each stream.

    The legacy MiniGrid collector stores ``histories_per_stream`` adjacent rows
    for one environment before moving to the next environment.  Episode state
    must therefore carry across internal row boundaries, but never across a
    stream-group boundary.
    """

    histories_per_stream = _positive_int(
        histories_per_stream,
        name="histories_per_stream",
    )
    if dones.ndim != 2:
        raise ValueError(f"dones must have shape [histories, sequence], got {dones.shape}.")
    history_count, sequence_length = dones.shape
    if history_count == 0 or sequence_length == 0:
        raise ValueError("dones must contain at least one history and one token.")
    if history_count % histories_per_stream != 0:
        raise ValueError(
            f"{history_count} histories are not divisible by "
            f"histories_per_stream={histories_per_stream}."
        )

    grouped_dones = dones.reshape(
        history_count // histories_per_stream,
        histories_per_stream * sequence_length,
    )
    grouped_steps = np.zeros(grouped_dones.shape, dtype=np.int32)
    # Loop over the temporal axis while vectorizing across independent streams.
    for time_index in range(1, grouped_dones.shape[1]):
        grouped_steps[:, time_index] = np.where(
            grouped_dones[:, time_index - 1],
            0,
            grouped_steps[:, time_index - 1] + 1,
        )
    return grouped_steps.reshape(dones.shape)


def _episode_step_diagnostics(
    dones: np.ndarray,
    episode_steps: np.ndarray,
    *,
    histories_per_stream: int,
) -> Dict[str, Any]:
    """Summarize the row-boundary labels corrected by grouped reconstruction."""

    continued_rows = episode_steps[:, 0] > 0
    continued_count = int(np.count_nonzero(continued_rows))
    affected_tokens = 0
    if continued_count:
        continued_dones = dones[continued_rows]
        has_done = continued_dones.any(axis=1)
        first_done = continued_dones.argmax(axis=1)
        affected_lengths = np.where(
            has_done,
            first_done + 1,
            continued_dones.shape[1],
        )
        affected_tokens = int(affected_lengths.sum())
    total_tokens = int(dones.size)
    internal_row_boundaries = int(
        dones.shape[0] - dones.shape[0] // histories_per_stream
    )
    return {
        "cross_row_continuation_count": continued_count,
        "cross_row_continuation_fraction": float(
            continued_count / max(1, internal_row_boundaries)
        ),
        "internal_row_boundary_count": internal_row_boundaries,
        "row_local_mislabeled_token_count": affected_tokens,
        "row_local_mislabeled_token_fraction": float(
            affected_tokens / total_tokens
        ),
        "max_episode_step": int(episode_steps.max()),
        "tokens_above_sequence_length": int(
            np.count_nonzero(episode_steps >= dones.shape[1])
        ),
        "tokens_above_sequence_length_fraction": float(
            np.count_nonzero(episode_steps >= dones.shape[1]) / total_tokens
        ),
    }


def _initial_stream_predecessors(
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    *,
    histories_per_stream: int,
) -> Dict[str, np.ndarray]:
    """Recover the transition immediately preceding each fixed-length row.

    Every stream-major group begins without an in-dataset predecessor.  Each
    later row in the same group receives the final action, reward, and done
    signal of the preceding row.  Placeholder zeros are paired with an explicit
    validity mask at group boundaries so they cannot be mistaken for data.
    """

    histories_per_stream = _positive_int(
        histories_per_stream,
        name="histories_per_stream",
    )
    if actions.shape != rewards.shape or actions.shape != dones.shape:
        raise ValueError(
            "actions, rewards, and dones must have the same [histories, sequence] shape."
        )
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ValueError(
            "actions, rewards, and dones must contain at least one history and one token."
        )
    history_count = int(actions.shape[0])
    if history_count % histories_per_stream != 0:
        raise ValueError(
            f"{history_count} histories are not divisible by "
            f"histories_per_stream={histories_per_stream}."
        )

    valid = np.arange(history_count) % histories_per_stream != 0
    current_rows = np.flatnonzero(valid)
    predecessor_rows = current_rows - 1

    initial_actions = np.zeros(history_count, dtype=np.int64)
    initial_rewards = np.zeros(history_count, dtype=np.float32)
    initial_dones = np.zeros(history_count, dtype=np.bool_)
    initial_actions[current_rows] = np.asarray(
        actions[predecessor_rows, -1],
        dtype=np.int64,
    )
    initial_rewards[current_rows] = np.asarray(
        rewards[predecessor_rows, -1],
        dtype=np.float32,
    )
    initial_dones[current_rows] = np.asarray(
        dones[predecessor_rows, -1],
        dtype=np.bool_,
    )
    return {
        "initial_previous_action": initial_actions,
        "initial_previous_reward": initial_rewards,
        "initial_previous_done": initial_dones,
        "initial_previous_valid": np.ascontiguousarray(valid, dtype=np.bool_),
    }


class MinigridICCQLDataset(torch.utils.data.Dataset):
    """Load complete MiniGrid histories with terminal and provenance metadata.

    Parameters
    ----------
    path:
        Pickle produced by ``collect_minigrid_data.py``.  It must contain
        ``observations``, ``actions``, ``rewards``, and ``dones``.
    max_histories:
        Optional deterministic prefix selection used by explicitly requested
        smoke runs.  The value may not exceed the number of stored histories.
    histories_per_stream:
        Number of adjacent fixed-length rows produced from each continuous
        environment stream.  Legacy payloads do not store this metadata, so it
        must be supplied explicitly and is recorded in the fingerprint.
    compute_sha256:
        Include a full-file SHA256 in :attr:`fingerprint`.  Formal runs should
        leave this enabled; disabling it is useful only for quick diagnostics.
    """

    def __init__(
        self,
        path: str | Path,
        max_histories: Optional[int] = None,
        histories_per_stream: Optional[int] = None,
        compute_sha256: bool = True,
        expected_sha256: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"IC-CQL dataset does not exist: {self.path}")
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
        if histories_per_stream is not None:
            histories_per_stream = _positive_int(
                histories_per_stream,
                name="histories_per_stream",
            )

        stat = self.path.stat()
        dataset_sha256 = _file_sha256(self.path) if compute_sha256 else None
        if expected_sha256 is not None and dataset_sha256 != expected_sha256:
            raise ValueError(
                f"Dataset SHA256 mismatch for {self.path}: "
                f"{dataset_sha256} != {expected_sha256}."
            )

        # Authenticate legacy pickle bytes before deserializing them.  The
        # configured hashes bind the external stream-layout contract as well
        # as failing fast on partial/corrupt transfers.
        with self.path.open("rb") as file_handle:
            payload = pickle.load(file_handle)
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"IC-CQL dataset must be a mapping, got {type(payload).__name__}."
            )

        missing_keys = [key for key in _REQUIRED_KEYS if key not in payload]
        if missing_keys:
            raise KeyError(
                "IC-CQL dataset is missing required keys: " + ", ".join(missing_keys)
            )

        observations = np.asarray(payload["observations"])
        if observations.dtype == object:
            raise ValueError(
                "observations must be a dense fixed-size numeric array, not dtype=object."
            )
        if not np.issubdtype(observations.dtype, np.number):
            raise ValueError("observations must contain numeric image values.")
        if np.issubdtype(observations.dtype, np.floating) and not _all_finite_in_chunks(
            observations
        ):
            raise ValueError("observations must contain only finite image values.")
        if observations.ndim != 5:
            raise ValueError(
                "observations must have shape [histories, sequence, channels, height, width], "
                f"got {observations.shape}."
            )
        if observations.shape[0] == 0 or observations.shape[1] < 2:
            raise ValueError(
                "IC-CQL requires at least one history and at least two observations per history."
            )
        if observations.shape[2] != 3:
            raise ValueError(
                f"MiniGrid image observations must have 3 channels, got {observations.shape[2]}."
            )

        actions = _as_bt_array(payload["actions"], name="actions")
        rewards = _as_bt_array(payload["rewards"], name="rewards")
        dones = _as_bt_array(payload["dones"], name="dones")
        expected_bt = observations.shape[:2]
        for name, array in (
            ("actions", actions),
            ("rewards", rewards),
            ("dones", dones),
        ):
            if array.shape != expected_bt:
                raise ValueError(
                    f"{name} shape {array.shape} does not match observations "
                    f"history/sequence shape {expected_bt}."
                )

        if not np.issubdtype(actions.dtype, np.integer):
            if not (
                np.issubdtype(actions.dtype, np.floating)
                and np.isfinite(actions).all()
                and np.equal(actions, np.floor(actions)).all()
            ):
                raise ValueError("actions must contain finite integer action indices.")
        if not np.issubdtype(rewards.dtype, np.number) or not np.isfinite(rewards).all():
            raise ValueError("rewards must contain only finite numeric values.")
        if not (
            np.issubdtype(dones.dtype, np.bool_)
            or np.issubdtype(dones.dtype, np.integer)
            or np.issubdtype(dones.dtype, np.floating)
        ):
            raise ValueError("dones must be boolean or numeric 0/1 values.")
        if not np.isin(dones, (0, 1)).all():
            raise ValueError("dones must contain only 0/1 values.")

        stored_histories = int(observations.shape[0])
        payload_histories_per_stream = payload.get("histories_per_stream")
        if payload_histories_per_stream is not None:
            payload_histories_per_stream = _positive_int(
                payload_histories_per_stream,
                name="payload histories_per_stream",
            )
            if (
                histories_per_stream is not None
                and histories_per_stream != payload_histories_per_stream
            ):
                raise ValueError(
                    "Configured histories_per_stream does not match the dataset payload: "
                    f"{histories_per_stream} != {payload_histories_per_stream}."
                )
            histories_per_stream = payload_histories_per_stream
            layout_source = "payload"
        else:
            layout_source = "explicit_legacy_contract"
        if histories_per_stream is None:
            raise ValueError(
                "Legacy IC-CQL data does not identify stream boundaries. "
                "Pass the verified histories_per_stream explicitly."
            )
        if stored_histories % histories_per_stream != 0:
            raise ValueError(
                f"Stored histories {stored_histories} are not divisible by "
                f"histories_per_stream={histories_per_stream}."
            )

        # Reconstruct on the complete authenticated layout before applying a
        # smoke-only prefix.  Otherwise a truncated prefix can silently create
        # a false stream boundary.
        episode_steps = _episode_steps_from_dones(
            np.ascontiguousarray(dones, dtype=np.bool_),
            histories_per_stream=histories_per_stream,
        )
        step_diagnostics = _episode_step_diagnostics(
            dones,
            episode_steps,
            histories_per_stream=histories_per_stream,
        )
        initial_predecessors = _initial_stream_predecessors(
            actions,
            rewards,
            dones,
            histories_per_stream=histories_per_stream,
        )

        selected_histories = stored_histories if max_histories is None else max_histories
        if selected_histories > stored_histories:
            raise ValueError(
                f"max_histories={selected_histories} exceeds the {stored_histories} "
                "histories stored in the file; IC-CQL does not pad histories."
            )

        selection = slice(0, selected_histories)
        # Keep the collector's compact uint8 representation in host memory.
        # ``encode_context`` casts only the active mini-batch to float.
        observations = np.ascontiguousarray(observations[selection])
        actions = np.ascontiguousarray(actions[selection], dtype=np.int64)
        rewards = np.ascontiguousarray(rewards[selection], dtype=np.float32)
        dones = np.ascontiguousarray(dones[selection], dtype=np.bool_)
        episode_steps = np.ascontiguousarray(episode_steps[selection], dtype=np.int32)
        initial_previous_action = np.ascontiguousarray(
            initial_predecessors["initial_previous_action"][selection],
            dtype=np.int64,
        )
        initial_previous_reward = np.ascontiguousarray(
            initial_predecessors["initial_previous_reward"][selection],
            dtype=np.float32,
        )
        initial_previous_done = np.ascontiguousarray(
            initial_predecessors["initial_previous_done"][selection],
            dtype=np.bool_,
        )
        initial_previous_valid = np.ascontiguousarray(
            initial_predecessors["initial_previous_valid"][selection],
            dtype=np.bool_,
        )

        self.context_states = torch.from_numpy(observations)
        self.context_actions = torch.from_numpy(actions)
        self.context_rewards = torch.from_numpy(rewards)
        self.context_dones = torch.from_numpy(dones)
        self.episode_steps = torch.from_numpy(episode_steps)
        self.initial_previous_action = torch.from_numpy(initial_previous_action)
        self.initial_previous_reward = torch.from_numpy(initial_previous_reward)
        self.initial_previous_done = torch.from_numpy(initial_previous_done)
        self.initial_previous_valid = torch.from_numpy(initial_previous_valid)
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
            "sequence_length": self.sequence_length,
            "observation_shape": [int(value) for value in observations.shape[2:]],
            "observation_dtype": str(observations.dtype),
            "source_keys": sorted(str(key) for key in payload.keys()),
            "history_layout": "stream_major_contiguous_chunks",
            "history_layout_source": layout_source,
            "histories_per_stream": histories_per_stream,
            "stream_count": stored_histories // histories_per_stream,
            "episode_steps_source": "grouped_dones",
            "episode_step_diagnostics": step_diagnostics,
            "stream_predecessor_source": "previous_row_final_transition",
            "internal_predecessor_count": int(
                np.count_nonzero(initial_predecessors["initial_previous_valid"])
            ),
            "selected_internal_predecessor_count": int(
                np.count_nonzero(initial_previous_valid)
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
            "episode_steps": self.episode_steps[index],
            "initial_previous_action": self.initial_previous_action[index],
            "initial_previous_reward": self.initial_previous_reward[index],
            "initial_previous_done": self.initial_previous_done[index],
            "initial_previous_valid": self.initial_previous_valid[index],
        }

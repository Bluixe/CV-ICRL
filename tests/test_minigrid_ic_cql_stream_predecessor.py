from __future__ import annotations

import pickle

import numpy as np
import torch

from minigrid_ic_cql_dataset import MinigridICCQLDataset


def _write_dataset(tmp_path, *, actions, rewards, dones):
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.bool_)
    assert actions.shape == rewards.shape == dones.shape
    path = tmp_path / "histories.pkl"
    payload = {
        "observations": np.zeros((*actions.shape, 3, 7, 7), dtype=np.uint8),
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def test_stream_predecessors_preserve_done_values_without_cross_stream_carry(
    tmp_path,
):
    path = _write_dataset(
        tmp_path,
        actions=[
            [0, 1, 2],
            [3, 4, 5],
            [1, 2, 6],
            [4, 3, 2],
        ],
        rewards=[
            [0.0, 0.0, 0.25],
            [0.0, 0.0, 1.25],
            [0.0, 0.0, 2.5],
            [0.0, 0.0, 3.5],
        ],
        dones=[
            [False, False, False],
            [False, False, False],
            [False, False, True],
            [False, False, False],
        ],
    )
    dataset = MinigridICCQLDataset(
        path,
        histories_per_stream=2,
        compute_sha256=False,
    )

    assert dataset.initial_previous_valid.tolist() == [False, True, False, True]
    assert dataset.initial_previous_action.tolist() == [0, 2, 0, 6]
    assert dataset.initial_previous_reward.tolist() == [0.0, 0.25, 0.0, 2.5]
    assert dataset.initial_previous_done.tolist() == [False, False, False, True]

    # DataLoader must collate row-level predecessor scalars without widening or
    # adding a temporal dimension.
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4)))
    assert batch["initial_previous_action"].shape == (4,)
    assert batch["initial_previous_reward"].shape == (4,)
    assert batch["initial_previous_done"].shape == (4,)
    assert batch["initial_previous_valid"].shape == (4,)
    assert batch["initial_previous_action"].dtype == torch.int64
    assert batch["initial_previous_reward"].dtype == torch.float32
    assert batch["initial_previous_done"].dtype == torch.bool
    assert batch["initial_previous_valid"].dtype == torch.bool

    assert dataset.fingerprint["stream_predecessor_source"] == (
        "previous_row_final_transition"
    )
    assert dataset.fingerprint["internal_predecessor_count"] == 2
    assert dataset.fingerprint["selected_internal_predecessor_count"] == 2


def test_max_histories_is_applied_after_full_predecessor_reconstruction(tmp_path):
    path = _write_dataset(
        tmp_path,
        actions=[
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 0],
            [1, 2],
            [3, 4],
        ],
        rewards=[
            [0.0, 0.5],
            [0.0, 1.5],
            [0.0, 2.5],
            [0.0, 3.5],
            [0.0, 4.5],
            [0.0, 5.5],
        ],
        dones=np.zeros((6, 2), dtype=np.bool_),
    )
    dataset = MinigridICCQLDataset(
        path,
        histories_per_stream=3,
        max_histories=2,
        compute_sha256=False,
    )

    # The selected two-row prefix is not divisible by three. It is nevertheless
    # valid because layout reconstruction occurred on all six stored rows.
    assert len(dataset) == 2
    assert dataset.initial_previous_valid.tolist() == [False, True]
    assert dataset.initial_previous_action.tolist() == [0, 1]
    assert dataset.initial_previous_reward.tolist() == [0.0, 0.5]
    assert dataset.initial_previous_done.tolist() == [False, False]
    assert dataset.fingerprint["internal_predecessor_count"] == 4
    assert dataset.fingerprint["selected_internal_predecessor_count"] == 1

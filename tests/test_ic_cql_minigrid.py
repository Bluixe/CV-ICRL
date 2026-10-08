from __future__ import annotations

import json
import math
import pickle

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("minigrid")

from eval_ic_cql_minigrid import OnlineHistory, aggregate_seed_metrics
from minigrid_ic_cql_dataset import MinigridICCQLDataset
from nets.ic_cql_minigrid import (
    MinigridICCQLTransformer,
    compute_ic_cql_loss,
)


def _config(*, tuple_mode: str = "paper") -> dict:
    del tuple_mode
    return {
        "horizon": 6,
        "n_embd": 16,
        "n_layer": 1,
        "n_head": 1,
        "action_dim": 7,
        "dropout": 0.0,
        "image_size": 7,
        "episode_step_scale": 1.0,
        "twin_q": True,
    }


def _batch(batch_size: int = 2, sequence_length: int = 6) -> dict:
    actions = torch.arange(sequence_length).repeat(batch_size, 1) % 7
    rewards = torch.arange(sequence_length, dtype=torch.float32).repeat(batch_size, 1)
    dones = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
    dones[:, 1] = True
    steps = torch.tensor([0, 1, 0, 1, 2, 3]).repeat(batch_size, 1)
    return {
        "context_states": torch.randn(batch_size, sequence_length, 3, 7, 7),
        "context_actions": actions,
        "context_rewards": rewards,
        "context_dones": dones,
        "episode_steps": steps,
        "initial_previous_action": torch.zeros(batch_size, dtype=torch.long),
        "initial_previous_reward": torch.zeros(batch_size, dtype=torch.float32),
        "initial_previous_done": torch.zeros(batch_size, dtype=torch.bool),
        "initial_previous_valid": torch.zeros(batch_size, dtype=torch.bool),
    }


def test_dataset_preserves_fixed_histories_and_derives_episode_steps(tmp_path):
    path = tmp_path / "histories.pkl"
    payload = {
        "observations": (
            np.arange(3 * 5 * 3 * 7 * 7).reshape(3, 5, 3, 7, 7) % 11
        ).astype(np.uint8),
        "actions": np.arange(15).reshape(3, 5) % 7,
        "rewards": np.arange(15, dtype=np.float32).reshape(3, 5),
        "dones": np.array(
            [
                [0, 1, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [1, 0, 0, 1, 0],
            ],
            dtype=bool,
        ),
        "values": np.zeros((3, 5)),
    }
    with path.open("wb") as file_handle:
        pickle.dump(payload, file_handle)

    dataset = MinigridICCQLDataset(
        path,
        max_histories=2,
        histories_per_stream=1,
    )
    assert len(dataset) == 2
    assert dataset.sequence_length == 5
    assert dataset[0]["context_states"].shape == (5, 3, 7, 7)
    assert dataset[0]["context_states"].dtype == torch.uint8
    assert dataset[0]["context_actions"].dtype == torch.long
    assert dataset[0]["context_dones"].dtype == torch.bool
    assert dataset[0]["initial_previous_action"].dtype == torch.long
    assert dataset[0]["initial_previous_reward"].dtype == torch.float32
    assert dataset[0]["initial_previous_done"].dtype == torch.bool
    assert dataset[0]["initial_previous_valid"].dtype == torch.bool
    assert not dataset[0]["initial_previous_valid"]
    assert dataset[0]["episode_steps"].tolist() == [0, 1, 0, 1, 2]
    assert dataset[0]["context_actions"].tolist() == payload["actions"][0].tolist()
    assert dataset.fingerprint["selected_histories"] == 2
    assert dataset.fingerprint["observation_dtype"] == "uint8"
    assert dataset.fingerprint["histories_per_stream"] == 1
    assert len(dataset.fingerprint["sha256"]) == 64
    json.dumps(dataset.fingerprint)

    with pytest.raises(ValueError, match="does not pad"):
        MinigridICCQLDataset(
            path,
            max_histories=4,
            histories_per_stream=1,
        )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        MinigridICCQLDataset(
            path,
            histories_per_stream=1,
            expected_sha256="0" * 64,
        )
    invalid_pickle = tmp_path / "invalid.pkl"
    invalid_pickle.write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        MinigridICCQLDataset(
            invalid_pickle,
            histories_per_stream=1,
            expected_sha256="0" * 64,
        )


def test_dataset_reconstructs_steps_across_rows_before_prefix_selection(tmp_path):
    path = tmp_path / "grouped-histories.pkl"
    payload = {
        "observations": np.zeros((4, 3, 3, 7, 7), dtype=np.uint8),
        "actions": np.zeros((4, 3), dtype=np.int64),
        "rewards": np.zeros((4, 3), dtype=np.float32),
        "dones": np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [1, 0, 0],
            ],
            dtype=bool,
        ),
    }
    with path.open("wb") as file_handle:
        pickle.dump(payload, file_handle)

    dataset = MinigridICCQLDataset(
        path,
        histories_per_stream=2,
        max_histories=3,
    )
    assert dataset[0]["episode_steps"].tolist() == [0, 1, 2]
    assert dataset[1]["episode_steps"].tolist() == [3, 4, 0]
    assert dataset[1]["initial_previous_valid"]
    assert dataset[1]["initial_previous_action"].item() == 0
    assert not dataset[1]["initial_previous_done"]
    # A new stream starts at row 2 even though row 1 ended with done=False.
    assert dataset[2]["episode_steps"].tolist() == [0, 1, 0]
    assert not dataset[2]["initial_previous_valid"]
    full_dataset = MinigridICCQLDataset(
        path,
        histories_per_stream=2,
    )
    assert full_dataset[3]["episode_steps"].tolist() == [1, 0, 1]
    diagnostics = dataset.fingerprint["episode_step_diagnostics"]
    assert diagnostics["internal_row_boundary_count"] == 2
    assert diagnostics["cross_row_continuation_count"] == 2
    assert diagnostics["cross_row_continuation_fraction"] == pytest.approx(1.0)
    assert diagnostics["row_local_mislabeled_token_count"] == 3
    assert diagnostics["row_local_mislabeled_token_fraction"] == pytest.approx(0.25)


def test_dataset_requires_verified_stream_layout_and_fails_on_bad_grouping(tmp_path):
    path = tmp_path / "legacy-histories.pkl"
    payload = {
        "observations": np.zeros((3, 3, 3, 7, 7), dtype=np.uint8),
        "actions": np.zeros((3, 3), dtype=np.int64),
        "rewards": np.zeros((3, 3), dtype=np.float32),
        "dones": np.zeros((3, 3), dtype=bool),
    }
    with path.open("wb") as file_handle:
        pickle.dump(payload, file_handle)

    with pytest.raises(ValueError, match="Pass the verified histories_per_stream"):
        MinigridICCQLDataset(path)
    with pytest.raises(ValueError, match="not divisible"):
        MinigridICCQLDataset(path, histories_per_stream=2)
    with pytest.raises(ValueError, match="positive integer"):
        MinigridICCQLDataset(path, histories_per_stream=True)


def test_dataset_rejects_missing_key_and_mismatched_fixed_batch(tmp_path):
    missing_path = tmp_path / "missing.pkl"
    with missing_path.open("wb") as file_handle:
        pickle.dump(
            {
                "observations": np.zeros((2, 4, 3, 7, 7)),
                "actions": np.zeros((2, 4), dtype=np.int64),
                "rewards": np.zeros((2, 4)),
            },
            file_handle,
        )
    with pytest.raises(KeyError, match="dones"):
        MinigridICCQLDataset(missing_path, histories_per_stream=1)

    mismatch_path = tmp_path / "mismatch.pkl"
    with mismatch_path.open("wb") as file_handle:
        pickle.dump(
            {
                "observations": np.zeros((2, 4, 3, 7, 7)),
                "actions": np.zeros((2, 3), dtype=np.int64),
                "rewards": np.zeros((2, 4)),
                "dones": np.zeros((2, 4), dtype=bool),
            },
            file_handle,
        )
    with pytest.raises(ValueError, match="does not match observations"):
        MinigridICCQLDataset(mismatch_path, histories_per_stream=1)


def test_matched_and_paper_tuples_shift_without_boundary_reset():
    batch = _batch(batch_size=1)
    batch["context_actions"][0] = torch.tensor([2, 3, 4, 5, 6, 0])
    batch["context_rewards"][0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    matched = MinigridICCQLTransformer(_config(), tuple_mode="matched")
    matched_signals = matched.prepare_history_signals(batch)
    assert matched_signals["previous_actions"][0, 0].sum() == 0
    assert matched_signals["previous_actions"][0, 2, 3] == 1
    assert matched_signals["previous_rewards"][0, :, 0].tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]
    assert "previous_dones" not in matched_signals

    paper = MinigridICCQLTransformer(_config(), tuple_mode="paper")
    paper_signals = paper.prepare_history_signals(batch)
    # done at t=1 resets episode_step at t=2, but terminal a/r remain visible.
    assert paper_signals["previous_dones"][0, :, 0].tolist() == [
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert paper_signals["episode_steps"][0, :, 0].tolist() == pytest.approx(
        [0.0, 1.0, 0.0, 1.0, 2.0, 3.0]
    )
    assert paper_signals["previous_actions"][0, 2, 3] == 1
    assert paper_signals["previous_rewards"][0, 2, 0] == 2


def test_history_token_zero_uses_only_valid_row_predecessor():
    batch = _batch(batch_size=2)
    batch["initial_previous_action"] = torch.tensor([6, 99])
    batch["initial_previous_reward"] = torch.tensor([2.5, 999.0])
    batch["initial_previous_done"] = torch.tensor([True, True])
    batch["initial_previous_valid"] = torch.tensor([True, False])

    model = MinigridICCQLTransformer(_config(), tuple_mode="paper")
    signals = model.prepare_history_signals(batch)
    assert signals["previous_actions"][0, 0, 6] == 1
    assert signals["previous_rewards"][0, 0, 0] == pytest.approx(2.5)
    assert signals["previous_dones"][0, 0, 0] == 1
    assert signals["previous_actions"][1, 0].sum() == 0
    assert signals["previous_rewards"][1, 0, 0] == 0
    assert signals["previous_dones"][1, 0, 0] == 0


def test_terminal_mask_removes_bootstrap():
    q1_values = torch.zeros(1, 2, 7, requires_grad=True)
    q2_values = torch.zeros(1, 2, 7, requires_grad=True)
    target_q1_values = torch.zeros(1, 2, 7)
    target_q2_values = torch.zeros(1, 2, 7)
    target_q1_values[:, 1, :] = 1000.0
    target_q2_values[:, 1, :] = 2000.0
    batch = {
        "context_actions": torch.tensor([[0, 0]]),
        "context_rewards": torch.tensor([[2.0, 0.0]]),
        "context_dones": torch.tensor([[True, False]]),
    }
    _, metrics = compute_ic_cql_loss(
        q1_values,
        q2_values,
        target_q1_values,
        target_q2_values,
        batch,
        gamma=0.9,
        cql_weight=0.0,
        cql_label_smoothing=0.3,
    )
    assert metrics["loss/td"].item() == pytest.approx(8.0)
    assert metrics["target/bellman_mean"].item() == pytest.approx(2.0)


def test_td_shift_never_crosses_batch_boundaries():
    q1_values = torch.zeros(2, 2, 7)
    q2_values = torch.zeros(2, 2, 7)
    target_q1_values = torch.zeros(2, 2, 7)
    target_q2_values = torch.zeros(2, 2, 7)
    target_q1_values[1, 0, :] = 10_000.0
    target_q2_values[1, 0, :] = 10_000.0
    batch = {
        "context_actions": torch.zeros(2, 2, dtype=torch.long),
        "context_rewards": torch.zeros(2, 2),
        "context_dones": torch.zeros(2, 2, dtype=torch.bool),
    }
    _, metrics = compute_ic_cql_loss(
        q1_values,
        q2_values,
        target_q1_values,
        target_q2_values,
        batch,
        gamma=1.0,
        cql_weight=0.0,
        cql_label_smoothing=0.3,
    )
    assert metrics["loss/td"].item() == pytest.approx(0.0)


def test_td_target_clips_between_twin_target_heads():
    q1_values = torch.zeros(1, 2, 7)
    q2_values = torch.zeros(1, 2, 7)
    target_q1_values = torch.zeros(1, 2, 7)
    target_q2_values = torch.zeros(1, 2, 7)
    target_q1_values[0, 1, 3] = 10.0
    target_q2_values[0, 1, 4] = 3.0
    batch = {
        "context_actions": torch.zeros(1, 2, dtype=torch.long),
        "context_rewards": torch.zeros(1, 2),
        "context_dones": torch.zeros(1, 2, dtype=torch.bool),
    }
    _, metrics = compute_ic_cql_loss(
        q1_values,
        q2_values,
        target_q1_values,
        target_q2_values,
        batch,
        gamma=1.0,
        cql_weight=0.0,
        cql_label_smoothing=0.3,
    )
    assert metrics["target/clipped_next_q_mean"].item() == pytest.approx(3.0)
    assert metrics["target/bellman_mean"].item() == pytest.approx(3.0)
    assert metrics["loss/td"].item() == pytest.approx(18.0)


def test_discrete_cql_of_zero_q_is_log_action_count():
    q1_values = torch.zeros(2, 3, 7, requires_grad=True)
    q2_values = torch.zeros(2, 3, 7, requires_grad=True)
    target_q1_values = torch.zeros_like(q1_values)
    target_q2_values = torch.zeros_like(q2_values)
    batch = {
        "context_actions": torch.zeros(2, 3, dtype=torch.long),
        "context_rewards": torch.zeros(2, 3),
        "context_dones": torch.zeros(2, 3, dtype=torch.bool),
    }
    loss, metrics = compute_ic_cql_loss(
        q1_values,
        q2_values,
        target_q1_values,
        target_q2_values,
        batch,
        gamma=0.9,
        cql_weight=1.0,
        cql_label_smoothing=0.3,
    )
    assert metrics["loss/td"].item() == pytest.approx(0.0)
    assert metrics["loss/cql"].item() == pytest.approx(2.0 * math.log(7.0))
    assert loss.item() == pytest.approx(2.0 * math.log(7.0))


def test_cql_regularizer_includes_final_token_without_td_target():
    q1_values = torch.zeros(1, 3, 7)
    q2_values = torch.zeros(1, 3, 7)
    q1_values[0, -1, 1:] = 4.0
    q2_values[0, -1, 1:] = 4.0
    target_q1_values = torch.zeros_like(q1_values)
    target_q2_values = torch.zeros_like(q2_values)
    batch = {
        "context_actions": torch.zeros(1, 3, dtype=torch.long),
        "context_rewards": torch.zeros(1, 3),
        "context_dones": torch.zeros(1, 3, dtype=torch.bool),
    }
    _, metrics = compute_ic_cql_loss(
        q1_values,
        q2_values,
        target_q1_values,
        target_q2_values,
        batch,
        gamma=0.0,
        cql_weight=1.0,
        cql_label_smoothing=0.0,
    )
    expected_last_penalty = math.log(1.0 + 6.0 * math.exp(4.0))
    expected = (2.0 * math.log(7.0) + expected_last_penalty) / 3.0
    assert metrics["loss/td"].item() == pytest.approx(0.0)
    assert metrics["loss/cql"].item() == pytest.approx(2.0 * expected)


def test_cql_label_smoothing_is_applied_to_both_q_heads():
    q1_values = torch.tensor(
        [[[4.0, 0.0], [0.0, 2.0]]],
        requires_grad=True,
    )
    q2_values = torch.tensor(
        [[[1.0, 3.0], [2.0, -1.0]]],
        requires_grad=True,
    )
    target_q1_values = torch.zeros_like(q1_values)
    target_q2_values = torch.zeros_like(q2_values)
    batch = {
        "context_actions": torch.tensor([[0, 1]]),
        "context_rewards": torch.zeros(1, 2),
        "context_dones": torch.ones(1, 2, dtype=torch.bool),
    }
    _, metrics = compute_ic_cql_loss(
        q1_values,
        q2_values,
        target_q1_values,
        target_q2_values,
        batch,
        gamma=0.0,
        cql_weight=1.0,
        cql_label_smoothing=0.3,
    )
    expected = torch.nn.functional.cross_entropy(
        q1_values.reshape(-1, 2),
        batch["context_actions"].reshape(-1),
        label_smoothing=0.3,
    ) + torch.nn.functional.cross_entropy(
        q2_values.reshape(-1, 2),
        batch["context_actions"].reshape(-1),
        label_smoothing=0.3,
    )
    assert metrics["loss/cql"].item() == pytest.approx(expected.item())


def test_polyak_updates_only_frozen_target_heads():
    model = MinigridICCQLTransformer(_config(), tuple_mode="matched")
    with torch.no_grad():
        for online_head in (model.q1_head, model.q2_head):
            for parameter in online_head.parameters():
                parameter.fill_(2.0)
        for target_head in (model.target_q1_head, model.target_q2_head):
            for parameter in target_head.parameters():
                parameter.zero_()

    transformer_before = [parameter.detach().clone() for parameter in model.transformer.parameters()]
    model.soft_update_target(0.25)
    for target_head in (model.target_q1_head, model.target_q2_head):
        for parameter in target_head.parameters():
            assert torch.allclose(parameter, torch.full_like(parameter, 0.5))
            assert not parameter.requires_grad
    for before, after in zip(transformer_before, model.transformer.parameters()):
        assert torch.equal(before, after)


def test_transformer_is_causal_for_future_context():
    torch.manual_seed(3)
    model = MinigridICCQLTransformer(_config(), tuple_mode="paper").eval()
    batch_a = _batch(batch_size=1)
    batch_b = {key: value.clone() for key, value in batch_a.items()}
    batch_b["context_states"][:, 4:] += 100.0
    batch_b["context_actions"][:, 4:] = 6
    batch_b["context_rewards"][:, 4:] = 99.0
    batch_b["context_dones"][:, 4:] = True
    batch_b["episode_steps"][:, 4:] = 0

    with torch.no_grad():
        hidden_a = model.encode_context(batch_a)
        hidden_b = model.encode_context(batch_b)
    assert torch.allclose(hidden_a[:, :4], hidden_b[:, :4], atol=1e-6, rtol=0.0)


def test_model_outputs_full_fixed_batch_and_has_q_heads_only():
    model = MinigridICCQLTransformer(_config(), tuple_mode="paper")
    q1_values, q2_values = model(_batch(batch_size=3))
    assert q1_values.shape == (3, 6, 7)
    assert q2_values.shape == (3, 6, 7)
    assert isinstance(model.q1_head[1], torch.nn.LeakyReLU)
    assert isinstance(model.q2_head[1], torch.nn.LeakyReLU)
    assert isinstance(model.target_q1_head[1], torch.nn.LeakyReLU)
    assert isinstance(model.target_q2_head[1], torch.nn.LeakyReLU)
    assert not hasattr(model, "pred_actions")
    assert not hasattr(model, "v_head")


def test_both_online_heads_receive_gradients_and_targets_remain_frozen():
    model = MinigridICCQLTransformer(_config(), tuple_mode="paper")
    batch = _batch(batch_size=2)
    hidden = model.encode_context(batch)
    q1_values = model.q1_head(hidden)
    q2_values = model.q2_head(hidden)
    with torch.no_grad():
        target_q1_values = model.target_q1_head(hidden.detach())
        target_q2_values = model.target_q2_head(hidden.detach())
    loss, _ = compute_ic_cql_loss(
        q1_values,
        q2_values,
        target_q1_values,
        target_q2_values,
        batch,
        gamma=0.9,
        cql_weight=0.01,
        cql_label_smoothing=0.3,
    )
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.q1_head.parameters())
    assert all(parameter.grad is not None for parameter in model.q2_head.parameters())
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for head in (model.target_q1_head, model.target_q2_head)
        for parameter in head.parameters()
    )


def _observation(value: int) -> np.ndarray:
    return np.full((7, 7, 3), value, dtype=np.uint8)


def test_online_history_keeps_terminal_signal_and_slides_in_alignment():
    history = OnlineHistory(horizon=3, initial_observation=_observation(0))
    history.append(
        action=2,
        reward=1.0,
        done=True,
        next_observation=_observation(1),
    )
    arrays = history.as_arrays()
    assert arrays["context_actions"].tolist() == [2, 0]
    assert arrays["context_rewards"].tolist() == [1.0, 0.0]
    assert arrays["context_dones"].tolist() == [True, False]
    assert arrays["episode_steps"].tolist() == [0.0, 0.0]
    assert not arrays["initial_previous_valid"]

    history.append(
        action=3,
        reward=0.0,
        done=False,
        next_observation=_observation(2),
    )
    history.append(
        action=4,
        reward=0.5,
        done=False,
        next_observation=_observation(3),
    )
    arrays = history.as_arrays()
    assert arrays["context_states"][:, 0, 0, 0].tolist() == [1, 2, 3]
    assert arrays["context_actions"].tolist() == [3, 4, 0]
    assert arrays["context_rewards"].tolist() == [0.0, 0.5, 0.0]
    assert arrays["episode_steps"].tolist() == [0.0, 1.0, 2.0]
    assert arrays["initial_previous_valid"]
    assert arrays["initial_previous_action"].item() == 2
    assert arrays["initial_previous_reward"].item() == pytest.approx(1.0)
    assert arrays["initial_previous_done"].item()

    history.append(
        action=5,
        reward=0.75,
        done=False,
        next_observation=_observation(4),
    )
    arrays = history.as_arrays()
    assert arrays["context_states"][:, 0, 0, 0].tolist() == [2, 3, 4]
    assert arrays["initial_previous_action"].item() == 3
    assert arrays["initial_previous_reward"].item() == pytest.approx(0.0)
    assert not arrays["initial_previous_done"].item()


def test_aer_ler_and_instability_frequency_match_hand_calculation():
    metrics = aggregate_seed_metrics([[1.0, 0.9, 0.8], [0.5, 1.0]])
    aggregate = metrics["aggregate"]
    assert aggregate["aer_mean"] == pytest.approx(
        ((1.0 + 0.9 + 0.8) / 3.0 + (0.5 + 1.0) / 2.0) / 2.0
    )
    assert aggregate["ler_mean"] == pytest.approx(0.9)
    assert aggregate["if_mean"] == pytest.approx((2.0 / 3.0) / 2.0)
    assert aggregate["if_valid_tasks"] == 2
    assert aggregate["no_completed_episode_tasks"] == 0


def test_seed_with_no_completed_episode_is_zero_return_and_excluded_from_if():
    metrics = aggregate_seed_metrics([[], [1.0, 0.9]])
    aggregate = metrics["aggregate"]
    assert aggregate["aer_mean"] == pytest.approx((0.0 + 0.95) / 2.0)
    assert aggregate["ler_mean"] == pytest.approx((0.0 + 0.9) / 2.0)
    assert aggregate["if_mean"] == pytest.approx(0.5)
    assert aggregate["if_valid_tasks"] == 1
    assert aggregate["no_completed_episode_tasks"] == 1
    assert metrics["per_seed"][0]["if"] is None


def test_if_is_explicitly_undefined_when_no_seed_completes_an_episode():
    metrics = aggregate_seed_metrics([[], []])
    aggregate = metrics["aggregate"]
    assert aggregate["aer_mean"] == 0.0
    assert aggregate["ler_mean"] == 0.0
    assert aggregate["if_mean"] is None
    assert aggregate["if_valid_tasks"] == 0
    assert aggregate["no_completed_episode_tasks"] == 2

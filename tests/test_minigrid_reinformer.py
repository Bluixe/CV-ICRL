import pickle

import numpy as np
import pytest
import torch


pytest.importorskip("transformers")
pytest.importorskip("gymnasium")
pytest.importorskip("minigrid")

from eval_reinformer_minigrid import checkpoint_result_contract  # noqa: E402
from minigrid_reinformer_dataset import (  # noqa: E402
    MinigridReinformerDataset,
    complete_episode_mask,
    episode_aware_return_to_go,
)
from nets.reinformer_minigrid import (  # noqa: E402
    REINFORMER_ACTION_HEAD_CONCAT_MLP_V1,
    REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3,
    REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2,
    REINFORMER_CHECKPOINT_FORMAT_VERSION,
    REINFORMER_CONTRACT_VERSION,
    REINFORMER_LEGACY_CHECKPOINT_FORMAT_VERSION,
    REINFORMER_LEGACY_CONTRACT_VERSION,
    REINFORMER_TWO_PASS_CHECKPOINT_FORMAT_VERSION,
    REINFORMER_TWO_PASS_CONTRACT_VERSION,
    MinigridReinformer,
    compute_reinformer_loss,
)


def _model_config(
    *,
    horizon: int = 4,
    n_embd: int = 16,
    action_head_type: str = REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3,
) -> dict:
    return {
        "horizon": horizon,
        "n_embd": n_embd,
        "n_layer": 1,
        "n_head": 1,
        "action_dim": 3,
        "dropout": 0.0,
        "image_size": 7,
        "action_head_type": action_head_type,
    }


def _model_batch(*, batch_size: int = 2, sequence_length: int = 4) -> dict:
    states = torch.linspace(
        -1.0,
        1.0,
        steps=batch_size * sequence_length * 3 * 7 * 7,
        dtype=torch.float32,
    ).reshape(batch_size, sequence_length, 3, 7, 7)
    token_indices = torch.arange(batch_size * sequence_length).reshape(
        batch_size,
        sequence_length,
    )
    return {
        "context_states": states,
        "context_actions": (token_indices % 3).long(),
        "context_rewards": token_indices.float() / 10.0,
        "context_dones": torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.bool,
        ),
        "context_rtgs": 1.5 - token_indices.float() / 10.0,
        "rtg_valid_mask": torch.ones(
            batch_size,
            sequence_length,
            dtype=torch.bool,
        ),
        "initial_previous_action": torch.arange(batch_size).long() % 3,
        "initial_previous_reward": torch.arange(batch_size).float() / 4.0,
        "initial_previous_done": torch.zeros(batch_size, dtype=torch.bool),
        "initial_previous_valid": torch.arange(batch_size).bool(),
        "initial_previous_rtg": torch.arange(batch_size).float() / 2.0,
    }


def test_episode_aware_rtg_crosses_rows_but_not_dones_or_streams(
    tmp_path,
) -> None:
    # Each pair of rows is one six-step stream.  In stream 0, the return at
    # row 0's final token must include rewards 4 and 5 from row 1, stop at its
    # terminal token, and exclude the trailing reward 6.  Stream 0's trailing
    # token must likewise exclude all rewards from stream 1.
    rewards = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
        ],
        dtype=np.float32,
    )
    dones = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.bool_,
    )
    expected = np.array(
        [
            [15.0, 14.0, 12.0],
            [9.0, 5.0, 6.0],
            [210.0, 200.0, 180.0],
            [150.0, 110.0, 60.0],
        ],
        dtype=np.float32,
    )

    actual = episode_aware_return_to_go(
        rewards,
        dones,
        histories_per_stream=2,
    )
    np.testing.assert_allclose(actual, expected)
    expected_valid_mask = np.array(
        [
            [True, True, True],
            [True, True, False],
            [False, False, False],
            [False, False, False],
        ],
        dtype=np.bool_,
    )
    np.testing.assert_array_equal(
        complete_episode_mask(dones, histories_per_stream=2),
        expected_valid_mask,
    )

    payload = {
        "observations": np.zeros((4, 3, 3, 7, 7), dtype=np.uint8),
        "actions": np.arange(12, dtype=np.int64).reshape(4, 3) % 3,
        "rewards": rewards,
        "dones": dones,
        "histories_per_stream": 2,
    }
    dataset_path = tmp_path / "synthetic_reinformer.pkl"
    with dataset_path.open("wb") as file_handle:
        pickle.dump(payload, file_handle)
    dataset = MinigridReinformerDataset(
        dataset_path,
        histories_per_stream=2,
        compute_sha256=False,
    )

    torch.testing.assert_close(dataset.context_rtgs, torch.from_numpy(expected))
    torch.testing.assert_close(
        dataset.rtg_valid_mask,
        torch.from_numpy(expected_valid_mask),
    )
    assert dataset[1]["initial_previous_valid"].item() is True
    assert dataset[1]["initial_previous_rtg"].item() == pytest.approx(12.0)
    assert dataset[2]["initial_previous_valid"].item() is False
    assert dataset[2]["initial_previous_rtg"].item() == pytest.approx(0.0)
    valid_values = expected[expected_valid_mask]
    assert dataset.fingerprint["rtg_valid_abs_mean"] == pytest.approx(
        float(np.abs(valid_values).mean())
    )


def test_teacher_forced_and_predicted_rtg_action_paths_are_finite() -> None:
    torch.manual_seed(3)
    model = MinigridReinformer(_model_config())
    model.eval()
    batch = _model_batch()

    with torch.no_grad():
        teacher_logits, teacher_rtg_predictions = model(
            batch,
            teacher_force_action_rtg=True,
        )
        predicted_logits, predicted_rtg_predictions = model(
            batch,
            teacher_force_action_rtg=False,
        )

    assert teacher_logits.shape == (2, 4, 3)
    assert predicted_logits.shape == (2, 4, 3)
    assert teacher_rtg_predictions.shape == (2, 4, 1)
    assert predicted_rtg_predictions.shape == (2, 4, 1)
    assert torch.isfinite(teacher_logits).all()
    assert torch.isfinite(predicted_logits).all()
    assert torch.isfinite(teacher_rtg_predictions).all()
    assert torch.isfinite(predicted_rtg_predictions).all()
    torch.testing.assert_close(
        teacher_rtg_predictions,
        predicted_rtg_predictions,
    )


def test_action_logits_change_when_conditioning_rtg_changes() -> None:
    torch.manual_seed(5)
    model = MinigridReinformer(_model_config(horizon=3))
    model.eval()
    batch = _model_batch(sequence_length=3)

    with torch.no_grad():
        low_rtg_logits = model.action_logits_for_rtgs(
            batch,
            torch.zeros(2, 3),
        )
        high_rtg_logits = model.action_logits_for_rtgs(
            batch,
            torch.full((2, 3), 3.0),
        )

    assert low_rtg_logits.shape == high_rtg_logits.shape == (2, 3, 3)
    assert torch.isfinite(low_rtg_logits).all()
    assert torch.isfinite(high_rtg_logits).all()
    assert (
        high_rtg_logits - low_rtg_logits
    ).abs().max().item() > 1e-6


def test_v3_action_head_exactly_matches_cv_linear_capacity() -> None:
    model = MinigridReinformer(_model_config(n_embd=16))
    audit = model.capacity_audit()

    assert isinstance(model.predict_action, torch.nn.Linear)
    assert model.predict_action.in_features == 17
    assert model.predict_action.out_features == 3
    assert model.predict_action.bias is None
    assert audit["action_output_head"] == "Linear(17, 3, bias=False)"
    assert audit["action_output_head_parameters"] == 51
    assert audit["cv_icrl_action_head_parameters"] == 51
    assert audit["action_conditioning_and_output_parameters"] == 51
    assert audit["rtg_prediction_head_parameters"] == 17
    assert audit["cv_icrl_value_head_parameters"] == 17
    assert audit["extra_transition_parameters_for_rtg_scalar"] == 16
    assert audit["parameter_delta_vs_cv_matched_architecture"] == 16
    assert audit["transformer_passes_per_action"] == 1


def test_v3_forward_uses_one_transformer_pass() -> None:
    model = MinigridReinformer(_model_config(horizon=3))
    model.eval()
    calls = []
    hook = model.transformer.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(1)
    )
    try:
        with torch.no_grad():
            model(_model_batch(sequence_length=3))
    finally:
        hook.remove()

    assert len(calls) == 1


def test_v2_forward_remains_loadable_and_uses_two_shared_transformer_passes() -> None:
    model = MinigridReinformer(
        _model_config(
            horizon=3,
            action_head_type=REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2,
        )
    )
    model.eval()
    calls = []
    hook = model.transformer.register_forward_hook(
        lambda _module, _inputs, _output: calls.append(1)
    )
    try:
        with torch.no_grad():
            model(_model_batch(sequence_length=3))
    finally:
        hook.remove()

    assert len(calls) == 2


def test_legacy_config_without_action_head_type_remains_loadable() -> None:
    config = _model_config()
    del config["action_head_type"]
    model = MinigridReinformer(config)
    reloaded = MinigridReinformer(config)
    reloaded.load_state_dict(model.state_dict(), strict=True)
    hidden = torch.randn(2, 4, model.n_embd)

    assert model.action_head_type == REINFORMER_ACTION_HEAD_CONCAT_MLP_V1
    with torch.no_grad():
        logits = model.action_logits_from_hidden(
            hidden,
            torch.zeros(2, 4),
        )
    assert logits.shape == (2, 4, 3)


def test_future_action_rtg_cannot_change_past_action_logits() -> None:
    torch.manual_seed(6)
    model = MinigridReinformer(_model_config())
    model.eval()
    batch = _model_batch(batch_size=1)
    original_action_rtgs = batch["context_rtgs"].clone()
    changed_action_rtgs = original_action_rtgs.clone()
    changed_action_rtgs[:, 2:] += 100.0

    with torch.no_grad():
        original_logits = model.action_logits_for_rtgs(
            batch,
            original_action_rtgs,
        )
        changed_logits = model.action_logits_for_rtgs(
            batch,
            changed_action_rtgs,
        )

    torch.testing.assert_close(
        original_logits[:, :2],
        changed_logits[:, :2],
        rtol=0,
        atol=0,
    )


def test_predecessor_rtg_is_zeroed_after_episode_boundary() -> None:
    model = MinigridReinformer(_model_config())
    batch = _model_batch(batch_size=1)
    batch["context_rtgs"] = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    batch["context_dones"] = torch.tensor([[False, True, False, False]])
    batch["initial_previous_valid"] = torch.tensor([True])
    batch["initial_previous_done"] = torch.tensor([True])
    batch["initial_previous_rtg"] = torch.tensor([9.0])

    signals = model.prepare_history_signals(batch)

    torch.testing.assert_close(
        signals["previous_rtgs"][0, :, 0],
        torch.tensor([0.0, 4.0, 0.0, 2.0]),
    )


def test_legacy_predecessor_rtg_alignment_is_unchanged() -> None:
    model = MinigridReinformer(
        _model_config(
            action_head_type=REINFORMER_ACTION_HEAD_CONCAT_MLP_V1,
        )
    )
    batch = _model_batch(batch_size=1)
    batch["context_rtgs"] = torch.tensor([[4.0, 3.0, 2.0, 1.0]])
    batch["context_dones"] = torch.tensor([[False, True, False, False]])
    batch["initial_previous_valid"] = torch.tensor([True])
    batch["initial_previous_done"] = torch.tensor([True])
    batch["initial_previous_rtg"] = torch.tensor([9.0])

    signals = model.prepare_history_signals(batch)

    torch.testing.assert_close(
        signals["previous_rtgs"][0, :, 0],
        torch.tensor([9.0, 4.0, 3.0, 2.0]),
    )


def _checkpoint_payload(*, version: int) -> dict:
    if version == 1:
        model_config = _model_config()
        del model_config["action_head_type"]
        format_version = REINFORMER_LEGACY_CHECKPOINT_FORMAT_VERSION
        contract_version = REINFORMER_LEGACY_CONTRACT_VERSION
        capacity_audit = None
    elif version == 2:
        model_config = _model_config(
            action_head_type=REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2,
        )
        format_version = REINFORMER_TWO_PASS_CHECKPOINT_FORMAT_VERSION
        contract_version = REINFORMER_TWO_PASS_CONTRACT_VERSION
        capacity_audit = MinigridReinformer(model_config).capacity_audit()
    elif version == 3:
        model_config = _model_config()
        format_version = REINFORMER_CHECKPOINT_FORMAT_VERSION
        contract_version = REINFORMER_CONTRACT_VERSION
        capacity_audit = MinigridReinformer(model_config).capacity_audit()
    else:
        raise ValueError(f"Unsupported test checkpoint version: {version}")
    contract = {
        "contract_version": contract_version,
        "checkpoint_format_version": format_version,
        "model_config": model_config,
        "git_commit": "test-commit",
    }
    if capacity_audit is not None:
        contract["capacity_audit"] = capacity_audit
    return {
        "algorithm": "Reinformer-MiniGrid",
        "format_version": format_version,
        "contract_version": contract_version,
        "model_config": model_config,
        "contract": contract,
        "provenance": {
            "git": {
                "commit": "test-commit",
                "dirty": False,
            }
        },
        "epoch": 4,
    }


@pytest.mark.parametrize("version", [1, 2, 3])
def test_eval_accepts_all_version_matched_contracts(version) -> None:
    result = checkpoint_result_contract(_checkpoint_payload(version=version))

    assert result["checkpoint_completed_epochs"] == 5


def test_eval_rejects_v3_version_with_legacy_action_head() -> None:
    payload = _checkpoint_payload(version=3)
    payload["model_config"]["action_head_type"] = (
        REINFORMER_ACTION_HEAD_CONCAT_MLP_V1
    )
    payload["contract"]["model_config"] = dict(payload["model_config"])

    with pytest.raises(ValueError, match="action-head type"):
        checkpoint_result_contract(payload)


def test_current_or_future_teacher_rtg_cannot_leak_into_its_prediction() -> None:
    torch.manual_seed(7)
    model = MinigridReinformer(_model_config())
    model.eval()
    batch = _model_batch(batch_size=1)
    changed = {key: value.clone() for key, value in batch.items()}
    changed["context_rtgs"][:, 2:] += 100.0

    with torch.no_grad():
        _, original_predictions = model(batch)
        _, changed_predictions = model(changed)

    # RTG_t enters the encoder only after a one-step right shift, so changing
    # targets at positions 2 and 3 cannot affect predictions through position 2.
    torch.testing.assert_close(
        original_predictions[:, :3],
        changed_predictions[:, :3],
        rtol=0,
        atol=0,
    )


def test_expectile_weights_underprediction_more_than_overprediction() -> None:
    action_logits = torch.zeros(1, 1, 2)
    batch = {
        "context_actions": torch.zeros(1, 1, dtype=torch.long),
        "context_rtgs": torch.ones(1, 1),
        "rtg_valid_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    _, under_metrics = compute_reinformer_loss(
        action_logits,
        torch.zeros(1, 1, 1),
        batch,
        expectile=0.9,
    )
    _, over_metrics = compute_reinformer_loss(
        action_logits,
        torch.full((1, 1, 1), 2.0),
        batch,
        expectile=0.9,
    )

    assert under_metrics["loss/rtg_expectile"].item() == pytest.approx(0.9)
    assert over_metrics["loss/rtg_expectile"].item() == pytest.approx(0.1)
    assert (
        under_metrics["loss/rtg_expectile"]
        > over_metrics["loss/rtg_expectile"]
    )


def test_fixed_rtg_normalizer_keeps_all_zero_target_batch_finite() -> None:
    action_logits = torch.zeros(2, 3, 2)
    predictions = torch.full((2, 3, 1), 0.5)
    batch = {
        "context_actions": torch.zeros(2, 3, dtype=torch.long),
        "context_rtgs": torch.zeros(2, 3),
        "rtg_valid_mask": torch.ones(2, 3, dtype=torch.bool),
    }

    loss, metrics = compute_reinformer_loss(
        action_logits,
        predictions,
        batch,
        expectile=0.99,
        rtg_normalizer=0.4,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["loss/rtg_expectile"])
    assert metrics["rtg/normalizer"].item() == pytest.approx(0.4)
    assert metrics["loss/rtg_expectile"].item() == pytest.approx(
        0.01 * (0.5 / 0.4) ** 2
    )


def test_tiny_cpu_batch_can_be_overfit() -> None:
    torch.manual_seed(11)
    model = MinigridReinformer(_model_config(horizon=3, n_embd=8))
    model.train()
    batch = _model_batch(batch_size=1, sequence_length=3)
    batch["context_actions"] = torch.tensor([[0, 1, 2]], dtype=torch.long)
    batch["context_rewards"] = torch.tensor([[0.0, 0.25, 1.0]])
    batch["context_rtgs"] = torch.tensor([[1.25, 1.25, 1.0]])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    with torch.no_grad():
        initial_logits, initial_predictions = model(batch)
        initial_loss, _ = compute_reinformer_loss(
            initial_logits,
            initial_predictions,
            batch,
            expectile=0.9,
        )
        initial_loss_value = initial_loss.item()

    for _ in range(120):
        action_logits, rtg_predictions = model(batch)
        loss, _ = compute_reinformer_loss(
            action_logits,
            rtg_predictions,
            batch,
            expectile=0.9,
        )
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_logits, final_predictions = model(batch)
        final_loss, final_metrics = compute_reinformer_loss(
            final_logits,
            final_predictions,
            batch,
            expectile=0.9,
        )

    assert final_loss.item() < 0.15 * initial_loss_value
    assert final_metrics["action/accuracy"].item() == pytest.approx(1.0)
    assert torch.isfinite(final_predictions).all()

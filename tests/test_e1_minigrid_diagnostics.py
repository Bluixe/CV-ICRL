import unittest

import numpy as np

from e1_minigrid_diagnostics import (
    ScalarWritebackState,
    aggregate_episode_metrics,
    context_arrays,
    horizon_metric_slices,
    linear_drift_slope,
    stable_array_bundle_sha256,
)


class ScalarWritebackStateTest(unittest.TestCase):
    def test_normal_reproduces_running_max(self):
        state = ScalarWritebackState("normal", num_steps=4)
        observed = [state.update(value, step) for step, value in enumerate(
            [0.2, 0.1, 0.6, 0.4]
        )]
        self.assertEqual(
            observed,
            [(0.2, 0.2), (0.2, 0.2), (0.6, 0.6), (0.6, 0.6)],
        )

    def test_zero_and_frozen_are_distinct(self):
        zero = ScalarWritebackState("zero", num_steps=3)
        frozen = ScalarWritebackState("frozen", num_steps=3)
        self.assertEqual(zero.update(0.3, 0), (0.3, 0.0))
        self.assertEqual(zero.update(0.8, 1), (0.8, 0.0))
        self.assertEqual(frozen.update(0.3, 0), (0.3, 0.3))
        self.assertEqual(frozen.update(0.8, 1), (0.8, 0.3))

    def test_noise_writeback_state_preserves_raw_running_max(self):
        state = ScalarWritebackState("noise", num_steps=3)
        self.assertEqual(state.update(0.5, 0), (0.5, 0.5))
        self.assertEqual(state.update(0.4, 1), (0.5, 0.5))


class ContextInterventionTest(unittest.TestCase):
    def setUp(self):
        self.states = np.arange(5 * 3 * 2 * 2, dtype=np.float32).reshape(
            5, 3, 2, 2
        )
        self.actions = np.asarray([1, 2, 3, 4], dtype=np.int64)
        self.tokens = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        self.transition_ids = np.asarray([10, 11, 12, 13], dtype=np.int64)

    def build(self, condition, **overrides):
        kwargs = {
            "states": self.states,
            "actions": self.actions,
            "tokens": self.tokens,
            "transition_ids": self.transition_ids,
            "condition": condition,
            "model_horizon": 5,
            "short_context_length": 3,
            "stream_index": 2,
            "global_step": 17,
            "shuffle_seed": 11,
            "noise_seed": 13,
            "noise_sigma": 0.05,
            "frozen_value": 0.25,
        }
        kwargs.update(overrides)
        return context_arrays(**kwargs)

    def test_archived_leading_zero_alignment(self):
        result = self.build("normal")
        np.testing.assert_array_equal(
            result["context_actions"],
            [0, 1, 2, 3, 4],
        )
        np.testing.assert_allclose(
            result["context_rewards"][:, 0],
            [0.0, 0.1, 0.2, 0.3, 0.4],
        )
        effective_model_tokens = np.concatenate(
            [[0.0], result["context_rewards"][:-1, 0]]
        )
        np.testing.assert_allclose(
            effective_model_tokens,
            [0.0, 0.0, 0.1, 0.2, 0.3],
        )
        self.assertAlmostEqual(float(result["fed_last_token"]), 0.3)
        self.assertAlmostEqual(float(result["staged_last_token"]), 0.4)

    def test_no_history_keeps_only_current_state(self):
        result = self.build("no_history")
        self.assertEqual(result["context_states"].shape[0], 1)
        np.testing.assert_array_equal(
            result["context_states"][0],
            self.states[-1],
        )
        np.testing.assert_array_equal(result["context_actions"], [0])
        np.testing.assert_array_equal(result["context_rewards"], [[0.0]])

    def test_short_context_keeps_aligned_suffix(self):
        result = self.build("short_context")
        np.testing.assert_array_equal(
            result["context_states"],
            self.states[-3:],
        )
        np.testing.assert_array_equal(result["context_actions"], [0, 3, 4])
        np.testing.assert_allclose(
            result["context_rewards"][:, 0],
            [0.0, 0.3, 0.4],
        )

    def test_shuffle_is_deterministic_and_preserves_observed_multiset(self):
        first = self.build("shuffle")
        second = self.build("shuffle")
        np.testing.assert_array_equal(
            first["context_rewards"],
            second["context_rewards"],
        )
        np.testing.assert_allclose(
            np.sort(first["context_rewards"][1:, 0]),
            np.sort(self.tokens),
        )
        self.assertGreaterEqual(
            float(first["intervention_changed_fraction"]),
            0.0,
        )

    def test_shuffle_keeps_existing_relative_priority_when_prefix_grows(self):
        short = self.build(
            "shuffle",
            states=self.states[:-1],
            actions=self.actions[:-1],
            tokens=self.tokens[:-1],
            transition_ids=self.transition_ids[:-1],
            global_step=16,
        )
        long = self.build("shuffle", global_step=17)
        short_values = short["context_rewards"][1:, 0]
        long_values = long["context_rewards"][1:, 0]
        long_without_new = long_values[
            ~np.isclose(long_values, self.tokens[-1])
        ]
        np.testing.assert_allclose(short_values, long_without_new)

    def test_zero_and_frozen_transform_only_tokens(self):
        zero = self.build("zero")
        frozen = self.build("frozen")
        np.testing.assert_array_equal(zero["context_actions"], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(zero["context_rewards"], np.zeros((5, 1)))
        np.testing.assert_allclose(
            frozen["context_rewards"][:, 0],
            [0.0, 0.25, 0.25, 0.25, 0.25],
        )

    def test_noise_is_deterministic_and_excludes_synthetic_zero(self):
        first = self.build("noise")
        second = self.build("noise")
        np.testing.assert_array_equal(
            first["context_rewards"],
            second["context_rewards"],
        )
        self.assertEqual(first["context_rewards"][0, 0], 0.0)
        self.assertFalse(
            np.array_equal(first["context_rewards"][1:, 0], self.tokens)
        )
        self.assertGreater(float(first["intervention_token_rms"]), 0.0)
        identity = self.build("noise", noise_sigma=0.0)
        np.testing.assert_allclose(
            identity["context_rewards"][:, 0],
            [0.0, *self.tokens],
        )

    def test_noise_is_persistent_per_transition_across_decisions(self):
        first = self.build("noise", global_step=17)
        later = self.build("noise", global_step=99)
        np.testing.assert_array_equal(
            first["context_rewards"],
            later["context_rewards"],
        )


class MetricContractTest(unittest.TestCase):
    def test_literal_if_uses_episode_count_denominator(self):
        result = aggregate_episode_metrics([[1.0, 0.9, 0.95]], seeds=[7])
        row = result["per_stream"][0]
        self.assertAlmostEqual(row["aer"], 0.95)
        self.assertAlmostEqual(row["ler"], 0.95)
        self.assertAlmostEqual(row["if"], 1.0 / 3.0)

    def test_horizon_slices_use_completion_steps(self):
        records = [
            [
                {"return": 0.2, "completion_step": 2},
                {"return": 0.8, "completion_step": 7},
            ],
            [{"return": 0.4, "completion_step": 4}],
        ]
        result = horizon_metric_slices(records, cutoffs=[5, 10], seeds=[1, 2])
        self.assertAlmostEqual(
            result["prefix"]["5"]["aggregate"]["aer_mean"],
            0.3,
        )
        self.assertAlmostEqual(
            result["stages"]["6-10"]["aggregate"]["aer_mean"],
            0.8,
        )

    def test_stage_if_assigns_cross_boundary_drop_to_later_episode(self):
        records = [
            [
                {"return": 1.0, "completion_step": 5},
                {"return": 0.8, "completion_step": 6},
            ]
        ]
        result = horizon_metric_slices(
            records,
            cutoffs=[5, 10],
            seeds=[1],
        )
        later = result["stages"]["6-10"]["per_stream"][0]
        self.assertEqual(later["episodes"], 1)
        self.assertEqual(later["instability_events"], 1)
        self.assertAlmostEqual(later["if"], 1.0)

    def test_linear_drift_slope_uses_normalized_time(self):
        values = np.asarray(
            [[0.0, 3.0], [0.5, 3.0], [1.0, 3.0]],
            dtype=np.float64,
        )
        slope = linear_drift_slope(values)
        np.testing.assert_allclose(slope, [1.0, 0.0], atol=1e-12)

    def test_core_trace_hash_ignores_mapping_insertion_order(self):
        first = stable_array_bundle_sha256(
            {
                "actions": np.asarray([[1, 2]], dtype=np.int16),
                "rewards": np.asarray([[0.0, 1.0]], dtype=np.float32),
            },
            episode_records=[
                [{"return": 0.0, "completion_step": 1}],
                [{"return": 1.0, "completion_step": 1}],
            ],
        )
        second = stable_array_bundle_sha256(
            {
                "rewards": np.asarray([[0.0, 1.0]], dtype=np.float32),
                "actions": np.asarray([[1, 2]], dtype=np.int16),
            },
            episode_records=[
                [{"completion_step": 1, "return": 0.0}],
                [{"completion_step": 1, "return": 1.0}],
            ],
        )
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

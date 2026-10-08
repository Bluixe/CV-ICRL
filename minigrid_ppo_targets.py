"""Source-checkpoint mapping and critic queries used by the reported PPO-state experiment."""

import gc
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence
import numpy as np

MODEL_SAMPLE_COUNT = 40
MODEL_PER_ROW = 8
STEPS_PER_MODEL = 50
ROWS_PER_STREAM = 17
STEPS_PER_ROW = 400

class ContractError(RuntimeError):
    """Raised whenever an input or output violates the locked E2 contract."""

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)

def _finite_float(value: Any, label: str) -> float:
    result = float(value)
    _require(math.isfinite(result), "{} must be finite.".format(label))
    return result

def derive_selected_eval_indices(eval_results: Sequence[float]) -> List[int]:
    """Reproduce the archived collector's exact 30+10 checkpoint selection."""

    values = np.asarray(eval_results, dtype=np.float64)
    _require(values.ndim == 1 and values.size > 0, "eval_results must be non-empty.")
    _require(np.all(np.isfinite(values)), "eval_results contains NaN or Inf.")
    max_score = float(np.max(values))
    _require(max_score > 0.0, "Archived threshold selection requires max score > 0.")
    lower = 0.05 * max_score
    upper = 0.95 * max_score
    lower_hits = np.flatnonzero(values >= lower)
    upper_hits = np.flatnonzero(values >= upper)
    _require(lower_hits.size > 0, "No eval result reaches the 5% threshold.")
    _require(upper_hits.size > 0, "No eval result reaches the 95% threshold.")
    init_idx = int(lower_hits[0])
    final_idx = int(upper_hits[0])
    _require(init_idx <= final_idx, "Threshold indices are reversed.")

    selected = np.round(np.linspace(init_idx, final_idx, 30)).astype(int).tolist()
    selected += (
        np.round(np.linspace(final_idx, values.size - 1, 11))
        .astype(int)
        .tolist()[1:]
    )
    _require(
        len(selected) == MODEL_SAMPLE_COUNT,
        "Archived selection must contain exactly 40 positions.",
    )
    _require(
        all(0 <= index < values.size for index in selected),
        "Selected eval index is out of range.",
    )
    return [int(index) for index in selected]

def derive_row_block_mapping(
    selected_eval_indices: Sequence[int],
) -> List[Dict[str, Any]]:
    """Recover the archived collector's 17 rows of eight 50-step blocks."""

    model_indices = [int(value) for value in selected_eval_indices]
    _require(
        len(model_indices) == MODEL_SAMPLE_COUNT,
        "The row mapping requires exactly 40 selected model indices.",
    )
    model_num = len(model_indices)
    max_offset = MODEL_PER_ROW // 2
    total_rows = (model_num - MODEL_PER_ROW) // 2 + 1
    rows: List[Dict[str, Any]] = []

    for offset in range(max_offset):
        max_idx = model_num - (model_num % MODEL_PER_ROW)
        if max_idx + offset * 2 > model_num:
            max_idx -= MODEL_PER_ROW
        sequence = [
            model_indices[index + offset * 2] for index in range(max_idx)
        ]
        _require(
            len(sequence) % MODEL_PER_ROW == 0,
            "Collector sequence cannot be split into 8-block rows.",
        )
        for start in range(0, len(sequence), MODEL_PER_ROW):
            blocks = sequence[start : start + MODEL_PER_ROW]
            rows.append(
                {
                    "row_pattern": len(rows),
                    "offset": offset,
                    "offset_sequence_start": start,
                    "eval_indices": blocks,
                }
            )

    _require(len(rows) == total_rows == ROWS_PER_STREAM, "Expected 17 row patterns.")
    expected_offsets = [0] * 5 + [1] * 4 + [2] * 4 + [3] * 4
    _require(
        [row["offset"] for row in rows] == expected_offsets,
        "Recovered row offsets differ from the archived 5+4+4+4 layout.",
    )

    coverage = np.zeros((ROWS_PER_STREAM, STEPS_PER_ROW), dtype=np.int8)
    for row in rows:
        row_index = int(row["row_pattern"])
        for block_index in range(MODEL_PER_ROW):
            start = block_index * STEPS_PER_MODEL
            coverage[row_index, start : start + STEPS_PER_MODEL] += 1
    _require(
        np.all(coverage == 1),
        "The 17-row mapping does not cover every token exactly once.",
    )
    return rows

class SB3ValueBackend:
    """Lazy Stable-Baselines3/Torch backend used only by the formal CLI."""

    def __init__(self, device: str) -> None:
        try:
            import torch
            from stable_baselines3 import PPO
        except ImportError as error:
            raise ContractError("Cannot import torch/SB3: {}".format(error))
        self.torch = torch
        self.PPO = PPO
        self.device = device

    def load(self, path: Path) -> Any:
        try:
            model = self.PPO.load(str(path), device=self.device)
        except Exception as error:
            raise ContractError("PPO.load failed for {}: {}".format(path, error))
        model.policy.set_training_mode(False)
        return model

    def predict_values(self, model: Any, observations_chw: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations_chw)
        _require(
            observations.ndim == 4 and observations.shape[1:] == (3, 7, 7),
            "Critic observations must have shape [batch, 3, 7, 7].",
        )
        model_shape = tuple(model.policy.observation_space.shape)
        if model_shape == (3, 7, 7):
            model_observations = observations
        elif model_shape == (7, 7, 3):
            model_observations = observations.transpose(0, 2, 3, 1)
        else:
            raise ContractError(
                "Loaded PPO observation shape is incompatible: {}.".format(model_shape)
            )
        with self.torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(model_observations)
            values = model.policy.predict_values(tensor).reshape(-1)
            result = values.detach().cpu().numpy().astype(np.float32, copy=False)
        _require(
            result.shape == (observations.shape[0],),
            "PPO critic returned an unexpected shape.",
        )
        _require(np.all(np.isfinite(result)), "PPO critic returned NaN or Inf.")
        return result

    def release(self, model: Any) -> None:
        del model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def runtime_info(self) -> Dict[str, Any]:
        try:
            import stable_baselines3

            sb3_version = stable_baselines3.__version__
        except (ImportError, AttributeError):
            sb3_version = "unknown"
        cuda_available = bool(self.torch.cuda.is_available())
        gpu_name = None
        if cuda_available and str(self.device).startswith("cuda"):
            gpu_name = self.torch.cuda.get_device_name(self.device)
        return {
            "torch_version": self.torch.__version__,
            "cuda_available": cuda_available,
            "torch_cuda_version": self.torch.version.cuda,
            "stable_baselines3_version": sb3_version,
            "device": self.device,
            "gpu_name": gpu_name,
        }

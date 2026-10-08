"""Architecture-matched discrete Reinformer adaptation for MiniGrid."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model


REINFORMER_LEGACY_CHECKPOINT_FORMAT_VERSION = 1
REINFORMER_LEGACY_CONTRACT_VERSION = "minigrid_reinformer_independent_v1"
REINFORMER_TWO_PASS_CHECKPOINT_FORMAT_VERSION = 2
REINFORMER_TWO_PASS_CONTRACT_VERSION = "minigrid_reinformer_two_pass_linear_v2"
REINFORMER_CHECKPOINT_FORMAT_VERSION = 3
REINFORMER_CONTRACT_VERSION = "minigrid_reinformer_single_pass_linear_v3"

REINFORMER_ACTION_HEAD_CONCAT_MLP_V1 = "concat_mlp_v1"
REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2 = "two_pass_linear_v2"
REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3 = "single_pass_linear_v3"
SUPPORTED_REINFORMER_CHECKPOINT_CONTRACTS = frozenset(
    {
        (
            REINFORMER_LEGACY_CHECKPOINT_FORMAT_VERSION,
            REINFORMER_LEGACY_CONTRACT_VERSION,
        ),
        (
            REINFORMER_TWO_PASS_CHECKPOINT_FORMAT_VERSION,
            REINFORMER_TWO_PASS_CONTRACT_VERSION,
        ),
        (REINFORMER_CHECKPOINT_FORMAT_VERSION, REINFORMER_CONTRACT_VERSION),
    }
)


def _require(config: Mapping[str, Any], name: str) -> Any:
    if name not in config:
        raise KeyError(f"Reinformer model config is missing {name!r}.")
    return config[name]


def _as_bt(
    tensor: torch.Tensor,
    *,
    name: str,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    if tensor.shape != (batch_size, sequence_length):
        raise ValueError(
            f"{name} must have shape [B, T] (or [B, T, 1]), "
            f"got {tuple(tensor.shape)}."
        )
    return tensor


def _as_b(
    tensor: torch.Tensor,
    *,
    name: str,
    batch_size: int,
) -> torch.Tensor:
    if tensor.ndim == 2 and tensor.shape[-1] == 1:
        tensor = tensor[:, 0]
    if tensor.shape != (batch_size,):
        raise ValueError(
            f"{name} must have shape [B] (or [B, 1]), got {tuple(tensor.shape)}."
        )
    return tensor


class MinigridReinformer(nn.Module):
    """Predict a high-expectile RTG, then condition the action on it.

    One causal token per environment step keeps the CNN and Transformer scale
    matched to the MiniGrid AD/CV-ICRL models.  Each token encodes
    ``(s_t, a_{t-1}, r_{t-1}, RTG_{t-1})``.  The causal hidden state predicts
    ``RTG_t``.  The compute-matched v3 concatenates the current teacher RTG
    during training, or the current predicted RTG during deployment, directly
    to that same hidden state and applies ``Linear(n_embd + 1, action_dim,
    bias=False)``.  For MiniGrid's ``n_embd=256`` and seven actions, this has
    exactly the same 1,799 action-head parameters as CV-ICRL's
    ``Linear(256, 7, bias=True)`` while using one Transformer pass per action.

    Checkpoints without ``action_head_type`` retain the legacy v1 concat-MLP
    path, and v2 checkpoints retain their two-pass path, so already-produced
    artifacts remain strictly loadable.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = dict(config)
        self.horizon = int(_require(config, "horizon"))
        self.n_embd = int(_require(config, "n_embd"))
        self.n_layer = int(_require(config, "n_layer"))
        self.n_head = int(_require(config, "n_head"))
        self.action_dim = int(_require(config, "action_dim"))
        self.dropout = float(_require(config, "dropout"))
        self.image_size = int(_require(config, "image_size"))
        self.action_head_type = str(
            config.get(
                "action_head_type",
                REINFORMER_ACTION_HEAD_CONCAT_MLP_V1,
            )
        )
        self.im_embd = 64

        if self.horizon < 1:
            raise ValueError("horizon must be positive.")
        if self.n_embd <= 0 or self.n_layer <= 0 or self.n_head <= 0:
            raise ValueError("n_embd, n_layer, and n_head must be positive.")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if self.action_dim < 2:
            raise ValueError("action_dim must be at least two.")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must lie in [0, 1].")
        if self.action_head_type not in {
            REINFORMER_ACTION_HEAD_CONCAT_MLP_V1,
            REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2,
            REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3,
        }:
            raise ValueError(
                "action_head_type must be one of "
                f"{REINFORMER_ACTION_HEAD_CONCAT_MLP_V1!r}, "
                f"{REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2!r}, or "
                f"{REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3!r}."
            )

        transformer_config = GPT2Config(
            n_positions=self.horizon,
            n_ctx=self.horizon,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(transformer_config)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Flatten(start_dim=1),
            nn.Linear(16 * self.image_size * self.image_size, self.im_embd),
            nn.ReLU(),
        )
        transition_dim = self.im_embd + self.action_dim + 2
        self.embed_transition = nn.Linear(transition_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)
        self.predict_rtg = nn.Linear(self.n_embd, 1)
        if self.action_head_type == REINFORMER_ACTION_HEAD_CONCAT_MLP_V1:
            self.embed_action_rtg = nn.Sequential(
                nn.Linear(1, self.n_embd),
                nn.GELU(),
            )
            self.predict_action = nn.Sequential(
                nn.Linear(2 * self.n_embd, self.n_embd),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.n_embd, self.action_dim),
            )
        elif self.action_head_type == REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2:
            self.predict_action = nn.Linear(self.n_embd, self.action_dim)
        else:
            self.predict_action = nn.Linear(
                self.n_embd + 1,
                self.action_dim,
                bias=False,
            )

    def prepare_history_signals(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        states = batch["context_states"]
        if states.ndim != 5:
            raise ValueError(
                "context_states must have shape [B, T, 3, H, W], "
                f"got {tuple(states.shape)}."
            )
        batch_size, sequence_length = states.shape[:2]
        if sequence_length > self.horizon:
            raise ValueError(
                f"Context length {sequence_length} exceeds H={self.horizon}."
            )
        actions = _as_bt(
            batch["context_actions"],
            name="context_actions",
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        rewards = _as_bt(
            batch["context_rewards"],
            name="context_rewards",
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        dones = _as_bt(
            batch["context_dones"],
            name="context_dones",
            batch_size=batch_size,
            sequence_length=sequence_length,
        ).bool()
        rtgs = _as_bt(
            batch["context_rtgs"],
            name="context_rtgs",
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        initial_action = _as_b(
            batch["initial_previous_action"],
            name="initial_previous_action",
            batch_size=batch_size,
        )
        initial_reward = _as_b(
            batch["initial_previous_reward"],
            name="initial_previous_reward",
            batch_size=batch_size,
        )
        initial_done = _as_b(
            batch["initial_previous_done"],
            name="initial_previous_done",
            batch_size=batch_size,
        ).bool()
        initial_rtg = _as_b(
            batch["initial_previous_rtg"],
            name="initial_previous_rtg",
            batch_size=batch_size,
        )
        initial_valid = _as_b(
            batch["initial_previous_valid"],
            name="initial_previous_valid",
            batch_size=batch_size,
        ).bool()

        if actions.dtype.is_floating_point or actions.dtype == torch.bool:
            raise TypeError("context_actions must use an integer dtype.")
        actions = actions.long()
        if torch.any(actions < 0) or torch.any(actions >= self.action_dim):
            raise ValueError(
                f"context_actions must lie in [0, {self.action_dim - 1}]."
            )
        initial_action = initial_action.long()
        valid_initial_actions = initial_action[initial_valid]
        if torch.any(valid_initial_actions < 0) or torch.any(
            valid_initial_actions >= self.action_dim
        ):
            raise ValueError("Valid initial previous actions are out of range.")

        float_dtype = states.dtype if states.dtype.is_floating_point else torch.float32
        previous_actions = torch.zeros(
            batch_size,
            sequence_length,
            self.action_dim,
            dtype=float_dtype,
            device=states.device,
        )
        if torch.any(initial_valid):
            previous_actions[initial_valid, 0] = F.one_hot(
                initial_action[initial_valid],
                num_classes=self.action_dim,
            ).to(previous_actions.dtype)
        if sequence_length > 1:
            previous_actions[:, 1:] = F.one_hot(
                actions[:, :-1],
                num_classes=self.action_dim,
            ).to(previous_actions.dtype)

        previous_rewards = torch.zeros(
            batch_size,
            sequence_length,
            1,
            dtype=float_dtype,
            device=states.device,
        )
        previous_rtgs = torch.zeros_like(previous_rewards)
        previous_rewards[initial_valid, 0, 0] = initial_reward[
            initial_valid
        ].to(dtype=float_dtype)
        initial_rtg_valid = initial_valid
        if self.action_head_type in {
            REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2,
            REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3,
        }:
            initial_rtg_valid = initial_rtg_valid & ~initial_done
        previous_rtgs[initial_rtg_valid, 0, 0] = initial_rtg[
            initial_rtg_valid
        ].to(dtype=float_dtype)
        if sequence_length > 1:
            previous_rewards[:, 1:, 0] = rewards[:, :-1].to(dtype=float_dtype)
            shifted_rtgs = rtgs[:, :-1].to(dtype=float_dtype)
            if self.action_head_type in {
                REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2,
                REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3,
            }:
                shifted_rtgs = torch.where(
                    dones[:, :-1],
                    torch.zeros_like(shifted_rtgs),
                    shifted_rtgs,
                )
            previous_rtgs[:, 1:, 0] = shifted_rtgs
        if not torch.isfinite(previous_rewards).all():
            raise FloatingPointError("Previous reward inputs contain NaN or Inf.")
        if not torch.isfinite(previous_rtgs).all():
            raise FloatingPointError("Previous RTG inputs contain NaN or Inf.")
        return {
            "previous_actions": previous_actions,
            "previous_rewards": previous_rewards,
            "previous_rtgs": previous_rtgs,
        }

    def _encode_with_rtg_inputs(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        rtg_inputs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        states = batch["context_states"]
        if not states.dtype.is_floating_point:
            states = states.float()
        if not torch.isfinite(states).all():
            raise FloatingPointError("context_states contains NaN or Inf.")
        if tuple(states.shape[2:]) != (
            3,
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                "context_states image shape must be "
                f"[3, {self.image_size}, {self.image_size}]."
            )
        signals = self.prepare_history_signals(batch)
        batch_size, sequence_length = states.shape[:2]
        if rtg_inputs is None:
            rtg_features = signals["previous_rtgs"]
        else:
            rtg_features = _as_bt(
                rtg_inputs,
                name="rtg_inputs",
                batch_size=batch_size,
                sequence_length=sequence_length,
            ).unsqueeze(-1)
            rtg_features = rtg_features.to(
                device=states.device,
                dtype=states.dtype,
            )
            if not torch.isfinite(rtg_features).all():
                raise FloatingPointError("Current RTG inputs contain NaN or Inf.")
        image_features = self.image_encoder(
            states.reshape(
                batch_size * sequence_length,
                3,
                self.image_size,
                self.image_size,
            )
        ).reshape(batch_size, sequence_length, self.im_embd)
        token_embeddings = self.embed_ln(
            self.embed_transition(
                torch.cat(
                    [
                        image_features,
                        signals["previous_actions"],
                        signals["previous_rewards"],
                        rtg_features,
                    ],
                    dim=-1,
                )
            )
        )
        return self.transformer(
            inputs_embeds=token_embeddings,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    def encode_context(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode the prediction pass with strictly predecessor-aligned RTGs."""

        return self._encode_with_rtg_inputs(batch, rtg_inputs=None)

    def encode_action_context(
        self,
        batch: Mapping[str, torch.Tensor],
        action_rtgs: torch.Tensor,
    ) -> torch.Tensor:
        """Encode the action pass after inserting current teacher/predicted RTG."""

        return self._encode_with_rtg_inputs(batch, rtg_inputs=action_rtgs)

    def action_logits_from_hidden(
        self,
        hidden: torch.Tensor,
        action_rtgs: torch.Tensor,
    ) -> torch.Tensor:
        if self.action_head_type == REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2:
            raise RuntimeError(
                "two_pass_linear_v2 cannot condition an already-computed hidden "
                "state. Use action_logits_for_rtgs(batch, action_rtgs) so the "
                "current RTG is inserted before the shared Transformer."
            )
        if action_rtgs.ndim == 2:
            action_rtgs = action_rtgs.unsqueeze(-1)
        if action_rtgs.shape != (*hidden.shape[:2], 1):
            raise ValueError(
                "action_rtgs must have shape [B, T, 1], "
                f"got {tuple(action_rtgs.shape)}."
            )
        action_rtgs = action_rtgs.to(device=hidden.device, dtype=hidden.dtype)
        if not torch.isfinite(action_rtgs).all():
            raise FloatingPointError("Action-conditioning RTG contains NaN or Inf.")
        if self.action_head_type == REINFORMER_ACTION_HEAD_CONCAT_MLP_V1:
            rtg_features = self.embed_action_rtg(action_rtgs)
            action_inputs = torch.cat([hidden, rtg_features], dim=-1)
        else:
            action_inputs = torch.cat([hidden, action_rtgs], dim=-1)
        return self.predict_action(action_inputs)

    def action_logits_for_rtgs(
        self,
        batch: Mapping[str, torch.Tensor],
        action_rtgs: torch.Tensor,
        *,
        prediction_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate RTG-conditioned action logits under the checkpoint contract."""

        if self.action_head_type != REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2:
            if prediction_hidden is None:
                prediction_hidden = self.encode_context(batch)
            return self.action_logits_from_hidden(prediction_hidden, action_rtgs)
        action_hidden = self.encode_action_context(batch, action_rtgs)
        return self.predict_action(action_hidden)

    def capacity_audit(self) -> Dict[str, Any]:
        """Return exact head and matched-architecture parameter counts."""

        action_output_parameters = sum(
            parameter.numel() for parameter in self.predict_action.parameters()
        )
        action_module_parameters = action_output_parameters
        if self.action_head_type == REINFORMER_ACTION_HEAD_CONCAT_MLP_V1:
            action_module_parameters += sum(
                parameter.numel()
                for parameter in self.embed_action_rtg.parameters()
            )
        cv_action_head_parameters = (
            self.n_embd * self.action_dim + self.action_dim
        )
        rtg_head_parameters = sum(
            parameter.numel() for parameter in self.predict_rtg.parameters()
        )
        parameter_delta_vs_cv = self.n_embd + (
            action_module_parameters - cv_action_head_parameters
        )
        total_parameters = sum(parameter.numel() for parameter in self.parameters())
        if self.action_head_type == REINFORMER_ACTION_HEAD_TWO_PASS_LINEAR_V2:
            action_output_head = f"Linear({self.n_embd}, {self.action_dim})"
            transformer_passes_per_action = 2
        elif (
            self.action_head_type
            == REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3
        ):
            action_output_head = (
                f"Linear({self.n_embd + 1}, {self.action_dim}, bias=False)"
            )
            transformer_passes_per_action = 1
        else:
            action_output_head = "RTG embedding + concat GELU MLP"
            transformer_passes_per_action = 1
        return {
            "action_head_type": self.action_head_type,
            "action_output_head": action_output_head,
            "action_output_head_parameters": int(action_output_parameters),
            "action_conditioning_and_output_parameters": int(
                action_module_parameters
            ),
            "cv_icrl_action_head_parameters": int(cv_action_head_parameters),
            "rtg_prediction_head_parameters": int(rtg_head_parameters),
            "cv_icrl_value_head_parameters": int(self.n_embd + 1),
            "extra_transition_parameters_for_rtg_scalar": int(self.n_embd),
            "parameter_delta_vs_cv_matched_architecture": int(
                parameter_delta_vs_cv
            ),
            "total_model_parameters": int(total_parameters),
            "cv_matched_architecture_parameters": int(
                total_parameters - parameter_delta_vs_cv
            ),
            "transformer_passes_per_action": transformer_passes_per_action,
        }

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        teacher_force_action_rtg: bool = True,
        action_rtgs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_context(batch)
        rtg_predictions = self.predict_rtg(hidden)
        if action_rtgs is not None and teacher_force_action_rtg:
            raise ValueError(
                "Pass either teacher_force_action_rtg=True or explicit action_rtgs, not both."
            )
        if action_rtgs is None:
            if teacher_force_action_rtg:
                action_rtgs = batch["context_rtgs"]
            else:
                action_rtgs = rtg_predictions
        action_logits = self.action_logits_for_rtgs(
            batch,
            action_rtgs,
            prediction_hidden=hidden,
        )
        return action_logits, rtg_predictions


def compute_reinformer_loss(
    action_logits: torch.Tensor,
    rtg_predictions: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    *,
    expectile: float,
    rtg_loss_weight: float = 1.0,
    rtg_normalizer: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Categorical action NLL plus fixed-scale expectile RTG regression."""

    if action_logits.ndim != 3 or rtg_predictions.ndim != 3:
        raise ValueError("action_logits and rtg_predictions must be rank-three.")
    batch_size, sequence_length, action_dim = action_logits.shape
    if rtg_predictions.shape != (batch_size, sequence_length, 1):
        raise ValueError(
            "rtg_predictions must have shape [B, T, 1], "
            f"got {tuple(rtg_predictions.shape)}."
        )
    if not torch.isfinite(action_logits).all():
        raise FloatingPointError("action_logits contains NaN or Inf.")
    if not torch.isfinite(rtg_predictions).all():
        raise FloatingPointError("rtg_predictions contains NaN or Inf.")
    expectile = float(expectile)
    rtg_loss_weight = float(rtg_loss_weight)
    rtg_normalizer = float(rtg_normalizer)
    if not 0.5 <= expectile < 1.0:
        raise ValueError("expectile must lie in [0.5, 1).")
    if rtg_loss_weight < 0.0:
        raise ValueError("rtg_loss_weight must be non-negative.")
    if not math.isfinite(rtg_normalizer) or rtg_normalizer <= 0.0:
        raise ValueError("rtg_normalizer must be finite and positive.")

    actions = _as_bt(
        batch["context_actions"],
        name="context_actions",
        batch_size=batch_size,
        sequence_length=sequence_length,
    ).to(device=action_logits.device, dtype=torch.long)
    target_rtgs = _as_bt(
        batch["context_rtgs"],
        name="context_rtgs",
        batch_size=batch_size,
        sequence_length=sequence_length,
    ).to(device=rtg_predictions.device, dtype=rtg_predictions.dtype)
    valid_mask = _as_bt(
        batch["rtg_valid_mask"],
        name="rtg_valid_mask",
        batch_size=batch_size,
        sequence_length=sequence_length,
    ).to(device=action_logits.device, dtype=torch.bool)
    if torch.any(actions < 0) or torch.any(actions >= action_dim):
        raise ValueError(f"context_actions must lie in [0, {action_dim - 1}].")
    if not torch.isfinite(target_rtgs).all():
        raise FloatingPointError("context_rtgs contains NaN or Inf.")
    if not torch.any(valid_mask):
        raise ValueError("rtg_valid_mask contains no complete-episode tokens.")

    action_loss = F.cross_entropy(
        action_logits[valid_mask],
        actions[valid_mask],
    )
    target_rtgs = target_rtgs.unsqueeze(-1)
    valid_target_rtgs = target_rtgs[valid_mask]
    valid_predictions = rtg_predictions[valid_mask]
    normalizer = torch.as_tensor(
        rtg_normalizer,
        device=valid_target_rtgs.device,
        dtype=valid_target_rtgs.dtype,
    )
    residual = (valid_target_rtgs - valid_predictions) / normalizer
    weights = torch.where(
        residual < 0,
        1.0 - expectile,
        expectile,
    )
    rtg_loss = (weights * residual.square()).mean()
    total_loss = action_loss + rtg_loss_weight * rtg_loss
    probabilities = torch.softmax(action_logits.detach()[valid_mask], dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(
        dim=-1
    ).mean()
    accuracy = (
        action_logits.detach()[valid_mask].argmax(dim=-1) == actions[valid_mask]
    ).float().mean()
    metrics = {
        "loss/total": total_loss.detach(),
        "loss/action": action_loss.detach(),
        "loss/rtg_expectile": rtg_loss.detach(),
        "loss/rtg_weighted": (rtg_loss_weight * rtg_loss).detach(),
        "action/accuracy": accuracy.detach(),
        "action/entropy": entropy.detach(),
        "rtg/target_mean": valid_target_rtgs.detach().mean(),
        "rtg/target_abs_mean": valid_target_rtgs.detach().abs().mean(),
        "rtg/pred_mean": valid_predictions.detach().mean(),
        "rtg/pred_min": valid_predictions.detach().min(),
        "rtg/pred_max": valid_predictions.detach().max(),
        "rtg/normalizer": normalizer.detach(),
        "data/valid_token_fraction": valid_mask.float().mean().detach(),
    }
    return total_loss, metrics

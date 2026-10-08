"""Matched-backbone in-context Conservative Q-Learning for MiniGrid."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model


IC_CQL_CHECKPOINT_FORMAT_VERSION = 2
IC_CQL_CONTRACT_VERSION = "minigrid_ic_cql_official_twin_q_v2"
IC_CQL_POLICY_Q_RULE = "q1_argmax"


def _require_config(config: Mapping[str, Any], name: str) -> Any:
    if name not in config:
        raise KeyError(f"IC-CQL model config is missing required key {name!r}.")
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
            f"expected {(batch_size, sequence_length)}, got {tuple(tensor.shape)}."
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
            f"{name} must have shape [B] (or [B, 1]), "
            f"expected {(batch_size,)}, got {tuple(tensor.shape)}."
        )
    return tensor


def _q_head(hidden_size: int, action_dim: int) -> nn.Sequential:
    # The paper uses lightweight two-layer value heads with LeakyReLU.
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.LeakyReLU(),
        nn.Linear(hidden_size, action_dim),
    )


class MinigridICCQLTransformer(nn.Module):
    """Causal context encoder with twin online and twin target Q heads.

    ``matched`` uses the exact AD tuple ``(s_t, a_{t-1}, r_{t-1})``.
    ``paper`` augments it with ``(done_{t-1}, episode_step_t)`` as proposed by
    Tarasov et al.  The formal configuration represents the episode step as an
    unnormalized float (``episode_step_scale=1``).  Previous action/reward are
    shifted only at the beginning of the complete stream, not at episode
    boundaries, so terminal transitions remain visible across episodes and
    fixed-row boundaries.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        tuple_mode: str = "paper",
    ) -> None:
        super().__init__()
        if tuple_mode not in {"paper", "matched"}:
            raise ValueError("tuple_mode must be either 'paper' or 'matched'.")

        self.config = dict(config)
        self.tuple_mode = tuple_mode
        self.horizon = int(_require_config(config, "horizon"))
        self.n_embd = int(_require_config(config, "n_embd"))
        self.n_layer = int(_require_config(config, "n_layer"))
        self.n_head = int(_require_config(config, "n_head"))
        self.action_dim = int(_require_config(config, "action_dim"))
        self.dropout = float(_require_config(config, "dropout"))
        self.image_size = int(_require_config(config, "image_size"))
        self.im_embd = 64
        self.episode_step_scale = float(config.get("episode_step_scale", 1.0))
        self.twin_q = bool(config.get("twin_q", True))

        if self.horizon < 2:
            raise ValueError("horizon must be at least 2 for one-step TD targets.")
        if self.n_embd <= 0 or self.n_layer <= 0 or self.n_head <= 0:
            raise ValueError("n_embd, n_layer, and n_head must be positive.")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if self.action_dim <= 1:
            raise ValueError("action_dim must be at least 2.")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive.")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must lie in [0, 1].")
        if self.episode_step_scale <= 0.0:
            raise ValueError("episode_step_scale must be positive.")
        if not self.twin_q:
            raise ValueError("Official IC-CQL requires twin_q=true.")

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

        # This is intentionally identical to the repository's MiniGrid AD CNN.
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

        tuple_dim = self.im_embd + self.action_dim + 1
        if self.tuple_mode == "paper":
            tuple_dim += 2  # previous done flag and float episode step
        self.embed_transition = nn.Linear(tuple_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)

        self.q1_head = _q_head(self.n_embd, self.action_dim)
        self.q2_head = _q_head(self.n_embd, self.action_dim)
        self.target_q1_head = deepcopy(self.q1_head)
        self.target_q2_head = deepcopy(self.q2_head)
        self.target_q1_head.requires_grad_(False)
        self.target_q2_head.requires_grad_(False)

    def prepare_history_signals(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Validate and right-shift transition signals used in each token."""

        if "context_states" not in batch:
            raise KeyError("batch is missing 'context_states'.")
        states = batch["context_states"]
        if states.ndim != 5:
            raise ValueError(
                "context_states must have shape [B, T, 3, H, W], "
                f"got {tuple(states.shape)}."
            )
        batch_size, sequence_length = states.shape[:2]
        if sequence_length > self.horizon:
            raise ValueError(
                f"Context length {sequence_length} exceeds model horizon {self.horizon}; "
                "IC-CQL does not silently crop context."
            )

        try:
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
        except KeyError as error:
            raise KeyError(f"batch is missing {error.args[0]!r}.") from error

        if actions.dtype.is_floating_point or actions.dtype == torch.bool:
            raise TypeError("context_actions must use an integer tensor dtype.")
        actions = actions.long()
        if torch.any(actions < 0) or torch.any(actions >= self.action_dim):
            raise ValueError(
                f"context_actions must lie in [0, {self.action_dim - 1}]."
            )
        if not rewards.dtype.is_floating_point:
            rewards = rewards.float()
        if not torch.isfinite(rewards).all():
            raise ValueError("context_rewards contains NaN or Inf.")

        try:
            initial_previous_action = _as_b(
                batch["initial_previous_action"],
                name="initial_previous_action",
                batch_size=batch_size,
            )
            initial_previous_reward = _as_b(
                batch["initial_previous_reward"],
                name="initial_previous_reward",
                batch_size=batch_size,
            )
            initial_previous_done = _as_b(
                batch["initial_previous_done"],
                name="initial_previous_done",
                batch_size=batch_size,
            )
            initial_previous_valid = _as_b(
                batch["initial_previous_valid"],
                name="initial_previous_valid",
                batch_size=batch_size,
            )
        except KeyError as error:
            raise KeyError(
                f"batch is missing predecessor field {error.args[0]!r}."
            ) from error

        if (
            initial_previous_action.dtype.is_floating_point
            or initial_previous_action.dtype == torch.bool
        ):
            raise TypeError("initial_previous_action must use an integer tensor dtype.")
        if not initial_previous_reward.dtype.is_floating_point:
            initial_previous_reward = initial_previous_reward.float()
        if not torch.isfinite(initial_previous_reward).all():
            raise ValueError("initial_previous_reward contains NaN or Inf.")
        if not (
            (initial_previous_done == 0) | (initial_previous_done == 1)
        ).all():
            raise ValueError("initial_previous_done must contain only 0/1 values.")
        if not (
            (initial_previous_valid == 0) | (initial_previous_valid == 1)
        ).all():
            raise ValueError("initial_previous_valid must contain only 0/1 values.")

        initial_previous_action = initial_previous_action.to(
            device=states.device,
            dtype=torch.long,
        )
        initial_previous_reward = initial_previous_reward.to(device=states.device)
        initial_previous_done = initial_previous_done.to(device=states.device)
        initial_previous_valid = initial_previous_valid.to(
            device=states.device,
            dtype=torch.bool,
        )
        valid_previous_actions = initial_previous_action[initial_previous_valid]
        if torch.any(valid_previous_actions < 0) or torch.any(
            valid_previous_actions >= self.action_dim
        ):
            raise ValueError(
                "Valid initial_previous_action values must lie in "
                f"[0, {self.action_dim - 1}]."
            )

        previous_actions = torch.zeros(
            batch_size,
            sequence_length,
            self.action_dim,
            dtype=states.dtype if states.dtype.is_floating_point else torch.float32,
            device=states.device,
        )
        if torch.any(initial_previous_valid):
            previous_actions[initial_previous_valid, 0] = F.one_hot(
                initial_previous_action[initial_previous_valid],
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
            dtype=states.dtype if states.dtype.is_floating_point else torch.float32,
            device=states.device,
        )
        previous_rewards[initial_previous_valid, 0, 0] = (
            initial_previous_reward[initial_previous_valid].to(
                dtype=previous_rewards.dtype
            )
        )
        if sequence_length > 1:
            previous_rewards[:, 1:, 0] = rewards[:, :-1].to(
                device=states.device,
                dtype=previous_rewards.dtype,
            )

        signals: Dict[str, torch.Tensor] = {
            "previous_actions": previous_actions,
            "previous_rewards": previous_rewards,
        }
        if self.tuple_mode == "paper":
            try:
                dones = _as_bt(
                    batch["context_dones"],
                    name="context_dones",
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                )
                episode_steps = _as_bt(
                    batch["episode_steps"],
                    name="episode_steps",
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                )
            except KeyError as error:
                raise KeyError(f"paper tuple batch is missing {error.args[0]!r}.") from error

            if not ((dones == 0) | (dones == 1)).all():
                raise ValueError("context_dones must contain only 0/1 values.")
            if episode_steps.dtype.is_floating_point:
                if not torch.equal(episode_steps, episode_steps.floor()):
                    raise ValueError("episode_steps must contain integer values.")
            if torch.any(episode_steps < 0):
                raise ValueError("episode_steps must be non-negative.")
            if not torch.isfinite(episode_steps).all():
                raise ValueError("episode_steps contains NaN or Inf.")

            previous_dones = torch.zeros(
                batch_size,
                sequence_length,
                1,
                dtype=previous_rewards.dtype,
                device=states.device,
            )
            previous_dones[initial_previous_valid, 0, 0] = (
                initial_previous_done[initial_previous_valid].to(
                    dtype=previous_dones.dtype
                )
            )
            if sequence_length > 1:
                previous_dones[:, 1:, 0] = dones[:, :-1].to(
                    device=states.device,
                    dtype=previous_dones.dtype,
                )
            float_steps = episode_steps.to(
                device=states.device,
                dtype=previous_rewards.dtype,
            ).unsqueeze(-1) / self.episode_step_scale
            signals["previous_dones"] = previous_dones
            signals["episode_steps"] = float_steps

        return signals

    def encode_context(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode every history token with causal attention."""

        states = batch["context_states"]
        if not states.dtype.is_floating_point:
            states = states.float()
        if not torch.isfinite(states).all():
            raise ValueError("context_states contains NaN or Inf.")
        if tuple(states.shape[2:]) != (
            3,
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                "context_states image shape must be "
                f"[3, {self.image_size}, {self.image_size}], got {tuple(states.shape[2:])}."
            )
        signals = self.prepare_history_signals(batch)
        batch_size, sequence_length = states.shape[:2]

        image_features = self.image_encoder(
            states.reshape(
                batch_size * sequence_length,
                3,
                self.image_size,
                self.image_size,
            )
        ).reshape(batch_size, sequence_length, self.im_embd)

        tuple_parts = [
            image_features,
            signals["previous_actions"],
            signals["previous_rewards"],
        ]
        if self.tuple_mode == "paper":
            tuple_parts.extend(
                [signals["previous_dones"], signals["episode_steps"]]
            )
        token_embeddings = self.embed_ln(
            self.embed_transition(torch.cat(tuple_parts, dim=-1))
        )
        return self.transformer(
            inputs_embeds=token_embeddings,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        use_target: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encode_context(batch)
        if use_target:
            return self.target_q1_head(hidden), self.target_q2_head(hidden)
        return self.q1_head(hidden), self.q2_head(hidden)

    @torch.no_grad()
    def soft_update_target(self, tau: float) -> None:
        """Polyak-update only the two lightweight target Q heads."""

        tau = float(tau)
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must lie in [0, 1].")
        for target_head, online_head in (
            (self.target_q1_head, self.q1_head),
            (self.target_q2_head, self.q2_head),
        ):
            for target_parameter, online_parameter in zip(
                target_head.parameters(),
                online_head.parameters(),
            ):
                target_parameter.lerp_(online_parameter, tau)


def compute_ic_cql_loss(
    q1_values: torch.Tensor,
    q2_values: torch.Tensor,
    target_q1_values: torch.Tensor,
    target_q2_values: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    gamma: float,
    cql_weight: float,
    cql_label_smoothing: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the official twin-Q terminal-masked TD plus discrete CQL loss.

    All Q tensors must be ``[B, T, A]``.  The Bellman target is
    ``r + gamma * (1-done) * min(max_a Q1_target(next), max_a Q2_target(next))``.
    The two online heads each receive an MSE TD loss.  Their conservative terms
    are label-smoothed cross entropies over dataset actions; both TD losses and
    both CE losses are summed.  Targets shift strictly within each history.
    """

    tensors = {
        "q1_values": q1_values,
        "q2_values": q2_values,
        "target_q1_values": target_q1_values,
        "target_q2_values": target_q2_values,
    }
    if q1_values.ndim != 3:
        raise ValueError(
            f"q1_values must have shape [B, T, A], got {q1_values.shape}."
        )
    for name, tensor in tensors.items():
        if tensor.shape != q1_values.shape:
            raise ValueError(
                f"{name} must have shape {tuple(q1_values.shape)}, "
                f"got {tuple(tensor.shape)}."
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains NaN or Inf.")

    batch_size, sequence_length, action_dim = q1_values.shape
    if sequence_length < 2:
        raise ValueError("IC-CQL loss requires T >= 2.")
    if action_dim < 2:
        raise ValueError("IC-CQL loss requires at least two actions.")
    gamma = float(gamma)
    cql_weight = float(cql_weight)
    cql_label_smoothing = float(cql_label_smoothing)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")
    if cql_weight < 0.0:
        raise ValueError("cql_weight must be non-negative.")
    if not 0.0 <= cql_label_smoothing < 1.0:
        raise ValueError("cql_label_smoothing must lie in [0, 1).")

    try:
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
        )
    except KeyError as error:
        raise KeyError(f"loss batch is missing {error.args[0]!r}.") from error

    if actions.dtype.is_floating_point or actions.dtype == torch.bool:
        raise TypeError("context_actions must use an integer tensor dtype.")
    actions = actions.to(device=q1_values.device, dtype=torch.long)
    rewards = rewards.to(device=q1_values.device, dtype=q1_values.dtype)
    dones = dones.to(device=q1_values.device)
    if torch.any(actions < 0) or torch.any(actions >= action_dim):
        raise ValueError(f"context_actions must lie in [0, {action_dim - 1}].")
    if not torch.isfinite(rewards).all():
        raise ValueError("context_rewards contains NaN or Inf.")
    if not ((dones == 0) | (dones == 1)).all():
        raise ValueError("context_dones must contain only 0/1 values.")

    td_actions = actions[:, :-1]
    td_data_q1 = q1_values[:, :-1, :].gather(
        -1,
        td_actions.unsqueeze(-1),
    ).squeeze(-1)
    td_data_q2 = q2_values[:, :-1, :].gather(
        -1,
        td_actions.unsqueeze(-1),
    ).squeeze(-1)

    with torch.no_grad():
        next_q1 = target_q1_values[:, 1:, :].max(dim=-1).values
        next_q2 = target_q2_values[:, 1:, :].max(dim=-1).values
        clipped_next_q = torch.minimum(next_q1, next_q2)
        not_terminal = 1.0 - dones[:, :-1].to(dtype=q1_values.dtype)
        bellman_target = (
            rewards[:, :-1] + gamma * not_terminal * clipped_next_q
        )

    td_loss_q1 = F.mse_loss(td_data_q1, bellman_target)
    td_loss_q2 = F.mse_loss(td_data_q2, bellman_target)
    td_loss = td_loss_q1 + td_loss_q2

    # Every token has a recorded dataset action, including the final token for
    # which no next observation is available.  It therefore contributes to the
    # conservative regularizer even though it cannot contribute a TD target.
    flat_actions = actions.reshape(-1)
    cql_loss_q1 = F.cross_entropy(
        q1_values.reshape(-1, action_dim),
        flat_actions,
        label_smoothing=cql_label_smoothing,
    )
    cql_loss_q2 = F.cross_entropy(
        q2_values.reshape(-1, action_dim),
        flat_actions,
        label_smoothing=cql_label_smoothing,
    )
    cql_loss = cql_loss_q1 + cql_loss_q2
    total_loss = td_loss + cql_weight * cql_loss

    all_data_q1 = q1_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    all_data_q2 = q2_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

    metrics = {
        "loss/total": total_loss.detach(),
        "loss/td": td_loss.detach(),
        "loss/td_q1": td_loss_q1.detach(),
        "loss/td_q2": td_loss_q2.detach(),
        "loss/cql": cql_loss.detach(),
        "loss/cql_q1": cql_loss_q1.detach(),
        "loss/cql_q2": cql_loss_q2.detach(),
        "loss/cql_weighted": (cql_weight * cql_loss).detach(),
        "q/data_mean": (
            0.5 * (all_data_q1.detach().mean() + all_data_q2.detach().mean())
        ),
        "q/all_mean": (
            0.5 * (q1_values.detach().mean() + q2_values.detach().mean())
        ),
        "q1/data_mean": all_data_q1.detach().mean(),
        "q2/data_mean": all_data_q2.detach().mean(),
        "q1/all_mean": q1_values.detach().mean(),
        "q2/all_mean": q2_values.detach().mean(),
        "target/bellman_mean": bellman_target.detach().mean(),
        "target/clipped_next_q_mean": clipped_next_q.detach().mean(),
        "data/terminal_fraction": dones[:, :-1].float().mean().detach(),
    }
    return total_loss, metrics

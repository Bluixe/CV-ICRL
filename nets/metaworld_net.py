"""Transformer for Metaworld AD / auto_relabel.

Input sequence: (a_{t-1}, r_{t-1}, s_t) per timestep (AD causal ordering).
Output: predicted action a_t (continuous, 4-dim).

Three variants:
  - MetaworldTransformer: predicts actions only (standard AD), input=(a, r, s)
  - MetaworldMultiheadTransformer: predicts actions + values (auto_relabel), input=(a, r, s)
  - MetaworldValueCondTransformer: predicts actions + values, input=(a, r, v, s) — value as extra dim
"""

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetaworldTransformer(nn.Module):
    """GPT-2 based transformer for AD on Metaworld (continuous actions)."""

    def __init__(self, config):
        super().__init__()
        self.horizon = config["horizon"]
        self.n_embd = config["n_embd"]
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]
        self.state_dim = config["state_dim"]   # 39
        self.action_dim = config["action_dim"]  # 4
        self.dropout = config["dropout"]

        gpt_config = GPT2Config(
            n_positions=self.horizon,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(gpt_config)

        # Each token: (action_dim + 1 + state_dim) = (4 + 1 + 39) = 44
        input_dim = self.action_dim + 1 + self.state_dim
        self.embed_transition = nn.Linear(input_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)

    def forward(self, x):
        context_states = x["context_states"]    # (B, T, state_dim)
        context_actions = x["context_actions"]  # (B, T, action_dim)
        context_rewards = x["context_rewards"]  # (B, T, 1)

        if len(context_rewards.shape) == 2:
            context_rewards = context_rewards.unsqueeze(-1)

        B, T, _ = context_states.shape

        # AD causal shift: at position t, input is (a_{t-1}, r_{t-1}, s_t)
        padding_action = torch.zeros(B, 1, self.action_dim, device=context_states.device)
        padding_reward = torch.zeros(B, 1, 1, device=context_states.device)
        shifted_actions = torch.cat([padding_action, context_actions[:, :-1, :]], dim=1)
        shifted_rewards = torch.cat([padding_reward, context_rewards[:, :-1, :]], dim=1)

        # Concatenate: (a_{t-1}, r_{t-1}, s_t)
        tokens = torch.cat([shifted_actions, shifted_rewards, context_states], dim=-1)
        tokens = self.embed_ln(self.embed_transition(tokens))

        out = self.transformer(inputs_embeds=tokens)["last_hidden_state"]
        pred_actions = self.pred_actions(out)  # (B, T, action_dim)
        return pred_actions


class MetaworldMultiheadTransformer(nn.Module):
    """Transformer that predicts both actions and values (for auto_relabel)."""

    def __init__(self, config):
        super().__init__()
        self.horizon = config["horizon"]
        self.n_embd = config["n_embd"]
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.dropout = config["dropout"]

        gpt_config = GPT2Config(
            n_positions=self.horizon,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(gpt_config)

        input_dim = self.action_dim + 1 + self.state_dim
        self.embed_transition = nn.Linear(input_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.pred_values = nn.Linear(self.n_embd, 1)

    def forward(self, x):
        context_states = x["context_states"]
        context_actions = x["context_actions"]
        context_rewards = x["context_rewards"]

        if len(context_rewards.shape) == 2:
            context_rewards = context_rewards.unsqueeze(-1)

        B, T, _ = context_states.shape

        padding_action = torch.zeros(B, 1, self.action_dim, device=context_states.device)
        padding_reward = torch.zeros(B, 1, 1, device=context_states.device)
        shifted_actions = torch.cat([padding_action, context_actions[:, :-1, :]], dim=1)
        shifted_rewards = torch.cat([padding_reward, context_rewards[:, :-1, :]], dim=1)

        tokens = torch.cat([shifted_actions, shifted_rewards, context_states], dim=-1)
        tokens = self.embed_ln(self.embed_transition(tokens))

        out = self.transformer(inputs_embeds=tokens)["last_hidden_state"]
        pred_actions = self.pred_actions(out)
        pred_values = self.pred_values(out)
        return pred_actions, pred_values


class MetaworldValueCondTransformer(nn.Module):
    """Transformer with value as additional input dim (not replacing reward).

    Input: (a_{t-1}, r_{t-1}, v_{t-1}, s_t) — input_dim = action_dim + 1 + 1 + state_dim
    Output: predicted actions + predicted values
    """

    def __init__(self, config):
        super().__init__()
        self.horizon = config["horizon"]
        self.n_embd = config["n_embd"]
        self.n_layer = config["n_layer"]
        self.n_head = config["n_head"]
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.dropout = config["dropout"]

        gpt_config = GPT2Config(
            n_positions=self.horizon,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            resid_pdrop=self.dropout,
            embd_pdrop=self.dropout,
            attn_pdrop=self.dropout,
            use_cache=False,
        )
        self.transformer = GPT2Model(gpt_config)

        # input: (action_dim + 1_reward + 1_value + state_dim) = 4 + 1 + 1 + 39 = 45
        input_dim = self.action_dim + 1 + 1 + self.state_dim
        self.embed_transition = nn.Linear(input_dim, self.n_embd)
        self.embed_ln = nn.LayerNorm(self.n_embd)
        self.pred_actions = nn.Linear(self.n_embd, self.action_dim)
        self.pred_values = nn.Linear(self.n_embd, 1)

    def forward(self, x):
        context_states = x["context_states"]      # (B, T, state_dim)
        context_actions = x["context_actions"]     # (B, T, action_dim)
        context_rewards = x["context_rewards"]     # (B, T, 1)
        context_values = x["context_values"]       # (B, T, 1)

        if len(context_rewards.shape) == 2:
            context_rewards = context_rewards.unsqueeze(-1)
        if len(context_values.shape) == 2:
            context_values = context_values.unsqueeze(-1)

        B, T, _ = context_states.shape

        padding_action = torch.zeros(B, 1, self.action_dim, device=context_states.device)
        padding_scalar = torch.zeros(B, 1, 1, device=context_states.device)
        shifted_actions = torch.cat([padding_action, context_actions[:, :-1, :]], dim=1)
        shifted_rewards = torch.cat([padding_scalar, context_rewards[:, :-1, :]], dim=1)
        shifted_values = torch.cat([padding_scalar, context_values[:, :-1, :]], dim=1)

        # (a_{t-1}, r_{t-1}, v_{t-1}, s_t)
        tokens = torch.cat([shifted_actions, shifted_rewards, shifted_values, context_states], dim=-1)
        tokens = self.embed_ln(self.embed_transition(tokens))

        out = self.transformer(inputs_embeds=tokens)["last_hidden_state"]
        pred_actions = self.pred_actions(out)
        pred_values = self.pred_values(out)
        return pred_actions, pred_values

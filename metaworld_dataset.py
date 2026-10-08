"""Dataset for Metaworld AD / auto_relabel training."""

import pickle

import numpy as np
import torch
from loguru import logger


class MetaworldDataset(torch.utils.data.Dataset):
    """Dataset for Metaworld AD training.

    Each sample is a sequence of (obs, action, reward) of length `horizon`.
    Data format in pkl: dict with keys
        observations: (N, horizon, obs_dim)
        actions:      (N, horizon, action_dim)
        rewards:      (N, horizon)
        values:       (N, horizon)  [optional, for relabel_reward mode]
    """

    def __init__(self, path, config, include_values=False):
        self.horizon = config["horizon"]

        if isinstance(path, list):
            trajs = []
            for p in path:
                with open(p, "rb") as f:
                    trajs.append(pickle.load(f))
            data = {k: np.concatenate([t[k] for t in trajs], axis=0) for k in trajs[0].keys()}
        else:
            with open(path, "rb") as f:
                data = pickle.load(f)

        self.observations = torch.from_numpy(data["observations"]).float()
        self.actions = torch.from_numpy(data["actions"]).float()

        rewards = data["rewards"]
        if len(rewards.shape) == 2:
            rewards = rewards[:, :, None]
        self.rewards = torch.from_numpy(rewards).float()

        if self.observations.shape[1] != self.horizon:
            raise ValueError("Dataset length and model horizon differ.")
        if include_values and "values" not in data:
            raise ValueError("Scalar training requires values in the dataset.")
        self.include_values = include_values and "values" in data
        if self.include_values:
            values = data["values"]
            if len(values.shape) == 2:
                values = values[:, :, None]
            self.values = torch.from_numpy(values).float()

        logger.info(f"Loaded {len(self)} samples | obs={self.observations.shape} act={self.actions.shape}")

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, index):
        res = {
            "context_states": self.observations[index],
            "context_actions": self.actions[index],
            "context_rewards": self.rewards[index],
        }
        if self.include_values:
            res["context_values"] = self.values[index]
        return res

"""MiniGrid AD/CV rollout and normal/zero/frozen feedback interventions."""
import argparse
import json
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import minigrid  # environment registration
import numpy as np
import torch
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from e1_minigrid_diagnostics import ScalarWritebackState, context_arrays, aggregate_episode_metrics
from nets.net import MinigridTransformer, MinigridMultiheadTransformer
from train_cv_minigrid import sha256


class History:
    """Cross-episode FIFO; leading-zero alignment matches the reported E1 evaluator."""
    def __init__(self, horizon, observation):
        self.horizon = horizon
        self.states = deque([observation], maxlen=horizon)
        self.actions = deque(maxlen=horizon - 1)
        self.tokens = deque(maxlen=horizon - 1)
        self.ids = deque(maxlen=horizon - 1)

    def add(self, observation, action, token, step):
        self.states.append(observation); self.actions.append(action)
        self.tokens.append(token); self.ids.append(step)

    def as_arrays(self, step, method="cv"):
        if method == "ad":
            return {"context_states": np.stack(self.states),
                    "context_actions": np.r_[np.asarray(self.actions, dtype=np.int64), np.int64(0)],
                    "context_rewards": np.r_[np.asarray(self.tokens, dtype=np.float32), np.float32(0)][:, None]}
        return context_arrays(states=np.stack(self.states), actions=np.asarray(self.actions, dtype=np.int64),
                              tokens=np.asarray(self.tokens, dtype=np.float32),
                              transition_ids=np.asarray(self.ids, dtype=np.int64), condition='normal',
                              model_horizon=self.horizon, short_context_length=self.horizon,
                              stream_index=0, global_step=step, shuffle_seed=0, noise_seed=0,
                              noise_sigma=0, frozen_value=None)


def preprocess(observation):
    return np.asarray(observation).transpose(2, 0, 1).astype(np.float32)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--eval-env', required=True)
    p.add_argument('--output-json', required=True)
    p.add_argument('--method', choices=['ad', 'cv'], default='cv')
    p.add_argument('--condition', choices=['normal', 'zero', 'frozen'], default='normal')
    p.add_argument('--num-envs', type=int, default=20)
    p.add_argument('--num-steps', type=int, default=8000)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--torch-seed', type=int, default=0)
    p.add_argument('--H', type=int, default=400)
    p.add_argument('--embd', type=int, default=256)
    p.add_argument('--layer', type=int, default=4)
    p.add_argument('--head', type=int, default=4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--trace-npz')
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if min(a.num_envs, a.num_steps) < 1: raise ValueError('Rollout budget must be positive.')
    if a.method == 'ad' and a.condition != 'normal': raise ValueError('Feedback interventions require CV.')
    out = Path(a.output_json)
    if out.exists() or (a.trace_npz and Path(a.trace_npz).exists()):
        raise FileExistsError('Choose a new output path.')
    random.seed(a.torch_seed); np.random.seed(a.torch_seed); torch.manual_seed(a.torch_seed)
    device = torch.device(a.device)
    checkpoint = torch.load(a.checkpoint, map_location=device, weights_only=False)
    config = dict(horizon=a.H, state_dim=2, action_dim=7, n_embd=a.embd, n_layer=a.layer,
                  n_head=a.head, dropout=0, shuffle=False, test=True, store_gpu=False, image_size=7)
    if 'model_config' in checkpoint:
        config.update(checkpoint['model_config'])
        if checkpoint.get('method', a.method) != a.method: raise ValueError('Checkpoint method mismatch.')
    model_type = MinigridMultiheadTransformer if a.method == 'cv' else MinigridTransformer
    model = model_type(config).to(device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint), strict=True); model.eval()
    seeds = list(range(a.seed_start, a.seed_start + a.num_envs))
    def make(seed):
        def factory():
            env = ImgObsWrapper(gym.make(a.eval_env)); env.reset(seed=seed); return env
        return factory
    envs = DummyVecEnv([make(seed) for seed in seeds]); envs.seed(a.seed_start)
    returns = [[] for _ in seeds]; running = np.zeros(a.num_envs)
    raw_trace = np.zeros((a.num_steps, a.num_envs), dtype=np.float32)
    token_trace = np.zeros_like(raw_trace)
    try:
        obs = envs.reset()
        histories = [History(config['horizon'], preprocess(o)) for o in obs]
        writebacks = [ScalarWritebackState(a.condition, a.num_steps) for _ in seeds]
        for step in range(a.num_steps):
            arrays = [h.as_arrays(step, a.method) for h in histories]
            batch = {k: torch.from_numpy(np.stack([b[k] for b in arrays])).to(device)
                     for k in ('context_states', 'context_actions', 'context_rewards')}
            with torch.no_grad():
                result = model(batch)
                if a.method == 'cv': logits, values = result
                else: logits = result
                actions = torch.multinomial(torch.softmax(logits[:, -1], -1), 1).squeeze(1).cpu().numpy()
            obs, reward, done, info = envs.step(actions)
            for i in range(a.num_envs):
                if a.method == 'cv':
                    raw = float(values[i, -1, 0].item())
                    _, token = writebacks[i].update(raw, step)
                    raw_trace[step, i] = raw
                else: token = float(reward[i])
                token_trace[step, i] = token
                histories[i].add(preprocess(obs[i]), int(actions[i]), token, step)
                running[i] += reward[i]
                if done[i]:
                    returns[i].append(float(running[i])); running[i] = 0
        metrics = aggregate_episode_metrics(returns, seeds=seeds)
        payload = {'args': vars(a), 'checkpoint_sha256': sha256(a.checkpoint),
                   'alignment': 'archived leading zero plus model causal shift',
                   'unfinished_episode_returns': running.tolist(), **metrics}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, allow_nan=False))
        if a.trace_npz:
            Path(a.trace_npz).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(a.trace_npz, raw_predictions=raw_trace, written_tokens=token_trace)
        print(json.dumps(metrics['aggregate'], indent=2))
    finally: envs.close()


if __name__ == '__main__': main()

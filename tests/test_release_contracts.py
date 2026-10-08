"""Contracts for the portable release entry points, including actual information flow."""
import json
import os
import hashlib
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from e1_minigrid_diagnostics import ScalarWritebackState, aggregate_episode_metrics
from eval_cv_minigrid import History
from train_cv_minigrid import attach_target, loss_for_batch
from build_ppo_state_targets import build, parse_args, normalize_ppo_values
from minigrid_ppo_targets import derive_selected_eval_indices, derive_row_block_mapping
from dataset import MinigridDataset
from nets.net import MinigridTransformer, MinigridMultiheadTransformer


def config():
    return dict(horizon=4, state_dim=2, action_dim=7, n_embd=16, n_layer=1,
                n_head=1, dropout=0, shuffle=False, test=False, store_gpu=False, image_size=7)


def test_core_checkpoint_keys_and_logits_unchanged():
    # Obtain the original public class without shipping a duplicated model implementation.
    baseline = os.environ.get('CV_ICRL_BASELINE_NET')
    source = Path(baseline).read_text() if baseline else subprocess.check_output(['git', 'show', 'fddde4d:nets/net.py'], text=True)
    assert hashlib.sha256(source.encode()).hexdigest() == 'b46f3af2e990d8ae0e33461ac56f0218dfea06e83c45151291791c447b7d30ba'
    namespace = {}; exec(compile(source, '<original-public-net>', 'exec'), namespace)
    for name in ('MinigridTransformer', 'MinigridMultiheadTransformer'):
        torch.manual_seed(7); old = namespace[name](config()).cpu().eval()
        torch.manual_seed(7); new = globals()[name](config()).cpu().eval()
        assert list(old.state_dict()) == list(new.state_dict())
        new.load_state_dict(old.state_dict())
        batch = dict(context_states=torch.ones(2, 4, 3, 7, 7),
                     context_actions=torch.tensor([[0,1,2,3]] * 2),
                     context_rewards=torch.tensor([[[.1],[.2],[.3],[.4]]] * 2))
        # Original public model selected CUDA at import; use the CPU smoke device.
        namespace['device'] = torch.device('cpu')
        with torch.no_grad(): x, y = old(batch), new(batch)
        if not isinstance(x, tuple): x, y = (x,), (y,)
        for a, b in zip(x, y): torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_feedback_fifo_reset_and_negative_initial_prediction():
    state = ScalarWritebackState('frozen', 5)
    h = History(3, np.zeros((3, 7, 7), dtype=np.float32))
    for t, prediction in enumerate([-.2, .3, .8, .1]):
        _, token = state.update(prediction, t)
        assert token == 0.
        # A reset observation is just another observation; no reset of feedback/history.
        h.add(np.full((3, 7, 7), t, dtype=np.float32), t, token, t)
    assert list(h.ids) == [2, 3]
    np.testing.assert_array_equal(h.as_arrays(4)['context_actions'], [0, 2, 3])
    np.testing.assert_array_equal(h.as_arrays(4, 'ad')['context_actions'], [2, 3, 0])
    assert state.raw_running_max == .8


def test_equal_task_metrics_not_episode_pooled():
    result = aggregate_episode_metrics([[1.], [0.,0.,0.]])['aggregate']
    assert result['aer_mean'] == .5 and result['aer_std'] == .5
    assert result['ler_mean'] == .5


class FakeCritic:
    def load(self, path): return None
    def predict_values(self, model, observations): return observations.mean(axis=(1,2,3)).astype(np.float32)
    def release(self, model): pass


@pytest.mark.parametrize("delimiter", ["\n", ", "])
def test_ppo_targets_training_moments_and_hash_lock(tmp_path, delimiter):
    scores = np.linspace(.01, 1., 40)
    rows = derive_row_block_mapping(derive_selected_eval_indices(scores))
    labels = np.empty((17,400), dtype=np.float64)
    for row in rows:
        labels[row['row_pattern']] = np.repeat(scores[row['eval_indices']], 50)
    states = (np.arange(17*400) % 251).reshape(17,400,1,1,1)
    states = np.broadcast_to(states, (17,400,3,7,7)).astype(np.uint8)
    def write(path, observations, reward):
        with path.open('wb') as f: pickle.dump(dict(observations=observations, rewards=reward,
                                                  actions=np.zeros((17,400),dtype=np.int64)), f)
    train, val = tmp_path/'train.pkl', tmp_path/'val.pkl'
    write(train, states, labels); write(val, np.full_like(states, 3), np.zeros_like(labels))
    ckpts = tmp_path/'checkpoints'; ckpts.mkdir()
    for i in range(40): (ckpts/f'ppo_{i}_steps.zip').write_bytes(b'fake')
    score_path = tmp_path/'scores.txt'; score_path.write_text(delimiter.join(map(str, scores)))
    out = tmp_path/'targets'
    args = parse_args(['--train-data',str(train),'--val-data',str(val),'--eval-results',str(score_path),
                       '--checkpoint-dir',str(ckpts),'--output-dir',str(out)])
    build(args, FakeCritic())
    target = np.load(out/'train_ppo_state.npy')
    np.testing.assert_allclose(target.mean(), labels.mean(), atol=1e-6)
    np.testing.assert_allclose(target.std(), labels.std(), atol=1e-6)
    moments = json.loads((out/'manifest.json').read_text())['training_moments']
    expected = moments['mu_J'] + moments['sigma_J'] * (3 - moments['mu_V']) / moments['sigma_V']
    np.testing.assert_allclose(np.load(out/'val_ppo_state.npy'), expected, atol=1e-6)
    data = MinigridDataset(str(train), {**config(),'horizon':400})
    attach_target(data, train, out/'train_ppo_state.npy', 'train')
    model = MinigridMultiheadTransformer({**config(), 'horizon': 400})
    batch = {key: value[None] for key, value in data[0].items()}
    loss = loss_for_batch(model, batch, 'cv'); loss.backward()
    assert torch.isfinite(loss) and model.pred_values.weight.grad is not None
    write(train, states, labels[::-1])
    with pytest.raises(ValueError, match='hashes'): attach_target(data, train, out/'train_ppo_state.npy', 'train')
    args.output_dir = str(tmp_path/'bad_targets')
    with pytest.raises(ValueError, match='mapping'): build(args, FakeCritic())


def test_source_critic_checkpoint_reload(tmp_path):
    import gymnasium as gym
    import minigrid
    from minigrid.wrappers import ImgObsWrapper
    from stable_baselines3 import PPO
    from nets.custom_net import CustomCNN
    from minigrid_ppo_targets import SB3ValueBackend
    env = ImgObsWrapper(gym.make('MiniGrid-SimpleCrossingS9N3-v0'))
    try:
        model = PPO('CnnPolicy', env, n_steps=8, batch_size=4, device='cpu',
                    policy_kwargs=dict(features_extractor_class=CustomCNN,
                                       features_extractor_kwargs=dict(features_dim=16)))
        path = tmp_path/'ppo.zip'; model.save(path)
        backend = SB3ValueBackend('cpu'); restored = backend.load(path)
        obs, _ = env.reset(seed=0)
        values = backend.predict_values(restored, obs.transpose(2,0,1)[None])
        assert values.shape == (1,) and np.isfinite(values).all()
    finally: env.close()

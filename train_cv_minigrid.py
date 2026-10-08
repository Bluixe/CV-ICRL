"""Portable AD/CV-ICRL training using the paper MiniGrid backbone and loss."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import MinigridDataset
from nets.net import MinigridTransformer, MinigridMultiheadTransformer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env', required=True)
    p.add_argument('--train-data', required=True)
    p.add_argument('--val-data', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--method', choices=['ad', 'cv'], default='cv')
    p.add_argument('--val-target-npy', help='Validation target sidecar computed with training moments.')
    p.add_argument('--target-npy', help='Optional PPO-state target sidecar for the training corpus.')
    p.add_argument('--H', type=int, default=400)
    p.add_argument('--embd', type=int, default=256)
    p.add_argument('--layer', type=int, default=4)
    p.add_argument('--head', type=int, default=4)
    p.add_argument('--num-epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-batches', type=int, help='Smoke-only cap on batches per epoch.')
    return p.parse_args(argv)


def loss_for_batch(model, batch, method):
    actions = batch['context_actions'].squeeze(-1).long()
    prediction = model(batch)
    if method == 'cv':
        logits, values = prediction
        target = batch['context_rewards']
        if values.shape != target.shape:
            raise ValueError('Scalar prediction and target shapes differ.')
        scalar_loss = torch.nn.functional.mse_loss(values, target, reduction='sum')
    else:
        logits = prediction
        scalar_loss = logits.new_zeros(())
    action_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 7), actions.reshape(-1), reduction='sum')
    total = action_loss + scalar_loss
    if not torch.isfinite(total):
        raise FloatingPointError('Training loss is NaN or Inf.')
    return total


def attach_target(dataset, data_path, target_path, split):
    path = Path(target_path)
    manifest = json.loads((path.parent / 'manifest.json').read_text())
    record = manifest['splits'][split]
    if record['data_sha256'] != sha256(data_path) or record['target_sha256'] != sha256(path):
        raise ValueError('Target manifest does not match the data and sidecar hashes.')
    target = np.load(path, allow_pickle=False)
    expected = dataset.dataset['context_rewards'].shape
    if target.shape == tuple(expected[:2]): target = target[..., None]
    if target.shape != tuple(expected) or not np.isfinite(target).all():
        raise ValueError('Target sidecar is not aligned with rewards.')
    dataset.dataset['context_rewards'] = torch.from_numpy(np.array(target, dtype=np.float32))


def main(argv=None):
    a = parse_args(argv)
    if min(a.H, a.batch_size, a.num_epochs) < 1:
        raise ValueError('H, batch size and epochs must be positive.')
    if a.target_npy and a.method != 'cv':
        raise ValueError('A scalar target sidecar requires --method cv.')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(a.device)
    config = dict(horizon=a.H, state_dim=2, action_dim=7, n_embd=a.embd,
                  n_layer=a.layer, n_head=a.head, dropout=0, shuffle=False,
                  test=False, store_gpu=False, image_size=7)
    model_type = MinigridMultiheadTransformer if a.method == 'cv' else MinigridTransformer
    model = model_type(config).to(device)
    train = MinigridDataset(a.train_data, config)
    val = MinigridDataset(a.val_data, config)
    for data in (train, val):
        if data.dataset['context_states'].shape[1] != a.H:
            raise ValueError('Dataset row length must equal H; prepare a separate H=200 corpus.')
    if a.target_npy:
        if not a.val_target_npy: raise ValueError('PPO-state training requires a validation sidecar.')
        attach_target(train, a.train_data, a.target_npy, 'train')
        attach_target(val, a.val_data, a.val_target_npy, 'val')
    elif a.val_target_npy:
        raise ValueError('Validation target requires a training target.')
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'args': vars(a), 'model_config': config,
                'scope': 'portable implementation; new runs, not an archived checkpoint reproduction',
                'data_sha256': {key: sha256(path) for key, path in
                               [('train', a.train_data), ('val', a.val_data)]}}
    if a.target_npy:
        manifest['data_sha256']['target'] = sha256(a.target_npy)
        manifest['data_sha256']['val_target'] = sha256(a.val_target_npy)
    (out / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, a.num_epochs + 1):
        metrics = {'epoch': epoch}
        for split, data in [('train', train), ('val', val)]:
            model.train(split == 'train')
            total, count = 0., 0
            loader = DataLoader(data, batch_size=a.batch_size,
                                shuffle=split == 'train', num_workers=a.num_workers)
            with torch.set_grad_enabled(split == 'train'):
                for i, batch in enumerate(loader):
                    if a.max_batches is not None and i >= a.max_batches: break
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss = loss_for_batch(model, batch, a.method)
                    if split == 'train':
                        opt.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
                        opt.step()
                    total += loss.item(); count += batch['context_states'].shape[0] * a.H
            if count == 0: raise ValueError('No batches were processed.')
            metrics[split + '_loss_per_token'] = total / count
        torch.save({'model_state_dict': model.state_dict(), 'model_config': config,
                    'method': a.method, 'epoch': epoch}, out / f'epoch_{epoch}.pt')
        history.append(metrics)
        (out / 'metrics.json').write_text(json.dumps(history, indent=2, allow_nan=False))
        print(json.dumps(metrics), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


if __name__ == '__main__': main()

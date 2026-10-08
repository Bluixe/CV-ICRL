"""Train in-context RL (AD / auto_relabel) on Metaworld reach-v3.

Usage:
    # Standard AD
    python train_icl_metaworld.py --train_data datasets/metaworld/reach-v3/train.pkl \
                                  --test_data datasets/metaworld/reach-v3/test.pkl

    # Auto relabel (predict actions + values, weight loss by value)
    python train_icl_metaworld.py --train_data datasets/metaworld/reach-v3/train_relabel.pkl \
                                  --test_data datasets/metaworld/reach-v3/test.pkl \
                                  --algorithm auto_relabel
"""

import argparse
import os
import time

import numpy as np
import torch
import wandb
os.environ.setdefault("WANDB_MODE", "disabled")
from loguru import logger
from tqdm import tqdm

from metaworld_dataset import MetaworldDataset
from nets.metaworld_net import MetaworldTransformer, MetaworldMultiheadTransformer, MetaworldValueCondTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Metaworld reach-v3 constants
STATE_DIM = 39
ACTION_DIM = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-name", choices=["reach-v3", "push-v3"], default="reach-v3")
    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=400)
    # Model
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Training
    parser.add_argument("--algorithm", type=str, default="ad", choices=["ad", "auto_relabel", "value_cond"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--value-token-input", action="store_true",
                        help="New CV runs: teacher-force checkpoint-return tokens instead of environment rewards.")
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="metaworld-icl")
    return parser.parse_args()


def model_batch(batch, algorithm, value_token_input):
    if value_token_input:
        if algorithm != "auto_relabel" or "context_values" not in batch:
            raise ValueError("Checkpoint-return input requires auto_relabel and values.")
        return {**batch, "context_rewards": batch["context_values"]}
    return batch


def train_epoch(model, loader, optimizer, loss_fn, algorithm, action_dim, value_token_input=False):
    model.train()
    total_loss = 0.0
    total_action_loss = 0.0
    total_value_loss = 0.0

    for batch in tqdm(loader, desc="Train"):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = model_batch(batch, algorithm, value_token_input)
        true_actions = batch["context_actions"]  # (B, T, action_dim)
        optimizer.zero_grad()

        if algorithm == "ad":
            pred_actions = model(batch)  # (B, T, action_dim)
            loss = loss_fn(pred_actions, true_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        else:  # auto_relabel
            pred_actions, pred_values = model(batch)
            action_loss = loss_fn(pred_actions, true_actions)

            # Value target: context_values if available, else context_rewards
            if "context_values" in batch:
                true_values = batch["context_values"]
            else:
                true_values = batch["context_rewards"]
            value_loss = torch.nn.functional.mse_loss(pred_values, true_values)

            # Weight action loss by value (higher value = more important)
            loss = action_loss + value_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_action_loss += action_loss.item()
            total_value_loss += value_loss.item()

    n = len(loader)
    metrics = {"train/loss": total_loss / n}
    if algorithm == "auto_relabel":
        metrics["train/action_loss"] = total_action_loss / n
        metrics["train/value_loss"] = total_value_loss / n
    return metrics


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, algorithm, action_dim, value_token_input=False):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc="Eval"):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = model_batch(batch, algorithm, value_token_input)
        true_actions = batch["context_actions"]

        if algorithm == "ad":
            pred_actions = model(batch)
            loss = loss_fn(pred_actions, true_actions)
        else:
            pred_actions, pred_values = model(batch)
            action_loss = loss_fn(pred_actions, true_actions)
            if "context_values" in batch:
                true_values = batch["context_values"]
            else:
                true_values = batch["context_rewards"]
            value_loss = torch.nn.functional.mse_loss(pred_values, true_values)
            loss = action_loss + value_loss

        total_loss += loss.item()

    return {"eval/loss": total_loss / len(loader)}


def main():
    args = parse_args()
    if args.value_token_input and args.algorithm != "auto_relabel":
        raise ValueError("--value-token-input requires --algorithm auto_relabel.")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Config
    config = {
        "horizon": args.horizon,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "n_embd": args.n_embd,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "dropout": args.dropout,
    }

    # Model
    include_values = args.algorithm in ("auto_relabel", "value_cond")
    if args.algorithm == "ad":
        model = MetaworldTransformer(config).to(device)
    elif args.algorithm == "auto_relabel":
        model = MetaworldMultiheadTransformer(config).to(device)
    elif args.algorithm == "value_cond":
        model = MetaworldValueCondTransformer(config).to(device)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Data
    train_dataset = MetaworldDataset(args.train_data, config, include_values=include_values)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_kwargs)

    test_loader = None
    if args.test_data:
        test_dataset = MetaworldDataset(args.test_data, config, include_values=include_values)
        test_loader = torch.utils.data.DataLoader(test_dataset, **loader_kwargs)

    # Optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()  # Continuous actions → MSE

    # Wandb
    wandb_name = f"{args.env_name}-{args.algorithm}-H{args.horizon}"
    wandb.init(
        project=args.wandb_project,
        name=wandb_name,
        config=vars(args),
    )

    # Training loop
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=False)
    import json
    from pathlib import Path
    (Path(save_dir) / "run_manifest.json").write_text(json.dumps({"args": vars(args), "scalar_input": "checkpoint_return" if args.value_token_input else "environment_reward"}, indent=2))

    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.num_epochs}")

        # Eval first
        if test_loader is not None:
            eval_metrics = eval_epoch(model, test_loader, loss_fn, args.algorithm, ACTION_DIM, args.value_token_input)
            logger.info(f"  Eval loss: {eval_metrics['eval/loss']:.6f}")
            wandb.log({"epoch": epoch + 1, **eval_metrics})

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, args.algorithm, ACTION_DIM, args.value_token_input)
        logger.info(f"  Train loss: {train_metrics['train/loss']:.6f}")
        wandb.log({"epoch": epoch + 1, **train_metrics})

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 5 == 0:
            path = os.path.join(save_dir, f"epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), path)
            logger.info(f"  Saved {path}")

    # Save final model
    final_path = os.path.join(save_dir, "final.pt")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training complete. Final model: {final_path}")
    wandb.finish()


if __name__ == "__main__":
    main()

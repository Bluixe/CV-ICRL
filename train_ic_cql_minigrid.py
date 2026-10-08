"""Train a context-conditioned Conservative Q-Learning policy on MiniGrid.

This is intentionally separate from the repository's legacy state-only CQL,
SICQL, and delta-v implementations.  It consumes the same cross-episode
learning histories as AD/CV-ICRL and learns Q(h_t, a_t) with a terminal-masked
Bellman loss plus the discrete CQL regularizer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler

from minigrid_ic_cql_dataset import MinigridICCQLDataset
from nets.ic_cql_minigrid import (
    IC_CQL_CHECKPOINT_FORMAT_VERSION,
    IC_CQL_CONTRACT_VERSION,
    IC_CQL_POLICY_Q_RULE,
    MinigridICCQLTransformer,
    compute_ic_cql_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train matched-backbone IC-CQL on MiniGrid histories."
    )
    parser.add_argument("--env", required=True)
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--val-data", default=None)
    parser.add_argument(
        "--train-histories-per-stream",
        type=int,
        required=True,
        help="Adjacent fixed-length rows belonging to one continuous training stream.",
    )
    parser.add_argument(
        "--val-histories-per-stream",
        type=int,
        required=True,
        help="Adjacent fixed-length rows belonging to one continuous validation stream.",
    )
    parser.add_argument(
        "--train-expected-sha256",
        default=None,
        help="Optional lowercase SHA256 that the complete training dataset must match.",
    )
    parser.add_argument(
        "--val-expected-sha256",
        default=None,
        help="Optional lowercase SHA256 that the complete validation dataset must match.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--H", type=int, default=400, dest="horizon")
    parser.add_argument("--embd", type=int, default=256)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--tuple-mode",
        choices=("paper", "matched"),
        default="paper",
        help=(
            "paper adds previous done and the current float episode step; "
            "matched uses exactly the AD (state, previous action, previous reward) tuple."
        ),
    )
    parser.add_argument(
        "--episode-step-scale",
        type=float,
        default=1.0,
        help=(
            "Divisor for the paper-mode float episode-step scalar. The official "
            "configuration uses 1.0, i.e. the raw step count."
        ),
    )

    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument(
        "--cql-weight",
        type=float,
        default=0.01,
        help=(
            "Weight on the discrete CQL regularizer. The default is the "
            "paper-selected XLand-MiniGrid value; formal launchers record it."
        ),
    )
    parser.add_argument(
        "--cql-label-smoothing",
        type=float,
        default=0.3,
        help=(
            "Label smoothing for each discrete-Q cross-entropy CQL term. "
            "The XLand formal configuration uses the inferred paper value 0.3."
        ),
    )
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--max-train-histories", type=int, default=None)
    parser.add_argument("--max-val-histories", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--skip-data-sha256",
        action="store_true",
        help="Only for quick diagnostics; formal runs must fingerprint the full files.",
    )

    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--wandb-project", default="minigrid-ic-cql")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def git_metadata() -> Dict[str, Any]:
    def run_git(*parts: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *parts],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    status = run_git("status", "--porcelain")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": status not in ("", "unknown"),
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def load_trusted_checkpoint(path: str | Path) -> Dict[str, Any]:
    """Load a checkpoint produced by this script across PyTorch 2.x defaults."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch releases before the weights_only argument was introduced.
        return torch.load(path, map_location="cpu")


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def finite_or_raise(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        detached = value.detach()
        raise FloatingPointError(
            f"{name} became non-finite: "
            f"min={detached.nan_to_num().min().item():.6g}, "
            f"max={detached.nan_to_num().max().item():.6g}"
        )


def online_parameters(model: MinigridICCQLTransformer) -> Iterable[torch.nn.Parameter]:
    for name, parameter in model.named_parameters():
        if (
            not name.startswith(("target_q1_head.", "target_q2_head."))
            and parameter.requires_grad
        ):
            yield parameter


def make_loader(
    dataset: MinigridICCQLDataset,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    sampler_generator: Optional[torch.Generator],
    worker_generator: torch.Generator,
) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
        "generator": worker_generator,
    }
    if shuffle:
        if sampler_generator is None:
            raise ValueError("A sampler generator is required for shuffled loading.")
        kwargs["sampler"] = RandomSampler(
            dataset,
            generator=sampler_generator,
        )
    if args.num_workers > 0:
        kwargs.update(
            {
                "prefetch_factor": args.prefetch_factor,
                "persistent_workers": True,
            }
        )
    return DataLoader(**kwargs)


@torch.no_grad()
def validation_epoch(
    model: MinigridICCQLTransformer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    batches = 0
    for batch_index, batch in enumerate(loader):
        if args.max_val_batches is not None and batch_index >= args.max_val_batches:
            break
        batch = move_batch(batch, device)
        hidden = model.encode_context(batch)
        q1_values = model.q1_head(hidden)
        q2_values = model.q2_head(hidden)
        target_q1_values = model.target_q1_head(hidden.detach())
        target_q2_values = model.target_q2_head(hidden.detach())
        loss, metrics = compute_ic_cql_loss(
            q1_values=q1_values,
            q2_values=q2_values,
            target_q1_values=target_q1_values,
            target_q2_values=target_q2_values,
            batch=batch,
            gamma=args.gamma,
            cql_weight=args.cql_weight,
            cql_label_smoothing=args.cql_label_smoothing,
        )
        finite_or_raise("validation loss", loss)
        batches += 1
        for key, value in metrics.items():
            scalar = float(value.detach().cpu().item() if torch.is_tensor(value) else value)
            totals[key] = totals.get(key, 0.0) + scalar
    if batches == 0:
        raise RuntimeError("Validation loader produced no batches.")
    return {f"val/{key}": value / batches for key, value in totals.items()}


def rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: Path,
    *,
    model: MinigridICCQLTransformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_batch: int,
    global_update: int,
    best_val_loss: float,
    args: argparse.Namespace,
    model_config: Dict[str, Any],
    run_id: Optional[str],
    provenance: Dict[str, Any],
    contract: Dict[str, Any],
    loader_generator_state: Dict[str, torch.Tensor],
) -> None:
    payload = {
        "format_version": IC_CQL_CHECKPOINT_FORMAT_VERSION,
        "algorithm": "IC-CQL",
        "twin_q": True,
        "policy_q_rule": IC_CQL_POLICY_Q_RULE,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "global_batch": global_batch,
        "global_update": global_update,
        "best_val_loss": best_val_loss,
        "args": vars(args),
        "model_config": model_config,
        "wandb_run_id": run_id,
        "provenance": provenance,
        "contract": contract,
        "loader_generator_state": loader_generator_state,
        "rng_state": rng_state(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def cuda_memory_stats(device: torch.device) -> Dict[str, int]:
    if device.type != "cuda":
        return {}
    return {
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be at least 1.")
    if args.cql_weight < 0.0:
        raise ValueError("--cql-weight must be non-negative.")
    if not 0.0 <= args.cql_label_smoothing < 1.0:
        raise ValueError("--cql-label-smoothing must be in [0, 1).")
    if args.episode_step_scale <= 0.0:
        raise ValueError("--episode-step-scale must be positive.")
    if not 0.0 <= args.target_tau <= 1.0:
        raise ValueError("--target-tau must be in [0, 1].")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        raise ValueError("--max-train-batches must be positive.")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(
        args.train_data
        or f"datasets/MiniGrid/{args.env}/train_traj-more.pkl"
    ).expanduser().resolve()
    val_path = Path(
        args.val_data
        or f"datasets/MiniGrid/{args.env}/test_traj.pkl"
    ).expanduser().resolve()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    if device.type == "cuda":
        logger.info(
            "Using CUDA device {} ({})",
            torch.cuda.current_device(),
            torch.cuda.get_device_name(torch.cuda.current_device()),
        )

    logger.info("Loading strict IC-CQL training dataset: {}", train_path)
    train_dataset = MinigridICCQLDataset(
        train_path,
        max_histories=args.max_train_histories,
        histories_per_stream=args.train_histories_per_stream,
        compute_sha256=not args.skip_data_sha256,
        expected_sha256=args.train_expected_sha256,
    )
    logger.info("Loading strict IC-CQL validation dataset: {}", val_path)
    val_dataset = MinigridICCQLDataset(
        val_path,
        max_histories=args.max_val_histories,
        histories_per_stream=args.val_histories_per_stream,
        compute_sha256=not args.skip_data_sha256,
        expected_sha256=args.val_expected_sha256,
    )
    if train_dataset.sequence_length != args.horizon:
        raise ValueError(
            f"Training sequence length {train_dataset.sequence_length} != H={args.horizon}; "
            "IC-CQL does not silently crop or pad histories."
        )
    if val_dataset.sequence_length != args.horizon:
        raise ValueError(
            f"Validation sequence length {val_dataset.sequence_length} != H={args.horizon}."
        )

    episode_step_scale = float(args.episode_step_scale)
    model_config: Dict[str, Any] = {
        "horizon": args.horizon,
        "state_dim": 2,
        "action_dim": 7,
        "n_layer": args.layer,
        "n_embd": args.embd,
        "n_head": args.head,
        "dropout": args.dropout,
        "image_size": 7,
        "episode_step_scale": episode_step_scale,
        "episode_step_encoding": (
            "raw_float" if episode_step_scale == 1.0 else "scaled_float"
        ),
        "twin_q": True,
    }
    model = MinigridICCQLTransformer(
        model_config,
        tuple_mode=args.tuple_mode,
    ).to(device)

    target_parameter_ids = {
        id(parameter)
        for head in (model.target_q1_head, model.target_q2_head)
        for parameter in head.parameters()
    }
    optim_parameters = list(online_parameters(model))
    if target_parameter_ids.intersection(map(id, optim_parameters)):
        raise AssertionError("Target Q-head parameters leaked into the optimizer.")
    optimizer = torch.optim.Adam(
        optim_parameters,
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    provenance: Dict[str, Any] = {
        "git": git_metadata(),
        "train_dataset": train_dataset.fingerprint,
        "val_dataset": val_dataset.fingerprint,
        "selected_train_histories": len(train_dataset),
        "selected_val_histories": len(val_dataset),
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "method_provenance": {
            "cql_label_smoothing": {
                "value": args.cql_label_smoothing,
                "evidence": (
                    "Strong inference from the paper hyperparameter table and "
                    "reuse of the AD XLand-MiniGrid hyperparameters; not directly "
                    "confirmed by the official-code audit."
                ),
                "evidence_strength": "strong_inference",
            }
        },
    }
    contract: Dict[str, Any] = {
        "contract_version": IC_CQL_CONTRACT_VERSION,
        "checkpoint_format_version": IC_CQL_CHECKPOINT_FORMAT_VERSION,
        "algorithm": "IC-CQL",
        "twin_q": True,
        "policy_q_rule": IC_CQL_POLICY_Q_RULE,
        "model_config": model_config,
        "tuple_mode": args.tuple_mode,
        "loss": {
            "td_target": (
                "r_t + gamma * (1-done_t) * "
                "min(max_a Q1_target(h_{t+1},a), max_a Q2_target(h_{t+1},a))"
            ),
            "td_loss": "MSE(Q1_data,y) + MSE(Q2_data,y)",
            "cql_loss": (
                "CE(Q1,a,label_smoothing=e) + CE(Q2,a,label_smoothing=e)"
            ),
        },
        "optimization": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "lr": args.lr,
            "adam_beta1": args.adam_beta1,
            "adam_beta2": args.adam_beta2,
            "weight_decay": args.weight_decay,
            "gamma": args.gamma,
            "cql_weight": args.cql_weight,
            "cql_label_smoothing": args.cql_label_smoothing,
            "target_tau": args.target_tau,
            "grad_clip": args.grad_clip,
            "amp": amp_enabled,
            "seed": args.seed,
        },
        "train_dataset": train_dataset.fingerprint,
        "val_dataset": val_dataset.fingerprint,
        "git_commit": provenance["git"]["commit"],
    }
    manifest = {
        "checkpoint_format_version": IC_CQL_CHECKPOINT_FORMAT_VERSION,
        "contract_version": IC_CQL_CONTRACT_VERSION,
        "algorithm": "IC-CQL",
        "twin_q": True,
        "policy_q_rule": IC_CQL_POLICY_Q_RULE,
        "args": vars(args),
        "model_config": model_config,
        "provenance": provenance,
        "contract": contract,
        "status": "initialized",
        "created_at_unix": time.time(),
    }
    resume_id: Optional[str] = None
    resume_payload: Optional[Dict[str, Any]] = None
    if args.resume:
        resume_payload = load_trusted_checkpoint(args.resume)
        if (
            resume_payload.get("format_version")
            != IC_CQL_CHECKPOINT_FORMAT_VERSION
            or resume_payload.get("twin_q") is not True
            or resume_payload.get("policy_q_rule") != IC_CQL_POLICY_Q_RULE
        ):
            raise RuntimeError(
                "Resume checkpoint is not an official twin-Q IC-CQL checkpoint."
            )
        if resume_payload.get("contract") != contract:
            raise RuntimeError(
                "Resume checkpoint contract does not exactly match the current "
                "model, optimizer, data fingerprints, and git commit."
            )
        resume_id = resume_payload.get("wandb_run_id")

    # Do not overwrite an existing run manifest until an explicitly requested
    # resume checkpoint has passed its immutable format and contract checks.
    atomic_json_dump(manifest, output_dir / "run_manifest.json")

    short_env = args.env.replace("MiniGrid-", "").replace("-v0", "")
    run_name = args.wandb_name or (
        f"ic-cql-{short_env}-{args.tuple_mode}-seed{args.seed}"
    )
    run_group = args.wandb_group or f"ic-cql-{short_env}-{args.tuple_mode}"
    start_epoch = 0
    global_batch = 0
    global_update = 0
    best_val_loss = math.inf
    train_sampler_generator = torch.Generator()
    train_sampler_generator.manual_seed(args.seed)
    train_worker_generator = torch.Generator()
    train_worker_generator.manual_seed(args.seed + 1)
    val_worker_generator = torch.Generator()
    val_worker_generator.manual_seed(args.seed + 2)
    run = None
    try:
        if resume_payload is not None:
            model.load_state_dict(resume_payload["model_state"])
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            scaler.load_state_dict(resume_payload.get("scaler_state", {}))
            start_epoch = int(resume_payload["epoch"]) + 1
            global_batch = int(resume_payload.get("global_batch", 0))
            global_update = int(resume_payload.get("global_update", 0))
            best_val_loss = float(resume_payload.get("best_val_loss", math.inf))
            loader_state = resume_payload.get("loader_generator_state")
            required_loader_states = {
                "train_sampler",
                "train_worker",
                "val_worker",
            }
            if not loader_state or not required_loader_states.issubset(loader_state):
                raise RuntimeError(
                    "Resume checkpoint lacks deterministic DataLoader generator state."
                )
            train_sampler_generator.set_state(loader_state["train_sampler"])
            train_worker_generator.set_state(loader_state["train_worker"])
            val_worker_generator.set_state(loader_state["val_worker"])
            logger.info(
                "Resumed exact contract from epoch {} at update {}",
                start_epoch,
                global_update,
            )

        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=run_group,
            mode=args.wandb_mode,
            id=resume_id,
            resume="allow" if resume_id else None,
            config={
                **vars(args),
                "model_config": model_config,
                "provenance": provenance,
                "contract": contract,
            },
            tags=["ic-cql", "twin-q", "cross-episode", args.tuple_mode],
        )
        train_loader = make_loader(
            train_dataset,
            args,
            shuffle=True,
            sampler_generator=train_sampler_generator,
            worker_generator=train_worker_generator,
        )
        val_loader = make_loader(
            val_dataset,
            args,
            shuffle=False,
            sampler_generator=None,
            worker_generator=val_worker_generator,
        )
        if resume_payload is not None:
            # Restore global RNG only after W&B and loader construction so their
            # initialization cannot perturb the resumed optimization stream.
            restore_rng_state(resume_payload.get("rng_state"))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(start_epoch, args.num_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_totals: Dict[str, float] = {}
            epoch_batches = 0
            epoch_start = time.time()

            expected_batches = len(train_loader)
            if args.max_train_batches is not None:
                expected_batches = min(expected_batches, args.max_train_batches)

            for batch_index, batch in enumerate(train_loader):
                if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                    break
                batch = move_batch(batch, device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    hidden = model.encode_context(batch)
                    q1_values = model.q1_head(hidden)
                    q2_values = model.q2_head(hidden)
                    with torch.no_grad():
                        target_q1_values = model.target_q1_head(hidden.detach())
                        target_q2_values = model.target_q2_head(hidden.detach())
                    loss, metrics = compute_ic_cql_loss(
                        q1_values=q1_values,
                        q2_values=q2_values,
                        target_q1_values=target_q1_values,
                        target_q2_values=target_q2_values,
                        batch=batch,
                        gamma=args.gamma,
                        cql_weight=args.cql_weight,
                        cql_label_smoothing=args.cql_label_smoothing,
                    )
                    finite_or_raise("training loss", loss)
                    group_start = (
                        (epoch_batches // args.grad_accum_steps)
                        * args.grad_accum_steps
                    )
                    actual_group_size = min(
                        args.grad_accum_steps,
                        expected_batches - group_start,
                    )
                    scaled_loss = loss / actual_group_size

                scaler.scale(scaled_loss).backward()
                global_batch += 1
                epoch_batches += 1
                should_step = (
                    (epoch_batches % args.grad_accum_steps == 0)
                    or (epoch_batches == expected_batches)
                )

                grad_norm_value = float("nan")
                if should_step:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        optim_parameters,
                        args.grad_clip,
                    )
                    finite_or_raise("gradient norm", torch.as_tensor(grad_norm, device=device))
                    grad_norm_value = float(torch.as_tensor(grad_norm).detach().cpu().item())
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    model.soft_update_target(args.target_tau)
                    global_update += 1

                scalar_metrics = {
                    key: float(
                        value.detach().cpu().item()
                        if torch.is_tensor(value)
                        else value
                    )
                    for key, value in metrics.items()
                }
                for key, value in scalar_metrics.items():
                    epoch_totals[key] = epoch_totals.get(key, 0.0) + value

                if (
                    should_step
                    and (global_update == 1 or global_update % args.log_every == 0)
                ):
                    log_payload = {
                        f"train/{key}": value for key, value in scalar_metrics.items()
                    }
                    log_payload.update(
                        {
                            "train/grad_norm": grad_norm_value,
                            "train/epoch": epoch,
                            "train/global_batch": global_batch,
                            "train/global_update": global_update,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        }
                    )
                    run.log(log_payload, step=global_update)

            if epoch_batches == 0:
                raise RuntimeError("Training loader produced no batches.")

            train_summary = {
                f"epoch/train_{key}": value / epoch_batches
                for key, value in epoch_totals.items()
            }
            val_summary = validation_epoch(model, val_loader, device, args)
            epoch_payload = {
                **train_summary,
                **val_summary,
                "epoch/index": epoch,
                "epoch/wall_seconds": time.time() - epoch_start,
                "epoch/global_update": global_update,
            }
            run.log(epoch_payload, step=global_update)
            logger.info(
                "Epoch {}/{}: train total={:.6f}, val total={:.6f}, {:.1f}s",
                epoch + 1,
                args.num_epochs,
                train_summary["epoch/train_loss/total"],
                val_summary["val/loss/total"],
                epoch_payload["epoch/wall_seconds"],
            )

            val_total = val_summary["val/loss/total"]
            if val_total < best_val_loss:
                best_val_loss = val_total
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_batch=global_batch,
                    global_update=global_update,
                    best_val_loss=best_val_loss,
                    args=args,
                    model_config=model_config,
                    run_id=run.id,
                    provenance=provenance,
                    contract=contract,
                    loader_generator_state={
                        "train_sampler": train_sampler_generator.get_state(),
                        "train_worker": train_worker_generator.get_state(),
                        "val_worker": val_worker_generator.get_state(),
                    },
                )

            save_checkpoint(
                output_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_batch=global_batch,
                global_update=global_update,
                best_val_loss=best_val_loss,
                args=args,
                model_config=model_config,
                run_id=run.id,
                provenance=provenance,
                contract=contract,
                loader_generator_state={
                    "train_sampler": train_sampler_generator.get_state(),
                    "train_worker": train_worker_generator.get_state(),
                    "val_worker": val_worker_generator.get_state(),
                },
            )
            if (epoch + 1) % args.checkpoint_every == 0:
                save_checkpoint(
                    output_dir / f"epoch_{epoch + 1}.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    global_batch=global_batch,
                    global_update=global_update,
                    best_val_loss=best_val_loss,
                    args=args,
                    model_config=model_config,
                    run_id=run.id,
                    provenance=provenance,
                    contract=contract,
                    loader_generator_state={
                        "train_sampler": train_sampler_generator.get_state(),
                        "train_worker": train_worker_generator.get_state(),
                        "val_worker": val_worker_generator.get_state(),
                    },
                )

        save_checkpoint(
            output_dir / "final.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=args.num_epochs - 1,
            global_batch=global_batch,
            global_update=global_update,
            best_val_loss=best_val_loss,
            args=args,
            model_config=model_config,
            run_id=run.id,
            provenance=provenance,
            contract=contract,
            loader_generator_state={
                "train_sampler": train_sampler_generator.get_state(),
                "train_worker": train_worker_generator.get_state(),
                "val_worker": val_worker_generator.get_state(),
            },
        )
        resource_metrics = cuda_memory_stats(device)
        manifest.update(
            {
                "status": "completed",
                "completed_at_unix": time.time(),
                "wandb_run_id": run.id,
                "global_update": global_update,
                "best_val_loss": best_val_loss,
                "resource_metrics": resource_metrics,
            }
        )
        atomic_json_dump(manifest, output_dir / "run_manifest.json")
        run.summary.update(
            {
                "status": "completed",
                "best_val_loss": best_val_loss,
                "global_update": global_update,
                "final_checkpoint": str(output_dir / "final.pt"),
                **resource_metrics,
            }
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at_unix": time.time(),
                "wandb_run_id": run.id if run else None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "global_update": global_update,
                "resource_metrics": cuda_memory_stats(device),
            }
        )
        atomic_json_dump(manifest, output_dir / "run_manifest.json")
        if run:
            run.summary.update(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        raise
    finally:
        if run:
            run.finish()


if __name__ == "__main__":
    main()

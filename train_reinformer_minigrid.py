"""Train an architecture-matched Reinformer adaptation on MiniGrid histories."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, RandomSampler

from minigrid_reinformer_dataset import MinigridReinformerDataset
from nets.reinformer_minigrid import (
    REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3,
    REINFORMER_CHECKPOINT_FORMAT_VERSION,
    REINFORMER_CONTRACT_VERSION,
    MinigridReinformer,
    compute_reinformer_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train an independent discrete Reinformer adaptation on the original "
            "cross-episode MiniGrid corpus."
        )
    )
    parser.add_argument("--env", required=True)
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--train-histories-per-stream", type=int, required=True)
    parser.add_argument("--val-histories-per-stream", type=int, required=True)
    parser.add_argument("--train-expected-sha256", default=None)
    parser.add_argument("--val-expected-sha256", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default=None)

    parser.add_argument("--H", type=int, default=400, dest="horizon")
    parser.add_argument("--embd", type=int, default=256)
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument("--head", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument(
        "--action-head-type",
        choices=(REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3,),
        default=REINFORMER_ACTION_HEAD_SINGLE_PASS_LINEAR_V3,
        help=(
            "The formal v3 contract uses one Transformer pass and an exact "
            "Linear(embd + 1, action_dim, bias=False) action head whose "
            "parameter count matches CV-ICRL."
        ),
    )

    parser.add_argument("--num-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--warmup-updates", type=int, default=5000)
    parser.add_argument("--expectile", type=float, default=0.99)
    parser.add_argument("--rtg-loss-weight", type=float, default=1.0)
    parser.add_argument("--rtg-gamma", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=("none", "float16", "bfloat16"),
        default="bfloat16",
    )

    parser.add_argument("--max-train-histories", type=int, default=None)
    parser.add_argument("--max-val-histories", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument(
        "--skip-data-sha256",
        action="store_true",
        help="Smoke-only option; formal runs must authenticate the complete pickle.",
    )

    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--wandb-project", default="minigrid-reinformer")
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
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


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
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json_dump(payload: Dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_trusted_checkpoint(path: str | Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def move_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def finite_or_raise(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        detached = value.detach().float()
        raise FloatingPointError(
            f"{name} became non-finite: "
            f"min={detached.nan_to_num().min().item():.6g}, "
            f"max={detached.nan_to_num().max().item():.6g}"
        )


def make_loader(
    dataset: MinigridReinformerDataset,
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
            raise ValueError("Shuffled loading requires a sampler generator.")
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


def amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def extra_diagnostics(
    model: MinigridReinformer,
    batch: Dict[str, torch.Tensor],
    *,
    perturbation: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """Measure teacher/self-feed shift and structural RTG conditioning."""

    with torch.no_grad():
        hidden = model.encode_context(batch)
        predictions = model.predict_rtg(hidden)
        targets = batch["context_rtgs"]
        if targets.ndim == 2:
            targets = targets.unsqueeze(-1)
        teacher_logits = model.action_logits_for_rtgs(
            batch,
            targets,
            prediction_hidden=hidden,
        )
        predicted_logits = model.action_logits_for_rtgs(
            batch,
            predictions,
            prediction_hidden=hidden,
        )
        perturbed_logits = model.action_logits_for_rtgs(
            batch,
            targets + perturbation,
            prediction_hidden=hidden,
        )
        valid_mask = batch["rtg_valid_mask"].bool()
        teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
        predicted_log_probs = torch.log_softmax(predicted_logits, dim=-1)
        teacher_probs = teacher_log_probs.exp()
        kl = (
            teacher_probs * (teacher_log_probs - predicted_log_probs)
        ).sum(dim=-1)[valid_mask].mean()
        changed = (
            teacher_logits.argmax(dim=-1) != predicted_logits.argmax(dim=-1)
        )[valid_mask].float().mean()
        perturb_delta = (
            perturbed_logits - teacher_logits
        ).abs()[valid_mask].mean()
        return {
            "diagnostic/teacher_predicted_action_kl": kl,
            "diagnostic/teacher_predicted_action_change": changed,
            "diagnostic/rtg_perturb_logit_abs_delta": perturb_delta,
            "diagnostic/rtg_prediction_mae": (
                predictions - targets
            ).abs()[valid_mask].mean(),
        }


@torch.no_grad()
def validation_epoch(
    model: MinigridReinformer,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    *,
    rtg_normalizer: float,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    batches = 0
    last_batch: Optional[Dict[str, torch.Tensor]] = None
    for batch_index, batch in enumerate(loader):
        if args.max_val_batches is not None and batch_index >= args.max_val_batches:
            break
        batch = move_batch(batch, device)
        with amp_context(device, args.amp_dtype):
            action_logits, rtg_predictions = model(
                batch,
                teacher_force_action_rtg=True,
            )
            loss, metrics = compute_reinformer_loss(
                action_logits,
                rtg_predictions,
                batch,
                expectile=args.expectile,
                rtg_loss_weight=args.rtg_loss_weight,
                rtg_normalizer=rtg_normalizer,
            )
        finite_or_raise("validation loss", loss)
        batches += 1
        last_batch = batch
        for key, value in metrics.items():
            scalar = float(value.detach().float().cpu().item())
            totals[key] = totals.get(key, 0.0) + scalar
    if batches == 0 or last_batch is None:
        raise RuntimeError("Validation loader produced no batches.")
    result = {f"val/{key}": value / batches for key, value in totals.items()}
    diagnostics = extra_diagnostics(model, last_batch)
    result.update(
        {
            f"val/{key}": float(value.detach().float().cpu().item())
            for key, value in diagnostics.items()
        }
    )
    return result


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
    model: MinigridReinformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
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
        "format_version": REINFORMER_CHECKPOINT_FORMAT_VERSION,
        "contract_version": REINFORMER_CONTRACT_VERSION,
        "algorithm": "Reinformer-MiniGrid",
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": int(epoch),
        "global_batch": int(global_batch),
        "global_update": int(global_update),
        "best_val_loss": float(best_val_loss),
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def cuda_memory_stats(device: torch.device) -> Dict[str, int]:
    if device.type != "cuda":
        return {}
    return {
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be at least one.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.num_epochs < 1:
        raise ValueError("--num-epochs must be positive.")
    if args.warmup_updates < 0:
        raise ValueError("--warmup-updates must be non-negative.")
    if not 0.5 <= args.expectile < 1.0:
        raise ValueError("--expectile must lie in [0.5, 1).")
    if args.rtg_loss_weight < 0.0:
        raise ValueError("--rtg-loss-weight must be non-negative.")
    if not 0.0 <= args.rtg_gamma <= 1.0:
        raise ValueError("--rtg-gamma must lie in [0, 1].")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be positive.")
    if args.checkpoint_every < 1:
        raise ValueError("--checkpoint-every must be positive.")
    for name in ("max_train_histories", "max_val_histories"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("max_train_batches", "max_val_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
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
        if args.amp_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 AMP was requested but this GPU does not support it.")

    logger.info("Loading Reinformer training data from {}", train_path)
    train_dataset = MinigridReinformerDataset(
        train_path,
        histories_per_stream=args.train_histories_per_stream,
        rtg_gamma=args.rtg_gamma,
        max_histories=args.max_train_histories,
        compute_sha256=not args.skip_data_sha256,
        expected_sha256=args.train_expected_sha256,
    )
    logger.info("Loading Reinformer validation data from {}", val_path)
    val_dataset = MinigridReinformerDataset(
        val_path,
        histories_per_stream=args.val_histories_per_stream,
        rtg_gamma=args.rtg_gamma,
        max_histories=args.max_val_histories,
        compute_sha256=not args.skip_data_sha256,
        expected_sha256=args.val_expected_sha256,
    )
    if train_dataset.sequence_length != args.horizon:
        raise ValueError(
            f"Training sequence length {train_dataset.sequence_length} != H={args.horizon}."
        )
    if val_dataset.sequence_length != args.horizon:
        raise ValueError(
            f"Validation sequence length {val_dataset.sequence_length} != H={args.horizon}."
        )
    train_rtg_abs_mean = float(
        train_dataset.fingerprint["rtg_valid_abs_mean"]
    )
    rtg_loss_normalizer = (
        train_rtg_abs_mean if train_rtg_abs_mean > 0.0 else 1.0
    )
    if not math.isfinite(rtg_loss_normalizer) or rtg_loss_normalizer <= 0.0:
        raise FloatingPointError(
            "Training-dataset RTG loss normalizer must be finite and positive."
        )

    model_config: Dict[str, Any] = {
        "horizon": args.horizon,
        "n_embd": args.embd,
        "n_layer": args.layer,
        "n_head": args.head,
        "action_dim": args.action_dim,
        "dropout": args.dropout,
        "image_size": 7,
        "action_head_type": args.action_head_type,
    }
    model = MinigridReinformer(model_config).to(device)
    capacity_audit = model.capacity_audit()
    expected_action_head_parameters = (
        args.embd * args.action_dim + args.action_dim
    )
    if (
        capacity_audit["action_output_head_parameters"]
        != expected_action_head_parameters
        or capacity_audit["cv_icrl_action_head_parameters"]
        != expected_action_head_parameters
    ):
        raise RuntimeError(
            "Reinformer action output head is not parameter-matched to CV-ICRL."
        )
    if (
        capacity_audit["parameter_delta_vs_cv_matched_architecture"]
        != args.embd
    ):
        raise RuntimeError(
            "The v3 Reinformer may exceed the matched CV architecture only by "
            "one RTG-scalar transition column."
        )
    if capacity_audit["transformer_passes_per_action"] != 1:
        raise RuntimeError(
            "The compute-matched v3 Reinformer must use one Transformer pass."
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    def warmup_lambda(update_index: int) -> float:
        if args.warmup_updates == 0:
            return 1.0
        return min((update_index + 1) / args.warmup_updates, 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and args.amp_dtype == "float16")
    )
    provenance = {
        "git": git_metadata(),
        "official_reinformer_reference": {
            "repository": "https://github.com/Dragon-Zhuang/Reinformer",
            "audited_commit": "61eaed92999239d9a3b1324a449e32c57a4c49ec",
            "adaptation_status": (
                "independent_discrete_minigrid_single_pass_linear_adaptation"
            ),
        },
        "created_at_unix": time.time(),
    }
    effective_batch_size = args.batch_size * args.grad_accum_steps
    contract: Dict[str, Any] = {
        "contract_version": REINFORMER_CONTRACT_VERSION,
        "checkpoint_format_version": REINFORMER_CHECKPOINT_FORMAT_VERSION,
        "algorithm": "Reinformer-MiniGrid",
        "comparison_scope": (
            "independent capacity- and evaluation-aligned MiniGrid adaptation"
        ),
        "model_config": model_config,
        "capacity_audit": capacity_audit,
        "history_token": "(state_t, action_t-1, reward_t-1, rtg_t-1)",
        "rtg_prediction": "causal hidden_t -> high-expectile RTG_t prediction",
        "rtg_history_input": (
            "predecessor RTG, zeroed whenever the predecessor transition is terminal"
        ),
        "action_conditioning": (
            "same hidden state concatenated with current teacher RTG_t during "
            "training or predicted RTG_t during evaluation"
        ),
        "action_output_head": (
            f"Linear({args.embd + 1}, {args.action_dim}, bias=False)"
        ),
        "transformer_passes_per_action": 1,
        "rtg_target": {
            "source": "original trajectory rewards and dones",
            "gamma": args.rtg_gamma,
            "reset_on_done": True,
            "cross_row_continuity": True,
            "cross_stream_continuity": False,
        },
        "loss": {
            "rtg": f"expectile regression, tau={args.expectile}",
            "action": "categorical negative log-likelihood",
            "rtg_loss_weight": args.rtg_loss_weight,
            "rtg_normalizer": rtg_loss_normalizer,
            "rtg_normalizer_source": (
                "training_dataset_valid_rtg_abs_mean"
                if train_rtg_abs_mean > 0.0
                else "unit_fallback_for_all_zero_training_rtgs"
            ),
            "batch_dependent_rtg_normalization": False,
        },
        "optimization": {
            "optimizer": "AdamW",
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_updates": args.warmup_updates,
            "num_epochs": args.num_epochs,
            "micro_batch_size": args.batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "effective_batch_size": effective_batch_size,
            "grad_clip": args.grad_clip,
            "amp_dtype": args.amp_dtype,
            "seed": args.seed,
        },
        "train_dataset": train_dataset.fingerprint,
        "val_dataset": val_dataset.fingerprint,
        "git_commit": provenance["git"]["commit"],
    }
    manifest: Dict[str, Any] = {
        "checkpoint_format_version": REINFORMER_CHECKPOINT_FORMAT_VERSION,
        "contract_version": REINFORMER_CONTRACT_VERSION,
        "algorithm": "Reinformer-MiniGrid",
        "args": vars(args),
        "model_config": model_config,
        "capacity_audit": capacity_audit,
        "provenance": provenance,
        "contract": contract,
        "status": "initialized",
        "created_at_unix": time.time(),
    }

    resume_payload: Optional[Dict[str, Any]] = None
    resume_id: Optional[str] = None
    if args.resume:
        resume_payload = load_trusted_checkpoint(args.resume)
        if (
            resume_payload.get("format_version")
            != REINFORMER_CHECKPOINT_FORMAT_VERSION
            or resume_payload.get("contract_version")
            != REINFORMER_CONTRACT_VERSION
        ):
            raise RuntimeError("Resume checkpoint is not a compatible Reinformer checkpoint.")
        if resume_payload.get("contract") != contract:
            raise RuntimeError("Resume checkpoint contract differs from this run.")
        resume_id = resume_payload.get("wandb_run_id")
    atomic_json_dump(manifest, output_dir / "run_manifest.json")

    short_env = args.env.replace("MiniGrid-", "").replace("-v0", "")
    run_name = args.wandb_name or f"reinformer-{short_env}-seed{args.seed}"
    run_group = args.wandb_group or f"reinformer-{short_env}"
    start_epoch = 0
    global_batch = 0
    global_update = 0
    best_val_loss = math.inf
    train_sampler_generator = torch.Generator().manual_seed(args.seed)
    train_worker_generator = torch.Generator().manual_seed(args.seed + 1)
    val_worker_generator = torch.Generator().manual_seed(args.seed + 2)
    run = None
    try:
        if resume_payload is not None:
            model.load_state_dict(resume_payload["model_state"])
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            scheduler.load_state_dict(resume_payload["scheduler_state"])
            scaler.load_state_dict(resume_payload.get("scaler_state", {}))
            start_epoch = int(resume_payload["epoch"]) + 1
            global_batch = int(resume_payload.get("global_batch", 0))
            global_update = int(resume_payload.get("global_update", 0))
            best_val_loss = float(resume_payload.get("best_val_loss", math.inf))
            loader_state = resume_payload.get("loader_generator_state")
            required_states = {"train_sampler", "train_worker", "val_worker"}
            if not loader_state or not required_states.issubset(loader_state):
                raise RuntimeError("Resume checkpoint lacks DataLoader generator state.")
            train_sampler_generator.set_state(loader_state["train_sampler"])
            train_worker_generator.set_state(loader_state["train_worker"])
            val_worker_generator.set_state(loader_state["val_worker"])
            logger.info(
                "Resuming from epoch {} and global update {}",
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
                "capacity_audit": capacity_audit,
                "provenance": provenance,
                "contract": contract,
            },
            tags=[
                "reinformer",
                "minigrid",
                "cross-episode",
                "independent-adaptation",
                "two-pass",
                "linear-action-head",
                "capacity-matched",
            ],
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
            last_batch: Optional[Dict[str, torch.Tensor]] = None

            for batch_index, batch in enumerate(train_loader):
                if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                    break
                batch = move_batch(batch, device)
                last_batch = batch
                with amp_context(device, args.amp_dtype):
                    action_logits, rtg_predictions = model(
                        batch,
                        teacher_force_action_rtg=True,
                    )
                    loss, metrics = compute_reinformer_loss(
                        action_logits,
                        rtg_predictions,
                        batch,
                        expectile=args.expectile,
                        rtg_loss_weight=args.rtg_loss_weight,
                        rtg_normalizer=rtg_loss_normalizer,
                    )
                    finite_or_raise("training loss", loss)
                    group_start = (
                        epoch_batches // args.grad_accum_steps
                    ) * args.grad_accum_steps
                    actual_group_size = min(
                        args.grad_accum_steps,
                        expected_batches - group_start,
                    )
                    scaled_loss = loss / actual_group_size

                scaler.scale(scaled_loss).backward()
                global_batch += 1
                epoch_batches += 1
                should_step = (
                    epoch_batches % args.grad_accum_steps == 0
                    or epoch_batches == expected_batches
                )
                grad_norm_value = float("nan")
                if should_step:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip,
                    )
                    finite_or_raise(
                        "gradient norm",
                        torch.as_tensor(grad_norm, device=device),
                    )
                    grad_norm_value = float(
                        torch.as_tensor(grad_norm).detach().float().cpu().item()
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_update += 1

                scalar_metrics = {
                    key: float(value.detach().float().cpu().item())
                    for key, value in metrics.items()
                }
                for key, value in scalar_metrics.items():
                    epoch_totals[key] = epoch_totals.get(key, 0.0) + value
                if (
                    should_step
                    and (global_update == 1 or global_update % args.log_every == 0)
                ):
                    run.log(
                        {
                            **{
                                f"train/{key}": value
                                for key, value in scalar_metrics.items()
                            },
                            "train/grad_norm": grad_norm_value,
                            "train/epoch": epoch,
                            "train/global_batch": global_batch,
                            "train/global_update": global_update,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        },
                        step=global_update,
                    )

            if epoch_batches == 0 or last_batch is None:
                raise RuntimeError("Training loader produced no batches.")
            train_summary = {
                f"epoch/train_{key}": value / epoch_batches
                for key, value in epoch_totals.items()
            }
            model.eval()
            diagnostics = extra_diagnostics(model, last_batch)
            train_summary.update(
                {
                    f"epoch/train_{key}": float(
                        value.detach().float().cpu().item()
                    )
                    for key, value in diagnostics.items()
                }
            )
            perturb_delta = train_summary[
                "epoch/train_diagnostic/rtg_perturb_logit_abs_delta"
            ]
            if not math.isfinite(perturb_delta) or perturb_delta <= 0.0:
                raise RuntimeError(
                    "Predicted/target RTG does not measurably condition action logits."
                )
            val_summary = validation_epoch(
                model,
                val_loader,
                device,
                args,
                rtg_normalizer=rtg_loss_normalizer,
            )
            epoch_payload = {
                **train_summary,
                **val_summary,
                "epoch/index": epoch,
                "epoch/wall_seconds": time.time() - epoch_start,
                "epoch/global_update": global_update,
            }
            run.log(epoch_payload, step=global_update)
            logger.info(
                "Epoch {}/{}: train={:.6f}, val={:.6f}, updates={}, {:.1f}s",
                epoch + 1,
                args.num_epochs,
                train_summary["epoch/train_loss/total"],
                val_summary["val/loss/total"],
                global_update,
                epoch_payload["epoch/wall_seconds"],
            )

            loader_states = {
                "train_sampler": train_sampler_generator.get_state(),
                "train_worker": train_worker_generator.get_state(),
                "val_worker": val_worker_generator.get_state(),
            }
            val_total = val_summary["val/loss/total"]
            if val_total < best_val_loss:
                best_val_loss = val_total
                save_checkpoint(
                    output_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
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
                    loader_generator_state=loader_states,
                )
            save_checkpoint(
                output_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
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
                loader_generator_state=loader_states,
            )
            if (epoch + 1) % args.checkpoint_every == 0:
                save_checkpoint(
                    output_dir / f"epoch_{epoch + 1}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
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
                    loader_generator_state=loader_states,
                )

        save_checkpoint(
            output_dir / "final.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
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

from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import ExperimentConfig
from .data import build_dataloaders
from .models import build_model
from .optim import build_optimizer, build_scheduler
from .privacy import attach_privacy, ensure_dp_compatible, get_epsilon
from .utils import create_run_dir, describe_device, set_seed, setup_logging, write_history, write_json, write_resolved_config

LOGGER = logging.getLogger(__name__)


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested_device)


def _configure_backend(config: ExperimentConfig, device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
        torch.backends.cudnn.benchmark = not config.runtime.deterministic
        torch.set_float32_matmul_precision("high" if config.runtime.allow_tf32 else "highest")


def _prepare_batch(inputs: torch.Tensor, device: torch.device, channels_last: bool) -> torch.Tensor:
    inputs = inputs.to(device, non_blocking=device.type == "cuda")
    if channels_last and device.type == "cuda" and inputs.ndim == 4:
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    return inputs


def _unwrap_model(model: nn.Module) -> nn.Module:
    current = model
    while True:
        next_model = getattr(current, "_module", None)
        if next_model is None:
            next_model = getattr(current, "_orig_mod", None)
        if next_model is None:
            return current
        current = next_model


def _autocast_context(enabled: bool, device: torch.device):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def _train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    channels_last: bool,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(data_loader, leave=False, desc="train")
    for inputs, targets in progress:
        inputs = _prepare_batch(inputs, device, channels_last)
        targets = targets.to(device, non_blocking=device.type == "cuda")

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(enabled=use_amp, device=device):
            logits = model(inputs)
            loss = criterion(logits, targets)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == targets).sum().item()
        total_samples += batch_size

        progress.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    channels_last: bool,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(data_loader, leave=False, desc="eval")
    for inputs, targets in progress:
        inputs = _prepare_batch(inputs, device, channels_last)
        targets = targets.to(device, non_blocking=device.type == "cuda")
        with _autocast_context(enabled=use_amp, device=device):
            logits = model(inputs)
            loss = criterion(logits, targets)

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == targets).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / max(1, total_samples),
        "accuracy": total_correct / max(1, total_samples),
    }


def run_experiment(config: ExperimentConfig) -> Path:
    setup_logging()
    set_seed(config.training.seed, config.runtime.deterministic)

    device = _resolve_device(config.runtime.device)
    _configure_backend(config, device)

    if config.runtime.amp and config.privacy.enabled:
        LOGGER.warning("AMP is disabled for DP-SGD runs to avoid unsupported mixed-precision edge cases.")
    use_amp = config.runtime.amp and device.type == "cuda" and not config.privacy.enabled
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None

    # Opacus per-sample gradient hooks are incompatible with:
    #   - multi-process DataLoader workers  (cudaErrorNotReady / CUDA race)
    #   - channels_last memory format       (non-contiguous strides break einsum)
    #   - grad_sample_mode="hooks" at large batch sizes (materialises [B, ...] grad tensors → OOM)
    # Override all three for private runs.
    import dataclasses
    if config.privacy.enabled:
        safe_dataset_config = dataclasses.replace(
            config.dataset,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        safe_runtime = dataclasses.replace(config.runtime, channels_last=False)
        # Ghost clipping is mathematically equivalent to hooks but uses O(1) extra
        # memory instead of O(batch_size) — switch automatically if hooks is set.
        safe_privacy = config.privacy
        if getattr(safe_privacy, "grad_sample_mode", "hooks") == "hooks":
            safe_privacy = dataclasses.replace(safe_privacy, grad_sample_mode="ghost")
            LOGGER.info("DP mode: switching grad_sample_mode hooks→ghost to avoid per-sample gradient OOM.")
        config = dataclasses.replace(
            config,
            dataset=safe_dataset_config,
            runtime=safe_runtime,
            privacy=safe_privacy,
        )
        LOGGER.info(
            "DP mode: forcing num_workers=0, pin_memory=False, channels_last=False for Opacus compatibility."
        )
    data_loaders = build_dataloaders(config.dataset, config.training, device.type)

    model = build_model(config.model, config.dataset.name)
    model = ensure_dp_compatible(model) if config.privacy.enabled else model
    model = model.to(device)

    if config.runtime.compile and config.privacy.enabled:
        LOGGER.warning("torch.compile is disabled for private runs because Opacus wrappers are not compile-stable across releases.")
    elif config.runtime.compile and hasattr(torch, "compile"):
        model = cast(nn.Module, torch.compile(model))

    optimizer = build_optimizer(model, config.optimizer)
    scheduler = build_scheduler(optimizer, config.scheduler, config.training)
    privacy_artifacts = attach_privacy(model, optimizer, data_loaders.train, config.privacy)

    model = privacy_artifacts.model
    optimizer = privacy_artifacts.optimizer
    train_loader = privacy_artifacts.train_loader
    criterion = nn.CrossEntropyLoss()

    run_dir = create_run_dir(config.training.output_dir, config.training.experiment_name)
    write_resolved_config(config, run_dir)

    LOGGER.info("Starting run in %s", run_dir)
    LOGGER.info("Device: %s", describe_device(device))
    LOGGER.info(
        "Dataset=%s | Train samples=%s | Eval samples=%s | Private=%s",
        config.dataset.name,
        data_loaders.train_size,
        data_loaders.eval_size,
        config.privacy.enabled,
    )

    best_accuracy = float("-inf")
    history: list[dict[str, Any]] = []

    for epoch in range(1, config.training.epochs + 1):
        train_metrics = _train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            channels_last=config.runtime.channels_last,
            use_amp=use_amp,
            scaler=scaler,
        )
        eval_metrics = _evaluate(
            model=model,
            data_loader=data_loaders.eval,
            criterion=criterion,
            device=device,
            channels_last=config.runtime.channels_last,
            use_amp=use_amp,
        )
        if scheduler is not None:
            scheduler.step()

        epsilon = get_epsilon(privacy_artifacts.privacy_engine, config.privacy.delta)
        learning_rate = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "eval_loss": round(eval_metrics["loss"], 6),
            "eval_accuracy": round(eval_metrics["accuracy"], 6),
            "epsilon": None if epsilon is None else round(epsilon, 6),
            "delta": config.privacy.delta if config.privacy.enabled else None,
            "learning_rate": learning_rate,
        }
        history.append(epoch_record)
        write_history(history, run_dir / "history.csv")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": _unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "metrics": epoch_record,
            "config": config.to_dict(),
        }
        torch.save(checkpoint, run_dir / "last.pt")

        if eval_metrics["accuracy"] >= best_accuracy:
            best_accuracy = eval_metrics["accuracy"]
            torch.save(checkpoint, run_dir / "best.pt")

        LOGGER.info(
            "Epoch %s/%s | train_loss=%.4f | train_acc=%.4f | eval_loss=%.4f | eval_acc=%.4f%s",
            epoch,
            config.training.epochs,
            train_metrics["loss"],
            train_metrics["accuracy"],
            eval_metrics["loss"],
            eval_metrics["accuracy"],
            "" if epsilon is None else f" | epsilon={epsilon:.4f}",
        )

    summary = {
        "run_dir": str(run_dir),
        "device": describe_device(device),
        "best_eval_accuracy": best_accuracy,
        "final_epsilon": history[-1]["epsilon"] if history else None,
        "trainable_parameters": sum(parameter.numel() for parameter in _unwrap_model(model).parameters()),
    }
    write_json({"history": history, "summary": summary}, run_dir / "metrics.json")
    LOGGER.info("Finished run. Best eval accuracy=%.4f", best_accuracy)
    return run_dir

from __future__ import annotations

from torch import nn
from torch.optim import AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .config import OptimizerConfig, SchedulerConfig, TrainingConfig


def build_optimizer(model: nn.Module, optimizer_config: OptimizerConfig) -> Optimizer:
    optimizer_name = optimizer_config.name.lower()
    parameters = model.parameters()

    if optimizer_name == "sgd":
        return SGD(
            parameters,
            lr=optimizer_config.lr,
            momentum=optimizer_config.momentum,
            weight_decay=optimizer_config.weight_decay,
            nesterov=optimizer_config.nesterov,
        )
    if optimizer_name == "adamw":
        return AdamW(
            parameters,
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_config.name}'.")


def build_scheduler(
    optimizer: Optimizer,
    scheduler_config: SchedulerConfig,
    training_config: TrainingConfig,
):
    scheduler_name = scheduler_config.name.lower()
    if scheduler_name in {"none", "off", "disabled"}:
        return None

    if scheduler_name != "cosine":
        raise ValueError(f"Unsupported scheduler '{scheduler_config.name}'.")

    total_epochs = max(1, training_config.epochs)
    warmup_epochs = max(0, scheduler_config.warmup_epochs)

    if warmup_epochs == 0:
        return CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=scheduler_config.min_lr,
        )

    if warmup_epochs >= total_epochs:
        return LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=total_epochs,
        )

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=scheduler_config.min_lr,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

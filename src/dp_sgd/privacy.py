from __future__ import annotations

from dataclasses import dataclass

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .config import PrivacyConfig


@dataclass(slots=True)
class PrivacyArtifacts:
    model: nn.Module
    optimizer: Optimizer
    train_loader: DataLoader
    privacy_engine: PrivacyEngine | None


def ensure_dp_compatible(model: nn.Module) -> nn.Module:
    validation_errors = ModuleValidator.validate(model, strict=False)
    if not validation_errors:
        return model

    fixed_model = ModuleValidator.fix(model)
    remaining_errors = ModuleValidator.validate(fixed_model, strict=False)
    if remaining_errors:
        formatted = "; ".join(str(error) for error in remaining_errors)
        raise ValueError(f"Model remains incompatible with Opacus after auto-fix: {formatted}")
    return fixed_model


def attach_privacy(
    model: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    privacy_config: PrivacyConfig,
) -> PrivacyArtifacts:
    if not privacy_config.enabled:
        return PrivacyArtifacts(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            privacy_engine=None,
        )

    privacy_engine = PrivacyEngine(
        accountant="rdp",
        secure_mode=privacy_config.secure_mode,
    )
    result = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=privacy_config.noise_multiplier,
        max_grad_norm=privacy_config.max_grad_norm,
        poisson_sampling=privacy_config.poisson_sampling,
        clipping=privacy_config.clipping,
        grad_sample_mode=privacy_config.grad_sample_mode,
    )

    # Opacus 1.x returns a 3-tuple (model, optimizer, loader) for hooks/flat mode
    # but a 4-tuple (model, optimizer, criterion_wrapper, loader) for ghost clipping.
    if len(result) == 4:
        private_model, private_optimizer, _, private_loader = result
    else:
        private_model, private_optimizer, private_loader = result

    return PrivacyArtifacts(
        model=private_model,
        optimizer=private_optimizer,
        train_loader=private_loader,
        privacy_engine=privacy_engine,
    )


def get_epsilon(privacy_engine: PrivacyEngine | None, delta: float) -> float | None:
    if privacy_engine is None:
        return None
    return float(privacy_engine.get_epsilon(delta))

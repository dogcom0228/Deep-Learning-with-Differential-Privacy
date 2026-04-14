from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetConfig:
    name: str = "mnist"
    root: str = "data"
    download: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    augment: bool = False


@dataclass(slots=True)
class ModelConfig:
    name: str = "auto"
    num_classes: int = 10
    width: int = 32
    dropout: float = 0.0


@dataclass(slots=True)
class OptimizerConfig:
    name: str = "sgd"
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0
    nesterov: bool = False


@dataclass(slots=True)
class SchedulerConfig:
    name: str = "cosine"
    warmup_epochs: int = 0
    min_lr: float = 0.0


@dataclass(slots=True)
class PrivacyConfig:
    enabled: bool = True
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    poisson_sampling: bool = True
    clipping: str = "flat"
    grad_sample_mode: str = "hooks"
    secure_mode: bool = False


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "auto"
    compile: bool = False
    allow_tf32: bool = True
    channels_last: bool = True
    amp: bool = False
    deterministic: bool = False


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 15
    batch_size: int = 256
    eval_batch_size: int = 512
    seed: int = 42
    log_every: int = 20
    output_dir: str = "results"
    experiment_name: str = "default"


@dataclass(slots=True)
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dicts(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_nested_value(target: dict[str, Any], path: str, value: Any) -> None:
    current = target
    segments = path.split(".")
    for segment in segments[:-1]:
        current = current.setdefault(segment, {})
    current[segments[-1]] = value


def apply_overrides(config_data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = dict(config_data)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}'. Expected KEY=VALUE.")
        key, raw_value = override.split("=", maxsplit=1)
        value = yaml.safe_load(raw_value)
        _set_nested_value(updated, key, value)
    return updated


def _build_experiment(data: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=DatasetConfig(**data.get("dataset", {})),
        model=ModelConfig(**data.get("model", {})),
        optimizer=OptimizerConfig(**data.get("optimizer", {})),
        scheduler=SchedulerConfig(**data.get("scheduler", {})),
        privacy=PrivacyConfig(**data.get("privacy", {})),
        runtime=RuntimeConfig(**data.get("runtime", {})),
        training=TrainingConfig(**data.get("training", {})),
    )


def load_config(path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    config_path = Path(path)
    raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_data, dict):
        raise ValueError(f"Config file {config_path} must contain a mapping at the top level.")
    merged_data = _merge_dicts({}, raw_data)
    if overrides:
        merged_data = apply_overrides(merged_data, overrides)
    return _build_experiment(merged_data)

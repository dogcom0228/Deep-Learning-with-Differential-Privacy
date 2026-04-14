from __future__ import annotations

import csv
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .config import ExperimentConfig


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def create_run_dir(output_dir: str, experiment_name: str) -> Path:
    sanitized_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in experiment_name.lower()
    ).strip("-")
    if not sanitized_name:
        sanitized_name = "experiment"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / sanitized_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_resolved_config(config: ExperimentConfig, run_dir: Path) -> None:
    resolved_config = yaml.safe_dump(config.to_dict(), sort_keys=False)
    (run_dir / "resolved-config.yaml").write_text(resolved_config, encoding="utf-8")


def write_history(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return device.type

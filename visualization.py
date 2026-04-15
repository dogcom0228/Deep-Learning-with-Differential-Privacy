from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import yaml

DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_FIGURES_DIR = Path("figures") / "comparisons"

IGNORED_CONFIG_FIELDS = {
    "cfg.dataset.root",
    "cfg.dataset.download",
    "cfg.dataset.num_workers",
    "cfg.dataset.pin_memory",
    "cfg.dataset.persistent_workers",
    "cfg.runtime.device",
    "cfg.training.output_dir",
    "cfg.training.experiment_name",
}

PREFERRED_LABEL_FIELDS = [
    "cfg.privacy.enabled",
    "cfg.privacy.noise_multiplier",
    "cfg.privacy.max_grad_norm",
    "cfg.optimizer.lr",
    "cfg.training.batch_size",
    "cfg.optimizer.momentum",
    "cfg.optimizer.weight_decay",
    "cfg.scheduler.warmup_epochs",
    "cfg.dataset.augment",
    "cfg.runtime.amp",
    "cfg.runtime.channels_last",
    "cfg.runtime.compile",
    "cfg.model.width",
    "cfg.model.dropout",
]

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "figure.figsize": (9, 5),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.2,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare all experiment runs under results/ and generate summary figures."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory that contains experiment run subfolders.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory where comparison figures and tables will be written.",
    )
    parser.add_argument(
        "--max-label-fields",
        type=int,
        default=3,
        help="Maximum number of varying config fields to include in plot labels.",
    )
    return parser.parse_args()


def flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=full_key))
        else:
            flat[full_key] = value
    return flat


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-")
    return slug.lower() or "plot"


def short_name(column_name: str) -> str:
    return column_name.removeprefix("cfg.")


def compact_name(column_name: str) -> str:
    mapping = {
        "privacy.noise_multiplier": "nm",
        "privacy.max_grad_norm": "clip",
        "privacy.enabled": "dp",
        "optimizer.lr": "lr",
        "optimizer.momentum": "mom",
        "optimizer.weight_decay": "wd",
        "training.batch_size": "bs",
        "scheduler.warmup_epochs": "warmup",
        "dataset.augment": "aug",
        "runtime.amp": "amp",
        "runtime.channels_last": "channels_last",
        "runtime.compile": "compile",
        "model.width": "width",
        "model.dropout": "dropout",
    }
    name = short_name(column_name)
    return mapping.get(name, name.split(".")[-1])


def format_value(value: Any) -> str:
    if pd.isna(value):
        return "na"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4g}"
    return str(value)


def last_valid_value(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def read_history(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "history.csv")


def read_config(run_dir: Path) -> dict[str, Any]:
    return yaml.safe_load((run_dir / "resolved-config.yaml").read_text())


def read_metrics_summary(run_dir: Path) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text())
    return metrics.get("summary", {})


def collect_runs(results_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    histories: dict[str, pd.DataFrame] = {}

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    for experiment_dir in sorted(path for path in results_dir.iterdir() if path.is_dir()):
        for run_dir in sorted(path for path in experiment_dir.iterdir() if path.is_dir()):
            history_path = run_dir / "history.csv"
            config_path = run_dir / "resolved-config.yaml"
            metrics_path = run_dir / "metrics.json"
            if not (history_path.exists() and config_path.exists() and metrics_path.exists()):
                continue

            history = read_history(run_dir)
            config = read_config(run_dir)
            summary = read_metrics_summary(run_dir)
            flat_config = {f"cfg.{key}": value for key, value in flatten_dict(config).items()}

            best_eval_accuracy = (
                float(history["eval_accuracy"].max())
                if "eval_accuracy" in history
                else float(summary.get("best_eval_accuracy", 0.0))
            )
            best_eval_epoch = None
            if "eval_accuracy" in history:
                best_eval_index = history["eval_accuracy"].idxmax()
                best_epoch_value = pd.to_numeric(history.loc[best_eval_index, "epoch"])
                best_eval_epoch = int(best_epoch_value)

            row = {
                "experiment": experiment_dir.name,
                "run_id": run_dir.name,
                "run_path": str(run_dir),
                "dataset": flat_config.get("cfg.dataset.name", experiment_dir.name),
                "privacy_enabled": bool(flat_config.get("cfg.privacy.enabled", False)),
                "best_eval_accuracy": best_eval_accuracy,
                "best_eval_epoch": best_eval_epoch,
                "final_eval_accuracy": last_valid_value(history.get("eval_accuracy")),
                "final_eval_loss": last_valid_value(history.get("eval_loss")),
                "final_train_accuracy": last_valid_value(history.get("train_accuracy")),
                "final_train_loss": last_valid_value(history.get("train_loss")),
                "final_epsilon": last_valid_value(history.get("epsilon")),
                "final_delta": last_valid_value(history.get("delta")),
                "final_learning_rate": last_valid_value(history.get("learning_rate")),
                "device": summary.get("device"),
                "trainable_parameters": summary.get("trainable_parameters"),
            }
            row.update(flat_config)
            rows.append(row)
            histories[str(run_dir)] = history

    summary_df = pd.DataFrame(rows)
    return summary_df, histories


def find_varying_config_fields(frame: pd.DataFrame) -> list[str]:
    varying: list[str] = []
    for column in frame.columns:
        if not column.startswith("cfg.") or column in IGNORED_CONFIG_FIELDS:
            continue
        non_null = frame[column].dropna()
        if non_null.astype(str).nunique() > 1:
            varying.append(column)
    return varying


def choose_label_fields(frame: pd.DataFrame, max_fields: int) -> list[str]:
    varying = find_varying_config_fields(frame)
    chosen: list[str] = []

    for column in PREFERRED_LABEL_FIELDS:
        if column in varying and column not in chosen:
            chosen.append(column)
        if len(chosen) >= max_fields:
            return chosen

    for column in varying:
        if column not in chosen:
            chosen.append(column)
        if len(chosen) >= max_fields:
            break
    return chosen


def build_label(
    row: pd.Series,
    label_fields: list[str],
    *,
    include_experiment: bool = True,
    include_run_id: bool = False,
) -> str:
    parts: list[str] = []
    if include_experiment:
        parts.append(str(row["experiment"]))
    if include_run_id:
        parts.append(str(row["run_id"]))
    for column in label_fields:
        value = row.get(column)
        if pd.isna(value):
            continue
        parts.append(f"{short_name(column)}={format_value(value)}")
    return " | ".join(parts) if parts else str(row["run_id"])


def build_point_label(row: pd.Series, label_fields: list[str]) -> str:
    if not label_fields:
        return ""
    parts: list[str] = []
    for column in label_fields[:2]:
        value = row.get(column)
        if pd.isna(value):
            continue
        parts.append(f"{compact_name(column)}={format_value(value)}")
    return " | ".join(parts)


def annotate_points(ax: Axes, x_values: pd.Series, y_values: pd.Series, labels: list[str]) -> None:
    x_list = [float(value) for value in x_values]
    y_list = [float(value) for value in y_values]
    if not x_list or not y_list:
        return

    x_span = max(max(x_list) - min(x_list), 1e-9)
    y_span = max(max(y_list) - min(y_list), 1e-9)
    placed: list[tuple[float, float, int]] = []
    offset_pattern = [10, -14, 18, -22, 26, -30, 34, -38]

    for x_value, y_value, label in zip(x_list, y_list, labels, strict=False):
        if not label:
            continue
        collision_level = 0
        for prev_x, prev_y, prev_level in placed:
            x_distance = abs(x_value - prev_x) / x_span
            y_distance = abs(y_value - prev_y) / y_span
            if x_distance < 0.12 and y_distance < 0.12:
                collision_level = max(collision_level, prev_level + 1)

        offset_y = offset_pattern[collision_level % len(offset_pattern)]
        offset_x = 6 + min(collision_level, 4) * 2
        ha = "left"
        if offset_x > 10 and collision_level % 2 == 1:
            ha = "right"
            offset_x = -offset_x

        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(offset_x, offset_y),
            textcoords="offset points",
            fontsize=8,
            ha=ha,
            va="center",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.7},
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.5", "alpha": 0.6},
        )
        placed.append((x_value, y_value, collision_level))


def save_summary_tables(summary_df: pd.DataFrame, figures_dir: Path) -> tuple[Path, Path, Path]:
    varying_fields = find_varying_config_fields(summary_df)
    preferred_fields = [field for field in PREFERRED_LABEL_FIELDS if field in summary_df.columns]
    ordered_fields = preferred_fields + [field for field in varying_fields if field not in preferred_fields]

    summary_columns = [
        "dataset",
        "experiment",
        "run_id",
        "privacy_enabled",
        "best_eval_accuracy",
        "best_eval_epoch",
        "final_eval_accuracy",
        "final_epsilon",
        "final_learning_rate",
    ] + ordered_fields
    summary_columns = [column for column in summary_columns if column in summary_df.columns]
    all_runs_path = figures_dir / "all_runs_summary.csv"
    summary_df.sort_values(["dataset", "experiment", "run_id"])[summary_columns].to_csv(
        all_runs_path, index=False
    )

    dataset_rows: list[dict[str, Any]] = []
    for dataset, group in summary_df.groupby("dataset", sort=True):
        best_overall = group.loc[group["best_eval_accuracy"].idxmax()]
        private_group = group[group["privacy_enabled"]]
        non_private_group = group[~group["privacy_enabled"]]
        best_private = (
            private_group.loc[private_group["best_eval_accuracy"].idxmax()]
            if not private_group.empty
            else None
        )
        best_non_private = (
            non_private_group.loc[non_private_group["best_eval_accuracy"].idxmax()]
            if not non_private_group.empty
            else None
        )
        dataset_rows.append(
            {
                "dataset": dataset,
                "total_runs": len(group),
                "best_overall_run": f"{best_overall['experiment']}/{best_overall['run_id']}",
                "best_overall_accuracy": best_overall["best_eval_accuracy"],
                "best_private_run": (
                    f"{best_private['experiment']}/{best_private['run_id']}" if best_private is not None else None
                ),
                "best_private_accuracy": (
                    best_private["best_eval_accuracy"] if best_private is not None else None
                ),
                "best_private_epsilon": (
                    best_private["final_epsilon"] if best_private is not None else None
                ),
                "best_non_private_run": (
                    f"{best_non_private['experiment']}/{best_non_private['run_id']}"
                    if best_non_private is not None
                    else None
                ),
                "best_non_private_accuracy": (
                    best_non_private["best_eval_accuracy"] if best_non_private is not None else None
                ),
                "accuracy_gap_private_vs_non_private": (
                    best_private["best_eval_accuracy"] - best_non_private["best_eval_accuracy"]
                    if best_private is not None and best_non_private is not None
                    else None
                ),
            }
        )

    dataset_summary = pd.DataFrame(dataset_rows).sort_values("dataset")
    dataset_summary_path = figures_dir / "dataset_summary.csv"
    dataset_summary.to_csv(dataset_summary_path, index=False)

    experiment_rows: list[dict[str, Any]] = []
    for experiment, group in summary_df.groupby("experiment", sort=True):
        varying = find_varying_config_fields(group)
        experiment_rows.append(
            {
                "experiment": experiment,
                "dataset": group["dataset"].iloc[0],
                "runs": len(group),
                "varying_parameters": ", ".join(short_name(field) for field in varying) or "none",
                "best_eval_accuracy": group["best_eval_accuracy"].max(),
                "lowest_epsilon": group["final_epsilon"].dropna().min() if group["final_epsilon"].notna().any() else None,
            }
        )

    experiment_summary = pd.DataFrame(experiment_rows).sort_values(["dataset", "experiment"])
    experiment_summary_path = figures_dir / "experiment_summary.csv"
    experiment_summary.to_csv(experiment_summary_path, index=False)

    return all_runs_path, dataset_summary_path, experiment_summary_path


def write_report(summary_df: pd.DataFrame, figures_dir: Path) -> Path:
    varying_fields = find_varying_config_fields(summary_df)
    lines = [
        "# Results Comparison Report",
        "",
        f"Total runs: {len(summary_df)}",
        f"Experiments: {summary_df['experiment'].nunique()}",
        f"Datasets: {summary_df['dataset'].nunique()}",
        "",
        "## Globally varying config fields",
    ]

    if varying_fields:
        lines.extend(f"- {short_name(field)}" for field in varying_fields)
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Dataset-level takeaways")
    for dataset, group in summary_df.groupby("dataset", sort=True):
        best_run = group.loc[group["best_eval_accuracy"].idxmax()]
        best_private = None
        lines.append(
            f"- {dataset}: best run is {best_run['experiment']}/{best_run['run_id']} "
            f"with best_eval_accuracy={best_run['best_eval_accuracy']:.4f}"
        )
        private_group = group[group["privacy_enabled"]]
        non_private_group = group[~group["privacy_enabled"]]
        if not private_group.empty:
            best_private = private_group.loc[private_group["best_eval_accuracy"].idxmax()]
            lines.append(
                f"- {dataset}: best private run is {best_private['experiment']}/{best_private['run_id']} "
                f"with epsilon={format_value(best_private['final_epsilon'])}"
            )
        if best_private is not None and not non_private_group.empty:
            best_non_private = non_private_group.loc[non_private_group["best_eval_accuracy"].idxmax()]
            gap = best_private["best_eval_accuracy"] - best_non_private["best_eval_accuracy"]
            lines.append(
                f"- {dataset}: private vs non-private best accuracy gap = {gap:.4f}"
            )

    lines.append("")
    lines.append("## Experiment-level sweeps")
    for experiment, group in summary_df.groupby("experiment", sort=True):
        varying = find_varying_config_fields(group)
        lines.append(f"- {experiment}: {len(group)} run(s)")
        lines.append(
            f"- {experiment}: varying parameters -> {', '.join(short_name(field) for field in varying) or 'none'}"
        )
        if group["final_epsilon"].notna().any():
            lowest_epsilon_run = group.loc[group["final_epsilon"].idxmin()]
            lines.append(
                f"- {experiment}: lowest epsilon run is {lowest_epsilon_run['run_id']} "
                f"with epsilon={lowest_epsilon_run['final_epsilon']:.4f}"
            )

    report_path = figures_dir / "comparison_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def plot_dataset_best_accuracy(summary_df: pd.DataFrame, figures_dir: Path, max_label_fields: int) -> list[Path]:
    output_paths: list[Path] = []
    for dataset, group in summary_df.groupby("dataset", sort=True):
        dataset_name = str(dataset)
        ordered = group.sort_values("best_eval_accuracy", ascending=True)
        label_fields = choose_label_fields(group, max_label_fields)
        labels = [build_label(row, label_fields) for _, row in ordered.iterrows()]
        colors = ["tab:orange" if row["privacy_enabled"] else "tab:blue" for _, row in ordered.iterrows()]

        fig, ax = plt.subplots(figsize=(11, max(4, 0.75 * len(ordered))))
        positions = range(len(ordered))
        ax.barh(list(positions), ordered["best_eval_accuracy"], color=colors)
        ax.set_yticks(list(positions), labels)
        ax.set_xlabel("Best eval accuracy")
        ax.set_title(f"{dataset_name}: best eval accuracy by run")

        for index, value in enumerate(ordered["best_eval_accuracy"]):
            ax.text(value + 0.002, index, f"{value:.4f}", va="center", fontsize=8)

        fig.tight_layout()
        output_path = figures_dir / f"dataset_{slugify(dataset_name)}_best_accuracy.png"
        fig.savefig(output_path)
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def plot_dataset_best_run_convergence(
    summary_df: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    figures_dir: Path,
    max_label_fields: int,
) -> list[Path]:
    output_paths: list[Path] = []
    for dataset, group in summary_df.groupby("dataset", sort=True):
        dataset_name = str(dataset)
        best_by_experiment = (
            group.sort_values("best_eval_accuracy", ascending=False)
            .groupby("experiment", as_index=False)
            .first()
        )
        if len(best_by_experiment) < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))
        for _, row in best_by_experiment.iterrows():
            experiment_group = summary_df[summary_df["experiment"] == row["experiment"]]
            label_fields = choose_label_fields(experiment_group, max_label_fields)
            label = build_label(row, label_fields, include_experiment=True)
            history = histories[row["run_path"]]
            ax.plot(history["epoch"], history["eval_accuracy"], label=label, linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Eval accuracy")
        ax.set_title(f"{dataset_name}: best run from each experiment")
        ax.legend(fontsize=8)
        fig.tight_layout()
        output_path = figures_dir / f"dataset_{slugify(dataset_name)}_best_run_convergence.png"
        fig.savefig(output_path)
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def plot_experiment_convergence(
    summary_df: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    figures_dir: Path,
    max_label_fields: int,
) -> list[Path]:
    output_paths: list[Path] = []
    for experiment, group in summary_df.groupby("experiment", sort=True):
        experiment_name = str(experiment)
        if len(group) < 2:
            continue
        label_fields = choose_label_fields(group, max_label_fields)
        ordered = group.sort_values(label_fields if label_fields else ["run_id"])

        fig, ax = plt.subplots(figsize=(10, 5))
        for _, row in ordered.iterrows():
            history = histories[row["run_path"]]
            label = build_label(row, label_fields, include_experiment=False)
            ax.plot(history["epoch"], history["eval_accuracy"], label=label)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Eval accuracy")
        ax.set_title(f"{experiment_name}: eval accuracy convergence")
        ax.legend(fontsize=8)
        fig.tight_layout()
        output_path = figures_dir / f"experiment_{slugify(experiment_name)}_eval_accuracy.png"
        fig.savefig(output_path)
        plt.close(fig)
        output_paths.append(output_path)

        if group["final_epsilon"].notna().any():
            fig, ax = plt.subplots(figsize=(10, 5))
            for _, row in ordered.iterrows():
                history = histories[row["run_path"]]
                if "epsilon" not in history or history["epsilon"].dropna().empty:
                    continue
                label = build_label(row, label_fields, include_experiment=False)
                ax.plot(history["epoch"], history["epsilon"], label=label)

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Epsilon")
            ax.set_title(f"{experiment_name}: privacy budget accumulation")
            ax.legend(fontsize=8)
            fig.tight_layout()
            output_path = figures_dir / f"experiment_{slugify(experiment_name)}_epsilon.png"
            fig.savefig(output_path)
            plt.close(fig)
            output_paths.append(output_path)
    return output_paths


def plot_privacy_tradeoff(summary_df: pd.DataFrame, figures_dir: Path, max_label_fields: int) -> list[Path]:
    output_paths: list[Path] = []
    private_df = summary_df[summary_df["privacy_enabled"] & summary_df["final_epsilon"].notna()].copy()
    for dataset, group in private_df.groupby("dataset", sort=True):
        dataset_name = str(dataset)
        if len(group) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        for experiment, experiment_group in group.groupby("experiment", sort=True):
            ordered = experiment_group.sort_values("final_epsilon")
            ax.plot(
                ordered["final_epsilon"],
                ordered["best_eval_accuracy"],
                marker="o",
                linewidth=1.5,
                label=experiment,
            )
            label_fields = choose_label_fields(experiment_group, max_label_fields)
            labels = [build_point_label(row, label_fields) for _, row in ordered.iterrows()]
            annotate_points(ax, ordered["final_epsilon"], ordered["best_eval_accuracy"], labels)

        non_private_group = summary_df[(summary_df["dataset"] == dataset) & (~summary_df["privacy_enabled"])]
        if not non_private_group.empty:
            baseline = non_private_group["best_eval_accuracy"].max()
            ax.axhline(baseline, color="tab:blue", linestyle="--", linewidth=1.5, label="best non-private")

        ax.set_xlabel("Final epsilon")
        ax.set_ylabel("Best eval accuracy")
        ax.set_title(f"{dataset_name}: privacy-utility tradeoff")
        ax.legend(fontsize=8)
        fig.tight_layout()
        output_path = figures_dir / f"dataset_{slugify(dataset_name)}_privacy_tradeoff.png"
        fig.savefig(output_path)
        plt.close(fig)
        output_paths.append(output_path)
    return output_paths


def plot_parameter_sweeps(summary_df: pd.DataFrame, figures_dir: Path, max_label_fields: int) -> list[Path]:
    output_paths: list[Path] = []
    for experiment, group in summary_df.groupby("experiment", sort=True):
        experiment_name = str(experiment)
        if len(group) < 2:
            continue

        varying_fields = find_varying_config_fields(group)
        numeric_fields: list[str] = []
        for field in varying_fields:
            numeric_values = pd.to_numeric(group[field], errors="coerce")
            if numeric_values.notna().sum() >= 2 and numeric_values.nunique(dropna=True) >= 2:
                numeric_fields.append(field)

        if not numeric_fields:
            continue

        for field in numeric_fields:
            ordered = group.copy()
            ordered["plot_x"] = pd.to_numeric(ordered[field], errors="coerce")
            ordered = ordered.dropna(subset=["plot_x"]).sort_values("plot_x")
            if len(ordered) < 2:
                continue

            has_epsilon = ordered["final_epsilon"].notna().any()
            ncols = 2 if has_epsilon else 1
            fig, axes = plt.subplots(1, ncols, figsize=(12 if has_epsilon else 6, 4.5))
            axes_list = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]

            label_fields = [column for column in choose_label_fields(group, max_label_fields) if column != field]
            labels = [build_point_label(row, label_fields) for _, row in ordered.iterrows()]

            axes_list[0].plot(ordered["plot_x"], ordered["best_eval_accuracy"], marker="o", linewidth=1.5)
            axes_list[0].set_xlabel(short_name(field))
            axes_list[0].set_ylabel("Best eval accuracy")
            axes_list[0].set_title(f"{experiment_name}: accuracy vs {short_name(field)}")
            annotate_points(axes_list[0], ordered["plot_x"], ordered["best_eval_accuracy"], labels)

            if has_epsilon:
                axes_list[1].plot(
                    ordered["plot_x"],
                    ordered["final_epsilon"],
                    marker="o",
                    linewidth=1.5,
                    color="tab:orange",
                )
                axes_list[1].set_xlabel(short_name(field))
                axes_list[1].set_ylabel("Final epsilon")
                axes_list[1].set_title(f"{experiment_name}: epsilon vs {short_name(field)}")
                annotate_points(axes_list[1], ordered["plot_x"], ordered["final_epsilon"], labels)

            fig.tight_layout()
            output_path = figures_dir / f"experiment_{slugify(experiment_name)}_{slugify(short_name(field))}_sweep.png"
            fig.savefig(output_path)
            plt.close(fig)
            output_paths.append(output_path)
    return output_paths


def print_console_summary(summary_df: pd.DataFrame) -> None:
    console_columns = [
        "dataset",
        "experiment",
        "run_id",
        "privacy_enabled",
        "best_eval_accuracy",
        "final_epsilon",
        "cfg.privacy.noise_multiplier",
        "cfg.optimizer.lr",
        "cfg.training.batch_size",
    ]
    console_columns = [column for column in console_columns if column in summary_df.columns]
    print(f"Loaded {len(summary_df)} runs across {summary_df['experiment'].nunique()} experiments.")
    print(summary_df.sort_values(["dataset", "experiment", "run_id"])[console_columns].to_string(index=False))


def main() -> None:
    args = parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    summary_df, histories = collect_runs(args.results_dir)
    if summary_df.empty:
        raise RuntimeError(f"No complete runs found under {args.results_dir}")

    all_runs_path, dataset_summary_path, experiment_summary_path = save_summary_tables(
        summary_df, args.figures_dir
    )
    report_path = write_report(summary_df, args.figures_dir)

    figure_paths: list[Path] = []
    figure_paths.extend(plot_dataset_best_accuracy(summary_df, args.figures_dir, args.max_label_fields))
    figure_paths.extend(
        plot_dataset_best_run_convergence(summary_df, histories, args.figures_dir, args.max_label_fields)
    )
    figure_paths.extend(plot_experiment_convergence(summary_df, histories, args.figures_dir, args.max_label_fields))
    figure_paths.extend(plot_privacy_tradeoff(summary_df, args.figures_dir, args.max_label_fields))
    figure_paths.extend(plot_parameter_sweeps(summary_df, args.figures_dir, args.max_label_fields))

    print_console_summary(summary_df)
    print(f"Saved run summary    : {all_runs_path}")
    print(f"Saved dataset summary: {dataset_summary_path}")
    print(f"Saved experiment summary: {experiment_summary_path}")
    print(f"Saved report         : {report_path}")
    for figure_path in figure_paths:
        print(f"Saved figure         : {figure_path}")


if __name__ == "__main__":
    main()

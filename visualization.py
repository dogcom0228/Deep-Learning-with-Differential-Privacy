from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "figure.figsize": (8, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_run(run_dir: Path) -> pd.DataFrame:
    """Load history.csv from a single experiment run directory."""
    csv_path = run_dir / "history.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No history.csv in {run_dir}")
    return pd.read_csv(csv_path)


def load_summary(run_dir: Path) -> dict:
    """Load the summary block from metrics.json."""
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.json in {run_dir}")
    return json.loads(metrics_path.read_text())["summary"]


def find_latest_run(experiment_name: str) -> Path:
    """Return the most recent timestamped sub-directory for an experiment."""
    exp_dir = RESULTS_DIR / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"No experiment directory: {exp_dir}")
    runs = sorted(exp_dir.iterdir(), reverse=True)
    if not runs:
        raise FileNotFoundError(f"No runs found under {exp_dir}")
    return runs[0]


def list_all_runs(experiment_name: str) -> list[Path]:
    """Return all timestamped run directories for an experiment, newest first."""
    exp_dir = RESULTS_DIR / experiment_name
    if not exp_dir.exists():
        return []
    return sorted(exp_dir.iterdir(), reverse=True)


def main() -> None:
    # ── Single-run: accuracy & loss convergence ───────────────────────────────
    # Change experiment_name to whichever run you want to inspect.
    experiment_name = "mnist-dp-sgd"

    run_dir = find_latest_run(experiment_name)
    df = load_run(run_dir)
    summary = load_summary(run_dir)

    print(f"Run  : {run_dir.name}")
    print(f"Best eval accuracy : {summary['best_eval_accuracy']:.4f}")
    print(f"Final epsilon      : {summary['final_epsilon']}")
    print(f"Device             : {summary['device']}")
    print(f"Parameters         : {summary['trainable_parameters']:,}")
    print(df.head())

    # ── Loss and accuracy curves ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(df["epoch"], df["train_loss"], label="Train loss")
    axes[0].plot(df["epoch"], df["eval_loss"], label="Eval loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss — {experiment_name}")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_accuracy"], label="Train accuracy")
    axes[1].plot(df["epoch"], df["eval_accuracy"], label="Eval accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Accuracy — {experiment_name}")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{experiment_name}_convergence.png")
    plt.close(fig)

    # ── Privacy budget (epsilon) curve ────────────────────────────────────────
    if df["epsilon"].notna().any():
        fig, ax = plt.subplots()
        ax.plot(df["epoch"], df["epsilon"], color="tab:orange")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("ε  (privacy budget spent)")
        ax.set_title(f"ε vs Epoch — {experiment_name}  (δ = {df['delta'].iloc[0]:.0e})")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{experiment_name}_epsilon.png")
        plt.close(fig)
    else:
        print("No epsilon data (non-private run).")

    # ── Accuracy vs. ε  (privacy-utility trade-off) ───────────────────────────
    if df["epsilon"].notna().any():
        fig, ax = plt.subplots()
        ax.plot(df["epsilon"], df["eval_accuracy"], marker="o", markersize=3, linewidth=1)
        ax.set_xlabel("ε  (privacy budget)")
        ax.set_ylabel("Eval accuracy")
        ax.set_title(f"Privacy–Utility trade-off — {experiment_name}")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"{experiment_name}_privacy_utility.png")
        plt.close(fig)
    else:
        print("No epsilon data (non-private run).")

    # ── Multi-run comparison: eval accuracy across experiments ────────────────
    # Lists the experiments you want to compare. Each entry is an experiment name.
    experiments_to_compare = [
        "mnist-dp-sgd",
        "mnist-sgd",
        "cifar10-dp-sgd",
        "cifar10-sgd",
    ]

    fig, ax = plt.subplots()
    available = []
    for exp in experiments_to_compare:
        runs = [r for r in list_all_runs(exp) if (r / "history.csv").exists()]
        if not runs:
            print(f"  (no runs found for {exp}, skipping)")
            continue
        _df = load_run(runs[0])
        ax.plot(_df["epoch"], _df["eval_accuracy"], label=exp)
        available.append(exp)

    if available:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Eval accuracy")
        ax.set_title("Eval accuracy: multi-experiment comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "comparison_accuracy.png")
        plt.close(fig)
    else:
        print("No experiment results found yet. Run at least one training job first.")

    # ── Summary table: best eval accuracy and final epsilon ───────────────────
    rows = []
    for exp in experiments_to_compare:
        runs = [r for r in list_all_runs(exp) if (r / "metrics.json").exists()]
        if not runs:
            continue
        s = load_summary(runs[0])
        rows.append({
            "experiment": exp,
            "best_eval_acc": round(s["best_eval_accuracy"], 4),
            "final_epsilon": s["final_epsilon"],
            "params": s["trainable_parameters"],
            "device": s["device"],
        })

    if rows:
        print(pd.DataFrame(rows).set_index("experiment").to_string())
    else:
        print("No result data found yet.")


if __name__ == "__main__":
    main()

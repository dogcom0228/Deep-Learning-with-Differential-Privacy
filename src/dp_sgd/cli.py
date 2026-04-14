from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import load_config
from .train import run_experiment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dp-sgd")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model from a YAML config.")
    train_parser.add_argument("--config", required=True, help="Path to a YAML configuration file.")
    train_parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override nested config values, for example training.epochs=10.",
    )
    train_parser.add_argument("--epochs", type=int, help="Override training.epochs.")
    train_parser.add_argument("--batch-size", type=int, help="Override training.batch_size.")
    train_parser.add_argument("--device", help="Override runtime.device.")
    train_parser.add_argument("--output-dir", help="Override training.output_dir.")
    train_parser.add_argument("--private", action="store_true", help="Force private training on.")
    train_parser.add_argument("--non-private", action="store_true", help="Force private training off.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command != "train":
        parser.error(f"Unsupported command: {args.command}")

    if args.private and args.non_private:
        parser.error("Choose either --private or --non-private, not both.")

    config = load_config(args.config, overrides=args.override)
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.device is not None:
        config.runtime.device = args.device
    if args.output_dir is not None:
        config.training.output_dir = args.output_dir
    if args.private:
        config.privacy.enabled = True
    if args.non_private:
        config.privacy.enabled = False

    run_dir = run_experiment(config)
    print(run_dir)
    return 0


def entrypoint() -> None:
    raise SystemExit(main())

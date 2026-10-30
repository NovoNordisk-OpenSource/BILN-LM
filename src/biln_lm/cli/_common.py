"""Shared helpers for the ``--config file key=value`` CLI style."""

from __future__ import annotations

import argparse
import logging

from omegaconf import DictConfig

from ..config import load_config


def build_parser(description: str, default_config: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        epilog="Any remaining 'key=value' arguments override the config, e.g. training.num_train_epochs=1",
    )
    parser.add_argument("--config", default=default_config, help="YAML config file or name in configs/")
    parser.add_argument("--print-config", action="store_true", help="Print the merged config and exit")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    return parser


def parse_config(parser: argparse.ArgumentParser, argv: list[str] | None = None) -> tuple[DictConfig, argparse.Namespace]:
    args = parser.parse_args(argv)
    invalid = [o for o in args.overrides if "=" not in o]
    if invalid:
        parser.error(f"Overrides must be key=value, got: {invalid}")
    config = load_config(args.config, args.overrides)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return config, args

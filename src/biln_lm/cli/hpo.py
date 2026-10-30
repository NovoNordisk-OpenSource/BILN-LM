"""Run an Optuna search over the pretraining configuration."""

from __future__ import annotations

import json

from ..config import describe
from ..hpo import run_search
from ._common import build_parser, parse_config


def main(argv: list[str] | None = None) -> None:
    parser = build_parser(__doc__ or "", default_config="hpo.yaml")
    config, args = parse_config(parser, argv)
    if args.print_config:
        print(describe(config))
        return
    print(json.dumps(run_search(config), indent=2))


if __name__ == "__main__":
    main()

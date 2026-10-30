"""Pretrain (or continue pretraining) a BILN masked language model."""

from __future__ import annotations

import json

from ..config import describe
from ..pretrain import run_pretraining
from ._common import build_parser, parse_config


def main(argv: list[str] | None = None) -> None:
    parser = build_parser(__doc__ or "", default_config="pretrain.yaml")
    config, args = parse_config(parser, argv)
    #print(describe(config))
    if args.print_config:
        print(describe(config))
        return
    print(json.dumps(run_pretraining(config), indent=2))


if __name__ == "__main__":
    main()

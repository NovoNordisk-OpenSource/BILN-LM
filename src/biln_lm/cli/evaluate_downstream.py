"""Evaluate pretrained BILN-LM embeddings on the downstream benchmarks."""

from __future__ import annotations

from ..config import describe
from ..downstream import run_benchmark
from ._common import build_parser, parse_config


def main(argv: list[str] | None = None) -> None:
    parser = build_parser(__doc__ or "", default_config="downstream.yaml")
    config, args = parse_config(parser, argv)
    if args.print_config:
        print(describe(config))
        return
    results = run_benchmark(config)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

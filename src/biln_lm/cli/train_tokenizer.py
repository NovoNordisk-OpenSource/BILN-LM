"""Build (or train) a BILN tokenizer and save it standalone."""

from __future__ import annotations

from pathlib import Path

from ..config import describe
from ..data import load_corpus
from ..tokenizer import build_tokenizer, unknown_token_rate
from ._common import build_parser, parse_config


def main(argv: list[str] | None = None) -> None:
    parser = build_parser(__doc__ or "", default_config="pretrain.yaml")
    config, args = parse_config(parser, argv)
    if args.print_config:
        print(describe(config))
        return

    sequences = load_corpus(
        files=list(config.data.files),
        data_dir=config.paths.data_dir,
        column=config.data.column,
        max_samples=config.data.get("max_samples"),
        seed=int(config.seed),
    )
    tokenizer = build_tokenizer(config, sequences)
    out_dir = Path(config.paths.output_dir) / "tokenizer"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(out_dir))

    print(f"kind={config.tokenizer.kind} vocab={len(tokenizer)} sequences={len(sequences)}")
    print(f"unknown token rate: {unknown_token_rate(tokenizer, sequences[:5000]):.5f}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()

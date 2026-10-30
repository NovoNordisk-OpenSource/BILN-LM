"""Generate embeddings for a file of BILN sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import read_sequences
from ..embeddings import POOLINGS, embed_biln


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with a BILN column, or a one-per-line text file")
    parser.add_argument("--model-dir", required=True, help="Directory produced by biln-pretrain")
    parser.add_argument("--out", required=True, help="Output .npy (or .csv to keep sequences alongside)")
    parser.add_argument("--column", default="biln")
    parser.add_argument("--pooling", default="mean", choices=POOLINGS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args(argv)

    sequences = read_sequences(args.input, args.column)
    embeddings = embed_biln(
        sequences,
        model_dir=args.model_dir,
        pooling=args.pooling,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
        show_progress=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".csv":
        frame = pd.DataFrame(embeddings, columns=[f"e{i}" for i in range(embeddings.shape[1])])
        frame.insert(0, args.column, sequences)
        frame.to_csv(out_path, index=False)
    else:
        np.save(out_path, embeddings)
    print(f"{embeddings.shape[0]} sequences x {embeddings.shape[1]} dims -> {out_path}")


if __name__ == "__main__":
    main()

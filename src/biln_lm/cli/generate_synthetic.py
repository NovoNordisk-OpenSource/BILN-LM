"""Generate a synthetic BILN corpus for pretraining."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from ..config import describe
from ..data import load_corpus
from ..monomers import MonomerLibrary
from ..synthetic import DeNovoSettings, augment_corpus, filter_sequences, sample_peptides
from ._common import build_parser, parse_config


def main(argv: list[str] | None = None) -> None:
    parser = build_parser(__doc__ or "", default_config="synthetic.yaml")
    config, args = parse_config(parser, argv)
    if args.print_config:
        print(describe(config))
        return

    settings = config.synthetic
    seed = int(settings.get("seed", config.seed))
    library = MonomerLibrary.from_csv(config.paths.get("monomer_library"))

    records: list[dict] = []

    de_novo = settings.de_novo
    if int(de_novo.n) > 0:
        sequences = sample_peptides(
            n=int(de_novo.n),
            library=library,
            settings=DeNovoSettings.from_config(OmegaConf.to_container(de_novo, resolve=True)),
            seed=seed,
        )
        records += [{"biln": s, "source": "de_novo"} for s in sequences]
        print(f"de-novo: {len(sequences)} sequences")

    augment = settings.augment
    if augment.enabled:
        source = load_corpus(
            files=list(augment.source_files),
            data_dir=config.paths.data_dir,
            column=augment.source_column,
            max_samples=augment.get("max_source_sequences"),
            seed=seed,
        )
        sequences = augment_corpus(
            source,
            library=library,
            n_per_sequence=int(augment.n_per_sequence),
            operations=list(augment.operations),
            p_mutation=float(augment.p_mutation),
            seed=seed,
        )
        records += [{"biln": s, "source": "augmented"} for s in sequences]
        print(f"augmented: {len(sequences)} sequences from {len(source)} source sequences")

    frame = pd.DataFrame(records)
    kept = set(
        filter_sequences(
            frame["biln"].tolist(),
            deduplicate=bool(settings.deduplicate),
        )
    )
    frame = frame[frame["biln"].isin(kept)].drop_duplicates(subset=["biln"])

    out_path = Path(config.paths.output_dir) / settings.out_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f"wrote {len(frame)} valid unique sequences -> {out_path}")


if __name__ == "__main__":
    main()

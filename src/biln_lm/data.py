"""Corpus loading and dataset construction for BILN masked language modelling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast

from .biln import is_valid_biln

TEXT_COLUMN = "text"


def resolve_files(files: list[str], data_dir: str | Path | None) -> list[Path]:
    """Interpret each entry as an absolute path or as relative to ``data_dir``."""
    resolved = []
    for name in files:
        path = Path(name)
        if not path.is_absolute() and data_dir is not None:
            path = Path(data_dir) / path
        if not path.is_file():
            raise FileNotFoundError(f"Corpus file not found: {path}")
        resolved.append(path)
    return resolved


def read_sequences(path: str | Path, column: str = "biln") -> list[str]:
    """Read BILN strings from a ``.csv`` column or from a one-per-line text file."""
    path = Path(path)
    if path.suffix.lower() in {".csv", ".tsv"}:
        separator = "\t" if path.suffix.lower() == ".tsv" else ","
        frame = pd.read_csv(path, sep=separator)
        if column not in frame.columns:
            raise KeyError(f"Column {column!r} not in {path} (found {list(frame.columns)})")
        values = frame[column]
    else:
        values = pd.Series(path.read_text().splitlines())
    return [str(v).strip() for v in values.dropna() if str(v).strip()]


def load_corpus(
    files: list[str],
    data_dir: str | Path | None = None,
    column: str = "biln",
    drop_invalid: bool = True,
    deduplicate: bool = True,
    max_samples: int | None = None,
    seed: int = 1,
) -> list[str]:
    """Collect BILN strings from one or more files into a single corpus."""
    sequences: list[str] = []
    for path in resolve_files(files, data_dir):
        sequences.extend(read_sequences(path, column))
    if drop_invalid:
        sequences = [s for s in sequences if is_valid_biln(s)]
    if deduplicate:
        sequences = list(dict.fromkeys(sequences))
    if max_samples is not None and len(sequences) > max_samples:
        sequences = (
            pd.Series(sequences).sample(n=max_samples, random_state=seed).tolist()
        )
    if not sequences:
        raise ValueError(f"No usable BILN sequences found in {files}")
    return sequences


def build_datasets(
    sequences: list[str],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 256,
    valid_fraction: float = 0.1,
    seed: int = 1,
    num_proc: int | None = None,
) -> DatasetDict:
    """Tokenize the corpus and split it into train/validation sets."""
    dataset = Dataset.from_dict({TEXT_COLUMN: sequences})

    def tokenize(batch: dict) -> dict:
        return tokenizer(batch[TEXT_COLUMN], truncation=True, max_length=max_length)

    dataset = dataset.map(
        tokenize, batched=True, num_proc=num_proc, remove_columns=[TEXT_COLUMN]
    )
    split = dataset.train_test_split(test_size=valid_fraction, seed=seed)
    return DatasetDict(train=split["train"], validation=split["test"])


def build_collator(
    tokenizer: PreTrainedTokenizerFast, mlm_probability: float = 0.15
) -> DataCollatorForLanguageModeling:
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
    )

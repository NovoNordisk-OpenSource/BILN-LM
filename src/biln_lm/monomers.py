"""Monomer library backing the tokenizer vocabulary and the synthetic generator."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .biln import CHAIN_SEPARATOR, MONOMER_SEPARATOR, BilnParseError, split_monomers, split_on_depth_zero

DEFAULT_LIBRARY_PATH = Path(__file__).resolve().parent / "data" / "helm_monomer_library.csv"

CANONICAL_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")

REQUIRED_COLUMNS = ("symbol", "natural_analog", "monomer_type")


def is_biln_safe(symbol: str) -> bool:
    """True if the symbol survives a BILN split unchanged when placed after a residue.

    ``-pip`` qualifies, ``1-Nal`` and ``Mono21-`` do not.
    """
    try:
        return (
            split_monomers(f"X{MONOMER_SEPARATOR}{symbol}") == ["X", symbol]
            and len(split_on_depth_zero(symbol, CHAIN_SEPARATOR)) == 1
        )
    except BilnParseError:
        return False


@dataclass
class MonomerLibrary:
    """Symbols available for building BILN strings, grouped by monomer type."""

    table: pd.DataFrame

    @classmethod
    def from_csv(cls, path: str | Path | None = None, biln_safe_only: bool = True) -> "MonomerLibrary":
        path = Path(path) if path is not None else DEFAULT_LIBRARY_PATH
        if not path.is_file():
            raise FileNotFoundError(f"Monomer library not found: {path}")
        table = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f"Monomer library {path} is missing columns: {missing}")
        table = table.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])
        if biln_safe_only:
            table = table[table["symbol"].astype(str).map(is_biln_safe)]
        return cls(table.reset_index(drop=True))

    def __len__(self) -> int:
        return len(self.table)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.symbol_set

    @property
    def symbols(self) -> list[str]:
        return self.table["symbol"].astype(str).tolist()

    @property
    def symbol_set(self) -> set[str]:
        return set(self.symbols)

    def symbols_of_type(self, monomer_type: str) -> list[str]:
        mask = self.table["monomer_type"].astype(str).str.lower() == monomer_type.lower()
        return self.table.loc[mask, "symbol"].astype(str).tolist()

    @property
    def backbone_symbols(self) -> list[str]:
        return self.symbols_of_type("Backbone")

    @property
    def terminal_symbols(self) -> list[str]:
        return self.symbols_of_type("Terminal")

    @property
    def n_terminal_symbols(self) -> list[str]:
        # A symbol such as ``-pip`` cannot open a BILN chain, only follow a residue.
        return [s for s in self.terminal_symbols if not s.startswith("-")]

    @property
    def canonical_symbols(self) -> list[str]:
        return [s for s in CANONICAL_AMINO_ACIDS if s in self.symbol_set]

    @property
    def non_canonical_symbols(self) -> list[str]:
        canonical = set(CANONICAL_AMINO_ACIDS)
        return [s for s in self.backbone_symbols if s not in canonical]

    def analogs_of(self, symbol: str) -> list[str]:
        """Symbols sharing the same natural analog, excluding ``symbol`` itself."""
        rows = self.table.loc[self.table["symbol"] == symbol, "natural_analog"]
        if rows.empty or pd.isna(rows.iloc[0]):
            return []
        analog = rows.iloc[0]
        mask = (self.table["natural_analog"] == analog) & (self.table["symbol"] != symbol)
        return self.table.loc[mask, "symbol"].astype(str).tolist()

    def sample(self, n: int, pool: list[str], rng: random.Random) -> list[str]:
        return [rng.choice(pool) for _ in range(n)]

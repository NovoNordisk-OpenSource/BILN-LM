"""Parsing utilities for BILN (Biological Identifier Language Notation) strings.

BILN grammar handled here::

    <biln>     := <chain> ("." <chain>)*
    <chain>    := <monomer> ("-" <monomer>)*
    <monomer>  := <symbol> <connection>*
    <connection> := "(" <bond_id> "," <rgroup> ")"

The symbol itself may contain parentheses and hyphens (e.g. ``Phe(4-Cl)``), so
splitting must be parenthesis-depth aware rather than a plain ``str.split``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CHAIN_SEPARATOR = "."
MONOMER_SEPARATOR = "-"

_CONNECTION_RE = re.compile(r"\((\d+),(\d+)\)")
_TRAILING_CONNECTIONS_RE = re.compile(r"(?:\(\d+,\d+\))+$")


@dataclass(frozen=True)
class Monomer:
    """A single BILN residue: a symbol plus its inter-monomer bond annotations."""

    symbol: str
    connections: tuple[tuple[int, int], ...] = ()

    def __str__(self) -> str:
        return self.symbol + "".join(f"({i},{j})" for i, j in self.connections)


@dataclass
class Peptide:
    """A parsed BILN string as a list of chains, each a list of monomers."""

    chains: list[list[Monomer]] = field(default_factory=list)

    def __str__(self) -> str:
        return CHAIN_SEPARATOR.join(
            MONOMER_SEPARATOR.join(str(m) for m in chain) for chain in self.chains
        )

    def __len__(self) -> int:
        return sum(len(chain) for chain in self.chains)

    @property
    def symbols(self) -> list[str]:
        return [m.symbol for m in self.monomers]

    @property
    def monomers(self) -> list[Monomer]:
        return [m for chain in self.chains for m in chain]


class BilnParseError(ValueError):
    """Raised when a string cannot be parsed as BILN."""


def split_on_depth_zero(text: str, separator: str) -> list[str]:
    """Split ``text`` on ``separator`` occurrences outside of any parentheses."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise BilnParseError(f"Unbalanced parentheses in {text!r}")
        if char == separator and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if depth != 0:
        raise BilnParseError(f"Unbalanced parentheses in {text!r}")
    parts.append("".join(current))
    return parts


def parse_monomer(token: str) -> Monomer:
    """Split a monomer token into its symbol and trailing ``(bond,rgroup)`` pairs."""
    token = token.strip()
    if not token:
        raise BilnParseError("Empty monomer token")
    match = _TRAILING_CONNECTIONS_RE.search(token)
    if match is None:
        return Monomer(token)
    symbol = token[: match.start()]
    if not symbol:
        raise BilnParseError(f"Monomer token {token!r} has no symbol")
    connections = tuple(
        (int(bond), int(rgroup)) for bond, rgroup in _CONNECTION_RE.findall(match.group())
    )
    return Monomer(symbol, connections)


def split_monomers(chain: str) -> list[str]:
    """Split a chain into monomer tokens.

    A few library symbols start with a hyphen (``-pip``, ``-aze``, ...), so a run of
    hyphens is one separator followed by a symbol that keeps the remaining hyphens.
    """
    tokens: list[str] = []
    prefix = ""
    for index, part in enumerate(split_on_depth_zero(chain, MONOMER_SEPARATOR)):
        if not part:
            if index:
                prefix += MONOMER_SEPARATOR
            continue
        tokens.append(prefix + part)
        prefix = ""
    return tokens


def parse_biln(biln: str) -> Peptide:
    """Parse a BILN string into a :class:`Peptide`."""
    if not isinstance(biln, str) or not biln.strip():
        raise BilnParseError(f"Not a BILN string: {biln!r}")
    chains = []
    for raw_chain in split_on_depth_zero(biln.strip(), CHAIN_SEPARATOR):
        if not raw_chain:
            raise BilnParseError(f"Empty chain in {biln!r}")
        chains.append([parse_monomer(token) for token in split_monomers(raw_chain)])
    return Peptide(chains)


def is_valid_biln(biln: str) -> bool:
    """Return True if ``biln`` parses and every bond id appears exactly twice."""
    try:
        peptide = parse_biln(biln)
    except BilnParseError:
        return False
    bond_counts: dict[int, int] = {}
    for monomer in peptide.monomers:
        for bond_id, _ in monomer.connections:
            bond_counts[bond_id] = bond_counts.get(bond_id, 0) + 1
    return all(count == 2 for count in bond_counts.values())


def pretokenize(biln: str) -> list[str]:
    """Split a BILN string into the atomic units used by the monomer tokenizer.

    Symbols, ``(bond,rgroup)`` annotations and chain separators become separate
    tokens; the monomer separator ``-`` is dropped as it carries no information.
    """
    tokens: list[str] = []
    for index, chain in enumerate(split_on_depth_zero(biln.strip(), CHAIN_SEPARATOR)):
        if index:
            tokens.append(CHAIN_SEPARATOR)
        for token in split_monomers(chain):
            monomer = parse_monomer(token)
            tokens.append(monomer.symbol)
            tokens.extend(f"({i},{j})" for i, j in monomer.connections)
    return tokens


def iter_symbols(biln_sequences: list[str], skip_invalid: bool = True) -> list[str]:
    """Collect every distinct monomer symbol used across a corpus."""
    symbols: set[str] = set()
    for sequence in biln_sequences:
        try:
            symbols.update(parse_biln(sequence).symbols)
        except BilnParseError:
            if not skip_invalid:
                raise
    return sorted(symbols)


# Some public datasets write residues as upper-case three-letter codes.
THREE_LETTER_CODES = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "NH2": "am", "ACE": "ac",
}


def normalize_biln(biln: str, aliases: dict[str, str] | None = None) -> str:
    """Rewrite known symbol aliases (e.g. ``HIS`` -> ``H``) to library symbols."""
    aliases = {**THREE_LETTER_CODES, **(aliases or {})}
    peptide = parse_biln(biln)
    peptide.chains = [
        [Monomer(aliases.get(m.symbol, m.symbol), m.connections) for m in chain]
        for chain in peptide.chains
    ]
    return str(peptide)

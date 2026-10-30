"""Synthetic BILN corpora: de-novo sampling and augmentation of an existing corpus.

Both routes exist because pretraining data for non-canonical peptides is scarce:
de-novo sampling covers the monomer library uniformly, while augmentation keeps
the residue statistics of real peptides.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .biln import Monomer, Peptide, is_valid_biln, parse_biln
from .monomers import MonomerLibrary

AUGMENTATION_OPERATIONS = ("point_mutation", "analog_swap", "truncation", "segment_shuffle")


@dataclass
class DeNovoSettings:
    n: int = 10000
    length_range: tuple[int, int] = (5, 30)
    p_non_canonical: float = 0.3
    p_n_terminal_cap: float = 0.15
    p_c_terminal_cap: float = 0.25
    p_cyclic: float = 0.2
    p_disulfide: float = 0.1
    p_multi_chain: float = 0.05
    max_chains: int = 3

    @classmethod
    def from_config(cls, config) -> "DeNovoSettings":
        settings = cls()
        for field_name in cls.__dataclass_fields__:
            if field_name in config:
                value = config[field_name]
                setattr(settings, field_name, tuple(value) if field_name == "length_range" else value)
        return settings


def _sample_residue(library: MonomerLibrary, settings: DeNovoSettings, rng: random.Random) -> str:
    if rng.random() < settings.p_non_canonical and library.non_canonical_symbols:
        return rng.choice(library.non_canonical_symbols)
    return rng.choice(library.canonical_symbols)


def _sample_chain(
    library: MonomerLibrary,
    settings: DeNovoSettings,
    rng: random.Random,
    next_bond_id: int,
) -> tuple[list[Monomer], int]:
    low, high = settings.length_range
    length = rng.randint(low, high)
    symbols = [_sample_residue(library, settings, rng) for _ in range(length)]

    connections: dict[int, list[tuple[int, int]]] = {}

    def add_bond(index_a: int, index_b: int, rgroup_a: int, rgroup_b: int, bond_id: int) -> None:
        connections.setdefault(index_a, []).append((bond_id, rgroup_a))
        connections.setdefault(index_b, []).append((bond_id, rgroup_b))

    head_to_tail = len(symbols) >= 4 and rng.random() < settings.p_cyclic
    if head_to_tail:
        # (n,1)/(n,2) is the BILN convention for a head-to-tail macrocycle.
        add_bond(0, len(symbols) - 1, 1, 2, next_bond_id)
        next_bond_id += 1
    else:
        # Free termini may be capped; a macrocycle has none to cap.
        n_terminals = library.n_terminal_symbols
        c_terminals = library.terminal_symbols
        if n_terminals and rng.random() < settings.p_n_terminal_cap:
            symbols.insert(0, rng.choice(n_terminals))
        if c_terminals and rng.random() < settings.p_c_terminal_cap:
            symbols.append(rng.choice(c_terminals))

    if len(symbols) >= 4 and rng.random() < settings.p_disulfide:
        first, second = sorted(rng.sample(range(len(symbols)), 2))
        if second - first >= 2:
            symbols[first] = "C"
            symbols[second] = "C"
            add_bond(first, second, 3, 3, next_bond_id)
            next_bond_id += 1

    chain = [
        Monomer(symbol, tuple(connections.get(index, ())))
        for index, symbol in enumerate(symbols)
    ]
    return chain, next_bond_id


def sample_peptide(library: MonomerLibrary, settings: DeNovoSettings, rng: random.Random) -> str:
    n_chains = 1
    if settings.max_chains > 1 and rng.random() < settings.p_multi_chain:
        n_chains = rng.randint(2, settings.max_chains)
    chains = []
    next_bond_id = 1
    for _ in range(n_chains):
        chain, next_bond_id = _sample_chain(library, settings, rng, next_bond_id)
        chains.append(chain)
    return str(Peptide(chains))


def sample_peptides(
    n: int,
    library: MonomerLibrary | None = None,
    settings: DeNovoSettings | None = None,
    seed: int = 1,
) -> list[str]:
    """Draw ``n`` de-novo BILN strings from the monomer library."""
    library = library or MonomerLibrary.from_csv()
    settings = settings or DeNovoSettings()
    rng = random.Random(seed)
    return [sample_peptide(library, settings, rng) for _ in range(n)]


def _mutate(peptide: Peptide, library: MonomerLibrary, p: float, rng: random.Random, analog_only: bool) -> Peptide:
    pool = library.backbone_symbols
    chains = []
    for chain in peptide.chains:
        new_chain = []
        for monomer in chain:
            # Connected residues carry the topology, so they are never replaced.
            if monomer.connections or rng.random() >= p:
                new_chain.append(monomer)
                continue
            if analog_only:
                analogs = library.analogs_of(monomer.symbol)
                replacement = rng.choice(analogs) if analogs else monomer.symbol
            else:
                replacement = rng.choice(pool)
            new_chain.append(Monomer(replacement, monomer.connections))
        chains.append(new_chain)
    return Peptide(chains)


def _truncate(peptide: Peptide, rng: random.Random) -> Peptide:
    chains = []
    for chain in peptide.chains:
        if len(chain) < 6 or any(m.connections for m in chain):
            chains.append(chain)
            continue
        max_cut = max(1, len(chain) // 4)
        start = rng.randint(0, max_cut)
        end = len(chain) - rng.randint(0, max_cut)
        chains.append(chain[start:end] or chain)
    return Peptide(chains)


def _shuffle_segment(peptide: Peptide, rng: random.Random) -> Peptide:
    chains = []
    for chain in peptide.chains:
        if len(chain) < 6 or any(m.connections for m in chain):
            chains.append(chain)
            continue
        start = rng.randint(0, len(chain) - 4)
        end = rng.randint(start + 3, len(chain))
        segment = chain[start:end]
        rng.shuffle(segment)
        chains.append(chain[:start] + segment + chain[end:])
    return Peptide(chains)


def augment_sequence(
    biln: str,
    library: MonomerLibrary,
    rng: random.Random,
    operations: list[str],
    p_mutation: float = 0.1,
) -> str:
    peptide = parse_biln(biln)
    operation = rng.choice(operations)
    if operation == "point_mutation":
        peptide = _mutate(peptide, library, p_mutation, rng, analog_only=False)
    elif operation == "analog_swap":
        peptide = _mutate(peptide, library, p_mutation, rng, analog_only=True)
    elif operation == "truncation":
        peptide = _truncate(peptide, rng)
    elif operation == "segment_shuffle":
        peptide = _shuffle_segment(peptide, rng)
    else:
        raise ValueError(f"Unknown augmentation operation: {operation!r}")
    return str(peptide)


def augment_corpus(
    sequences: list[str],
    library: MonomerLibrary | None = None,
    n_per_sequence: int = 1,
    operations: list[str] | None = None,
    p_mutation: float = 0.1,
    seed: int = 1,
) -> list[str]:
    """Derive new BILN strings from an existing corpus."""
    library = library or MonomerLibrary.from_csv()
    operations = list(operations or AUGMENTATION_OPERATIONS)
    unknown = set(operations) - set(AUGMENTATION_OPERATIONS)
    if unknown:
        raise ValueError(f"Unknown augmentation operations: {sorted(unknown)}")
    rng = random.Random(seed)
    generated = []
    for sequence in sequences:
        for _ in range(n_per_sequence):
            try:
                generated.append(augment_sequence(sequence, library, rng, operations, p_mutation))
            except ValueError:
                continue
    return generated


def filter_sequences(
    sequences: list[str], deduplicate: bool = True) -> list[str]:
    """Keep syntactically valid BILN, optionally deduplicated and pepfunn-checked."""
    kept = [s for s in sequences if is_valid_biln(s)]
    if deduplicate:
        kept = list(dict.fromkeys(kept))
    
    return kept

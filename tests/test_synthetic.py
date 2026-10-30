import pytest

from biln_lm.biln import is_valid_biln, parse_biln
from biln_lm.monomers import MonomerLibrary
from biln_lm.synthetic import (
    AUGMENTATION_OPERATIONS,
    DeNovoSettings,
    augment_corpus,
    filter_sequences,
    sample_peptides,
)

SOURCE = [
    "C-P-W-H-L-L-P-F-C",
    "S-W-L-A-Y-P-G-A-V-A-Y-R",
    "ac-dPhe(4-Cl)-Me_dR-G-am",
    "L(1,1)-dL-dL-dL-P-Y(1,2)",
]


@pytest.fixture(scope="module")
def library():
    return MonomerLibrary.from_csv()


def test_de_novo_sequences_are_valid(library):
    sequences = sample_peptides(200, library=library, seed=0)
    assert len(sequences) == 200
    assert all(is_valid_biln(s) for s in sequences)


def test_de_novo_is_deterministic(library):
    assert sample_peptides(20, library=library, seed=7) == sample_peptides(20, library=library, seed=7)


def test_length_range_is_respected(library):
    settings = DeNovoSettings(length_range=(6, 8), p_cyclic=0.0, p_multi_chain=0.0,
                              p_n_terminal_cap=0.0, p_c_terminal_cap=0.0)
    for sequence in sample_peptides(50, library=library, settings=settings, seed=3):
        assert 6 <= len(parse_biln(sequence)) <= 8


def test_cyclic_peptides_have_paired_bonds(library):
    settings = DeNovoSettings(length_range=(8, 12), p_cyclic=1.0, p_disulfide=0.0, p_multi_chain=0.0)
    for sequence in sample_peptides(30, library=library, settings=settings, seed=4):
        peptide = parse_biln(sequence)
        assert peptide.chains[0][0].connections == ((1, 1),)
        assert peptide.chains[0][-1].connections == ((1, 2),)
        assert is_valid_biln(sequence)


@pytest.mark.parametrize("operation", AUGMENTATION_OPERATIONS)
def test_each_augmentation_produces_valid_biln(library, operation):
    generated = augment_corpus(SOURCE * 10, library=library, operations=[operation], seed=1)
    assert generated
    assert all(is_valid_biln(s) for s in generated)


def test_augmentation_rejects_unknown_operation(library):
    with pytest.raises(ValueError):
        augment_corpus(SOURCE, library=library, operations=["nope"])


def test_filter_deduplicates_and_drops_invalid():
    assert filter_sequences(["A-B", "A-B", "A-(1,1)"]) == ["A-B"]

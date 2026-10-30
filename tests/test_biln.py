import pytest

from biln_lm.biln import (
    BilnParseError,
    is_valid_biln,
    normalize_biln,
    parse_biln,
    pretokenize,
    split_on_depth_zero,
)


@pytest.mark.parametrize(
    "biln, n_monomers, symbols",
    [
        ("C-P-W-H", 4, ["C", "P", "W", "H"]),
        ("ac-S-E-K-P", 5, ["ac", "S", "E", "K", "P"]),
        ("Y-P-F-F_OtBu", 4, ["Y", "P", "F", "F_OtBu"]),
        # a hyphen inside a symbol must not split the residue
        ("dPhe(4-Cl)-A", 2, ["dPhe(4-Cl)", "A"]),
        ("A.B", 2, ["A", "B"]),
    ],
)
def test_parse_basic(biln, n_monomers, symbols):
    peptide = parse_biln(biln)
    assert len(peptide) == n_monomers
    assert peptide.symbols == symbols


def test_parse_connections():
    peptide = parse_biln("L(1,1)-dL-P-Y(1,2)")
    assert peptide.chains[0][0].connections == ((1, 1),)
    assert peptide.chains[0][-1].connections == ((1, 2),)
    assert peptide.chains[0][1].connections == ()


def test_multiple_connections_on_one_monomer():
    peptide = parse_biln("Dpr(1,3)(3,1)-A-G(1,1)-C(3,3)")
    assert peptide.chains[0][0].connections == ((1, 3), (3, 1))


@pytest.mark.parametrize("bad", ["", "   ", "A-(1,1)", "A-B)", "A..B"])
def test_invalid_input(bad):
    with pytest.raises(BilnParseError):
        parse_biln(bad)


def test_is_valid_requires_paired_bonds():
    assert is_valid_biln("L(1,1)-A-Y(1,2)")
    assert not is_valid_biln("L(1,1)-A-Y")


def test_split_on_depth_zero_ignores_nested():
    assert split_on_depth_zero("A-Phe(4-Cl)-B", "-") == ["A", "Phe(4-Cl)", "B"]


def test_symbol_starting_with_hyphen():
    peptide = parse_biln("A(1,1)-L-D(1,3)--pip")
    assert peptide.symbols == ["A", "L", "D", "-pip"]
    assert str(peptide) == "A(1,1)-L-D(1,3)--pip"


def test_pretokenize_emits_symbols_and_connections():
    assert pretokenize("ac-L(1,1)-A.G") == ["ac", "L", "(1,1)", "A", ".", "G"]


def test_normalize_three_letter_codes():
    assert normalize_biln("HIS-PRO-ARG-NH2") == "H-P-R-am"


def test_roundtrip_on_corpus(corpus):
    for sequence in corpus:
        assert str(parse_biln(sequence)) == sequence

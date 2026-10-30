import pytest

from biln_lm.biln import pretokenize
from biln_lm.monomers import MonomerLibrary, is_biln_safe
from biln_lm.tokenizer import (
    MASK_TOKEN,
    PAD_TOKEN,
    build_monomer_tokenizer,
    train_bpe_tokenizer,
    unknown_token_rate,
)

SEQUENCES = [
    "C-P-W-H-L-L-P-F-C",
    "ac-dPhe(4-Cl)(1,3)-Me_dR-G-am",
    "L(1,1)-dL-dL-P-Y(1,2)",
    "A-K-C-R-A.G-G-G",
    "A(1,1)-L-meV-D(1,3)--pip",
]


@pytest.fixture(scope="module")
def tokenizer():
    return build_monomer_tokenizer(sequences=SEQUENCES)


def test_special_tokens_present(tokenizer):
    assert tokenizer.pad_token == PAD_TOKEN
    assert tokenizer.mask_token == MASK_TOKEN
    assert tokenizer.pad_token_id is not None


def test_tokenizes_one_token_per_unit(tokenizer):
    assert tokenizer.tokenize("ac-dPhe(4-Cl)(1,3)-Me_dR-G-am") == [
        "ac",
        "dPhe(4-Cl)",
        "(1,3)",
        "Me_dR",
        "G",
        "am",
    ]


def test_pre_tokenizer_matches_parser(tokenizer):
    for sequence in SEQUENCES:
        produced = [t for t, _ in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sequence)]
        assert produced == pretokenize(sequence)


def test_library_symbols_are_biln_safe():
    library = MonomerLibrary.from_csv()
    assert all(is_biln_safe(symbol) for symbol in library.symbols)
    assert not is_biln_safe("1-Nal")
    assert not is_biln_safe("Mono21-")
    assert is_biln_safe("-pip")


def test_corpus_coverage(corpus, tokenizer):
    """The library-backed vocabulary should cover essentially the whole corpus."""
    full = build_monomer_tokenizer(sequences=corpus)
    assert unknown_token_rate(full, corpus) < 0.01


def test_bpe_tokenizer_trains():
    bpe = train_bpe_tokenizer(SEQUENCES * 20, vocab_size=120)
    assert len(bpe) > len(bpe.all_special_tokens)
    assert bpe.tokenize(SEQUENCES[0])

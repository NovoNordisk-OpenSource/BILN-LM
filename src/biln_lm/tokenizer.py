"""Tokenizers for BILN strings.

Two flavours are available:

``monomer``
    A word-level tokenizer whose vocabulary is the monomer library (plus any
    extra symbols seen in the corpus) and the ``(bond,rgroup)`` annotations.
    One token per residue, so the model sees chemistry rather than substrings.

``bpe``
    Byte-pair encoding trained on the corpus.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from tokenizers import Regex, Tokenizer, pre_tokenizers, trainers
from tokenizers.models import BPE, WordLevel
from transformers import PreTrainedTokenizerFast

from .biln import BilnParseError, parse_biln
from .monomers import MonomerLibrary

UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]

CHAIN_TOKEN = "."

# Isolate the connection annotations first, then split the residue separator only
# where it is not inside a symbol such as ``Phe(4-Cl)``. The look-behind keeps the
# second hyphen of ``D(1,3)--pip`` attached to the ``-pip`` symbol.
_CONNECTION_PATTERN = Regex(r"\(\d+,\d+\)")
_RESIDUE_SEPARATOR_PATTERN = Regex(r"(?<!-)-(?![^()]*\))")

_BILN_PRE_TOKENIZER = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Split(pattern=_CONNECTION_PATTERN, behavior="isolated"),
        pre_tokenizers.Split(pattern=_RESIDUE_SEPARATOR_PATTERN, behavior="removed"),
        pre_tokenizers.Split(pattern=CHAIN_TOKEN, behavior="isolated"),
        pre_tokenizers.WhitespaceSplit(),
    ]
)


def _wrap(tokenizer: Tokenizer, max_length: int) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=max_length,
        unk_token=UNK_TOKEN,
        cls_token=CLS_TOKEN,
        sep_token=SEP_TOKEN,
        pad_token=PAD_TOKEN,
        mask_token=MASK_TOKEN,
    )


def connection_tokens(max_bond_id: int, max_rgroup: int) -> list[str]:
    return [
        f"({bond},{rgroup})"
        for bond in range(1, max_bond_id + 1)
        for rgroup in range(1, max_rgroup + 1)
    ]


def corpus_symbols(sequences: list[str], min_frequency: int = 1) -> list[str]:
    """Monomer symbols occurring at least ``min_frequency`` times in ``sequences``."""
    counts: Counter[str] = Counter()
    for sequence in sequences:
        try:
            counts.update(parse_biln(sequence).symbols)
        except BilnParseError:
            continue
    return sorted(symbol for symbol, count in counts.items() if count >= min_frequency)


def build_monomer_tokenizer(
    sequences: list[str] | None = None,
    library: MonomerLibrary | None = None,
    max_bond_id: int = 8,
    max_rgroup: int = 3,
    min_symbol_frequency: int = 1,
    include_library_symbols: bool = True,
    max_length: int = 256,
) -> PreTrainedTokenizerFast:
    """Build a word-level tokenizer over monomer symbols and bond annotations."""
    vocab_tokens: list[str] = list(SPECIAL_TOKENS)
    vocab_tokens.append(CHAIN_TOKEN)
    vocab_tokens.extend(connection_tokens(max_bond_id, max_rgroup))

    known = set(vocab_tokens)
    if include_library_symbols:
        library = library or MonomerLibrary.from_csv()
        for symbol in library.symbols:
            if symbol not in known:
                known.add(symbol)
                vocab_tokens.append(symbol)
    if sequences:
        for symbol in corpus_symbols(sequences, min_symbol_frequency):
            if symbol not in known:
                known.add(symbol)
                vocab_tokens.append(symbol)

    vocab = {token: index for index, token in enumerate(vocab_tokens)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = _BILN_PRE_TOKENIZER
    return _wrap(tokenizer, max_length)


def train_bpe_tokenizer(
    sequences: list[str],
    vocab_size: int = 3000,
    max_length: int = 256,
    min_frequency: int = 2,
) -> PreTrainedTokenizerFast:
    """Train byte-pair encoding on a BILN corpus."""
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = _BILN_PRE_TOKENIZER
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(sequences, trainer=trainer, length=len(sequences))
    return _wrap(tokenizer, max_length)


def load_tokenizer(path: str | Path, max_length: int = 256) -> PreTrainedTokenizerFast:
    """Load a tokenizer from a ``save_pretrained`` directory or a ``tokenizer.json``."""
    path = Path(path)
    if path.is_dir():
        return PreTrainedTokenizerFast.from_pretrained(str(path), model_max_length=max_length)
    return _wrap(Tokenizer.from_file(str(path)), max_length)


def build_tokenizer(config, sequences: list[str] | None = None) -> PreTrainedTokenizerFast:
    """Create the tokenizer described by ``config.tokenizer``."""
    settings = config.tokenizer
    max_length = int(config.data.max_length)
    if settings.get("path"):
        return load_tokenizer(settings.path, max_length)

    kind = str(settings.kind).lower()
    if kind == "monomer":
        library = (
            MonomerLibrary.from_csv(config.paths.get("monomer_library"))
            if settings.include_library_symbols
            else None
        )
        return build_monomer_tokenizer(
            sequences=sequences,
            library=library,
            max_bond_id=int(settings.max_bond_id),
            max_rgroup=int(settings.max_rgroup),
            min_symbol_frequency=int(settings.min_symbol_frequency),
            include_library_symbols=bool(settings.include_library_symbols),
            max_length=max_length,
        )
    if kind == "bpe":
        if not sequences:
            raise ValueError("BPE tokenizer training requires a corpus")
        return train_bpe_tokenizer(sequences, int(settings.vocab_size), max_length)
    raise ValueError(f"Unknown tokenizer kind: {settings.kind!r} (use 'monomer' or 'bpe')")


def unknown_token_rate(tokenizer: PreTrainedTokenizerFast, sequences: list[str]) -> float:
    """Fraction of produced tokens that fall back to ``[UNK]``; a vocabulary sanity check."""
    unk_id = tokenizer.unk_token_id
    total = 0
    unknown = 0
    for encoding in tokenizer(list(sequences), add_special_tokens=False)["input_ids"]:
        total += len(encoding)
        unknown += sum(1 for token_id in encoding if token_id == unk_id)
    return unknown / total if total else 0.0

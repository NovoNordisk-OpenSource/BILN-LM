import numpy as np
import pytest
import transformers as hf

from biln_lm.embeddings import BilnEmbedder, embed_biln
from biln_lm.tokenizer import build_monomer_tokenizer

SEQUENCES = ["C-P-W-H-L-L-P-F-C", "ac-S-E-K-P", "L(1,1)-dL-dL-P-Y(1,2)"]
HIDDEN_SIZE = 32


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """A tiny randomly initialised model saved in the standard layout."""
    tokenizer = build_monomer_tokenizer(sequences=SEQUENCES, max_length=64)
    config = hf.EsmConfig(
        vocab_size=len(tokenizer),
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=66,
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        position_embedding_type="rotary",
    )
    model = hf.AutoModelForMaskedLM.from_config(config)
    path = tmp_path_factory.mktemp("model")
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return path


def test_shape_and_finiteness(model_dir):
    embeddings = embed_biln(SEQUENCES, model_dir)
    assert embeddings.shape == (len(SEQUENCES), HIDDEN_SIZE)
    assert np.isfinite(embeddings).all()


def test_deterministic(model_dir):
    assert np.array_equal(embed_biln(SEQUENCES, model_dir), embed_biln(SEQUENCES, model_dir))


def test_batch_size_does_not_change_result(model_dir):
    embedder = BilnEmbedder(model_dir)
    single = embedder.encode(SEQUENCES, batch_size=1)
    batched = embedder.encode(SEQUENCES, batch_size=len(SEQUENCES))
    assert np.allclose(single, batched, atol=1e-5)


@pytest.mark.parametrize("pooling", ["mean", "cls", "max", "sum"])
def test_pooling_modes(model_dir, pooling):
    embeddings = embed_biln(SEQUENCES, model_dir, pooling=pooling)
    assert embeddings.shape == (len(SEQUENCES), HIDDEN_SIZE)


def test_unknown_pooling_rejected(model_dir):
    with pytest.raises(ValueError):
        embed_biln(SEQUENCES, model_dir, pooling="median")


def test_empty_input(model_dir):
    assert embed_biln([], model_dir).shape == (0, HIDDEN_SIZE)


def test_cache_is_reused(model_dir, tmp_path):
    embedder = BilnEmbedder(model_dir, cache_dir=tmp_path / "cache")
    first = embedder.encode(SEQUENCES)
    assert list((tmp_path / "cache").glob("*.npy"))
    assert np.array_equal(first, embedder.encode(SEQUENCES))

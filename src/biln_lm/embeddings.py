"""Generate fixed-size embeddings for BILN peptides from a pretrained BILN-LM."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .model import load_pretrained, resolve_device

POOLINGS = ("mean", "cls", "max", "sum")


def _pool(hidden: torch.Tensor, mask: torch.Tensor, pooling: str) -> torch.Tensor:
    """Pool token states to one vector per sequence, ignoring padded positions."""
    if pooling == "cls":
        return hidden[:, 0]
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    if pooling == "max":
        return hidden.masked_fill(mask == 0, torch.finfo(hidden.dtype).min).max(dim=1).values
    summed = (hidden * mask).sum(dim=1)
    if pooling == "sum":
        return summed
    return summed / mask.sum(dim=1).clamp(min=1)


class BilnEmbedder:
    """Reusable encoder wrapper; load once, embed many batches."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "auto",
        max_length: int | None = None,
        cache_dir: str | Path | None = None,
    ):
        self.model_dir = str(model_dir)
        self.device = resolve_device(device)
        self.model, self.tokenizer = load_pretrained(model_dir, encoder_only=True)
        self.model.eval().to(self.device)
        self.max_length = max_length or int(self.tokenizer.model_max_length)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def embedding_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def _cache_path(self, sequences: list[str], pooling: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha1(
            "\n".join([self.model_dir, pooling, *sequences]).encode()
        ).hexdigest()
        return self.cache_dir / f"{digest}.npy"

    @torch.inference_mode()
    def encode(
        self,
        sequences: list[str],
        pooling: str = "mean",
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        if pooling not in POOLINGS:
            raise ValueError(f"Unknown pooling {pooling!r}, expected one of {POOLINGS}")
        sequences = [str(s) for s in sequences]
        if not sequences:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        cache_path = self._cache_path(sequences, pooling)
        if cache_path is not None and cache_path.is_file():
            return np.load(cache_path)

        vectors = []
        batches = range(0, len(sequences), batch_size)
        for start in tqdm(batches, disable=not show_progress, desc="embedding"):
            batch = sequences[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                padding="longest",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            inputs.pop("token_type_ids", None)
            hidden = self.model(**inputs).last_hidden_state
            pooled = _pool(hidden, inputs["attention_mask"], pooling)
            vectors.append(pooled.float().cpu().numpy())

        embeddings = np.concatenate(vectors, axis=0)
        if cache_path is not None:
            np.save(cache_path, embeddings)
        return embeddings


def embed_biln(
    sequences: list[str],
    model_dir: str | Path,
    pooling: str = "mean",
    batch_size: int = 32,
    device: str = "auto",
    max_length: int | None = None,
    cache_dir: str | Path | None = None,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed BILN strings with a pretrained model.

    Returns an array of shape ``(len(sequences), hidden_size)``.
    """
    embedder = BilnEmbedder(model_dir, device=device, max_length=max_length, cache_dir=cache_dir)
    return embedder.encode(sequences, pooling=pooling, batch_size=batch_size, show_progress=show_progress)

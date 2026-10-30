"""ESM-encoder masked language model for BILN sequences."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForMaskedLM, EsmConfig, PreTrainedTokenizerFast

from .tokenizer import load_tokenizer

ARCHITECTURE_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "intermediate_size",
    "position_embedding_type",
    "hidden_dropout_prob",
    "attention_probs_dropout_prob",
)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_config(config, tokenizer: PreTrainedTokenizerFast) -> EsmConfig:
    architecture = {k: config.model[k] for k in ARCHITECTURE_KEYS if k in config.model}
    return EsmConfig(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        max_position_embeddings=int(config.data.max_length) + 2,
        **architecture,
    )


def build_model(config, tokenizer: PreTrainedTokenizerFast):
    """Create a fresh MLM, or load ``config.model.pretrained_path`` to continue pretraining."""
    pretrained = config.model.get("pretrained_path")
    if pretrained:
        model = AutoModelForMaskedLM.from_pretrained(str(pretrained))
        if model.config.vocab_size != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
        return model
    return AutoModelForMaskedLM.from_config(build_config(config, tokenizer))


def load_pretrained(
    model_dir: str | Path, encoder_only: bool = True
) -> tuple[torch.nn.Module, PreTrainedTokenizerFast]:
    """Load a model saved by :func:`biln_lm.pretrain.run_pretraining`."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    loader = AutoModel if encoder_only else AutoModelForMaskedLM
    model = loader.from_pretrained(str(model_dir))
    return model, load_tokenizer(model_dir)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def format_parameter_count(model: torch.nn.Module) -> str:
    n_params = count_parameters(model)
    if n_params < 1e9:
        return f"{n_params / 1e6:.1f}M"
    return f"{n_params / 1e9:.2f}B"

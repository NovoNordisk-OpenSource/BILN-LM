"""Masked language model pretraining on a BILN corpus."""

from __future__ import annotations

import inspect
import json
import logging
import math
import shutil
from pathlib import Path

import pandas as pd
import transformers as hf
from omegaconf import DictConfig, OmegaConf

from .config import save_config
from .data import build_collator, build_datasets, load_corpus
from .model import build_model, format_parameter_count, resolve_device
from .tokenizer import build_tokenizer, unknown_token_rate

logger = logging.getLogger(__name__)

# Names we accept in configs that differ between transformers versions.
_STRATEGY_ALIASES = {"eval_strategy": "evaluation_strategy"}


def _training_arguments(settings: dict, output_dir: Path, seed: int) -> hf.TrainingArguments:
    """Build TrainingArguments, dropping keys the installed transformers does not accept."""
    supported = set(inspect.signature(hf.TrainingArguments.__init__).parameters)
    kwargs = {"output_dir": str(output_dir), "seed": seed}
    for key, value in settings.items():
        if key in supported:
            kwargs[key] = value
        elif _STRATEGY_ALIASES.get(key) in supported:
            kwargs[_STRATEGY_ALIASES[key]] = value
        else:
            logger.debug("Ignoring unsupported TrainingArguments key %r", key)
    return hf.TrainingArguments(**kwargs)


def run_pretraining(config: DictConfig) -> dict:
    """Train (or continue training) a BILN masked language model.

    Returns a summary dict with the output directory, best eval loss and model size.
    The output directory is a standard ``save_pretrained`` folder, so the model can be
    reloaded with :func:`biln_lm.model.load_pretrained` or plain ``AutoModel``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir = Path(config.paths.output_dir)
    if output_dir.exists() and config.training.get("overwrite_output_dir"):
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hf.set_seed(int(config.seed))

    sequences = load_corpus(
        files=list(config.data.files),
        data_dir=config.paths.data_dir,
        column=config.data.column,
        max_samples=config.data.get("max_samples"),
        seed=int(config.seed),
    )
    logger.info("Corpus: %d unique BILN sequences", len(sequences))

    tokenizer = build_tokenizer(config, sequences)
    unk_rate = unknown_token_rate(tokenizer, sequences[:5000])
    logger.info("Tokenizer: %s, vocab=%d, unk rate=%.4f", config.tokenizer.kind, len(tokenizer), unk_rate)

    datasets = build_datasets(
        sequences,
        tokenizer,
        max_length=int(config.data.max_length),
        valid_fraction=float(config.data.valid_fraction),
        seed=int(config.seed),
    )

    model = build_model(config, tokenizer)
    device = resolve_device(config.hardware.device)
    logger.info("Model: %s parameters on %s", format_parameter_count(model), device)

    training_settings = OmegaConf.to_container(config.training, resolve=True)
    training_settings.pop("overwrite_output_dir", None)
    resume = bool(training_settings.pop("resume_from_checkpoint", False))
    patience = int(training_settings.pop("early_stopping_patience", 0))
    mlm_probability = float(training_settings.pop("mlm_probability", 0.15))
    training_settings.setdefault("eval_strategy", training_settings.get("logging_strategy", "epoch"))
    training_settings.setdefault("save_strategy", training_settings.get("logging_strategy", "epoch"))
    training_settings.setdefault("load_best_model_at_end", True)
    training_settings.setdefault("metric_for_best_model", "eval_loss")
    training_settings.setdefault("greater_is_better", False)
    training_settings["use_cpu"] = device.type == "cpu"

    callbacks = []
    if patience > 0:
        callbacks.append(hf.EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = hf.Trainer(
        model=model,
        args=_training_arguments(training_settings, output_dir / "checkpoints", int(config.seed)),
        data_collator=build_collator(tokenizer, mlm_probability),
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train(resume_from_checkpoint=resume or None)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    save_config(config, output_dir / "config.yaml")

    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(output_dir / "training_history.csv", index=False)
    best_loss = float(history["eval_loss"].min()) if "eval_loss" in history else float("nan")

    summary = {
        "output_dir": str(output_dir),
        "best_eval_loss": best_loss,
        "perplexity": math.exp(best_loss) if math.isfinite(best_loss) else float("nan"),
        "vocab_size": len(tokenizer),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "n_sequences": len(sequences),
        "unk_rate": unk_rate,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Done. Best eval loss %.4f -> %s", best_loss, output_dir)
    return summary

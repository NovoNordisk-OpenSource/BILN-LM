"""Optional Optuna search over the pretraining config.

The search space is declared in the config as ``hpo.search_space``: a list of
sampler specs, each naming a dotted config key, so no code changes are needed to
tune a different parameter. Every trial starts from the config as it was given on
the command line, with the sampled values merged on top.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from omegaconf import OmegaConf

from .pretrain import run_pretraining

logger = logging.getLogger(__name__)


def _suggest(trial, spec: dict):
    key, kind = spec["key"], spec["type"]
    if kind == "int":
        return trial.suggest_int(key, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False)))
    if kind == "float":
        return trial.suggest_float(key, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False)))
    if kind == "categorical":
        return trial.suggest_categorical(key, list(spec["choices"]))
    raise ValueError(f"Unknown sampler type {kind!r} for {key!r}")


def run_search(config) -> dict:
    try:
        import optuna
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("optuna is required: pip install -r requirements-optional.txt") from error

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    search_space = OmegaConf.to_container(config.hpo.search_space, resolve=True)
    root_dir = Path(config.paths.output_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial) -> float:
        overrides = [f"{spec['key']}={_suggest(trial, spec)}" for spec in search_space]
        overrides.append(f"paths.output_dir={root_dir / f'trial_{trial.number:03d}'}")
        trial_config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
        trial_config.pop("hpo", None)
        # num_attention_heads must divide hidden_size; skip invalid draws cheaply.
        if trial_config.model.hidden_size % trial_config.model.num_attention_heads:
            raise optuna.TrialPruned()
        summary = run_pretraining(trial_config)
        trial.set_user_attr("summary", summary)
        return summary["best_eval_loss"]

    study = optuna.create_study(direction="minimize", study_name=config.hpo.study_name)
    study.optimize(objective, n_trials=int(config.hpo.n_trials), gc_after_trial=True)

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_summary": study.best_trial.user_attrs.get("summary", {}),
    }
    (root_dir / "best_trial.json").write_text(json.dumps(best, indent=2))
    study.trials_dataframe().to_csv(root_dir / "trials.csv", index=False)
    logger.info("Best eval loss %.4f with %s", study.best_value, study.best_params)
    return best

"""Simplified downstream benchmark: BILN-LM embeddings + a scikit-learn head."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR

from .biln import is_valid_biln, normalize_biln
from .embeddings import BilnEmbedder
from .metrics import evaluate

logger = logging.getLogger(__name__)

ESTIMATORS = {
    "svm": (SVR, SVC),
    "rf": (RandomForestRegressor, RandomForestClassifier),
    "gb": (GradientBoostingRegressor, GradientBoostingClassifier),
    "knn": (KNeighborsRegressor, KNeighborsClassifier),
}


def build_estimator(algorithm: str, task_type: str, params: dict | None = None, seed: int = 1):
    """One place where every estimator is constructed, for both task types."""
    if algorithm not in ESTIMATORS:
        raise ValueError(f"Unknown algorithm {algorithm!r}, expected one of {sorted(ESTIMATORS)}")
    regressor, classifier = ESTIMATORS[algorithm]
    estimator_cls = regressor if task_type == "regression" else classifier
    params = dict(params or {})
    if algorithm in {"rf", "gb"}:
        params.setdefault("random_state", seed)
    if algorithm == "svm":
        params.setdefault("C", 1.0)
        if task_type == "classification":
            params.setdefault("probability", True)
    return estimator_cls(**params)


def encode_proteins(sequences: list[str], model_name: str, device: str, batch_size: int = 8) -> np.ndarray:
    """Mean-pooled protein embeddings for the multi-instance binding tasks."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    unique = list(dict.fromkeys(sequences))
    lookup: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            batch = unique[start : start + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = ((hidden * mask).sum(1) / mask.sum(1)).float().cpu().numpy()
            lookup.update(dict(zip(batch, pooled)))
    return np.stack([lookup[s] for s in sequences])


def featurize(frame: pd.DataFrame, task: dict, embedder: BilnEmbedder, config) -> np.ndarray:
    sequences = frame[task["biln_column"]].astype(str).tolist()
    if config.downstream.get("normalize_biln", True):
        sequences = [normalize_biln(s) for s in sequences]
    features = embedder.encode(
        sequences,
        pooling=config.downstream.pooling,
        batch_size=int(config.downstream.batch_size),
        show_progress=True,
    )
    protein_column = task.get("protein_column")
    if protein_column:
        proteins = encode_proteins(
            frame[protein_column].astype(str).tolist(),
            config.downstream.protein_model,
            str(embedder.device),
        )
        features = np.concatenate([features, proteins], axis=1)
    return features


def run_task(task: dict, config, embedder: BilnEmbedder) -> list[dict]:
    path = Path(config.paths.data_dir) / task["file"]
    frame = pd.read_csv(path).dropna(subset=[task["biln_column"], task["label_column"]])
    usable = frame[task["biln_column"]].astype(str).map(is_valid_biln)
    if not usable.all():
        logger.warning("%s: dropping %d rows with unparseable BILN", task["name"], (~usable).sum())
        frame = frame[usable]
    labels = frame[task["label_column"]].to_numpy()
    if task["type"] == "classification":
        labels = labels.astype(int)

    features = featurize(frame, task, embedder, config)
    seed = int(config.seed)
    stratify = labels if task["type"] == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=stratify
    )

    results = []
    for algorithm in config.downstream.algorithms:
        estimator = build_estimator(algorithm, task["type"], seed=seed)
        estimator.fit(x_train, y_train)
        predictions = estimator.predict(x_test)
        scores = None
        if task["type"] == "classification" and hasattr(estimator, "predict_proba"):
            scores = estimator.predict_proba(x_test)[:, 1]
        results.append(
            {
                "task": task["name"],
                "type": task["type"],
                "algorithm": algorithm,
                "n_train": len(y_train),
                "n_test": len(y_test),
                **evaluate(task["type"], y_test, predictions, scores),
            }
        )
        logger.info("%s / %s: %s", task["name"], algorithm, results[-1])
    return results


def run_benchmark(config) -> pd.DataFrame:
    """Evaluate the pretrained embeddings on every task listed in the config."""
    embedder = BilnEmbedder(
        config.downstream.model_dir,
        device=config.hardware.device,
        cache_dir=config.downstream.get("cache_dir"),
    )
    selected = set(config.downstream.get("tasks") or [])
    rows: list[dict] = []
    for task in config.downstream.task_definitions:
        task = dict(task)
        if selected and task["name"] not in selected:
            continue
        rows.extend(run_task(task, config, embedder))

    results = pd.DataFrame(rows)
    out_dir = Path(config.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "results.csv", index=False)
    return results

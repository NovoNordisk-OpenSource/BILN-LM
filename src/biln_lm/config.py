"""OmegaConf-based configuration loading with ``key=value`` CLI overrides.

A config may declare ``_base_: <file>`` to inherit from another config; the chain
is merged root-first, then ``configs/default.yaml`` underneath everything.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_CONFIG = "default.yaml"
BASE_KEY = "_base_"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    candidate = CONFIG_DIR / path.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Config file not found: {path}")


def _load_with_bases(path: str | Path, seen: set[Path] | None = None) -> DictConfig:
    resolved = _resolve(path).resolve()
    seen = set() if seen is None else seen
    if resolved in seen:
        raise ValueError(f"Circular {BASE_KEY} reference at {resolved}")
    seen.add(resolved)
    config = OmegaConf.load(resolved)
    base = config.pop(BASE_KEY, None)
    if base is None:
        return config  # type: ignore[return-value]
    return OmegaConf.merge(_load_with_bases(base, seen), config)  # type: ignore[return-value]


def load_config(
    config_path: str | Path | None = None, overrides: list[str] | None = None
) -> DictConfig:
    """Merge ``configs/default.yaml``, the task config (with its bases) and CLI overrides."""
    config = OmegaConf.load(_resolve(DEFAULT_CONFIG))
    if config_path is not None:
        config = OmegaConf.merge(config, _load_with_bases(config_path))
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    return config  # type: ignore[return-value]


def save_config(config: DictConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(config))


def describe(config: DictConfig) -> str:
    return OmegaConf.to_yaml(config)

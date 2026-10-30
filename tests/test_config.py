import pytest
from omegaconf import OmegaConf

from biln_lm.config import load_config


def test_defaults_load():
    config = load_config()
    assert config.tokenizer.kind == "monomer"
    assert config.data.column == "biln"


def test_task_config_merges_over_defaults():
    config = load_config("pretrain.yaml")
    assert config.model.num_hidden_layers == 6
    assert config.data.column == "biln"


def test_base_inheritance():
    config = load_config("pretrain_bpe.yaml")
    assert config.tokenizer.kind == "bpe"
    # inherited from pretrain.yaml through _base_
    assert config.model.num_hidden_layers == 6
    assert "_base_" not in config


def test_cli_overrides_win():
    config = load_config("pretrain.yaml", ["training.num_train_epochs=1", "model.hidden_size=64"])
    assert config.training.num_train_epochs == 1
    assert config.model.hidden_size == 64


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("does_not_exist.yaml")


def test_config_is_serialisable():
    assert OmegaConf.to_yaml(load_config("pretrain.yaml"))

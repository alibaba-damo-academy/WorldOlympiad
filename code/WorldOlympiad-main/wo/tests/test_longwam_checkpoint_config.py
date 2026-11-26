from pathlib import Path
import json
import pytest
from omegaconf import OmegaConf
from wo.models.longwam_server import inference_model_config


def test_missing_training_field_does_not_change_released_evaluation_or_architecture(tmp_path):
    config = {"model": {"_target_": "fastwam.runtime.create_arwam", "arwam_past_obs_size": 48},
              "data": {"train": {"processor": {"_target_": "original.Processor", "shape_meta": {"action": 14},
                       "val_transforms": [{"resize": [240, 320]}]}}}}
    original = tmp_path / "released.yaml"
    OmegaConf.save(OmegaConf.create(config), original)
    original_bytes = original.read_bytes()
    effective = inference_model_config(original, tmp_path)
    actual = OmegaConf.to_container(OmegaConf.load(effective))
    assert actual["data"]["train"]["processor"].pop("train_transforms") == []
    assert actual == config
    assert original.read_bytes() == original_bytes
    assert json.loads((tmp_path / "config_compatibility.json").read_text())["added"] == {"data.train.processor.train_transforms": []}


def test_complete_checkpoint_is_used_without_an_overlay(tmp_path):
    original = tmp_path / "released.yaml"
    OmegaConf.save(OmegaConf.create({"data": {"train": {"processor": {"train_transforms": ["original"], "val_transforms": []}}}}), original)
    assert inference_model_config(original, tmp_path) == original
    assert not (tmp_path / "inference_model_config.yaml").exists()

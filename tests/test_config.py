import json
from pathlib import Path
import tempfile
import pytest

from config import UnlockConfig


def test_unlock_config_defaults():
    cfg = UnlockConfig()
    assert cfg.camera == 0
    assert cfg.ear == 0.21
    assert cfg.blinks == 1
    assert cfg.sound is True
    assert cfg.tolerance == 0.50


def test_unlock_config_save_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        cfg = UnlockConfig(camera=2, ear=0.19, blinks=3, sound=False)
        cfg.save(config_path)

        assert config_path.exists()
        loaded = UnlockConfig.load(config_path)
        assert loaded.camera == 2
        assert loaded.ear == 0.19
        assert loaded.blinks == 3
        assert loaded.sound is False


def test_unlock_config_ignores_unknown():
    data = {"camera": 1, "unknown_field": "something"}
    cfg = UnlockConfig.from_dict(data)
    assert cfg.camera == 1
    assert not hasattr(cfg, "unknown_field")

"""Configuration management for Face-Eye-blink-unlock-demo.

Allows settings to be stored in and loaded from JSON configuration files,
or configured via command-line arguments.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class UnlockConfig:
    camera: int = 0
    scale: float = 0.5
    ear: float = 0.21
    consec: int = 2
    blinks: int = 1
    tolerance: float = 0.50
    exit_on_unlock: bool = False
    use_mediapipe: bool = False
    log: Optional[str] = None
    save_unlock_shots: Optional[str] = None
    headless: bool = False
    sound: bool = True
    challenge: bool = False
    challenge_min: int = 1
    challenge_max: int = 3
    challenge_timeout: float = 8.0
    head_pose: bool = False

    # Hardware & Physical Biometrics Settings
    gpio: bool = False
    gpio_relay: int = 18
    gpio_led_green: Optional[int] = 23
    gpio_led_red: Optional[int] = 24
    gpio_led_blue: Optional[int] = 22
    gpio_buzzer: Optional[int] = 25
    gpio_button: Optional[int] = 17
    unlock_duration: float = 3.0
    serial_port: Optional[str] = None
    serial_baud: int = 115200
    webhook_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UnlockConfig:
        valid_fields = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def load(cls, path: str | Path) -> UnlockConfig:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

import csv
import json
from pathlib import Path
import tempfile
import pytest

from utils import sanitize_name, log_unlock_event, play_sound


def test_sanitize_name():
    assert sanitize_name("Alice") == "Alice"
    assert sanitize_name("John Doe") == "John_Doe"
    assert sanitize_name("   Bob   Smith   ") == "Bob_Smith"
    assert sanitize_name("Alice/../../etc/passwd") == "Aliceetcpasswd"
    assert sanitize_name('Bad:Name*?"<>|') == "BadName"
    assert sanitize_name("") == "unnamed"
    assert sanitize_name("   ") == "unnamed"
    assert sanitize_name(None) == "unnamed"


def test_log_unlock_event_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "unlocks.csv"
        log_unlock_event(csv_file, "Alice", method="blink", details={"ear": 0.18})
        log_unlock_event(csv_file, "Bob", method="challenge", details={"target": 2})

        assert csv_file.exists()
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))

        # Header + 2 rows
        assert len(reader) == 3
        assert reader[0] == ["timestamp", "name", "method", "details"]
        assert reader[1][1] == "Alice"
        assert reader[1][2] == "blink"
        assert "0.18" in reader[1][3]
        assert reader[2][1] == "Bob"
        assert reader[2][2] == "challenge"


def test_log_unlock_event_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "unlocks.json"
        log_unlock_event(json_file, "Alice", method="blink")
        log_unlock_event(json_file, "Charlie", method="challenge", details={"target": 3})

        assert json_file.exists()
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["name"] == "Alice"
        assert data[1]["name"] == "Charlie"
        assert data[1]["details"]["target"] == 3


def test_play_sound_graceful():
    # Should not raise any exception whether sound is enabled or disabled
    play_sound("unlock", enabled=False)
    play_sound("blink", enabled=False)
    play_sound("fail", enabled=False)

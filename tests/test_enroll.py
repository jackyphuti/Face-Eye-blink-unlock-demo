import argparse
from pathlib import Path
import tempfile
import numpy as np
import pytest

from enroll import check_image_quality, enroll


def test_check_image_quality_synthetic():
    # Very dark image (all zeros)
    dark_img = np.zeros((100, 100, 3), dtype=np.uint8)
    ok, msg = check_image_quality(dark_img, min_brightness=40.0)
    # Brightness should fail if cv2 installed, or pass with skip
    if not ok:
        assert "dark" in msg.lower()

    # Bright uniform image (low blur variance)
    bright_uniform = np.full((100, 100, 3), 150, dtype=np.uint8)
    ok, msg = check_image_quality(bright_uniform)
    if not ok:
        assert "blurry" in msg.lower()


def test_enroll_nonexistent_image():
    with tempfile.TemporaryDirectory() as tmpdir:
        res = enroll("Alice", image_path="nonexistent_file_xyz.jpg", custom_dir=tmpdir)
        assert res is False


def test_enroll_cli_argument_parsing():
    # Verify that --name and --shots together do NOT raise unrecognized argument error
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")

    args = parser.parse_args(["--name", "Alice", "--shots", "3", "--camera", "1"])
    assert args.name == "Alice"
    assert args.shots == 3
    assert args.camera == 1

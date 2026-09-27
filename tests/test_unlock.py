from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from unlock import (
    load_known_faces,
    match_face,
    find_matching_mesh,
    build_arg_parser,
)


def test_load_known_faces():
    with tempfile.TemporaryDirectory() as tmpdir:
        k_dir = Path(tmpdir)
        np.save(k_dir / "Alice.npy", np.zeros(128))
        np.save(k_dir / "Bob.npy", np.ones(128))

        names, encs = load_known_faces(k_dir)
        assert len(names) == 2
        assert "Alice" in names
        assert "Bob" in names
        assert len(encs) == 2


def test_match_face_picks_closest():
    known_encs = [np.array([0.0, 0.0]), np.array([1.0, 1.0])]
    names = ["Alice", "Bob"]
    target = np.array([0.95, 0.95])  # Closer to Bob

    with patch("unlock.face_recognition") as mock_fr:
        # Distance to Alice is 1.34, to Bob is 0.07
        mock_fr.face_distance.return_value = np.array([1.34, 0.07])

        name, conf = match_face(known_encs, names, target, tolerance=0.50)
        assert name == "Bob"
        assert conf > 0.8


def test_match_face_unknown_when_distance_exceeds_tolerance():
    known_encs = [np.array([0.0, 0.0])]
    names = ["Alice"]
    target = np.array([5.0, 5.0])

    with patch("unlock.face_recognition") as mock_fr:
        mock_fr.face_distance.return_value = np.array([7.07])

        name, conf = match_face(known_encs, names, target, tolerance=0.50)
        assert name == "Unknown"
        assert conf == 0.0


def test_find_matching_mesh():
    # Mock Mediapipe landmark
    mesh_1 = MagicMock()
    nose_1 = MagicMock()
    nose_1.x = 0.5
    nose_1.y = 0.5
    mesh_1.landmark = {1: nose_1}

    mesh_2 = MagicMock()
    nose_2 = MagicMock()
    nose_2.x = 0.1
    nose_2.y = 0.1
    mesh_2.landmark = {1: nose_2}

    # Bounding box centered near (500, 500) in a (1000, 1000) frame
    # box = (top, right, bottom, left)
    box = (400, 600, 600, 400)
    matched = find_matching_mesh([mesh_1, mesh_2], box, 1000, 1000)
    assert matched == mesh_1


def test_unlock_arg_parser():
    parser = build_arg_parser()
    args = parser.parse_args([
        "--camera", "2",
        "--ear", "0.19",
        "--blinks", "2",
        "--tolerance", "0.45",
        "--headless",
        "--no-sound",
        "--challenge",
        "--challenge-min", "2",
        "--challenge-max", "4",
        "--challenge-timeout", "10.0"
    ])
    assert args.camera == 2
    assert args.ear == 0.19
    assert args.blinks == 2
    assert args.tolerance == 0.45
    assert args.headless is True
    assert args.sound is False
    assert args.challenge is True
    assert args.challenge_min == 2
    assert args.challenge_max == 4
    assert args.challenge_timeout == 10.0

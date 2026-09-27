import json
from pathlib import Path
import tempfile
import numpy as np
import pytest

from manage_faces import (
    list_faces,
    info_face,
    remove_face,
    rename_face,
    clear_faces,
    export_faces,
    import_faces,
)


@pytest.fixture
def temp_known_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        k_dir = Path(tmpdir)
        # Create sample faces
        np.save(k_dir / "Alice.npy", np.zeros(128))
        (k_dir / "Alice.jpg").write_text("fake_jpg_alice")

        np.save(k_dir / "Bob.npy", np.ones(128))
        (k_dir / "Bob.jpg").write_text("fake_jpg_bob")

        yield k_dir


def test_list_faces(temp_known_dir):
    faces = list_faces(temp_known_dir)
    assert "Alice" in faces
    assert "Bob" in faces


def test_info_face(temp_known_dir):
    assert info_face("Alice", temp_known_dir) is True
    assert info_face("NonExistent", temp_known_dir) is False


def test_remove_face(temp_known_dir):
    assert remove_face("Alice", temp_known_dir) is True
    assert not (temp_known_dir / "Alice.npy").exists()
    assert not (temp_known_dir / "Alice.jpg").exists()

    # Removing again should fail
    assert remove_face("Alice", temp_known_dir) is False


def test_rename_face(temp_known_dir):
    assert rename_face("Alice", "Alicia", temp_known_dir) is True
    assert not (temp_known_dir / "Alice.npy").exists()
    assert (temp_known_dir / "Alicia.npy").exists()
    assert (temp_known_dir / "Alicia.jpg").exists()

    # Renaming to existing should fail
    assert rename_face("Alicia", "Bob", temp_known_dir) is False


def test_clear_faces(temp_known_dir):
    assert clear_faces(temp_known_dir, force=True) is True
    assert len(list(temp_known_dir.glob("*"))) == 0


def test_export_and_import(temp_known_dir):
    with tempfile.TemporaryDirectory() as out_tmp:
        zip_path = Path(out_tmp) / "backup.zip"
        assert export_faces(str(zip_path), temp_known_dir) is True
        assert zip_path.exists()

        # Import into fresh directory
        target_dir = Path(out_tmp) / "imported"
        assert import_faces(str(zip_path), target_dir) is True
        assert (target_dir / "Alice.npy").exists()
        assert (target_dir / "Bob.npy").exists()

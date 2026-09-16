from pathlib import Path

import pytest

from app.stt.whisper import prepare_model_source


def test_bundled_model_does_not_create_download_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "whisper-tiny-en"
    model_path.mkdir()
    download_root = tmp_path / "read-only-deployment" / "whisper"

    source = prepare_model_source(str(model_path), download_root, local_files_only=True)

    assert source == str(model_path)
    assert not download_root.exists()


def test_missing_local_model_fails_without_creating_download_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "missing-model"
    download_root = tmp_path / "read-only-deployment" / "whisper"

    with pytest.raises(FileNotFoundError, match="scripts/download_models.py"):
        prepare_model_source(str(model_path), download_root, local_files_only=True)

    assert not download_root.exists()


def test_remote_model_creates_download_directory(tmp_path: Path) -> None:
    download_root = tmp_path / "cache" / "whisper"

    source = prepare_model_source("tiny.en", download_root, local_files_only=False)

    assert source == "tiny.en"
    assert download_root.is_dir()

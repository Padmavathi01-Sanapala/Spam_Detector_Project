import os
import pathlib

import pytest

from src.train import ensure_data_directory


def test_ensure_data_directory_replaces_stale_file(tmp_path):
    stale_data = tmp_path / 'data'
    stale_data.write_text('not a directory')

    result = ensure_data_directory(str(tmp_path))

    assert result == tmp_path / 'data'
    assert result.is_dir()
    assert stale_data.exists()
    assert stale_data.is_dir()


def test_ensure_data_directory_keeps_existing_directory(tmp_path):
    target_dir = tmp_path / 'data'
    target_dir.mkdir()

    result = ensure_data_directory(str(tmp_path))

    assert result == target_dir
    assert result.is_dir()

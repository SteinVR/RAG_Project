"""Shared pytest fixtures for the Modular RAG test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import AppConfig, ConfigManager


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    """Return a deep-copied AppConfig whose paths point into a temp dir."""
    settings = ConfigManager.load().settings.model_copy(deep=True)
    settings.paths.docs_dir = tmp_path / "docs"
    settings.paths.vector_store_dir = tmp_path / "vector_store"
    settings.paths.file_registry = tmp_path / "registry" / "file_registry.json"
    settings.logging.file = tmp_path / "logs" / "test.log"
    settings.logging.console = False
    settings.vector_store.persist_directory = tmp_path / "vector_store"
    settings.materialize()
    return settings



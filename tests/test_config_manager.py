"""Tests for the configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from src.core.config import ConfigManager


def _write_config_file(
    config_path: Path,
    docs_dir: Path,
    vector_dir: Path,
    registry_path: Path,
    log_file: Path,
    *,
    level: str = "DEBUG",
) -> None:
    """Create a fully-populated settings file pointing at a temp directory."""
    config_template = f"""
app:
  name: "Test Console"
paths:
  docs_dir: "{docs_dir}"
  vector_store_dir: "{vector_dir}"
  file_registry: "{registry_path}"
logging:
  level: "{level}"
  format: "[%(levelname)s] %(message)s"
  file: "{log_file}"
  console: false
ingestion:
  chunk_size: 1000
  chunk_overlap: 10
  retriever_k: 5
  glob_patterns:
    - "*.txt"
vector_store:
  collection_name: "test"
  recreate: false
  embedding_batch_size: 16
embeddings:
  model_name: "test-model"
  device: "cpu"
  normalize_embeddings: false
llm:
  provider: "google"
  model_name: "models/test"
  temperature: 0.3
  max_output_tokens: 512
  api_key_env: "TEST_API_KEY"
modules:
  rewriter: true
  hyde: true
  reranker: true
rewriter:
  enabled: true
  prompt_template: "Rewrite the query"
hyde:
  enabled: true
  prompt_template: "Hyde prompt"
  max_tokens: 128
generator:
  max_context_documents: 2
  answer_prompt: "Answer prompt"
  citations_required: true
reranker:
  enabled: true
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k: 3
"""
    config_path.write_text(textwrap.dedent(config_template).strip(), encoding="utf-8")


def test_loads_custom_config_and_materializes_paths(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "settings.yaml"
    docs_dir = tmp_path / "docs"
    vector_dir = tmp_path / "vector_store"
    registry_path = tmp_path / "registry" / "file_registry.json"
    log_file = tmp_path / "logs" / "app.log"

    _write_config_file(config_path, docs_dir, vector_dir, registry_path, log_file)
    monkeypatch.setattr(ConfigManager, "_instance", None, raising=False)

    manager = ConfigManager.load(config_path)
    settings = manager.settings

    assert settings.paths.docs_dir == docs_dir
    assert settings.paths.vector_store_dir == vector_dir
    assert settings.paths.file_registry == registry_path
    assert settings.vector_store.persist_directory == vector_dir
    assert docs_dir.exists()
    assert vector_dir.exists()
    assert registry_path.parent.exists()
    assert log_file.exists()
    assert settings.ingestion.chunk_size == 1000

    _write_config_file(
        config_path,
        docs_dir,
        vector_dir,
        registry_path,
        log_file,
        level="WARNING",
    )
    manager.reload()
    assert manager.settings.logging.level == "WARNING"


def test_load_rejects_non_mapping_yaml(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("- not-a-mapping\n", encoding="utf-8")
    monkeypatch.setattr(ConfigManager, "_instance", None, raising=False)

    with pytest.raises(ValueError, match="Top-level YAML structure"):
        ConfigManager.load(config_path)


def test_load_raises_for_missing_file(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "missing.yaml"
    monkeypatch.setattr(ConfigManager, "_instance", None, raising=False)

    with pytest.raises(FileNotFoundError):
        ConfigManager.load(config_path)


"""Tests covering the VectorStoreManager incremental sync logic."""

from __future__ import annotations

import hashlib
import json

import pytest
from langchain_core.documents import Document

from src.ingestion.loader import DocumentLoader
from src.ingestion.vector_store import VectorStoreManager


class StubRetriever:
    """Minimal retriever stub used by the fake Chroma client."""

    def __init__(self) -> None:
        self.invocations: list[str] = []

    def invoke(self, query: str):
        self.invocations.append(query)
        return []


class StubChroma:
    """Track interactions performed by the vector store manager."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - signature matches real class
        self.init_args = args
        self.init_kwargs = kwargs
        self.added_documents: list[Document] = []
        self.delete_calls: list[dict[str, str]] = []
        self.persist_calls = 0
        self.retriever_kwargs: dict[str, int] | None = None
        self._retriever = StubRetriever()

    def add_documents(self, documents):
        self.added_documents.extend(documents)

    def delete(self, where):
        self.delete_calls.append(dict(where))

    def persist(self):
        self.persist_calls += 1

    def as_retriever(self, search_kwargs):
        self.retriever_kwargs = search_kwargs
        return self._retriever


@pytest.fixture(autouse=True)
def stub_vector_dependencies(monkeypatch):
    """Patch expensive dependencies so tests can run quickly."""

    class StubEmbeddings:
        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - mimic API
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(
        "src.ingestion.vector_store.HuggingFaceEmbeddings",
        StubEmbeddings,
    )
    monkeypatch.setattr(
        "src.ingestion.vector_store.resolve_device",
        lambda *_args, **_kwargs: "cpu",
    )
    monkeypatch.setattr(
        "src.ingestion.vector_store.Chroma",
        StubChroma,
    )


def _write_doc(app_config, name: str, content: str = "sample text") -> None:
    docs_dir = app_config.paths.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / name
    path.write_text(content, encoding="utf-8")


def test_sync_adds_new_documents_and_updates_registry(app_config) -> None:
    _write_doc(app_config, "note.txt", "A short document.")
    loader = DocumentLoader(app_config)

    manager = VectorStoreManager(config=app_config, loader=loader)
    stats = manager.sync()

    assert stats == {"added_chunks": 1, "removed_sources": 0}
    vector_store = manager.get_vector_store()
    assert isinstance(vector_store, StubChroma)
    assert len(vector_store.added_documents) == 1

    chunk = vector_store.added_documents[0]
    assert chunk.metadata["file_path"] == "note.txt"
    expected_checksum = hashlib.sha256(
        (app_config.paths.docs_dir / "note.txt").read_bytes(),
    ).hexdigest()
    assert chunk.metadata["checksum"] == expected_checksum
    assert "chunk_id" in chunk.metadata
    assert vector_store.persist_calls == 1

    registry_data = json.loads(app_config.paths.file_registry.read_text(encoding="utf-8"))
    assert "note.txt" in registry_data


def test_sync_removes_missing_sources(app_config) -> None:
    _write_doc(app_config, "orphan.txt", "Original content.")
    loader = DocumentLoader(app_config)
    manager = VectorStoreManager(config=app_config, loader=loader)
    manager.sync()
    vector_store = manager.get_vector_store()
    initial_delete_calls = len(vector_store.delete_calls)

    (app_config.paths.docs_dir / "orphan.txt").unlink()
    stats = manager.sync()

    assert stats["added_chunks"] == 0
    assert stats["removed_sources"] == 1
    assert len(vector_store.delete_calls) == initial_delete_calls + 1
    assert vector_store.persist_calls == 2

    registry_data = json.loads(app_config.paths.file_registry.read_text(encoding="utf-8"))
    assert "orphan.txt" not in registry_data


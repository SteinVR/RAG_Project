"""Tests for the document ingestion helpers."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.loader import DocumentLoader


def test_iter_source_files_filters_supported_extensions(app_config) -> None:
    docs_dir = app_config.paths.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "a.txt").write_text("alpha", encoding="utf-8")
    (docs_dir / "b.pdf").write_text("fake pdf bytes", encoding="utf-8")
    nested = docs_dir / "nested"
    nested.mkdir()
    (nested / "c.md").write_text("## Heading", encoding="utf-8")
    (docs_dir / "ignore.jpg").write_text("image", encoding="utf-8")

    loader = DocumentLoader(app_config)
    names = [path.name for path in loader.iter_source_files()]
    assert names == ["a.txt", "b.pdf", "c.md"]


def test_load_file_returns_text_document_metadata(app_config) -> None:
    docs_dir = app_config.paths.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    text_file = docs_dir / "note.txt"
    text_file.write_text("Hello world\n", encoding="utf-8")

    loader = DocumentLoader(app_config)
    documents = loader.load_file(text_file)
    assert len(documents) == 1
    document = documents[0]
    assert document.page_content == "Hello world"
    assert document.metadata["source"] == "note.txt"
    assert document.metadata["page"] == 1
    assert document.metadata["doc_id"]


def test_load_file_skips_empty_text_file(app_config) -> None:
    docs_dir = app_config.paths.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    empty_file = docs_dir / "empty.txt"
    empty_file.write_text("\n\n", encoding="utf-8")

    loader = DocumentLoader(app_config)
    assert loader.load_file(empty_file) == []


def test_load_file_unsupported_extension_returns_empty(app_config) -> None:
    docs_dir = app_config.paths.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    unknown = docs_dir / "diagram.svg"
    unknown.write_text("<svg></svg>", encoding="utf-8")

    loader = DocumentLoader(app_config)
    assert loader.load_file(unknown) == []



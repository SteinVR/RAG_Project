"""Tests covering the FileRegistry's incremental indexing logic."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.vector_store import FileRegistry


def test_file_registry_detects_changes(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    file_path = docs_dir / "example.txt"
    file_path.write_text("hello", encoding="utf-8")

    registry = FileRegistry(registry_path)
    snapshot = registry.snapshot(file_path)
    rel_path = "example.txt"

    assert registry.has_changed(rel_path, snapshot)
    registry.update(rel_path, snapshot)
    assert not registry.has_changed(rel_path, snapshot)

    file_path.write_text("hello world", encoding="utf-8")
    new_snapshot = registry.snapshot(file_path)
    assert registry.has_changed(rel_path, new_snapshot)


def test_file_registry_removed_entries_and_persistence(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    file_a = docs_dir / "a.txt"
    file_b = docs_dir / "b.txt"
    file_a.write_text("a", encoding="utf-8")
    file_b.write_text("b", encoding="utf-8")

    registry = FileRegistry(registry_path)
    snap_a = registry.snapshot(file_a)
    snap_b = registry.snapshot(file_b)
    registry.update("a.txt", snap_a)
    registry.update("b.txt", snap_b)
    registry.save()

    removed = registry.removed_entries(["a.txt"])
    assert removed == ["b.txt"]

    # Reload from disk to ensure persistence
    registry_reloaded = FileRegistry(registry_path)
    assert not registry_reloaded.has_changed("a.txt", snap_a)
    assert registry_reloaded.has_changed("b.txt", snap_b)



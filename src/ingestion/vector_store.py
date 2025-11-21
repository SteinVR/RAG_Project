"""Vector store management with incremental indexing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.core.config import AppConfig, ConfigManager
from src.ingestion.loader import DocumentLoader
from src.utils.logger import get_logger
from src.utils.device import resolve_device


class FileRegistry:
    """Persist metadata about the files that have already been indexed."""

    def __init__(self, path: Path) -> None:
        """Initialize the registry and read existing state if available."""
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict[str, str | int]] = self._read()

    def snapshot(self, file_path: Path) -> Dict[str, str | int]:
        """Return the current metadata snapshot for a file."""
        stat = file_path.stat()
        return {
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
            "checksum": self._checksum(file_path),
        }

    def has_changed(self, rel_path: str, snapshot: Dict[str, str | int]) -> bool:
        """Check whether the file is new or modified since last indexing."""
        stored = self._data.get(rel_path)
        return stored != snapshot

    def update(self, rel_path: str, snapshot: Dict[str, str | int]) -> None:
        """Store the latest snapshot for a file."""
        self._data[rel_path] = snapshot

    def remove(self, rel_path: str) -> None:
        """Remove a file entry from the registry."""
        self._data.pop(rel_path, None)

    def removed_entries(self, active_paths: Iterable[str]) -> List[str]:
        """Return registry entries that point to missing files."""
        active = set(active_paths)
        removed = [key for key in list(self._data.keys()) if key not in active]
        if removed:
            for key in removed:
                self._data.pop(key, None)
            self.save()
        return removed

    def save(self) -> None:
        """Persist the registry JSON to disk."""
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2, sort_keys=True)

    def _read(self) -> Dict[str, Dict[str, str | int]]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _checksum(path: Path) -> str:
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()


class VectorStoreManager:
    """Coordinate document ingestion and persistence with ChromaDB."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        loader: Optional[DocumentLoader] = None,
    ) -> None:
        """Create the manager using application settings."""
        self._config = config or ConfigManager.load().settings
        self._logger = get_logger(self.__class__.__name__)
        self._loader = loader or DocumentLoader(self._config)
        self._registry = FileRegistry(self._config.paths.file_registry)
        self._parent_retriever_enabled = self._config.modules.parent_page_retriever
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._config.ingestion.chunk_size,
            chunk_overlap=self._config.ingestion.chunk_overlap,
        )
        # Resolve device for optimal hardware acceleration
        device = resolve_device(self._config.embeddings.device)
        self._logger.info(f"Initializing embeddings on device: {device}")
        self._embeddings = HuggingFaceEmbeddings(
            model_name=self._config.embeddings.model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": self._config.embeddings.normalize_embeddings},
        )
        self._vector_store: Optional[Chroma] = None

    def sync(self) -> Dict[str, int]:
        """Ensure the vector store reflects the documents folder."""
        vector_store = self._ensure_vector_store()
        docs_root = self._config.paths.docs_dir
        additions: List[Tuple[Path, str, Dict[str, str | int]]] = []
        active_rel_paths: List[str] = []

        for file_path in self._loader.iter_source_files(docs_root):
            rel_path = str(file_path.relative_to(docs_root))
            snapshot = self._registry.snapshot(file_path)
            active_rel_paths.append(rel_path)
            if self._registry.has_changed(rel_path, snapshot):
                additions.append((file_path, rel_path, snapshot))

        removed = self._registry.removed_entries(active_rel_paths)
        added_chunks = self._ingest_new_files(vector_store, additions)
        removed_sources = self._remove_missing_sources(vector_store, removed)

        if added_chunks or removed_sources:
            vector_store.persist()
            self._registry.save()

        return {"added_chunks": added_chunks, "removed_sources": removed_sources}

    def get_vector_store(self) -> Chroma:
        """Return the underlying Chroma vector store."""
        return self._ensure_vector_store()

    def as_retriever(self, search_kwargs: Optional[Dict[str, int]] = None):
        """Return the vector store retriever with provided search parameters."""
        kwargs = search_kwargs or {"k": self._config.ingestion.retriever_k}
        return self._ensure_vector_store().as_retriever(search_kwargs=kwargs)

    def _ensure_vector_store(self) -> Chroma:
        if self._vector_store is not None:
            return self._vector_store
        
        persist_dir = str(self._config.vector_store.persist_directory)
        
        # Handle recreation if requested
        if self._config.vector_store.recreate:
            self._logger.warning("Recreate flag is set. Resetting vector store.")
            import shutil
            path = Path(persist_dir)
            if path.exists():
                shutil.rmtree(path)
            # Reset the registry as well since we are wiping the DB
            self._registry = FileRegistry(self._config.paths.file_registry)
            # Prevent infinite recreation loop in runtime by flipping the flag in memory (optional)
            # self._config.vector_store.recreate = False 

        self._vector_store = Chroma(
            collection_name=self._config.vector_store.collection_name,
            persist_directory=persist_dir,
            embedding_function=self._embeddings,
        )
        return self._vector_store

    def _ingest_new_files(
        self,
        vector_store: Chroma,
        additions: Sequence[Tuple[Path, str, Dict[str, str | int]]],
    ) -> int:
        added_chunks = 0
        for file_path, rel_path, snapshot in additions:
            documents = self._loader.load_file(file_path)
            if not documents:
                continue
            self._clear_existing_source(vector_store, rel_path)
            parent_metadata: Dict[str, Dict[str, Any]] = {}
            for doc in documents:
                metadata = doc.metadata or {}
                metadata["file_path"] = rel_path
                metadata["checksum"] = snapshot["checksum"]
                if self._parent_retriever_enabled:
                    parent_payload = {
                        "parent_page_content": doc.page_content,
                        "parent_doc_id": metadata.get("doc_id"),
                        "parent_source": metadata.get("source", rel_path),
                        "parent_page": metadata.get("page", 0),
                    }
                    metadata.update(parent_payload)
                    doc_id = parent_payload["parent_doc_id"]
                    if doc_id:
                        parent_metadata[str(doc_id)] = parent_payload
                doc.metadata = metadata
            chunks = self._splitter.split_documents(documents)
            for index, chunk in enumerate(chunks):
                chunk.metadata.setdefault("file_path", rel_path)
                chunk.metadata.setdefault("checksum", snapshot["checksum"])
                chunk.metadata["chunk_id"] = f"{chunk.metadata.get('doc_id', 'doc')}-{index}"
                if self._parent_retriever_enabled:
                    parent_doc_id = chunk.metadata.get("parent_doc_id") or chunk.metadata.get("doc_id")
                    parent_payload = parent_metadata.get(str(parent_doc_id)) if parent_doc_id else None
                    if parent_payload:
                        chunk.metadata.setdefault(
                            "parent_page_content",
                            parent_payload["parent_page_content"],
                        )
                        chunk.metadata.setdefault(
                            "parent_doc_id",
                            parent_payload["parent_doc_id"],
                        )
                        chunk.metadata.setdefault(
                            "parent_source",
                            parent_payload["parent_source"],
                        )
                        chunk.metadata.setdefault(
                            "parent_page",
                            parent_payload["parent_page"],
                        )
                    else:
                        self._logger.debug(
                            "Missing parent metadata for chunk %s (%s)",
                            chunk.metadata.get("chunk_id"),
                            rel_path,
                        )
            if chunks:
                self._logger.info(
                    "Adding %d chunks from %s",
                    len(chunks),
                    rel_path,
                )
                vector_store.add_documents(chunks)
                self._registry.update(rel_path, snapshot)
                added_chunks += len(chunks)
        return added_chunks

    def _remove_missing_sources(self, vector_store: Chroma, removed: Sequence[str]) -> int:
        removed_sources = 0
        for rel_path in removed:
            self._logger.info("Removing stale source %s", rel_path)
            vector_store.delete(where={"file_path": rel_path})
            self._registry.remove(rel_path)
            removed_sources += 1
        return removed_sources

    def _clear_existing_source(self, vector_store: Chroma, rel_path: str) -> None:
        """Delete existing chunks for a source to avoid duplicates."""
        self._logger.info("Clearing existing chunks for %s", rel_path)
        vector_store.delete(where={"file_path": rel_path})


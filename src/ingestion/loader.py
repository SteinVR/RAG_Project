"""Document loading utilities for the ingestion pipeline."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Iterator, List, Sequence

import fitz
from langchain_core.documents import Document

from src.core.config import AppConfig, ConfigManager
from src.utils.logger import get_logger


class DocumentLoader:
    """Load PDF/TXT/MD files into LangChain Document instances."""

    SUPPORTED_EXTENSIONS: Sequence[str] = (".pdf", ".txt", ".md")

    def __init__(self, config: AppConfig | None = None) -> None:
        """Create a loader bound to the application configuration."""
        self._config = config or ConfigManager.load().settings
        self._logger = get_logger(self.__class__.__name__)

    def iter_source_files(self, root: Path | None = None) -> Iterator[Path]:
        """Yield all supported files inside the docs directory."""
        docs_root = Path(root) if root else self._config.paths.docs_dir
        docs_root.mkdir(parents=True, exist_ok=True)
        for path in sorted(docs_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                yield path

    def load_file(self, path: Path) -> List[Document]:
        """Load a single source file into parent Document objects."""
        loader = self._resolve_loader(path)
        if loader is None:
            self._logger.debug("Skipping unsupported file %s", path)
            return []
        try:
            documents = loader(path)
            self._logger.info("Loaded %s (%d documents)", path.name, len(documents))
            return documents
        except Exception as exc:  # pragma: no cover - defensive logging
            self._logger.error("Failed to load %s: %s", path, exc, exc_info=True)
            return []

    def load_all(self) -> List[Document]:
        """Load every supported file inside the docs directory."""
        documents: List[Document] = []
        for path in self.iter_source_files():
            documents.extend(self.load_file(path))
        return documents

    def _resolve_loader(self, path: Path):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf
        if suffix in {".txt", ".md"}:
            return self._load_text
        return None

    def _load_pdf(self, path: Path) -> List[Document]:
        documents: List[Document] = []
        with fitz.open(path) as pdf:  # type: ignore[call-arg]
            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()
                if not text:
                    continue
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "doc_id": str(uuid.uuid4()),
                            "source": path.name,
                            "page": page_index + 1,
                        },
                    ),
                )
        return documents

    def _load_text(self, path: Path) -> List[Document]:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return []
        return [
            Document(
                page_content=text,
                metadata={
                    "doc_id": str(uuid.uuid4()),
                    "source": path.name,
                    "page": 1,
                },
            ),
        ]


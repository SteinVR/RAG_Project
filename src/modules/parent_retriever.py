"""Parent page reconstruction module."""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

from langchain_core.documents import Document

from src.core.config import AppConfig, ConfigManager
from src.modules.base import PipelineModule


class ParentPageRetriever(PipelineModule):
    """Replace chunk-level retrieval results with their parent pages."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize the module with application configuration."""
        super().__init__(config or ConfigManager.load().settings)

    def is_enabled(self) -> bool:
        """Return whether the parent page retriever should run."""
        return self.config.modules.parent_page_retriever

    def retrieve_parent_pages(self, chunks: List[Document]) -> List[Document]:
        """Group chunks by parent page and return reconstructed documents."""
        if not self.is_enabled() or not chunks:
            return chunks

        grouped: "OrderedDict[str, Document]" = OrderedDict()
        for chunk in chunks:
            metadata = chunk.metadata or {}
            parent_source = metadata.get("parent_source") or metadata.get("source")
            parent_page = metadata.get("parent_page") or metadata.get("page")
            parent_doc_id = metadata.get("parent_doc_id") or metadata.get("doc_id")
            parent_content = metadata.get("parent_page_content")
            if parent_content is None:
                self.logger.debug(
                    "Skipping chunk %s due to missing parent content",
                    metadata.get("chunk_id"),
                )
                continue
            key = self._build_key(parent_source, parent_page)
            if key in grouped:
                continue
            doc_metadata: Dict[str, object] = {
                "source": parent_source or metadata.get("source", "unknown"),
                "page": parent_page or metadata.get("page", 0),
                "doc_id": parent_doc_id or metadata.get("doc_id", "unknown"),
                "chunk_id": f"{parent_doc_id or metadata.get('doc_id', 'unknown')}-p{parent_page or metadata.get('page', 0)}",
            }
            grouped[key] = Document(page_content=parent_content, metadata=doc_metadata)

        parent_pages = list(grouped.values())
        if parent_pages:
            self.logger.info(
                "Parent page retriever collapsed %d chunks into %d pages",
                len(chunks),
                len(parent_pages),
            )
            return parent_pages

        self.logger.warning(
            "Parent page retriever failed to build parent documents; falling back to chunks",
        )
        return chunks

    @staticmethod
    def _build_key(source: Optional[str], page: Optional[int]) -> str:
        """Return a deterministic grouping key for a source page."""
        safe_source = source or "unknown"
        safe_page = page if page is not None else 0
        return f"{safe_source}::p{safe_page}"


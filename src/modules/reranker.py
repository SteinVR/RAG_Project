"""Cross-encoder reranking module."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.core.config import AppConfig, ConfigManager
from src.modules.base import PipelineModule


class Reranker(PipelineModule):
    """Score retrieved documents with a cross-encoder."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Load the configured cross-encoder model."""
        super().__init__(config or ConfigManager.load().settings)
        self._model = CrossEncoder(
            self.config.reranker.model_name,
            device=self.config.embeddings.device,
        )

    def is_enabled(self) -> bool:
        """Return whether the reranker is active."""
        return self.config.modules.reranker and self.config.reranker.enabled

    def rerank(self, query: str, documents: Sequence[Document]) -> List[Document]:
        """Return the top documents sorted by cross-encoder score."""
        if not self.is_enabled() or not documents:
            return list(documents)

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self._model.predict(pairs)
        scored_docs: List[Tuple[float, Document]] = list(zip(scores, documents))
        scored_docs.sort(key=lambda pair: pair[0], reverse=True)
        top_k = self.config.reranker.top_k
        reranked = [doc for _, doc in scored_docs[:top_k]]
        self.logger.info("Reranked %d documents using the cross-encoder", len(reranked))
        return reranked


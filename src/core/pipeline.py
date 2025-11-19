"""Pipeline orchestrator that wires ingestion, retrieval, and RAG modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document

from src.core.config import AppConfig, ConfigManager
from src.ingestion.vector_store import VectorStoreManager
from src.modules.generator import Generator, AnswerSchema
from src.modules.hyde import HyDEGenerator
from src.modules.reranker import Reranker
from src.modules.rewriter import QueryRewriter, RewriteResult
from src.utils.logger import get_logger


@dataclass
class PipelineResult:
    """Structured response returned by the RAG pipeline."""

    question: str
    rewritten_query: str
    classification: str
    hyde_prompt: str
    context: List[Document]
    answer: AnswerSchema


class RAGPipeline:
    """High-level orchestrator used by the CLI or other front-ends."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._config = config or ConfigManager.load().settings
        self._logger = get_logger(self.__class__.__name__)
        self._vector_store_manager = VectorStoreManager(self._config)
        self._rewriter = QueryRewriter(self._config)
        self._hyde = HyDEGenerator(self._config)
        self._reranker = Reranker(self._config)
        self._generator = Generator(self._config)
        self._retriever = None
        self._prepare_vector_store()

    def _prepare_vector_store(self) -> None:
        stats = self._vector_store_manager.sync()
        self._retriever = self._vector_store_manager.as_retriever(
            search_kwargs={"k": self._config.ingestion.retriever_k},
        )
        self._logger.info(
            "Vector store synchronized (added=%s removed=%s)",
            stats["added_chunks"],
            stats["removed_sources"],
        )

    def execute(self, query: str) -> PipelineResult:
        """Run the end-to-end pipeline for a user query."""
        rewrite = self._run_rewriter(query)
        effective_query = rewrite.rewritten_query if rewrite else query
        classification = rewrite.classification if rewrite else "direct"
        hyde_prompt = self._run_hyde(effective_query)
        documents = self._retrieve_documents(effective_query, hyde_prompt)
        answer = self._generator.generate(query, documents)
        return PipelineResult(
            question=query,
            rewritten_query=effective_query,
            classification=classification,
            hyde_prompt=hyde_prompt,
            context=documents,
            answer=answer,
        )

    def _run_rewriter(self, query: str) -> Optional[RewriteResult]:
        if not self._rewriter.is_enabled():
            return None
        self._logger.info("Running QueryRewriter")
        return self._rewriter.rewrite(query)

    def _run_hyde(self, query: str) -> str:
        if not self._hyde.is_enabled():
            return ""
        self._logger.info("Running HyDE generator")
        return self._hyde.generate(query)

    def _retrieve_documents(self, query: str, hyde_prompt: str) -> List[Document]:
        if self._retriever is None:
            raise RuntimeError("Retriever not initialized. Call _prepare_vector_store first.")
        base_results: List[Document] = list(self._invoke_retriever(query))
        hyde_results: List[Document] = []
        if hyde_prompt:
            hyde_results = list(self._invoke_retriever(hyde_prompt))
        combined = self._deduplicate(base_results + hyde_results)
        self._logger.info("Retrieved %d unique documents", len(combined))
        if self._reranker.is_enabled():
            combined = self._reranker.rerank(query, combined)
        return combined

    def _invoke_retriever(self, query: str) -> Sequence[Document]:
        assert self._retriever is not None
        results = self._retriever.invoke(query)
        if isinstance(results, Document):
            return [results]
        return results

    @staticmethod
    def _deduplicate(documents: Sequence[Document]) -> List[Document]:
        seen: Dict[str, Document] = {}
        ordered: List[Document] = []
        for doc in documents:
            metadata = doc.metadata or {}
            key = metadata.get("chunk_id") or metadata.get("doc_id") or doc.page_content
            if key in seen:
                continue
            seen[key] = doc
            ordered.append(doc)
        return ordered


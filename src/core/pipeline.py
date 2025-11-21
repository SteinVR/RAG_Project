"""Pipeline orchestrator that wires ingestion, retrieval, and RAG modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document

from src.core.config import AppConfig, ConfigManager
from src.ingestion.vector_store import VectorStoreManager
from src.modules.generator import Generator, AnswerSchema
from src.modules.hyde import HyDEGenerator
from src.modules.parent_retriever import ParentPageRetriever
from src.modules.reranker import Reranker
from src.modules.rewriter import QueryRewriter, RewriteResult
from src.utils.console import ConsoleProgress
from src.utils.logger import get_logger
from src.utils.pipeline_logger import PipelineLogger


@dataclass
class PipelineResult:
    """Structured response returned by the RAG pipeline."""

    question: str
    rewritten_query: str
    rewrite_decision: str
    hyde_seed_query: str
    hyde_prompt: str
    context: List[Document]
    answer: AnswerSchema


class RAGPipeline:
    """High-level orchestrator used by the CLI or other front-ends."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        show_progress: bool = True,
        enable_pipeline_logging: Optional[bool] = None,
    ) -> None:
        self._config = config or ConfigManager.load().settings
        self._logger = get_logger(self.__class__.__name__)
        self._progress = ConsoleProgress(enabled=show_progress)
        pipeline_logging_enabled = (
            enable_pipeline_logging
            if enable_pipeline_logging is not None
            else self._config.logging.extended
        )
        self._pipeline_logger = PipelineLogger() if pipeline_logging_enabled else None
        self._vector_store_manager = VectorStoreManager(self._config)
        self._rewriter = QueryRewriter(self._config)
        self._hyde = HyDEGenerator(self._config)
        self._parent_retriever = ParentPageRetriever(self._config)
        self._reranker = Reranker(self._config)
        self._generator = Generator(self._config)
        self._retriever = None
        self._prepare_vector_store()

    def _prepare_vector_store(self) -> None:
        with self._progress.stage("Vector store synchronization"):
            stats = self._vector_store_manager.sync()
            self._retriever = self._vector_store_manager.as_retriever(
                search_kwargs={"k": self._config.ingestion.retriever_k},
            )
            self._logger.info(
                "Vector store synchronized (added=%s removed=%s)",
                stats["added_chunks"],
                stats["removed_sources"],
            )
        if stats["added_chunks"] > 0 or stats["removed_sources"] > 0:
            self._progress.info(
                f"Indexed {stats['added_chunks']} chunks, removed {stats['removed_sources']} sources"
            )

    def execute(self, query: str) -> PipelineResult:
        """Run the end-to-end pipeline for a user query."""
        if self._pipeline_logger:
            self._pipeline_logger.start_query(
                query=query,
                rewriter_enabled=self._rewriter.is_enabled(),
                hyde_enabled=self._hyde.is_enabled(),
                reranker_enabled=self._reranker.is_enabled(),
                parent_retriever_enabled=self._parent_retriever.is_enabled(),
            )
        
        with self._progress.stage("Processing query"):
            hyde_enabled = self._hyde.is_enabled()
            rewrite = self._run_rewriter(query, hyde_enabled)
            effective_query = rewrite.retriever_query if rewrite else query
            rewrite_decision = rewrite.rewrite_decision if rewrite else "use_original"
            hyde_seed = ""
            if hyde_enabled:
                if rewrite and rewrite.hyde_query.strip():
                    hyde_seed = rewrite.hyde_query.strip()
                else:
                    hyde_seed = effective_query
            hyde_prompt = self._run_hyde(hyde_seed, query)
            documents = self._retrieve_documents(effective_query, hyde_prompt)
            
            if self._pipeline_logger:
                self._pipeline_logger.log_final_chunks(documents)
            
            with self._progress.stage("Generating answer"):
                answer = self._generator.generate(query, documents)
            
            if self._pipeline_logger:
                self._pipeline_logger.log_answer(answer.answer_markdown)
                log_file = self._pipeline_logger.save()
                self._logger.info("Pipeline execution log saved to %s", log_file)
            
            return PipelineResult(
                question=query,
                rewritten_query=effective_query,
                rewrite_decision=rewrite_decision,
                hyde_seed_query=hyde_seed,
                hyde_prompt=hyde_prompt,
                context=documents,
                answer=answer,
            )

    def _run_rewriter(self, query: str, hyde_enabled: bool) -> Optional[RewriteResult]:
        if not self._rewriter.is_enabled():
            return None
        with self._progress.stage("Query rewriting"):
            self._logger.info("Running QueryRewriter")
            result = self._rewriter.rewrite(query, hyde_enabled=hyde_enabled)
            if self._pipeline_logger and result:
                self._pipeline_logger.log_rewriter(
                    decision=result.rewrite_decision,
                    retriever_query=result.retriever_query,
                    hyde_seed=result.hyde_query if hyde_enabled else "",
                    rationale=result.rationale,
                )
            return result

    def _run_hyde(self, hyde_seed: str, original_question: str) -> str:
        if not self._hyde.is_enabled():
            return ""
        with self._progress.stage("HyDE generation"):
            self._logger.info("Running HyDE generator")
            hyde_answer = self._hyde.generate(hyde_seed, original_question)
            if self._pipeline_logger:
                self._pipeline_logger.log_hyde(hyde_answer)
            return hyde_answer

    def _retrieve_documents(self, query: str, hyde_prompt: str) -> List[Document]:
        if self._retriever is None:
            raise RuntimeError("Retriever not initialized. Call _prepare_vector_store first.")
        
        with self._progress.stage("Document retrieval"):
            base_results: List[Document] = list(self._invoke_retriever(query))
            if self._pipeline_logger:
                self._pipeline_logger.log_base_retrieval(base_results)
            
            hyde_results: List[Document] = []
            if hyde_prompt:
                hyde_results = list(self._invoke_retriever(hyde_prompt))
                if self._pipeline_logger:
                    self._pipeline_logger.log_hyde_retrieval(hyde_results)
            
            combined = self._deduplicate(base_results + hyde_results)
            self._logger.info("Retrieved %d unique documents", len(combined))
            
            if self._pipeline_logger:
                self._pipeline_logger.log_combined_count(len(combined))
        
        documents = combined
        if self._parent_retriever.is_enabled():
            with self._progress.stage("Parent page retrieval"):
                documents = self._parent_retriever.retrieve_parent_pages(combined)
                self._logger.info(
                    "Parent page retriever returned %d parent documents",
                    len(documents),
                )
                if self._pipeline_logger:
                    self._pipeline_logger.log_parent_pages(documents)
        
        if self._reranker.is_enabled():
            with self._progress.stage("Reranking documents", f"{len(documents)} candidates"):
                combined_with_scores = self._reranker.rerank_with_scores(query, documents)
                documents = [doc for doc, _ in combined_with_scores]
                scores = [score for _, score in combined_with_scores]
                
                if self._pipeline_logger:
                    self._pipeline_logger.log_reranker(documents, scores)
        
        return documents

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


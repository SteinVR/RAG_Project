"""Detailed pipeline execution logger for debugging and analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    """Captured retrieval result."""
    
    doc_id: str
    source: str
    page: int
    score: Optional[float]
    content_preview: str
    
    @classmethod
    def from_document(cls, doc: Document, score: Optional[float] = None) -> RetrievalResult:
        """Create from LangChain Document."""
        metadata = doc.metadata or {}
        content_preview = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content
        return cls(
            doc_id=metadata.get("chunk_id", "unknown"),
            source=metadata.get("source", "unknown"),
            page=metadata.get("page", 0),
            score=score,
            content_preview=content_preview,
        )


@dataclass
class PipelineLog:
    """Complete log entry for a single query execution."""
    
    timestamp: str
    query: str
    rewriter_enabled: bool
    rewriter_decision: Optional[str] = None
    rewritten_query: Optional[str] = None
    rewrite_rationale: Optional[str] = None
    hyde_seed_query: Optional[str] = None
    hyde_enabled: bool = False
    hyde_answer: Optional[str] = None
    base_retrieval_results: List[RetrievalResult] = None
    hyde_retrieval_results: List[RetrievalResult] = None
    combined_results_count: int = 0
    reranker_enabled: bool = False
    parent_retriever_enabled: bool = False
    reranker_scores: Optional[List[Dict[str, Any]]] = None
    final_chunks: List[RetrievalResult] = None
    final_answer: Optional[str] = None
    parent_page_results: List[RetrievalResult] = None
    
    def __post_init__(self) -> None:
        if self.base_retrieval_results is None:
            self.base_retrieval_results = []
        if self.hyde_retrieval_results is None:
            self.hyde_retrieval_results = []
        if self.final_chunks is None:
            self.final_chunks = []
        if self.parent_page_results is None:
            self.parent_page_results = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PipelineLogger:
    """Logger for detailed pipeline execution tracking."""
    
    def __init__(self, log_dir: str = "logs/Pipeline") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log: Optional[PipelineLog] = None
    
    def start_query(
        self,
        query: str,
        rewriter_enabled: bool,
        hyde_enabled: bool,
        reranker_enabled: bool,
        parent_retriever_enabled: bool,
    ) -> None:
        """Initialize a new query log."""
        self.current_log = PipelineLog(
            timestamp=datetime.now().isoformat(),
            query=query,
            rewriter_enabled=rewriter_enabled,
            hyde_enabled=hyde_enabled,
            reranker_enabled=reranker_enabled,
            parent_retriever_enabled=parent_retriever_enabled,
        )
    
    def log_rewriter(self, decision: str, retriever_query: str, hyde_seed: str, rationale: str) -> None:
        """Log query rewriter output."""
        if self.current_log:
            self.current_log.rewriter_decision = decision
            self.current_log.rewritten_query = retriever_query
            self.current_log.hyde_seed_query = hyde_seed or None
            self.current_log.rewrite_rationale = rationale
    
    def log_hyde(self, hyde_answer: str) -> None:
        """Log HyDE generated answer."""
        if self.current_log:
            self.current_log.hyde_answer = hyde_answer
    
    def log_base_retrieval(self, documents: List[Document]) -> None:
        """Log base retrieval results."""
        if self.current_log:
            self.current_log.base_retrieval_results = [
                RetrievalResult.from_document(doc) for doc in documents
            ]
    
    def log_hyde_retrieval(self, documents: List[Document]) -> None:
        """Log HyDE retrieval results."""
        if self.current_log:
            self.current_log.hyde_retrieval_results = [
                RetrievalResult.from_document(doc) for doc in documents
            ]
    
    def log_combined_count(self, count: int) -> None:
        """Log combined deduplicated results count."""
        if self.current_log:
            self.current_log.combined_results_count = count
    
    def log_parent_pages(self, documents: List[Document]) -> None:
        """Log parent page consolidation results."""
        if self.current_log:
            self.current_log.parent_page_results = [
                RetrievalResult.from_document(doc) for doc in documents
            ]
    
    def log_reranker(self, documents: List[Document], scores: List[float]) -> None:
        """Log reranker scores."""
        if self.current_log:
            self.current_log.reranker_scores = [
                {
                    "doc_id": doc.metadata.get("chunk_id", "unknown"),
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 0),
                    "score": float(score),
                    "content_preview": doc.page_content[:100] + "...",
                }
                for doc, score in zip(documents, scores)
            ]
    
    def log_final_chunks(self, documents: List[Document]) -> None:
        """Log final selected chunks."""
        if self.current_log:
            self.current_log.final_chunks = [
                RetrievalResult.from_document(doc) for doc in documents
            ]
    
    def log_answer(self, answer: str) -> None:
        """Log final generated answer."""
        if self.current_log:
            self.current_log.final_answer = answer
    
    def save(self) -> str:
        """Save current log to file and return filename."""
        if not self.current_log:
            raise ValueError("No active log to save")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.log_dir / f"query_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.current_log.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Also save human-readable version
        txt_filename = self.log_dir / f"query_{timestamp}.txt"
        self._save_readable(txt_filename)
        
        self.current_log = None
        return str(filename)
    
    def _save_readable(self, filename: Path) -> None:
        """Save human-readable version of the log."""
        if not self.current_log:
            return
        
        log = self.current_log
        lines = [
            "=" * 80,
            "PIPELINE EXECUTION LOG",
            "=" * 80,
            f"Timestamp: {log.timestamp}",
            f"Query: {log.query}",
            "",
            "CONFIGURATION",
            "-" * 80,
            f"Rewriter enabled: {log.rewriter_enabled}",
            f"HyDE enabled: {log.hyde_enabled}",
            f"Reranker enabled: {log.reranker_enabled}",
            f"Parent retriever enabled: {log.parent_retriever_enabled}",
            "",
        ]
        
        if log.rewriter_enabled and log.rewritten_query:
            lines.extend([
                "QUERY REWRITER",
                "-" * 80,
                f"Decision: {log.rewriter_decision}",
                f"Rationale: {log.rewrite_rationale}",
                f"Retrieval query: {log.rewritten_query}",
                f"HyDE seed: {log.hyde_seed_query or '(none)'}",
                "",
            ])
        
        if log.hyde_enabled and log.hyde_answer:
            lines.extend([
                "HYDE GENERATION",
                "-" * 80,
                f"Generated answer:\n{log.hyde_answer}",
                "",
            ])
        
        lines.extend([
            "BASE RETRIEVAL",
            "-" * 80,
            f"Retrieved {len(log.base_retrieval_results)} documents:",
        ])
        for i, result in enumerate(log.base_retrieval_results, 1):
            lines.append(f"  {i}. {result.source} (p.{result.page}) - {result.doc_id}")
            lines.append(f"     {result.content_preview}")
        lines.append("")
        
        if log.hyde_enabled and log.hyde_retrieval_results:
            lines.extend([
                "HYDE RETRIEVAL",
                "-" * 80,
                f"Retrieved {len(log.hyde_retrieval_results)} documents:",
            ])
            for i, result in enumerate(log.hyde_retrieval_results, 1):
                lines.append(f"  {i}. {result.source} (p.{result.page}) - {result.doc_id}")
                lines.append(f"     {result.content_preview}")
            lines.append("")
        
        lines.extend([
            "COMBINED RESULTS",
            "-" * 80,
            f"After deduplication: {log.combined_results_count} unique documents",
            "",
        ])
        
        if log.parent_retriever_enabled:
            lines.extend([
                "PARENT PAGE RETRIEVAL",
                "-" * 80,
                f"Parent pages returned: {len(log.parent_page_results)}",
            ])
            for i, result in enumerate(log.parent_page_results, 1):
                lines.append(f"  {i}. {result.source} (p.{result.page}) - {result.doc_id}")
                lines.append(f"     {result.content_preview}")
            lines.append("")
        
        if log.reranker_enabled and log.reranker_scores:
            lines.extend([
                "RERANKER SCORES",
                "-" * 80,
            ])
            for i, score_info in enumerate(log.reranker_scores, 1):
                lines.append(
                    f"  {i}. Score: {score_info['score']:.4f} | "
                    f"{score_info['source']} (p.{score_info['page']})"
                )
                lines.append(f"     {score_info['content_preview']}")
            lines.append("")
        
        lines.extend([
            "FINAL CHUNKS SELECTED",
            "-" * 80,
            f"Total: {len(log.final_chunks)} chunks",
        ])
        for i, chunk in enumerate(log.final_chunks, 1):
            lines.append(f"  {i}. {chunk.source} (p.{chunk.page})")
            lines.append(f"     {chunk.content_preview}")
        lines.append("")
        
        if log.final_answer:
            lines.extend([
                "FINAL ANSWER",
                "-" * 80,
                log.final_answer,
                "",
            ])
        
        lines.append("=" * 80)
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


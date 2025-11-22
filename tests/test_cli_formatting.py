"""Tests for CLI formatting helpers."""

from __future__ import annotations

from langchain_core.documents import Document

from main import format_answer
from src.core.pipeline import PipelineResult
from src.modules.generator import AnswerSchema, Citation, RelevantSource


def _make_result(citations, sources=None):
    relevant_sources = sources if sources is not None else []
    return PipelineResult(
        question="What is RAG?",
        rewritten_query="What is RAG?",
        rewrite_decision="use_original",
        hyde_seed_query="",
        hyde_prompt="",
        context=[],
        answer=AnswerSchema(
            answer_markdown="RAG explained.",
            citations=citations,
            relevant_sources=relevant_sources,
        ),
    )


def test_format_answer_includes_citations_section() -> None:
    citations = [
        Citation(document="doc.pdf", page=2, text="Important passage"),
        Citation(document="guide.md", page=5, text="Another passage"),
    ]
    sources = [
        RelevantSource(
            source="doc.pdf",
            page=2,
            snippet="Important passage about RAG in doc.pdf section 2 describing basics.",
        ),
        RelevantSource(
            source="guide.md",
            page=5,
            snippet="Another passage from guide.md page 5 summarizing details for readers.",
        ),
    ]
    result = _make_result(citations, sources)

    formatted = format_answer(result)

    assert "Citations:" in formatted
    assert "- doc.pdf (p.2): Important passage" in formatted
    assert "- guide.md (p.5): Another passage" in formatted
    assert "RelevantSources = [" in formatted
    assert '"source": "doc.pdf"' in formatted
    assert '"page": 5' in formatted


def test_format_answer_without_citations() -> None:
    result = _make_result([], [])

    formatted = format_answer(result)

    assert formatted.strip() == "RAG explained."
    assert "Citations:" not in formatted
    assert "RelevantSources" not in formatted



"""Tests for pipeline helper utilities."""

from __future__ import annotations

from langchain_core.documents import Document

from src.core.pipeline import RAGPipeline


def test_deduplicate_prefers_first_occurrence() -> None:
    docs = [
        Document(page_content="A", metadata={"chunk_id": "1"}),
        Document(page_content="A duplicate", metadata={"chunk_id": "1"}),
        Document(page_content="B", metadata={"doc_id": "doc-1"}),
        Document(page_content="B duplicate", metadata={"doc_id": "doc-1"}),
        Document(page_content="C"),
        Document(page_content="C"),
    ]

    deduped = RAGPipeline._deduplicate(docs)

    assert len(deduped) == 3
    assert [doc.page_content for doc in deduped] == ["A", "B", "C"]


def test_deduplicate_preserves_original_order() -> None:
    docs = [
        Document(page_content="First", metadata={"chunk_id": "a"}),
        Document(page_content="Second", metadata={"chunk_id": "b"}),
        Document(page_content="Third", metadata={"chunk_id": "c"}),
        Document(page_content="First again", metadata={"chunk_id": "a"}),
        Document(page_content="Second again", metadata={"chunk_id": "b"}),
    ]

    deduped = RAGPipeline._deduplicate(docs)

    assert [doc.page_content for doc in deduped] == ["First", "Second", "Third"]



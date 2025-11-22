"""Final answer generator module."""

from __future__ import annotations

from typing import List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from src.core.config import AppConfig, ConfigManager
from src.modules.base import LLMClientFactory, PipelineModule


class Citation(BaseModel):
    """Representation of a single cited passage."""

    document: str = Field(description="Human readable source filename.")
    page: int = Field(description="Page number inside the original document.")
    text: str = Field(description="Quoted or paraphrased fragment.")


class RelevantSource(BaseModel):
    """Structured representation of a referenced context chunk."""

    source: str = Field(description="Exact filename of the cited document.")
    page: int = Field(description="Page number containing the snippet.")
    snippet: str = Field(
        description="Summary of the cited passage with newlines removed (target 100-200 chars, but flexible for safety).",
        min_length=1,
        max_length=2048,
    )


class AnswerSchema(BaseModel):
    """Structured final answer schema."""

    answer_markdown: str = Field(description="Markdown formatted response.")
    citations: List[Citation]
    relevant_sources: List[RelevantSource]


class Generator(PipelineModule):
    """LLM-based synthesis of the final response."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize the generator chain."""
        super().__init__(config or ConfigManager.load().settings)
        llm_factory = LLMClientFactory(self.config)
        self._parser = PydanticOutputParser(pydantic_object=AnswerSchema)
        self._chain: RunnableSequence = (
            self._build_prompt()
            | llm_factory.build_chat_model()
            | self._parser
        )

    def generate(self, question: str, contexts: Sequence[Document]) -> AnswerSchema:
        """Generate an answer from the supplied context documents."""
        top_contexts = list(contexts)[: self.config.generator.max_context_documents]
        payload = {
            "question": question,
            "context": self._format_context(top_contexts),
            "format_instructions": self._parser.get_format_instructions(),
        }
        return self._chain.invoke(payload)

    def _build_prompt(self) -> ChatPromptTemplate:
        """Compose the chat prompt for the generator."""
        template = (
            f"{self.config.generator.answer_prompt}\n\n"
            "Context:\n{context}\n\n"
            "{format_instructions}\n\n"
            "User question: {question}"
        )
        return ChatPromptTemplate.from_template(template)

    @staticmethod
    def _format_context(contexts: Sequence[Document]) -> str:
        """Format retrieved documents for inclusion in the prompt."""
        formatted_sections = []
        for index, doc in enumerate(contexts, start=1):
            source = doc.metadata.get("source") or doc.metadata.get("file_path", "unknown")
            page = doc.metadata.get("page", "?")
            formatted_sections.append(
                f"[{index}] Source: {source} (page {page})\n{doc.page_content}",
            )
        return "\n\n".join(formatted_sections)


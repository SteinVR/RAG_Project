"""Query rewriter and classifier module."""

from __future__ import annotations

from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from src.core.config import AppConfig, ConfigManager
from src.modules.base import LLMClientFactory, PipelineModule


class RewriteResult(BaseModel):
    """Structured response from the query rewriter."""

    classification: str = Field(description="One of the configured labels.")
    rewritten_query: str = Field(description="Self-contained reformulation.")


class QueryRewriter(PipelineModule):
    """LLM-powered query classifier and rewriter."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize the rewriter and prompt chain."""
        super().__init__(config or ConfigManager.load().settings)
        self._parser = PydanticOutputParser(pydantic_object=RewriteResult)
        llm_factory = LLMClientFactory(self.config)
        self._chain: RunnableSequence = (
            self._build_prompt()
            | llm_factory.build_chat_model()
            | self._parser
        )

    def is_enabled(self) -> bool:
        """Return whether the rewriter should be used."""
        return self.config.modules.rewriter and self.config.rewriter.enabled

    def rewrite(self, query: str) -> RewriteResult:
        """Rewrite and classify the incoming user query."""
        if not self.is_enabled():
            return RewriteResult(
                classification="direct",
                rewritten_query=query,
            )
        result = self._chain.invoke({"query": query})
        return RewriteResult(
            classification=result.classification,
            rewritten_query=result.rewritten_query.strip() or query,
        )

    def _build_prompt(self) -> ChatPromptTemplate:
        """Return the chat prompt with format instructions injected."""
        instructions = self._parser.get_format_instructions()
        template = (
            f"{self.config.rewriter.prompt_template}\n\n"
            f"{instructions}\n\n"
            "User question: {{query}}"
        )
        return ChatPromptTemplate.from_template(template)


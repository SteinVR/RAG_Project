"""Query rewriter and classifier module."""

from __future__ import annotations

from typing import Literal, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field

from src.core.config import AppConfig, ConfigManager
from src.modules.base import LLMClientFactory, PipelineModule


class RewriteResult(BaseModel):
    """Structured response from the query rewriter."""

    rewrite_decision: Literal["use_original", "rewrite"] = Field(
        description="Whether to keep the original query or apply the rewrite.",
    )
    retriever_query: str = Field(
        description="Self-contained query to send to the dense retriever.",
    )
    hyde_query: str = Field(
        default="",
        description="Directive for the HyDE module when enabled, else empty.",
    )
    rationale: str = Field(
        description="Short reason explaining the decision.",
    )


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

    def rewrite(self, query: str, hyde_enabled: bool = False) -> RewriteResult:
        """Rewrite and classify the incoming user query."""
        if not self.is_enabled():
            return RewriteResult(
                rewrite_decision="use_original",
                retriever_query=query,
                hyde_query=query if hyde_enabled else "",
                rationale="Query rewriter disabled.",
            )
        result = self._chain.invoke(
            {
                "query": query,
                "hyde_enabled": "true" if hyde_enabled else "false",
            },
        )
        return RewriteResult(
            rewrite_decision=result.rewrite_decision,
            retriever_query=result.retriever_query.strip() or query,
            hyde_query=result.hyde_query.strip() if hyde_enabled else "",
            rationale=result.rationale.strip(),
        )

    def _build_prompt(self) -> ChatPromptTemplate:
        """Return the chat prompt with format instructions injected."""
        instructions = self._parser.get_format_instructions()
        # Escape curly braces in instructions to prevent template variable conflicts
        instructions_escaped = instructions.replace("{", "{{").replace("}", "}}")
        template = (
            f"{self.config.rewriter.prompt_template}\n\n"
            f"{instructions_escaped}\n\n"
            "HyDE enabled: {{hyde_enabled}}\n"
            "User question: {{query}}"
        )
        return ChatPromptTemplate.from_template(template)


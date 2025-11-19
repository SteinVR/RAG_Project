"""HyDE (Hypothetical Document Embeddings) helper module."""

from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate

from src.core.config import AppConfig, ConfigManager
from src.modules.base import LLMClientFactory, PipelineModule


class HyDEGenerator(PipelineModule):
    """Generate hypothetical answers used for retrieval seeding."""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        """Initialize the HyDE prompt chain."""
        super().__init__(config or ConfigManager.load().settings)
        llm_factory = LLMClientFactory(self.config)
        self._prompt = ChatPromptTemplate.from_template(
            f"{self.config.hyde.prompt_template}\n\nUser question: {{query}}",
        )
        self._llm = llm_factory.build_chat_model(
            max_output_tokens=self.config.hyde.max_tokens,
        )

    def is_enabled(self) -> bool:
        """Return whether HyDE is globally enabled."""
        return self.config.modules.hyde and self.config.hyde.enabled

    def generate(self, query: str) -> str:
        """Generate a hypothetical answer for the supplied query."""
        if not self.is_enabled():
            return ""
        response = self._prompt | self._llm
        result = response.invoke({"query": query})
        return result.content.strip() if hasattr(result, "content") else str(result).strip()


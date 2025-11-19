"""Base abstractions shared by RAG modules."""

from __future__ import annotations

import logging
import os
from abc import ABC
from typing import Any, List, Optional

from google import genai
from google.genai import types as genai_types
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.core.config import AppConfig
from src.utils.logger import get_logger


class PipelineModule(ABC):
    """Base class for all configurable pipeline modules."""

    def __init__(self, config: AppConfig) -> None:
        """Store configuration and initialize the module logger."""
        self._config = config
        self._logger = get_logger(self.__class__.__name__)

    @property
    def config(self) -> AppConfig:
        """Return the global application configuration."""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """Return the logger bound to this module."""
        return self._logger

    def is_enabled(self) -> bool:
        """Return whether this module should run."""
        return True


class GoogleGenAIChatModel(BaseChatModel):
    """Adapter that enables LangChain to call the native google-genai SDK."""

    def __init__(
        self,
        *,
        client: genai.Client,
        model_name: str,
        temperature: float,
        max_output_tokens: int,
    ) -> None:
        super().__init__()
        self._client = client
        self._model_name = model_name
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        contents, system_instruction = self._convert_messages(messages)
        config = genai_types.GenerateContentConfig(
            temperature=self._temperature,
            max_output_tokens=self._max_output_tokens,
            stop_sequences=stop,
            system_instruction=system_instruction,
        )
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
        )
        text = self._extract_text(response)
        generation = ChatGeneration(message=AIMessage(content=text))
        return ChatResult(generations=[generation])

    def _convert_messages(
        self,
        messages: List[BaseMessage],
    ) -> tuple[List[genai_types.Content], Optional[str]]:
        contents: List[genai_types.Content] = []
        system_instruction: Optional[str] = None
        for message in messages:
            if isinstance(message, SystemMessage):
                instruction = message.content
                system_instruction = (
                    instruction
                    if system_instruction is None
                    else f"{system_instruction}\n{instruction}"
                )
                continue
            role = "user"
            if isinstance(message, AIMessage):
                role = "model"
            parts = [
                genai_types.Part.from_text(
                    text=message.content if isinstance(message.content, str) else str(message.content),
                ),
            ]
            contents.append(genai_types.Content(role=role, parts=parts))
        return contents, system_instruction

    @staticmethod
    def _extract_text(response: Any) -> str:
        if getattr(response, "text", None):
            return response.text
        candidates = getattr(response, "candidates", None) or []
        fragments: List[str] = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", []) if content else []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    fragments.append(text)
        return "\n".join(fragments)

    @property
    def _llm_type(self) -> str:
        return "google-genai-native"


class LLMClientFactory:
    """Helper for instantiating chat models according to configuration."""

    def __init__(self, config: AppConfig) -> None:
        """Create the factory with application settings."""
        self._config = config

    def build_chat_model(
        self,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> BaseChatModel:
        """Return a configured google-genai-backed chat model."""
        llm_cfg = self._config.llm
        if llm_cfg.provider.lower() != "google":
            raise ValueError(f"Unsupported LLM provider: {llm_cfg.provider}")
        api_key = os.getenv(llm_cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key for provider: set {llm_cfg.api_key_env}",
            )

        client = genai.Client(api_key=api_key)
        return GoogleGenAIChatModel(
            client=client,
            model_name=llm_cfg.model_name,
            temperature=temperature if temperature is not None else llm_cfg.temperature,
            max_output_tokens=(
                max_output_tokens
                if max_output_tokens is not None
                else llm_cfg.max_output_tokens
            ),
        )
"""Configuration management utilities for the Modular RAG console app."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, ClassVar, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class PathSettings(BaseModel):
    """Filesystem locations used throughout the application."""

    docs_dir: Path = Field(default=Path("data/docs"))
    vector_store_dir: Path = Field(default=Path("data/vector_store"))
    file_registry: Path = Field(default=Path("data/file_registry.json"))

    model_config = ConfigDict(validate_assignment=True)

    def ensure_directories(self) -> None:
        """Create directories required for ingestion and persistence."""
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.file_registry.parent.mkdir(parents=True, exist_ok=True)


class LoggingSettings(BaseModel):
    """Logging configuration shared across modules."""

    level: str = Field(default="INFO")
    format: str = Field(
        default="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )
    file: Path = Field(default=Path("logs/prototype.log"))
    console: bool = Field(default=True)

    model_config = ConfigDict(validate_assignment=True)

    def ensure_directories(self) -> None:
        """Create the log directory and file placeholder."""
        self.file.parent.mkdir(parents=True, exist_ok=True)
        if not self.file.exists():
            self.file.touch()


class IngestionSettings(BaseModel):
    """Chunking and scanning parameters for document ingestion."""

    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)
    retriever_k: int = Field(default=10, ge=1)
    glob_patterns: List[str] = Field(
        default_factory=lambda: ["*.pdf", "*.txt", "*.md"],
    )


class VectorStoreSettings(BaseModel):
    """ChromaDB persistence and collection settings."""

    collection_name: str = Field(default="rag_documents")
    recreate: bool = Field(default=False)
    embedding_batch_size: int = Field(default=64, ge=1)
    persist_directory: Optional[Path] = None


class EmbeddingSettings(BaseModel):
    """Embedding model configuration."""

    model_name: str = Field(default="BAAI/bge-m3")
    device: str = Field(default="cpu")
    normalize_embeddings: bool = Field(default=True)


class LLMSettings(BaseModel):
    """LLM provider configuration for the generator and helpers."""

    provider: str = Field(default="google")
    model_name: str = Field(default="models/gemini-flash-lite-latest")
    temperature: float = Field(default=0.2, ge=0.0)
    max_output_tokens: int = Field(default=2048, ge=64)
    api_key_env: str = Field(default="GOOGLE_API_KEY")


class ModuleToggles(BaseModel):
    """Feature flags for optional pipeline modules."""

    rewriter: bool = Field(default=True)
    hyde: bool = Field(default=True)
    reranker: bool = Field(default=True)


class RewriterSettings(BaseModel):
    """Prompt template for the Query Rewriter."""

    enabled: bool = Field(default=True)
    prompt_template: str


class HydeSettings(BaseModel):
    """Prompt parameters for HyDE generation."""

    enabled: bool = Field(default=True)
    prompt_template: str
    max_tokens: int = Field(default=256, ge=1)


class GeneratorSettings(BaseModel):
    """Final answer synthesis settings."""

    max_context_documents: int = Field(default=4, ge=1)
    answer_prompt: str
    citations_required: bool = Field(default=True)


class RerankerSettings(BaseModel):
    """Cross-encoder reranker configuration."""

    enabled: bool = Field(default=True)
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    top_k: int = Field(default=5, ge=1)


class AppConfig(BaseModel):
    """Aggregate configuration object loaded from YAML."""

    app: Dict[str, Any] = Field(default_factory=lambda: {"name": "Modular RAG Console"})
    paths: PathSettings = Field(default_factory=PathSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    modules: ModuleToggles = Field(default_factory=ModuleToggles)
    rewriter: RewriterSettings
    hyde: HydeSettings
    generator: GeneratorSettings
    reranker: RerankerSettings

    model_config = ConfigDict(validate_assignment=True)

    def materialize(self) -> None:
        """Ensure required directories exist and derived fields are populated."""
        self.paths.ensure_directories()
        self.logging.ensure_directories()
        if self.vector_store.persist_directory is None:
            self.vector_store.persist_directory = self.paths.vector_store_dir


class ConfigManager:
    """Singleton-style configuration loader."""

    _instance: ClassVar[Optional["ConfigManager"]] = None
    _lock: ClassVar[Lock] = Lock()

    def __init__(self, settings: AppConfig, source_path: Path) -> None:
        """Initialize the manager with validated settings."""
        self._settings = settings
        self._source_path = source_path

    @classmethod
    def load(cls, path: Optional[Path | str] = None) -> "ConfigManager":
        """Load configuration from disk if needed and return the shared instance."""
        cfg_path = Path(path) if path else Path("config/settings.yaml")
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        if cls._instance is None or cls._instance._source_path != cfg_path:
            with cls._lock:
                settings = cls._read_settings(cfg_path)
                cls._instance = cls(settings=settings, source_path=cfg_path)
        return cls._instance

    @property
    def settings(self) -> AppConfig:
        """Return the cached application settings."""
        return self._settings

    def reload(self) -> None:
        """Force reloading the configuration file."""
        with self._lock:
            self._settings = self._read_settings(self._source_path)

    @classmethod
    def _read_settings(cls, path: Path) -> AppConfig:
        """Parse YAML file and return a validated AppConfig instance."""
        data = cls._load_yaml(path)
        try:
            settings = AppConfig.model_validate(data)
        except ValidationError as exc:
            raise ValueError(f"Invalid configuration: {exc}") from exc
        settings.materialize()
        return settings

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load YAML content from disk with safety checks."""
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError("Top-level YAML structure must be a mapping.")
        return data


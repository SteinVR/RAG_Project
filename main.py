"""Console entry point for the Modular RAG application."""

from __future__ import annotations

import json
from typing import Iterable

from dotenv import load_dotenv

from src.core.config import ConfigManager
from src.core.pipeline import PipelineResult, RAGPipeline
from src.utils.console import ConsoleProgress
from src.utils.logger import configure_logging, get_logger


EXIT_COMMANDS = {"exit", "quit", "q"}


def format_answer(result: PipelineResult) -> str:
    """Format the pipeline output for CLI display."""
    answer_lines = [result.answer.answer_markdown.strip()]
    if result.answer.citations:
        answer_lines.append("\nCitations:")
        for citation in result.answer.citations:
            answer_lines.append(
                f"- {citation.document} (p.{citation.page}): {citation.text}",
            )
    if getattr(result.answer, "relevant_sources", None):
        answer_lines.append("\nRelevantSources = [")
        for source in result.answer.relevant_sources:
            payload = {
                "source": source.source,
                "page": source.page,
                "snippet": source.snippet.replace("\n", " "),
            }
            answer_lines.append(f"  {json.dumps(payload, ensure_ascii=False)}")
        answer_lines.append("]")
    return "\n".join(answer_lines)


def iter_cli_inputs() -> Iterable[str]:
    """Yield user inputs until they exit."""
    while True:
        try:
            user_input = input("\n>>> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        yield user_input


def main() -> None:
    """CLI bootstrap."""
    progress = ConsoleProgress()
    
    print("=" * 60)
    print("Modular RAG Console")
    print("=" * 60)
    
    with progress.stage("Loading environment"):
        load_dotenv()
    
    with progress.stage("Loading configuration"):
        config_manager = ConfigManager.load()
        settings = config_manager.settings
        configure_logging(settings)
        logger = get_logger("CLI")
    
    with progress.stage("Initializing pipeline"):
        pipeline = RAGPipeline(settings, show_progress=True)
    
    progress.info("Ready! Type your question or 'exit' to quit")
    print("=" * 60)
    
    for user_input in iter_cli_inputs():
        if user_input.lower() in EXIT_COMMANDS:
            print("\nGoodbye!")
            break
        try:
            print()
            result = pipeline.execute(user_input)
            print("\n" + "=" * 60)
            print(format_answer(result))
            print("=" * 60)
        except Exception as exc:  # pragma: no cover - interactive safeguard
            logger.exception("Pipeline execution failed: %s", exc)
            print("An error occurred. Check logs/prototype.log for details.")


if __name__ == "__main__":
    main()


"""Console entry point for the Modular RAG application."""

from __future__ import annotations

from typing import Iterable

from dotenv import load_dotenv

from src.core.config import ConfigManager
from src.core.pipeline import PipelineResult, RAGPipeline
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
    load_dotenv()
    config_manager = ConfigManager.load()
    settings = config_manager.settings
    configure_logging(settings)
    logger = get_logger("CLI")
    pipeline = RAGPipeline(settings)
    print("Modular RAG Console (type 'exit' to quit)")
    for user_input in iter_cli_inputs():
        if user_input.lower() in EXIT_COMMANDS:
            print("Goodbye!")
            break
        try:
            result = pipeline.execute(user_input)
            print(format_answer(result))
        except Exception as exc:  # pragma: no cover - interactive safeguard
            logger.exception("Pipeline execution failed: %s", exc)
            print("An error occurred. Check logs/prototype.log for details.")


if __name__ == "__main__":
    main()


"""Console progress utilities for visual feedback during pipeline execution."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Generator, Optional


class ConsoleProgress:
    """Utility for displaying timed progress stages in the console."""

    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled

    @contextmanager
    def stage(self, name: str, details: Optional[str] = None) -> Generator[None, None, None]:
        """
        Context manager that displays a stage name, executes code, and prints elapsed time.

        Args:
            name: Stage name to display (e.g., "Query Rewriting")
            details: Optional additional info to display alongside the stage name

        Example:
            >>> progress = ConsoleProgress()
            >>> with progress.stage("Loading documents", "from disk"):
            ...     load_documents()
            [▶] Loading documents (from disk)...
            [✓] Loading documents completed in 1.23s
        """
        if not self._enabled:
            yield
            return

        stage_label = name
        if details:
            stage_label = f"{name} ({details})"

        self._print_start(stage_label)
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self._print_complete(name, elapsed)

    def _print_start(self, stage: str) -> None:
        print(f"[▶] {stage}...", flush=True)

    def _print_complete(self, stage: str, elapsed: float) -> None:
        print(f"[✓] {stage} completed in {elapsed:.2f}s", flush=True)

    def info(self, message: str) -> None:
        """Print an informational message if progress display is enabled."""
        if self._enabled:
            print(f"[ℹ] {message}", flush=True)

    def warning(self, message: str) -> None:
        """Print a warning message if progress display is enabled."""
        if self._enabled:
            print(f"[⚠] {message}", file=sys.stderr, flush=True)


# Modular RAG Console

Retrieval-Augmented Generation console app that assembles a configurable pipeline (Rewriter → HyDE → Retriever → Reranker → Generator) around a persistent local Chroma vector store.

## Features
- Config-driven pipeline defined in `config/settings.yaml`.
- Incremental ingestion with checksum-based file registry to avoid duplicate chunks.
- Optional Query Rewriter, HyDE bootstrapper, and Cross-Encoder reranker.
- Streaming-ready integration with Google Gemini via the native `google-genai` SDK.
- Instrumented logging (`logs/prototype.log`) for observability.

## Project Layout
```
├── config/                # Settings, API key template
├── data/docs/             # Source documents to ingest
├── data/vector_store/     # Chroma persistence
├── src/
│   ├── core/              # Config + pipeline orchestrator
│   ├── ingestion/         # Loader + vector store manager
│   ├── modules/           # Rewriter, HyDE, Reranker, Generator
│   └── utils/             # Logging helpers
├── main.py                # CLI entry point
├── pyproject.toml         # Dependency graph (managed by uv)
└── requirements.txt       # Convenience shim (`pip install -r requirements.txt`)
```

## Environment Setup (uv)
1. **Install uv** (one time):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Sync dependencies** (reads `pyproject.toml`):
   ```bash
   uv sync
   ```
   This creates a virtual environment under `.venv` (managed by uv).
3. **Configure API keys**:
   - Ensure the environment variable is exported before running (`uv run` automatically loads `.env` if you `export $(cat .env)` or use your shell RC).
4. **Prepare data**:
   - Drop PDFs/TXTs/MDs into `data/docs/`.
   - First run will chunk + index them into `data/vector_store/`.
5. **Launch the CLI**:
   ```bash
   uv run python main.py
   ```
   Type your questions; use `exit`/`quit` to stop.

### Updating or Adding Dependencies
- Edit the `dependencies` list in `pyproject.toml`.
- Re-run `uv sync` to lock + install the updated graph.
- The minimal `requirements.txt` simply points to the project (`-e .`) for `pip` compatibility, but uv should be the primary workflow.

## Configuration Highlights
- `config/settings.yaml` toggles modules (`modules.*`), adjusts chunk sizes, and selects Gemini model names.
- Logging format/location lives under `logging.*`.
- Change HyDE, prompt, or reranker settings without touching code.

## Running the Pipeline
1. `main.py` boots, loads config, configures logging, and syncs the vector store.
2. Each query runs through:
   - Query Rewriter (classification + rewrite) – optional.
   - HyDE generator – optional hypothetical answer for better recall.
   - Retriever (Chroma) merges base + HyDE queries and deduplicates results.
   - Reranker (BAAI/bge-reranker-v2-m3) scores top documents – optional.
   - Generator formats the final Markdown answer with citations.
3. Output prints to the console; detailed traces are written to `logs/prototype.log`.

## Next Steps
- After environment setup, consider scripting evaluation prompts once test data and API keys are ready.
- For deployment or batch workloads, the `RAGPipeline` in `src/core/pipeline.py` can be reused directly without the CLI.


# Modular RAG Console

Retrieval-Augmented Generation console app built around a configurable pipeline (Rewriter → HyDE → Retriever → Reranker → Generator) and a persistent Chroma vector store.

## Environment Setup
1. Install uv  
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Sync dependencies  
   ```bash
   uv sync
   ```
3. Install PyTorch (pick one)
   - Apple Silicon  
     ```bash
     uv pip install torch torchvision torchaudio
     ```  
   - NVIDIA CUDA  
     ```bash
     uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
     ```  
   - CPU only  
     ```bash
     uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     ```
4. Verify device mapping (optional)  
   ```bash
   uv run python -m src.utils.check_device
   ```
5. Set Google Gemini credentials  
   ```bash
   export GOOGLE_API_KEY="your-api-key"
   ```
6. Drop PDFs/TXTs/MDs in `data/docs/`.
7. Launch  
   ```bash
   uv run python main.py
   ```

## Project Layout
```
├── config/
│   └── settings.yaml      # Pipeline config, model settings, API keys
├── data/
│   ├── docs/              # Source documents (PDFs, TXTs, MDs)
│   ├── vector_store/      # Chroma persistence (auto-created)
│   └── file_registry.json # Incremental indexing state (auto-created)
├── logs/
│   ├── prototype.log      # Application logs
│   └── pipeline/          # Query execution artifacts (JSON + TXT)
├── src/
│   ├── core/
│   │   ├── config.py      # Pydantic config loader
│   │   └── pipeline.py    # RAGPipeline orchestrator
│   ├── ingestion/
│   │   ├── loader.py      # DocumentLoader (PDF/TXT/MD)
│   │   └── vector_store.py # VectorStoreManager, FileRegistry
│   ├── modules/
│   │   ├── base.py        # PipelineModule, LLMClientFactory
│   │   ├── generator.py   # Final answer generator
│   │   ├── hyde.py        # HyDE hypothetical document generator
│   │   ├── parent_retriever.py # Chunk-to-page reconstruction
│   │   ├── reranker.py    # CrossEncoder reranker
│   │   └── rewriter.py    # Query rewriter
│   └── utils/
│       ├── check_device.py # Hardware detection utility
│       ├── console.py     # Progress display
│       ├── device.py      # Device resolution logic
│       ├── logger.py      # Logging setup
│       └── pipeline_logger.py # Structured pipeline logging
├── tests/                 # Pytest test suite
├── main.py                # CLI entry point
├── pyproject.toml         # Project metadata + dependencies (uv)
└── uv.lock                # Lockfile (uv-managed)
```

## Pipeline Overview
- **Ingestion Layer**: `DocumentLoader` parses PDFs/TXTs/MDs; `VectorStoreManager` chunks content, stores embeddings in Chroma, and uses a checksum registry to skip unchanged files.
- **Retrieval Layer**: Base retriever pulls nearest chunks; HyDE prompt can seed an auxiliary search; deduplication merges both result sets.
- **Parent Page Retriever (optional)**: Replaces chunk lists with full source pages before reranking.
- **Reranker**: Cross-Encoder scores remaining candidates.
- **Generator**: Gemini-powered answer synthesizer that cites sources.

### Execution Flow
1. `main.py` loads config and syncs the vector store.
2. For each query, the pipeline conditionally runs Query Rewriter, HyDE, Retrieval, Parent Page Retriever, Reranker, and Generator.
3. Responses stream to the console; detailed traces land in `logs/prototype.log` and per-query files under `logs/pipeline/`.

## Configuration Highlights
- `config/settings.yaml` toggles modules (`modules.rewriter`, `modules.hyde`, `modules.reranker`, `modules.parent_page_retriever`) and defines chunk sizes, batch sizes, and prompts.
- `embeddings.device` selects `cpu`, `mps`, `cuda`, or `auto`.
- HyDE, Rewriter, and Generator prompts live entirely in config for quick experimentation.
- Logging format and destination are declared under `logging.*`.

## Operations
- Run `uv run python -m src.utils.check_device` whenever hardware changes.
- `data/file_registry.json` tracks source checksums; deleting it forces a full reindex.
- `logs/prototype.log` captures high-level events, while `PipelineLogger` writes per-query JSON/text transcripts for regression analysis.
- Update dependencies via `pyproject.toml` + `uv sync`.

## Features
- Config-driven, switchable pipeline defined in YAML.
- Incremental ingestion with persistent Chroma vector store.
- Optional Query Rewriter, HyDE bootstrapper, Parent Page Retriever, and Cross-Encoder reranker.
- Gemini-based generator with citation enforcement.

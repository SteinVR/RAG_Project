# Architecture: Modular RAG Console Application

## 1. Project Idea & Philosophy
The goal is to refactor a legacy monolithic RAG script into a **modular, configurable console application**. 
The system follows a "Constructor" philosophy: the user defines the pipeline structure (Rewriter, HyDE, Reranker, etc.) via a configuration file, and the application assembles itself at runtime.
It is designed for **local usage**, efficiently handling document indexing, persistence (avoiding re-indexing), and interactive question-answering.

## 2. Body (Form Factor)
- **Type**: Console Application (CLI).
- **Interface**: Interactive command-line loop.
- **Configuration**: `config/settings.yaml`.
- **Input**: Documents (PDF, TXT, MD) in a local folder.
- **Output**: Text answers in the console (JSON/Markdown).

## 3. User Workflow
1.  **Setup**: User places documents in `data/docs/` and adjusts `config/settings.yaml` (toggling modules like HyDE or Reranker).
2.  **Initialization**: User runs `python main.py`.
3.  **Loading/Indexing**:
    - System loads configuration.
    - System checks `data/vector_store/` for an existing index.
    - **Incremental Indexing**: System scans `data/docs/`.
        - If new files are found (not in DB), they are processed (chunked, embedded) and added.
        - If the DB is missing, a full build is triggered.
    - System initializes enabled modules (LLM, Embeddings, Reranker).
4.  **Interaction Loop**:
    - System displays "Ready" prompt.
    - User types a question.
    - **Pipeline Execution**:
        1.  *Query Processing*: Query is optionally rewritten/classified (if enabled).
        2.  *Retrieval*:
            - Base retrieval finds candidate chunks.
            - (Optional) HyDE generates hypothetical answer -> embedded -> searched.
        3.  *Post-Processing*:
            - Results are deduplicated.
            - (Optional) Reranker scores and sorts results.
        4.  *Generation*: Top context chunks are sent to LLM to generate the final answer.
    - System prints the answer and citations.
5.  **Termination**: User types `exit` or `quit`.

## 4. Technical Decisions
- **Language**: Python 3.10+.
- **Framework**: LangChain (Core/Community) for orchestration.
- **Vector Database**: **ChromaDB** (Local, Persistent).
- **Embeddings**: **HuggingFace** (`BAAI/bge-m3` or similar from config).
- **LLM**: **Google Gemini** (via `langchain-google-genai`).
- **Reranker**: **SentenceTransformers** (`CrossEncoder`).
- **Configuration**: **YAML** (via `PyYAML`) + Pydantic Settings.
- **Persistence Strategy**:
    - ChromaDB persists vectors to disk.
    - A localized `KVStore` (e.g., simple JSON/Pickle or SQL wrapper) persists the mapping of Document IDs to filenames to support incremental indexing (checking what's already indexed).

## 5. Component Design

### A. Core Infrastructure (`src/core/`)
- **`ConfigManager`**: Singleton that validates and provides settings from `settings.yaml`.
- **`Pipeline`**: The orchestrator that holds references to active modules and runs the `execute(query)` flow.

### B. Data Layer (`src/ingestion/`)
- **`DocumentLoader`**: Scans directories, parses PDFs/TXTs into `Document` objects.
- **`VectorStoreManager`**:
    - Manages ChromaDB instance.
    - Handles **Incremental Indexing**:
        - `get_indexed_files()`: Returns list of already indexed source files.
        - `add_documents()`: Adds new docs.
    - *Note*: We will simplify from `ParentDocumentRetriever` to standard `RecursiveCharacterTextSplitter` + Metadata for MVP modularity, unless Parent Retrieval is strictly required. *Decision: Stick to standard retrieval for robustness in this refactor, or use Chroma's native metadata for source tracking.*

### C. RAG Modules (`src/modules/`)
All modules should inherit from a base interface where applicable.
- **`QueryRewriter`**: Uses LLM to classify/rewrite queries.
- **`HyDEGenerator`**: Uses LLM to hallucinate an answer for retrieval.
- **`Reranker`**: Uses Cross-Encoder to re-score retrieved docs.
- **`Generator`**: The final RAG synthesizer (Context + Question -> Answer).

### D. Main Entry (`main.py`)
- Handles CLI arguments (optional).
- Initializes `ConfigManager`.
- Calls `VectorStoreManager` to ensure data is ready.
- Instantiates `Pipeline`.
- Runs the `while True` input loop.

## 6. Directory Structure
```
project_root/
├── config/
│   └── settings.yaml
├── data/
│   ├── docs/             # Input files
│   ├── vector_store/     # ChromaDB persistence
│   └── file_registry.json # Tracks indexed files
├── src/
│   ├── core/
│   │   ├── config.py
│   │   └── pipeline.py
│   ├── ingestion/
│   │   ├── loader.py
│   │   └── vector_store.py
│   ├── modules/
│   │   ├── generator.py
│   │   ├── hyde.py
│   │   ├── reranker.py
│   │   └── rewriter.py
│   └── utils/
│       └── logger.py
├── main.py
└── requirements.txt
```

## 7. Key Conventions & Logging

- **Modularity:** Code must be logically separated into classes and modules within `src/`.
- **Logging:** All significant events must be logged to `logs/prototype.log`.
    - **Format:** `[YYYY-MM-DD HH:MM:SS] [LEVEL] - Message`
- **Tools:** Reusable scripts created during development should be saved in `AGENT_TOOLS/`.

## 8. Legacy Code Analysis & Migration Strategy

The legacy code (`legacy/RAG_Prototype_6.1.py`) contains valuable logic but suffers from monolithic structure and lack of persistence. The Lead Engineer is encouraged to **adapt** working snippets but **rewrite** the architecture.

### A. What to Adapt (Reuse)
- **Document Processing**: The `fitz` (PyMuPDF) logic in `DocumentProcessor` is efficient. Adapt this into `src/ingestion/loader.py`.
- **Pydantic Models**: `Citation` and `AnswerSchema` are well-defined. Reuse in `src/modules/generator.py`.
- **Prompts**: The `rewrite_prompt` and system messages are tuned. Move these to `config/settings.yaml` or constants in modules.
- **Deduplication**: The `deduplicate` helper function logic is valid.
- **Reranker Logic**: The `CrossEncoder` implementation is standard. Adapt into `src/modules/reranker.py`.

### B. What to Rewrite (Do Not Copy)
- **ParentDocumentRetriever + InMemoryStore**:
    - *Issue*: The legacy code uses `InMemoryStore` for parent documents, which wipes data on restart, breaking the "persistence" requirement.
    - *Solution*: **Rewrite retrieval.** Use standard `Chroma` persistence. If "Parent Retrieval" is needed, store parent text in metadata or implement a persistent DocStore (e.g., SQLite). For MVP, standard chunk retrieval is preferred.
- **Global State & Constants**:
    - *Issue*: `DATA_DIR`, `USE_HYDE`, etc., are global variables.
    - *Solution*: **Rewrite.** All configuration must come from `ConfigManager` (loaded from `settings.yaml`).
- **Monolithic Control Flow**:
    - *Issue*: `dynamic_retrieve` mixes routing, retrieval, HyDE, and reranking in one massive function.
    - *Solution*: **Refactor.** Split into distinct pipeline steps in `src/core/pipeline.py`.
- **Incremental Logic**:
    - *Issue*: Legacy code often recreates the DB (`recreate=True` or manual deletion).
    - *Solution*: **Implement New Logic.** Use a file registry (JSON/DB) to compare file hashes/names and only index *new* files.

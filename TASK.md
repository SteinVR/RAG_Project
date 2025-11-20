# Task Log & Backlog: RAG_Project

## 1. Feature Backlog

### Phase 1: Foundation & Configuration
- [ ] **Project Setup**: Create directory structure (`src/`, `config/`, `data/`).
- [ ] **Dependencies**: Create `requirements.txt` based on legacy imports and new architecture.
- [ ] **Configuration**:
    - [ ] Create `config/settings.yaml` template.
    - [ ] Implement `src/core/config.py` (Pydantic settings).
    - [ ] Implement `src/utils/logger.py`.

### Phase 2: Ingestion & Persistence
- [ ] **Document Loader**:
    - [ ] Implement `src/ingestion/loader.py` (PDF/TXT support using `fitz`/`langchain`).
- [ ] **Vector Store**:
    - [ ] Implement `src/ingestion/vector_store.py`.
    - [ ] Implement ChromaDB initialization with persistence.
    - [ ] Implement **Incremental Indexing logic** (check existing sources vs new files).

### Phase 3: RAG Modules
- [ ] **Base Interfaces**: Define interfaces for pipeline components if necessary.
- [ ] **Generator**: Implement `src/modules/generator.py` (Gemini integration).
- [ ] **Rewriter**: Implement `src/modules/rewriter.py` (Rewrite guard + HyDE directive).
- [ ] **HyDE**: Implement `src/modules/hyde.py`.
- [ ] **Reranker**: Implement `src/modules/reranker.py` (CrossEncoder).

### Phase 4: Orchestration & CLI
- [ ] **Pipeline**:
    - [ ] Implement `src/core/pipeline.py` to stitch modules together based on config.
- [ ] **Main Application**:
    - [ ] Implement `main.py` (Entry point, initialization, interaction loop).


### Phase 5: Polish
- [x] **Testing**: Verify toggling modules in `settings.yaml`.
- [x] **Documentation**: Add usage instructions to README (optional).
- [x] **Environment Optimization**: Multi-platform GPU acceleration support (M1 MPS + NVIDIA CUDA).

---

<!-- Updated by the User/Lead Engineer to focus effort -->
## 2. Current Task in Focus

> Purpose: The single active task being worked on right now.

> **Current Task:** Refresh prompts + rewriter behavior for RAG modules.

---

<!-- Owned by Lead Engineer -->
## 4. Implementation Plan (Scratchpad)

> Purpose: The Lead Engineer's dynamic plan for the Current Task in Focus. This section will be cleared and re-written for each new task 
from the backlog.
> 
- [x] Audit existing prompts and module flow (rewriter, HyDE, generator)
- [x] Design improved prompt briefs + JSON schemas per module
- [x] Update code/config to use new prompts and dual-mode rewriting
- [x] Adjust pipeline logic for HyDE-aware rewrite decisions
- [x] Smoke test modules and document rationale

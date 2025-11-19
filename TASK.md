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
- [ ] **Rewriter**: Implement `src/modules/rewriter.py` (Classification + Rewrite logic).
- [ ] **HyDE**: Implement `src/modules/hyde.py`.
- [ ] **Reranker**: Implement `src/modules/reranker.py` (CrossEncoder).

### Phase 4: Orchestration & CLI
- [ ] **Pipeline**:
    - [ ] Implement `src/core/pipeline.py` to stitch modules together based on config.
- [ ] **Main Application**:
    - [ ] Implement `main.py` (Entry point, initialization, interaction loop).


### Phase 5: Polish
- [ ] **Testing**: Verify toggling modules in `settings.yaml`.
- [ ] **Documentation**: Add usage instructions to README (optional).

---

<!-- Updated by the User/Lead Engineer to focus effort -->
## 2. Current Task in Focus

> Purpose: The single active task being worked on right now.

> **Current Task:** Phase 4: Orchestration & CLI, Phase 5: Polish

---

<!-- Owned by Lead Engineer -->
## 4. Implementation Plan (Scratchpad)

> Purpose: The Lead Engineer's dynamic plan for the Current Task in Focus. This section will be cleared and re-written for each new task 
from the backlog.
> 
- [x] Phase 4: implement `src/core/pipeline.py` orchestrating ingestion, modules, and retrieval
- [x] Phase 4: add `main.py` CLI wiring + logging/bootstrap
- [x] Phase 5: document env setup, add uv/pyproject + README + API key template

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
- [ ] **Parent Page Retriever**: Implement `src/modules/parent_retriever.py` (Replace chunks with full pages).
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

## Feature Request: Parent Page Retriever Module

### Objective
Implement an optional module that replaces retrieved chunks with their full parent pages before reranking. This provides richer context to the LLM at the cost of larger input size.

### Implementation Tasks

#### Task 1: Extend Ingestion to Store Parent Content
- [ ] Modify `src/ingestion/vector_store.py`:
    - [ ] Check if `parent_page_retriever` is enabled in config.
    - [ ] If enabled, before chunking each parent Document (page), store its full content.
    - [ ] After chunking, attach parent metadata to each chunk:
        - `parent_page_content`: Full text of the parent page.
        - `parent_doc_id`: `doc_id` of the parent Document.
        - `parent_source`: Source filename.
        - `parent_page`: Page number.
    - [ ] Ensure these fields are added to chunk metadata before indexing.

#### Task 2: Create Parent Page Retriever Module
- [ ] Create `src/modules/parent_retriever.py`:
    - [ ] Inherit from `PipelineModule` (base class).
    - [ ] Implement `retrieve_parent_pages(chunks: List[Document]) -> List[Document]`:
        - [ ] Group chunks by `(parent_source, parent_page)`.
        - [ ] For each unique page, create a new Document:
            - `page_content` = `parent_page_content` from any chunk of that page.
            - `metadata` = `{"source": parent_source, "page": parent_page, "doc_id": parent_doc_id}`.
        - [ ] Return list of parent Documents.
    - [ ] Implement `is_enabled()` to check `config.modules.parent_page_retriever`.

#### Task 3: Integrate into Pipeline
- [ ] Modify `src/core/pipeline.py`:
    - [ ] Import `ParentPageRetriever`.
    - [ ] Instantiate `self._parent_retriever` in `__init__`.
    - [ ] In `_retrieve_documents()` method:
        - [ ] After deduplication and before reranking, check if parent retriever is enabled.
        - [ ] If enabled, call `self._parent_retriever.retrieve_parent_pages(combined)`.
        - [ ] Pass result to reranker (or generator if reranker is disabled).
    - [ ] Update pipeline logging to reflect parent page retrieval step.

#### Task 4: Configuration
- [ ] Add to `config/settings.yaml`:
    ```yaml
    modules:
      parent_page_retriever: false  # Optional: Replace chunks with full pages
    ```
- [ ] Update `src/core/config.py`:
    - [ ] Add `parent_page_retriever: bool = False` to `ModulesConfig` class.

#### Task 5: Testing & Validation
- [ ] Test with `parent_page_retriever: true`:
    - [ ] Verify chunks are replaced with full pages.
    - [ ] Check that Reranker scores pages, not chunks.
    - [ ] Ensure Generator receives correct context.
- [ ] Test with `parent_page_retriever: false`:
    - [ ] Verify default chunk-based behavior is unchanged.
- [ ] Compare answer quality and context size for both modes.

---

<!-- Updated by the User/Lead Engineer to focus effort -->
## 2. Current Task in Focus

> Purpose: The single active task being worked on right now.

> **Current Task:** Implement Parent Page Retriever module (optional chunk-to-page replacement before reranking).

---

<!-- Owned by Lead Engineer -->
## 4. Implementation Plan (Scratchpad)

> Purpose: The Lead Engineer's dynamic plan for the Current Task in Focus. This section will be cleared and re-written for each new task 
from the backlog.
> 
- [x] Review architecture + existing pipeline code for parent metadata requirements
- [x] Extend config + ingestion to persist parent page metadata on chunks
- [x] Implement `ParentPageRetriever` module with grouping logic
- [x] Wire module into pipeline execution + logging
- [x] Run e2e test with parent retriever on/off, gather logs, write report

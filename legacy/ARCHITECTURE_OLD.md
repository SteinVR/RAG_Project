# Architecture of an LLM-Based Intelligent Assistant

## 1. Introduction

This document describes the architecture of an LLM-based intelligent assistant aimed at optimizing manufacturing processes. The system is designed to consult on standards (ISO and others) and internal documentation, with a focus on answer accuracy and providing source citations.

Development will proceed iteratively, starting with a basic Retrieval-Augmented Generation (RAG) pipeline (MVP) and gradually adding more advanced modules to enhance quality and functionality.

## 2. Overall Architecture and Components

The system consists of two main parts:

* **Data Ingestion Pipeline:** Responsible for preparing and loading documents into the knowledge base.
* **Query Processing Pipeline:** Accepts a user query, finds relevant information, and generates a response.

### 2.1. Data Ingestion Pipeline

This pipeline is triggered when new documents are added or existing ones are updated.

1. **Document Loader:**
   * Reads files in various formats (PDF, TXT, MD for the MVP).
2. **Document Parser:**
   * Extracts textual content from the loaded documents.
   * For PDF: extracts text page-by-page to retain page numbers.
3. **Text Chunker:**
   * Splits the extracted text into smaller, manageable fragments (chunks).
   * **Key aspect:** Each chunk must inherit metadata from its source (file name, page number).
4. **Embedding Model:**
   * Converts text chunks into numerical vectors (embeddings).
5. **Vector Database:**
   * Stores chunk embeddings and their metadata.
   * Provides efficient semantic-similarity search.

### 2.2. Query Processing Pipeline (RAG Pipeline)

#### 2.2.1. Basic MVP RAG Pipeline

```
User Query
      |
      v
[Query Embedding Module] -- (Embedding Model)
      |
      v
[Vector DB Search Module] -- (Vector Database)
      | (Retrieved chunks + metadata)
      v
[Context Assembler Module]
      | (Context for LLM)
      v
[LLM Generation Module] -- (LLM API/Local Model + Prompt)
      | (LLM answer + citation info)
      v
[Output Formatting Module]
      |
      v
Answer to User with Citations
```

* **Query Embedding Module:** Vectorizes the user query using the same model as for the documents.
* **Vector DB Search Module:** Finds the N most relevant chunks in the Vector DB.
* **Context Assembler Module:** Builds a context string from the retrieved chunks for the LLM.
* **LLM Generation Module:** Sends the user query and assembled context to the LLM to generate an answer. The prompt must instruct the LLM to use the context and reference the sources.
* **Output Formatting Module:** Extracts from the LLM response the answer text and the citation information (based on metadata passed in the context).

#### 2.2.2. Extended ("Ideal") RAG Pipeline

This pipeline is built by adding modules to the basic one.

```
User Query (+ Dialogue History)
       |
       v
[Query Understanding & Rewriting Agent (LLM + System Prompt)]  // Step 0 from earlier discussion
       | (Classified and/or rewritten query, strategy)
       v
[Strategy: HyDE / Direct Search]
       |
       v
[Query/Hypothetical Doc Embedding Module]
       |
       v
[Initial Retrieval (Vector DB Search on small chunks)]
       | (Top-N small chunks)
       v
[Parent Context Fetching Module]  // Fetch full pages / larger contexts
       | (Unique pages/contexts)
       v
[LLM Reranker Module]  // Re-rank pages/contexts
       | (Top-K relevant pages/contexts)
       v
[Context Assembler Module (with source metadata)]
       |
       v
[LLM Generation Module (CoT + System Prompt, citation instructions)]
       | (JSON with answer, CoT, citations)
       v
[Output Parsing & Validation Module (System-Orchestrated Re-parser)]
       |
       v
Answer to User with Formatted Citations
```

* **Query Understanding & Rewriting Agent:** Analyzes the query, classifies it, rewrites it if necessary, and recommends a strategy (HyDE / direct search).
* **HyDE Module (Hypothetical Document Embeddings):** Generates a hypothetical document to improve search for fuzzy queries.
* **Parent Context Fetching Module:** Retrieves parent pages/contexts for the selected small chunks.
* **LLM Reranker Module:** Uses an LLM to more accurately assess the relevance of retrieved pages/contexts.

### 2.3. Evaluation Module

* Calculates quality metrics for the RAG system (e.g., using RAGAs: faithfulness, answer relevancy, context precision/recall, etc.).
* Will be used to assess the impact of each added module.

## 3. Technology Stack

A detailed selection of libraries will be justified separately. The main categories are:

* Document handling.
* Text splitting.
* Vectorization (embedding models).
* Vector databases.
* LLM interaction (API and local models).
* RAG orchestration (frameworks such as LangChain/LlamaIndex or a custom implementation).
* RAG evaluation.

## 4. Modularity and Experiments

The architecture is designed with modularity in mind. Each block will be implemented as a separate function or class, which allows you to:

* Easily swap components (e.g., different embedding models, different LLMs).
* Enable/disable advanced modules (HyDE, Reranker) to assess their contribution.
* Experiment with various parameters of each module.
* Develop and test parts of the system independently.

Development in `.ipynb` during the initial stages will allow for rapid prototyping and visualization of each step’s results.

## 5. Choice of Libraries and Technologies

1. **Document Handling (Loading and Parsing):**
   * **PDF:**
     * **Library:** `PyMuPDF (fitz)`
     * **Rationale:** A very fast and powerful library for working with PDFs. Makes it easy to extract text page-by-page, access metadata, and work with images (if needed in the future). Has fewer dependencies than some more comprehensive parsers.
   * **TXT:**
     * **Library:** Built-in Python functions (`open().read()`).
     * **Rationale:** Simple and effective.
   * **MD (Markdown):**
     * **Library:** `markdown` (for converting to HTML if structural analysis is needed) or simple reading as TXT (if structure is not important for the MVP). To extract plain text, `mistune` or `markdown-it-py` may be more efficient if you need to remove markup. For the MVP, reading as plain text is likely sufficient.
     * **Rationale:** Simplicity for the MVP.

2. **Text Chunking:**
   * **Library:** `LangChain` (`langchain_text_splitters`)
   * **Class:** `RecursiveCharacterTextSplitter`
   * **Rationale:** `LangChain` provides convenient, well-tested splitters. `RecursiveCharacterTextSplitter` is flexible, attempts to split along semantically meaningful delimiters (paragraphs, sentences), and works well with chunk size and overlap settings. It is the de facto standard for many RAG systems.

3. **Embedding Models:**
   * **Initial phase (API, for development speed):**
     * **Option 1 (if Gemini API offers high-quality embeddings):** `google-genai` (for embeddings from Google).
       * **Rationale:** If you already plan to use Gemini for generation, using their embeddings may be consistent. Need to check their quality on MTEB or similar retrieval benchmarks.
     * **Option 2 (standard, well-studied):** `OpenAI` (`text-embedding-3-small` or `text-embedding-3-large`).
       * **Rationale:** `text-embedding-3-small` offers a good balance of quality and cost, outperforming `ada-002`. `large` is even better but more expensive. Widely used with many examples.
   * **Later phase (local models):**
     * **Library:** `sentence-transformers` (Hugging Face).
     * **Models:**
       * `all-MiniLM-L6-v2` (fast, good quality for its size, excellent for prototyping).
       * `bge-base-en-v1.5` / `bge-large-en-v1.5` (BAAI General Embedding – top open-source models for retrieval).
       * `gte-large` (General Text Embeddings from Alibaba, also among the top).
     * **Rationale:** `sentence-transformers` provides easy access to a huge number of pre-trained models. Model choice depends on the "quality-speed-resources" trade-off. BGE and GTE are current leaders among open models.

4. **Vector Database:**
   * **Library:** `ChromaDB`
   * **Rationale:**
     * **Ease for local development:** Very easy to set up and use in `.ipynb`. Does not require a separate server.
     * **Persistence:** Data are saved to disk between sessions.
     * **Built for RAG:** Has a good API focused on RAG tasks.
     * **Integration:** Integrates well with `LangChain` and `LlamaIndex`.
   * **Alternative:** `FAISS` (from Facebook/Meta).
     * **Rationale:** Extremely fast, especially for search. Excellent if maximum performance is required and data can be kept in memory (or if you are ready to manage index persistence yourself). In `.ipynb` it may be slightly less convenient for out-of-the-box persistence than Chroma.

5. **LLM Interaction (Generation, HyDE, Reranker):**
   * **API (MVP):**
     * **Library:** `google-genai` (for Gemini).
     * **Rationale:** A straightforward choice since you plan to use Gemini.
   * **Local models (later phase):**
     * **Libraries:** `Hugging Face Transformers`, `accelerate`, `bitsandbytes` (for QLoRA/quantization).
     * **Server tools (if needed):** `Ollama`, `vLLM`, `Text Generation Inference (TGI)`.
     * **Rationale:** The standard stack for working with open-source LLMs locally. `Ollama` greatly simplifies running and managing local models.

6. **RAG Orchestration:**
   * **Library:** `LangChain`
   * **Rationale:** Even though you want a modular build, `LangChain` can provide ready-made abstractions for recurring tasks (e.g., working with a VectorStore, prompt formatting, simple chains). You can use its components selectively without building the entire system on it. This can speed up the development of basic parts.
   * **If you prefer full control:** You can do without `LangChain`, using the above libraries directly. For `.ipynb` this is entirely feasible.

7. **RAG Evaluation:**
   * **Library:** `RAGAs`
   * **Rationale:** A specialized framework for evaluating RAG systems. Includes metrics such as faithfulness (faithfulness to context), answer relevancy (answer relevance to the query), context precision/recall, and context relevancy. Can use an LLM for evaluation. Well suited for comparing versions of your pipeline.
   * **Alternative:** `TruLens` (more general for LLM applications).

8. **Supporting libraries:**
   * `python-dotenv`: Safe management of API keys.
   * `pandas` and `numpy`: For possible work with data and evaluation results.
   * `tqdm`: For progress bars during lengthy operations (indexing, evaluation).
   * `ipynb` / `jupyterlab`: Development environment.

**Rationale for the MVP choice (Gemini API, ChromaDB, LangChain components):**
This combination allows you to quickly build a first prototype. The Gemini API removes the complexity of setting up a local LLM at the start. ChromaDB is simple for vector storage. LangChain components (splitters, VectorDB integrations) speed up coding of the basic steps. PyMuPDF provides quality PDF parsing.
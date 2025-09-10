# Architecture of an LLM-Based Intelligent Assistant

## 1. Introduction

This document describes the architecture of an LLM-based intelligent assistant aimed at optimizing manufacturing processes. The system is designed to consult on standards (ISO and others) and internal documentation, with a focus on answer accuracy and providing source citations.

Development will proceed iteratively, starting with a basic Retrieval-Augmented Generation (RAG) pipeline (MVP) and gradually adding more advanced modules to enhance quality and functionality. The current version represents an advanced prototype with features like query rewriting, hypothetical document embeddings, and reranking.

## 2. Overall Architecture and Components

The system consists of two main parts:

* **Data Ingestion Pipeline:** Responsible for preparing and loading documents into the knowledge base.
* **Query Processing Pipeline:** Accepts a user query, finds relevant information, and generates a response.

### 2.1. Data Ingestion Pipeline

This pipeline is triggered when new documents are added or existing ones are updated.

1. **Document Loader:**
   * Reads files in various formats (PDF, TXT, MD).
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

#### 2.2.2. Extended ("Current Advanced") RAG Pipeline

This pipeline is built by adding modules to the basic one, reflecting the `dynamic_retrieve` function in `RAG_Prototype_8.py`.

```
User Query
       |
       v
[Query Understanding & Rewriting Agent (LLM + System Prompt + Pydantic Output)] // Classifies query, rewrites it, decides initial HyDE strategy based on CLASS2HYDE rules.
       | (Classified and/or rewritten query, use_hyde_decision)
       v
[Base Retrieval (Vector DB Search on child chunks, returns parent docs)] // Uses rewritten query, k=BASE_RETRIEVER_K
       | (Top-N parent documents from base search)
       v
[Optional HyDE Augmentation] // If use_hyde_decision is true
       |
       v
   [HyDE Module (HypotheticalDocumentEmbedder + LLM)] // Generates hypothetical document from rewritten query
         | (Hypothetical document embedding)
         v
   [HyDE Retrieval (Vector DB Search on child chunks, returns parent docs)] // k=HYDE_RETRIEVER_K
         | (Top-M parent documents from HyDE search)
         v
[Combine & Deduplicate Documents] // Combines base and HyDE results, removes duplicates by ID
       | (Combined, unique parent documents, limited by RERANKER_INPUT_LIMIT)
       v
[Optional Reranker Module (CrossEncoder)] // If enabled and documents exist
       | (Top-K relevant parent documents, e.g., FINAL_K)
       v
[Context Assembler Module (with source metadata)]
       |
       v
[LLM Generation Module (LLM + System Prompt with citation instructions)] // Supports API (Gemini) and local (Unsloth: Gemma, Qwen3 with thinking mode)
       | (JSON with answer, citations, potentially thinking process for Qwen3)
       v
[Output Parsing & Validation Module (PydanticOutputParser)]
       |
       v
Answer to User with Formatted Citations
```

* **Query Understanding & Rewriting Agent:** Analyzes the user query using an LLM.
    * Classifies the query into types: `specific`, `fuzzy`, `comparative`, `procedural`, `creative`.
    * Rewrites the query for clarity and self-containment.
    * Determines if HyDE should be used based on predefined rules (`CLASS2HYDE` dictionary) overriding the LLM's initial suggestion.
    * Outputs a structured Pydantic object with classification, rewritten query, and HyDE decision.
* **Base Retrieval:** Uses a `ParentDocumentRetriever` with a vector store (ChromaDB) to find documents relevant to the (potentially rewritten) query. Retrieves `BASE_RETRIEVER_K` documents.
* **HyDE Module (Hypothetical Document Embeddings):** If activated by the Query Rewriter's logic and `USE_HYDE` toggle:
    * Uses `langchain.chains.hyde.base.HypotheticalDocumentEmbedder` with an LLM to generate a hypothetical document based on the rewritten query.
    * Embeds this hypothetical document.
* **HyDE Retrieval:** Searches the vector store using the hypothetical document's embedding, retrieving `HYDE_RETRIEVER_K` documents.
* **Combine & Deduplicate Documents:** Merges documents from base retrieval and (if used) HyDE retrieval. Duplicates are removed based on document IDs. The list is then truncated to `RERANKER_INPUT_LIMIT` if it exceeds this.
* **Reranker Module:** If `USE_RERANKER` is true and documents are available:
    * Uses a `sentence-transformers.CrossEncoder` model (e.g., `BAAI/bge-reranker-v2-m3`) to re-score the relevance of the (deduplicated and limited) list of documents against the rewritten query.
    * Selects the top documents (e.g., `FINAL_K * 2` before final slicing to `FINAL_K`) based on these new scores.
* **Context Assembler Module:** Prepares the final set of `FINAL_K` documents for the LLM, formatting them with source metadata.
* **LLM Generation Module:**
    * Uses an LLM (API-based like Gemini, or local models like Unsloth-optimized Gemma/Qwen3) to generate an answer based on the assembled context and the original query.
    * For Qwen3, can optionally include a "thinking process" in the output if `QWEN3_THINKING_MODE` is enabled.
    * The prompt instructs the LLM to provide citations and output in a specific JSON format.
* **Output Parsing & Validation Module:** Uses a `PydanticOutputParser` to parse the LLM's JSON output into a structured `AnswerSchema` object. For Qwen3 with thinking mode, an additional parsing step extracts the thinking content and final answer.
* **Parent Context Fetching Module:** This is implicitly handled by the `ParentDocumentRetriever`, which stores full parent documents and retrieves them based on child chunk matches.

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
* **Reranking models.**
* **Local LLM optimization libraries (e.g., Unsloth).**

## 4. Modularity and Experiments

The architecture is designed with modularity in mind. Each block will be implemented as a separate function or class, which allows you to:

* Easily swap components (e.g., different embedding models, different LLMs, different rerankers).
* Enable/disable advanced modules (Query Rewriter, HyDE, Reranker) to assess their contribution.
* Experiment with various parameters of each module (e.g., `CHUNK_SIZE`, `BASE_RETRIEVER_K`, `HYDE_RETRIEVER_K`, `RERANKER_INPUT_LIMIT`, `FINAL_K`).
* Develop and test parts of the system independently.
* Switch between API-based and local LLMs.

Development in `.ipynb` during the initial stages will allow for rapid prototyping and visualization of each step's results. The current `.py` script format allows for more structured and robust implementation.

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
   * **Parameters:** `CHUNK_SIZE` (e.g., 800), `CHUNK_OVERLAP` (e.g., 150).
   * **Rationale:** `LangChain` provides convenient, well-tested splitters. `RecursiveCharacterTextSplitter` is flexible, attempts to split along semantically meaningful delimiters (paragraphs, sentences), and works well with chunk size and overlap settings. It is the de facto standard for many RAG systems.

3. **Embedding Models:**
   * **Current Local Model:**
     * **Library:** `sentence-transformers` (via `langchain_community.embeddings.HuggingFaceEmbeddings`)
     * **Model:** `BAAI/bge-m3`
     * **Rationale:** `BAAI/bge-m3` is a powerful, multilingual embedding model offering strong performance for retrieval tasks. `sentence-transformers` provides easy access. Normalization is typically enabled for better search quality.
   * **Initial API options (still viable for quick tests/fallback):**
     * **Option 1 (Google):** `google-genai` (for embeddings from Google).
       * **Rationale:** If using Gemini for generation, their embeddings might be consistent.
     * **Option 2 (OpenAI):** `OpenAI` (`text-embedding-3-small` or `text-embedding-3-large`).
       * **Rationale:** `text-embedding-3-small` offers a good balance of quality and cost.
   * **Other Local Model Examples (previously considered/alternative):**
     * `all-MiniLM-L6-v2` (fast, good quality for its size, excellent for prototyping).
     * `bge-base-en-v1.5` / `bge-large-en-v1.5`
     * `gte-large` (General Text Embeddings from Alibaba).
     * **Rationale:** Choice depends on the "quality-speed-resources" trade-off. BGE and GTE are current leaders among open models.

4. **Vector Database:**
   * **Library:** `ChromaDB`
   * **Rationale:**
     * **Ease for local development:** Very easy to set up and use in `.ipynb`. Does not require a separate server.
     * **Persistence:** Data are saved to disk between sessions.
     * **Built for RAG:** Has a good API focused on RAG tasks.
     * **Integration:** Integrates well with `LangChain` and `LlamaIndex`.
   * **Alternative:** `FAISS` (from Facebook/Meta).
     * **Rationale:** Extremely fast, especially for search. Excellent if maximum performance is required and data can be kept in memory (or if you are ready to manage index persistence yourself). In `.ipynb` it may be slightly less convenient for out-of-the-box persistence than Chroma.

5. **LLM Interaction (Generation, Query Rewriting, HyDE):**
   * **API (Primary for Gemini):**
     * **Library:** `langchain-google-genai`
     * **Model:** `gemini-1.5-flash-preview-04-17` (as per `RAG_Prototype_8.py`)
     * **Rationale:** A powerful and cost-effective model from Google, suitable for generation and structured output tasks.
   * **Local models (Advanced Prototype):**
     * **Optimization Library:** `unsloth` (using `FastLanguageModel`)
     * **Integration Library:** `langchain_huggingface.HuggingFacePipeline`
     * **Underlying Library:** `transformers.pipelines`
     * **Models:**
       * `unsloth/gemma-3-4b-it-qat-int4-unsloth-bnb-4bit` (Gemma 3 4B QAT Unsloth optimized)
       * `unsloth/Qwen3-8B-unsloth-bnb-4bit` (Qwen3 8B Dynamic 2.0 optimized)
     * **Features:**
        * Support for 4-bit quantization (`load_in_4bit=True`).
        * Qwen3 can use a "thinking mode" which might affect generation style (configurable via `QWEN3_THINKING_MODE` and specific temperature/top_p settings).
     * **Rationale:** `unsloth` significantly accelerates inference and reduces memory usage for Hugging Face models, making local LLMs more practical. Gemma and Qwen3 are capable open-source models.
   * **Server tools (if scaling beyond direct local loading):** `Ollama`, `vLLM`, `Text Generation Inference (TGI)`.

6. **RAG Orchestration:**
   * **Library:** `LangChain`
   * **Components Used:**
     * `ParentDocumentRetriever`: For efficient retrieval of parent documents.
     * `HypotheticalDocumentEmbedder`: For HyDE.
     * `PydanticOutputParser`: For structured output from LLMs.
     * `ChatPromptTemplate`: For creating flexible prompts.
     * `RunnablePassthrough`, `RunnableLambda`: For custom chain logic.
     * `InMemoryStore`: For temporarily storing parent documents for the retriever.
   * **Rationale:** LangChain provides robust and well-tested components that accelerate the development of complex RAG pipelines. It allows for modular construction while offering abstractions for common tasks like prompt management, output parsing, and integrating various services (vector stores, LLMs).

7. **Reranking:**
    * **Library:** `sentence-transformers`
    * **Class:** `CrossEncoder`
    * **Model Example:** `BAAI/bge-reranker-v2-m3`
    * **Rationale:** Cross-encoder models provide a more accurate relevance score by performing full attention between the query and each document. This step significantly improves the quality of documents fed to the LLM after initial retrieval, especially when the initial retrieval pool (`RERANKER_INPUT_LIMIT`) is larger.

8. **RAG Evaluation:**
   * **Library:** `RAGAs`
   * **Rationale:** A specialized framework for evaluating RAG systems. Includes metrics such as faithfulness (faithfulness to context), answer relevancy (answer relevance to the query), context precision/recall, and context relevancy. Can use an LLM for evaluation. Well suited for comparing versions of your pipeline.
   * **Alternative:** `TruLens` (more general for LLM applications).
   * **Supporting libraries:**
     * `python-dotenv`: Safe management of API keys.
     * `pandas` and `numpy`: For possible work with data and evaluation results.
     * `tqdm`: For progress bars during lengthy operations (indexing, evaluation).
     * `torch`: Required by `sentence-transformers` and `unsloth`.
     * `rich`: For potentially enhanced console output (though not explicitly used in core RAG logic of the script).
     * `ipynb` / `jupyterlab`: Development environment for initial exploration.

**Rationale for the Current Advanced Prototype Choices:**
This setup aims to maximize answer quality and flexibility by incorporating several advanced RAG techniques:
*   **Multi-stage Retrieval:**
    *   **Query Rewriting:** Improves the clarity and effectiveness of the initial query.
    *   **HyDE (Optional):** Addresses fuzzy queries by searching for a generated hypothetical answer.
    *   **Parent Document Retrieval:** Ensures that while search is performed on smaller chunks, the LLM receives context from larger, more coherent parent documents (pages).
    *   **Reranking (Cross-Encoder):** Provides a fine-grained re-assessment of relevance for documents retrieved in earlier stages, leading to a higher quality context for the LLM.
*   **Flexible LLM Backend:** Supports both powerful API-based models (Gemini) for robust generation and Unsloth-optimized local models (Gemma, Qwen3) for experimentation, cost-saving, or offline use. Qwen3's thinking mode offers another dimension for generation style.
*   **High-Quality Embeddings and Rerankers:** Uses state-of-the-art open-source models like `BAAI/bge-m3` for embeddings and `BAAI/bge-reranker-v2-m3` for reranking.
*   **Structured Output:** Employs Pydantic models and LangChain parsers to ensure reliable and structured JSON output from the LLM, which is crucial for citation accuracy and downstream processing.
*   **Modularity:** LangChain components and custom functions allow for individual modules to be enabled, disabled, or swapped for experimentation and evaluation.
*   **Detailed Logging:** Integration of a detailed logger (`rag_detailed_logger.py`) allows for in-depth analysis of each step in the RAG pipeline.

ChromaDB remains a good choice for its ease of use and persistence in local development. `PyMuPDF` is effective for PDF parsing. The combination of these tools, orchestrated by LangChain and custom Python logic, creates a powerful and adaptable RAG system.
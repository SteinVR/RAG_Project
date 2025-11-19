# !pip install langchain==0.1.16 langchain-community==0.0.26 langchain-google-genai==0.0.10 \
#     chromadb==0.4.24 python-dotenv PyMuPDF rich pydantic

import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from sentence_transformers import CrossEncoder
import torch

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
logger = logging.getLogger("RAG_SO")

#-------------------------------------------------------------------------------
# Markdown Cell:
# ## Configuration
#-------------------------------------------------------------------------------

DATA_DIR = Path('docs')         # Place your documents here
CHROMA_DIR = Path('chroma_db_mvp')   # Chroma persistence directory
# COLLECTION_NAME = 'chroma_rag_mvp'
COLLECTION = 'rag_demo_collection'

# Text splitting
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RETRIEVER_K = 10

EMBEDDING_MODEL = "BAAI/bge-m3"  # Локальная мультиязычная модель
LLM_MODEL = 'gemini-2.5-flash-preview-04-17'

# === Feature toggles ====================================================== #
USE_QR        = True    # Query Rewriter & Classifier
USE_HYDE      = True    # HyDE‑style hypothetical doc embeddings
USE_RERANKER  = True    # Cross‑encoder reranking

# === Query Rewriter Rules ================================================= #
# Словарь для контроля использования HyDE в зависимости от типа запроса
CLASS2HYDE = {
    "specific":     False,  # Точные запросы с конкретными терминами
    "fuzzy":        True,   # Нечёткие запросы, требующие контекстного поиска
    "comparative":  False,  # Сравнительные запросы - лучше прямой поиск
    "procedural":   False,  # Процедурные вопросы - нужны точные инструкции
    "creative":     True,   # Творческие запросы - HyDE может помочь
}

# Допустимые типы классификации (для валидации)
ALLOWED_CLASSIFICATIONS = set(CLASS2HYDE.keys())

#-------------------------------------------------------------------------------
# Markdown Cell:
# ## 1 · Structured-output schema
#-------------------------------------------------------------------------------

class Citation(BaseModel):
    """Single citation referring to a fragment in the source docs."""
    document: str = Field(..., description='Filename, e.g. handbook.pdf')
    page: int = Field(..., description='Page number within the document')
    text: str = Field(..., description='Quoted or paraphrased fragment')

class AnswerSchema(BaseModel):
    """Final model answer with citations list."""
    answer_markdown: str = Field(..., description='Full answer in Russian Markdown')
    citations: List[Citation]
    
# LangChain parser that will both produce format instructions and parse the JSON back
answer_parser = PydanticOutputParser(pydantic_object=AnswerSchema)

format_instructions = answer_parser.get_format_instructions()
format_instructions_escaped = format_instructions.replace("{", "{{").replace("}", "}}")

#-------------------------------------------------------------------------------
# Markdown Cell:
# ## 1.1 · Query Rewriter schema & chain (NEW)
#-------------------------------------------------------------------------------

class RewriteSchema(BaseModel):
    """Schema for the query rewriter."""
    classification: str = Field(..., description='Тип запроса из списка: specific|fuzzy|comparative|procedural|creative')
    rewritten_query: str = Field(..., description='Самодостаточный запрос')
    use_hyde: bool      = Field(..., description='Нужно ли применять HyDE (будет перезаписано правилами)')

rewrite_parser = PydanticOutputParser(pydantic_object=RewriteSchema)
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Ты модуль Query Rewriter для RAG-системы. "
     "Классифицируй запрос пользователя, выбрав СТРОГО ОДИН тип из списка: "
     "specific | fuzzy | comparative | procedural | creative.\n\n"
     "Типы запросов:\n"
     "• specific: точные запросы с конкретными терминами, названиями, цифрами\n"
     "• fuzzy: нечёткие запросы, требующие контекстного понимания\n" 
     "• comparative: сравнительные вопросы (что лучше, различия, сходства)\n"
     "• procedural: вопросы о процедурах, инструкциях, как что-то сделать\n"
     "• creative: открытые, исследовательские, творческие вопросы\n\n"
     "Поле use_hyde заполняй согласно правилам: "
     "specific→false, fuzzy→true, comparative→false, procedural→false, creative→true.\n\n"
     "Верни JSON, соответствующий схеме: "
     f"{rewrite_parser.get_format_instructions().replace('{','{{').replace('}','}}')}"),
    ("user", "{query}")
])

#-------------------------------------------------------------------------------
# Markdown Cell:
# ## 2 · Document loading — produce **parent pages**
#-------------------------------------------------------------------------------

class DocumentProcessor:
    """Turn files into `Document`s representing *pages* (parent docs)."""
    SUPPORTED = {'.pdf', '.txt', '.md'}
    
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
    
    def _pdf_pages(self, path: Path) -> List[Document]:
        docs = []
        with fitz.open(path) as pdf:
            for idx, page in enumerate(pdf): # type: ignore
                text = page.get_text('text').strip()
                if text:
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                'id': str(uuid.uuid4()),
                                'source': path.name,
                                'page': idx + 1
                            }
                        )
                    )
        self.log.info("Loaded %s – %d pages", path.name, len(docs))
        return docs
    
    def _text_file(self, path: Path) -> List[Document]:
        text = path.read_text(encoding='utf-8').strip()
        if not text:
            return []
        # Treat entire text file as a single "page"
        return [Document(page_content=text,
                         metadata={
                             'id': str(uuid.uuid4()),
                             'source': path.name,
                             'page': 1
                         })]
    
    def load(self, source: Path | str) -> List[Document]:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        docs = []
        if path.is_file():
            docs.extend(self._dispatch(path))
        else:
            for p in path.rglob('*'):
                if p.is_file():
                    docs.extend(self._dispatch(p))
        if not docs:
            self.log.warning("No supported docs found in %s", path)
        return docs
    
    def _dispatch(self, path: Path) -> List[Document]:
        suff = path.suffix.lower()
        if suff == '.pdf':
            return self._pdf_pages(path)
        if suff in {'.txt', '.md'}:
            return self._text_file(path)
        self.log.debug("Skipping %s", path)
        return []

# Helper to deduplicate documents by ID (NEW)
def deduplicate(docs: List[Document]) -> List[Document]:
    seen = set()
    out  = []
    for d in docs:
        # Ensure metadata and 'id' key exist
        if hasattr(d, 'metadata') and isinstance(d.metadata, dict) and 'id' in d.metadata:
            pid = d.metadata['id']
            if pid not in seen:
                out.append(d)
                seen.add(pid)
        else:
            # Handle docs without proper metadata if necessary, or log a warning
            logger.warning(f"Document missing 'id' in metadata: {d}")
            # Optionally, still add the document if deduplication by ID is not possible
            # out.append(d) 
    return out

#-------------------------------------------------------------------------------
# Markdown Cell:
# ## 3 · Vector store & **Parent-Page Retriever**
#-------------------------------------------------------------------------------

def build_retriever(parent_docs: List[Document],
                    embeddings: HuggingFaceEmbeddings,
                    recreate: bool = True):
    """Index child chunks but return the full parent page on retrieval."""
    # 1) Optional fresh start
    if recreate and CHROMA_DIR.exists():
        import shutil, time
        shutil.rmtree(CHROMA_DIR)
        logger.info("Removed existing Chroma directory for fresh index")
    
    # 2) Split parents into *child* chunks for dense retrieval
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    
    # 3) Build Chroma from children
    vector_store = Chroma.from_documents(
        documents=child_splitter.split_documents(parent_docs),
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR)
    )
    
    # 4) Map parent IDs to full pages in a docstore
    docstore = InMemoryStore()
    docstore.mset([(d.metadata['id'], d) for d in parent_docs])
    
    # 5) Parent retriever wraps vector store
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        child_splitter=child_splitter,
        search_kwargs={'k': RETRIEVER_K},
        id_key='id'
    )
    return retriever

#-------------------------------------------------------------------------------
# Markdown Cell:
# ## 4 · Prompt & chain assembly
#-------------------------------------------------------------------------------
# Combined (dynamic) retrieval logic (NEW)
def dynamic_retrieve(input_data: dict, # Modified to accept dict
                     retriever: ParentDocumentRetriever,
                     query_rewriter: Optional[RunnablePassthrough], # Optional
                     hyde_embedder: Optional[HypotheticalDocumentEmbedder], # Optional
                     reranker_model: Optional[CrossEncoder] # Optional
                     ) -> List[Document]:
    query = input_data if isinstance(input_data, str) else input_data.get("question", "") # Extract query
    if not query:
        return []
    print(f"\n--- Original Query ---\n{query}")

    # --- 1) rewrite & decide on HyDE ------------------------------------ #
    rewritten = query
    use_hyde_local = USE_HYDE   # global default
    if USE_QR and query_rewriter:
        print("\n--- Query Rewriter ---")
        rewrite_out = query_rewriter.invoke({"query": query}) # Removed history
        print(f"  Classification: {rewrite_out.classification}")
        print(f"  Rewritten Query: {rewrite_out.rewritten_query}")
        
        # Валидация классификации
        classification = rewrite_out.classification.lower().strip()
        if classification not in ALLOWED_CLASSIFICATIONS:
            logger.warning(f"Unknown classification '{classification}', defaulting to 'fuzzy'")
            classification = "fuzzy"
        
        # Применяем строгие правила из CLASS2HYDE (игнорируем ответ LLM)
        use_hyde_from_rules = CLASS2HYDE.get(classification, False)
        use_hyde_local = USE_HYDE and use_hyde_from_rules
        
        print(f"  LLM suggested HyDE: {rewrite_out.use_hyde}")
        print(f"  Rules-based HyDE: {use_hyde_from_rules}")
        print(f"  Final HyDE decision: {use_hyde_local}")
        
        rewritten = rewrite_out.rewritten_query or query
        logger.debug("QR → %s | Class=%s | HyDE=%s", rewritten, classification, use_hyde_local)
    else:
        print("\n--- Query Rewriter (Skipped) ---")

    # --- 2) plain retrieval --------------------------------------------- #
    print(f"\n--- Base Retriever (Query: '{rewritten}') ---")
    base_docs = retriever.get_relevant_documents(rewritten)
    print(f"  Retrieved {len(base_docs)} documents.")
    combined  = base_docs

    # --- 3) HyDE augmentation ------------------------------------------- #
    if use_hyde_local and hyde_embedder:
        print(f"\n--- HyDE Augmentation (Query: '{rewritten}') ---")
        hyde_vec = hyde_embedder.embed_query(rewritten)
        hyde_children = retriever.vectorstore.similarity_search_by_vector(
            hyde_vec, k=RETRIEVER_K)
        print(f"  HyDE retrieved {len(hyde_children)} child documents.")
        # Correctly retrieve parent documents
        hyde_parent_ids = [child.metadata.get(retriever.id_key) for child in hyde_children if child.metadata and retriever.id_key in child.metadata]
        valid_hyde_parent_ids = [pid for pid in hyde_parent_ids if pid is not None]
        
        # Ensure mget receives a list of strings (IDs)
        hyde_parents_list_of_optional_docs = retriever.docstore.mget(valid_hyde_parent_ids)
        # Filter out None values that mget might return if an ID is not found
        hyde_parents = [doc for doc in hyde_parents_list_of_optional_docs if doc is not None]
        print(f"  Converted to {len(hyde_parents)} parent documents.")
        
        combined.extend(hyde_parents)
    else:
        print("\n--- HyDE Augmentation (Skipped or Disabled) ---")
        
    print("\n--- Deduplication ---")
    len_before_dedup = len(combined)
    combined = deduplicate(combined)
    print(f"  Documents before: {len_before_dedup}, After: {len(combined)}")

    # --- 4) Rerank ------------------------------------------------------- #
    if USE_RERANKER and reranker_model and combined: # Added check for combined not empty
        print("\n--- Reranker ---")
        pairs  = [[rewritten, d.page_content[:512]] for d in combined]
        print(f"  Reranking {len(pairs)} pairs.")
        scores = reranker_model.predict(pairs)
        combined = [d for d,_ in sorted(zip(combined, scores),
                                        key=lambda x: x[1], reverse=True)]
        print(f"  Reranked {len(combined)} documents.")
    else:
        print("\n--- Reranker (Skipped or Disabled) ---")

    final_docs = combined[:RETRIEVER_K]
    print(f"\n--- Final Selected Documents ({len(final_docs)}) ---")
    for i, doc in enumerate(final_docs):
        print(f"  Doc {i+1}: {doc.metadata['source']} (Page {doc.metadata['page']})") # Removed content preview for brevity

    return final_docs

def format_context(docs: List[Document]) -> str:
    if not docs:
        return "[контекст не найден]"
    out = []
    for idx, d in enumerate(docs, 1):
        src = d.metadata['source']
        page = d.metadata['page']
        out.append(f"[DOC:{idx} | {src} / стр.{page}]\n{d.page_content}\n---")
    return "\n".join(out)

SYSTEM_MESSAGE = (
    "Ты ИИ-ассистент. Используй **только** предоставленный контекст. "
    "Отвечай на вопрос на русском языке. "
    f"Верни JSON, соответствующий этой схеме: {format_instructions_escaped}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MESSAGE),
    ("user", "Контекст:\n{context}\n\nВопрос: {question}")
]).partial(format_instructions=answer_parser.get_format_instructions())

def build_chain(retriever, llm, query_rewriter, hyde_embedder, reranker_model): # Added new params
    
    # Create a RunnableLambda for dynamic retrieval
    # It now correctly expects a dictionary with a "question" key
    dynamic_retriever_runnable = RunnableLambda(
        lambda input_dict: dynamic_retrieve(
            input_dict,  # Pass the whole dict # type: ignore
            retriever=retriever,
            query_rewriter=query_rewriter,
            hyde_embedder=hyde_embedder,
            reranker_model=reranker_model
        )
    )
    
    return (
        {
            "context": dynamic_retriever_runnable | format_context, # Use the new runnable
            "question": RunnablePassthrough() # Will pass the original input_dict
        }
        | prompt
        | llm
        | answer_parser  # parses JSON -> AnswerSchema object
    )


#-------------------------------------------------------------------------------
# Markdown Cell:
# ## 5 · Model initialization
#-------------------------------------------------------------------------------

load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')  # keep the same env var
if not api_key:
    raise EnvironmentError("GOOGLE_API_KEY not set (.env or env var)")

# Настройка устройства для эмбеддингов
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device for embeddings: {device}")

# Настройки для BAAI/bge-m3
model_kwargs = {
    'device': device,
    'trust_remote_code': True  # Требуется для некоторых моделей BGE
}
encode_kwargs = {
    'normalize_embeddings': True  # Нормализация для лучшего качества поиска
}

print(f"Loading embedding model: {EMBEDDING_MODEL}")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("Embedding model loaded successfully!")

llm = ChatGoogleGenerativeAI(model=LLM_MODEL,
                             google_api_key=SecretStr(api_key),
                             temperature=0.05)

# --- Query Rewriter chain (if enabled) (NEW) ----------------------------------- #
if USE_QR:
    qr_llm = ChatGoogleGenerativeAI(model=LLM_MODEL,
                                    google_api_key=SecretStr(api_key),
                                    temperature=0.0)
    query_rewriter_chain = (rewrite_prompt
                      | qr_llm
                      | rewrite_parser)
else:
    query_rewriter_chain = None

# --- HyDE embedder (lazy only if toggled) (NEW) -------------------------------- #
if USE_HYDE:
    hyde_llm = ChatGoogleGenerativeAI(model=LLM_MODEL,
                                      google_api_key=SecretStr(api_key),
                                      temperature=0.7) # Higher temperature for creative generation
    # Correct initialization of HypotheticalDocumentEmbedder
    hyde_embedder_instance = HypotheticalDocumentEmbedder.from_llm(
        llm=hyde_llm,
        base_embeddings=embeddings,  # Теперь используется HuggingFaceEmbeddings
        prompt_key="web_search" # Using a default prompt key
    )
else:
    hyde_embedder_instance = None

# --- Reranker (NEW) ------------------------------------------------------------ #
if USE_RERANKER:
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        reranker_model_instance = CrossEncoder('BAAI/bge-reranker-v2-m3', device=device, trust_remote_code=True) # Added trust_remote_code
    except Exception as e:
        logger.error(f"Failed to load Reranker model: {e}. Reranking will be disabled.")
        reranker_model_instance = None
        USE_RERANKER = False # Disable if loading fails
else:
    reranker_model_instance = None
#-------------------------------------------------------------------------------
# Markdown Cell:
# ## 6 · Index & retriever
#-------------------------------------------------------------------------------

processor = DocumentProcessor()
parent_pages = processor.load(DATA_DIR)

# ВАЖНО: При смене модели эмбеддингов нужно пересоздать векторную базу!
# Установите recreate=True при первом запуске с новой моделью эмбеддингов
print("Building retriever with new embedding model...")
print("IMPORTANT: If you changed embedding model, set recreate=True to rebuild vector DB")
retriever = build_retriever(parent_pages, embeddings, recreate=True)  # ИЗМЕНИЛИ ОБРАТНО НА False
print("Retriever built successfully!")

#-------------------------------------------------------------------------------
# Markdown Cell:
# ## 7 · RAG chain with structured output
#-------------------------------------------------------------------------------
import time
start_time = time.time()

print("Building RAG chain...")
# Pass the initialized modules to build_chain
rag_chain = build_chain(retriever, 
                        llm, 
                        query_rewriter_chain, 
                        hyde_embedder_instance, 
                        reranker_model_instance)
print(f"RAG chain built successfully. Time taken: {time.time() - start_time:.2f} seconds")
#-------------------------------------------------------------------------------
# Markdown Cell:
# ## 8 · Quick test
#-------------------------------------------------------------------------------

print("\n--- Starting RAG Chain Invocation ---")
q = "What rooms are used for acoustic tests, and what are their volumes?"
response = rag_chain.invoke({"question": q}) # Invoke with a dictionary
print("\n--- Final Output ---")
print(response.model_dump_json(indent=2))

#-------------------------------------------------------------------------------
# Markdown Cell:
# ### Optional − interactive loop
#-------------------------------------------------------------------------------

# while True:
#     q_input = input("\nAsk (or 'exit'): ")
#     if q_input.lower() in {'exit', 'quit'}:
#         break
#     # Invoke with a dictionary
#     # print(rag_chain.invoke({"question": q_input}).json(indent=2, ensure_ascii=False))
#     print("\n--- Final Output ---") # Added for interactive loop
#     print(rag_chain.invoke({"question": q_input}).model_dump_json(indent=2, ensure_ascii=False))

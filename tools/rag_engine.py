from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import re

from config import config
from core.logger import get_logger

logger = get_logger(__name__)

embeddings = (
    OpenAIEmbeddings(
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_EMBEDDING_MODEL,
    )
    if config.OPENAI_API_KEY
    else None
)

def _lexical_search_fallback(chunks: list[str], query: str, top_k: int = 3) -> str:
    """Fallback retrieval via keyword frequency when vector search is unavailable."""
    terms = [token for token in re.split(r"\W+", query.lower()) if token]
    if not terms:
        return "\n\n---\n\n".join(chunks[:top_k])

    scored: list[tuple[int, str]] = []
    for chunk in chunks:
        lowered = chunk.lower()
        score = sum(lowered.count(term) for term in terms)
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [chunk for score, chunk in scored if score > 0][:top_k]
    if not selected:
        selected = chunks[:top_k]
    return "\n\n---\n\n".join(selected)

def build_and_retrieve_context(full_text: str, query: str, top_k: int = 5) -> str:
    """
    Hàm chính được gọi bởi Node RAG.
    Chia nhỏ văn bản và tìm top K chunks liên quan nhất với query.
    """
    if not full_text.strip():
        return ""

    logger.debug("[rag] Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.RAG_CHUNK_SIZE,
        chunk_overlap=config.RAG_CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_text(full_text)
    
    # Kích hoạt fallback ngay nếu không có OpenAI API key
    if not embeddings:
        logger.warning("[rag] Missing OpenAI key for embeddings; using lexical fallback")
        return _lexical_search_fallback(chunks, query, top_k)

    try:
        logger.debug("[rag] Building FAISS index with OpenAI embeddings")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        docs = vectorstore.similarity_search(query, k=top_k)
        
        retrieved_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        logger.info("[rag] Retrieved %s chunks from vector search", len(docs))
        return retrieved_text
        
    except Exception as e:
        logger.error("[rag] Vector retrieval failed; using lexical fallback: %s", e, exc_info=True)
        return _lexical_search_fallback(chunks, query, top_k)
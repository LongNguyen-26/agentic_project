from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

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

def build_and_retrieve_context(full_text: str, query: str, top_k: int = 5) -> str:
    """
    Hàm chính được gọi bởi Node RAG.
    Chia nhỏ văn bản và tìm top K chunks liên quan nhất với query bằng Hybrid Search (BM25 + FAISS).
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
    
    # 1. Khởi tạo Retriever tìm kiếm từ khóa (BM25)
    logger.debug("[rag] Building BM25 retriever")
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = top_k

    # Nếu không có API Key cho Embeddings, chỉ sử dụng BM25
    if not embeddings:
        logger.warning("[rag] Missing OpenAI key for embeddings; using ONLY BM25 lexical search")
        try:
            docs = bm25_retriever.invoke(query)
            return "\n\n---\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            logger.error("[rag] BM25 retrieval failed: %s", e, exc_info=True)
            return ""

    # 2. Khởi tạo Retriever tìm kiếm ngữ nghĩa (FAISS Vector Search)
    try:
        logger.debug("[rag] Building FAISS index with OpenAI embeddings")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        
        # 3. Kết hợp bằng EnsembleRetriever (Hybrid Search)
        logger.debug("[rag] Executing Hybrid Search (BM25 + FAISS)")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5] # Cân bằng 50% từ khóa - 50% ngữ nghĩa
        )
        
        # Invoke sẽ tự động gọi cả 2 retriever, lấy kết quả và tính điểm chéo (RRF)
        docs = ensemble_retriever.invoke(query)
        
        # EnsembleRetriever có thể trả về nhiều hơn top_k, ta cắt lại cho đúng số lượng
        final_docs = docs[:top_k]
        
        retrieved_text = "\n\n---\n\n".join([doc.page_content for doc in final_docs])
        logger.info("[rag] Retrieved %s chunks from hybrid search", len(final_docs))
        return retrieved_text
        
    except Exception as e:
        logger.error("[rag] Hybrid retrieval failed; falling back to BM25 ONLY: %s", e, exc_info=True)
        # Fallback an toàn: Nếu FAISS hoặc Ensemble lỗi, vẫn cố gắng trả về kết quả từ BM25
        try:
            docs = bm25_retriever.invoke(query)
            return "\n\n---\n\n".join([doc.page_content for doc in docs[:top_k]])
        except Exception as fallback_err:
            logger.error("[rag] Fallback BM25 also failed: %s", fallback_err)
            return ""
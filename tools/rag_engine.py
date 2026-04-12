from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownTextSplitter
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

_reranker_model: Optional[Any] = None
_reranker_failed = False


def _should_rerank() -> bool:
    return bool(getattr(config, "RAG_RERANK_ENABLED", False))


def _resolve_retrieval_k(top_k: int) -> int:
    if not _should_rerank():
        return top_k
    pre_top_k = int(getattr(config, "RAG_RERANK_PRE_TOP_K", 20))
    return max(top_k, pre_top_k)


def _build_chunk_documents(parsed_documents: List[Dict[str, Any]]) -> List[Document]:
    text_splitter = MarkdownTextSplitter(
        chunk_size=config.RAG_CHUNK_SIZE,
        chunk_overlap=config.RAG_CHUNK_OVERLAP,
    )

    chunk_documents: List[Document] = []
    for doc in parsed_documents:
        file_path = doc.get("file_path", "unknown")
        summary = (doc.get("summary") or "").strip()
        text = doc.get("text", "")

        if not text or not text.strip():
            continue

        for chunk_index, chunk in enumerate(text_splitter.split_text(text)):
            chunk_documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "file_path": file_path,
                        "summary": summary,
                        "chunk_index": chunk_index,
                    },
                )
            )
    return chunk_documents


def _render_documents_for_prompt(documents: List[Document]) -> str:
    rendered_chunks: List[str] = []
    summarized_files = set()

    def _short_summary(raw_summary: str) -> str:
        limit = max(int(config.RAG_SUMMARY_MAX_CHARS), 80)
        summary = raw_summary.strip()
        if len(summary) <= limit:
            return summary
        return f"{summary[:limit]}..."

    for doc in documents:
        file_path = str(doc.metadata.get("file_path", "unknown"))
        summary = str(doc.metadata.get("summary", "")).strip()
        if summary and file_path not in summarized_files:
            prefix = f"[Source: {file_path} - Summary: {_short_summary(summary)}]"
            summarized_files.add(file_path)
        else:
            prefix = f"[Source: {file_path}]"
        rendered_chunks.append(f"{prefix}\n{doc.page_content}")
    return "\n\n---\n\n".join(rendered_chunks)


def _get_reranker_model() -> Optional[Any]:
    global _reranker_model
    global _reranker_failed

    if not _should_rerank() or _reranker_failed:
        return None
    if _reranker_model is not None:
        return _reranker_model

    try:
        from sentence_transformers import SentenceTransformer

        cache_dir = Path(config.STORAGE_ROOT) / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_name = getattr(config, "RAG_RERANK_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        device = getattr(config, "RAG_RERANK_DEVICE", "cpu")

        _reranker_model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(cache_dir),
        )
        logger.info("[rag] Reranker ready model=%s device=%s", model_name, device)
        return _reranker_model
    except Exception:
        logger.warning("[rag] Failed to load reranker model; disabling reranker", exc_info=True)
        _reranker_failed = True
        return None


def _rerank_documents(query: str, documents: List[Document]) -> List[Document]:
    reranker = _get_reranker_model()
    if reranker is None or not documents:
        return documents

    try:
        from sentence_transformers import util as st_util

        batch_size = max(int(getattr(config, "RAG_RERANK_BATCH_SIZE", 16)), 1)
        query_embedding = reranker.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        doc_embeddings = reranker.encode(
            [doc.page_content for doc in documents],
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=batch_size,
        )

        scores = st_util.cos_sim(query_embedding, doc_embeddings)[0].tolist()
        ranked_pairs = sorted(
            zip(scores, documents),
            key=lambda pair: pair[0],
            reverse=True,
        )
        logger.info(
            "[rag] Reranked %s candidates; best_score=%.4f",
            len(documents),
            ranked_pairs[0][0] if ranked_pairs else 0.0,
        )
        return [doc for _, doc in ranked_pairs]
    except Exception:
        logger.warning("[rag] Reranker scoring failed; keep ensemble order", exc_info=True)
        return documents

def build_and_retrieve_context(parsed_documents: List[Dict[str, Any]], query: str, top_k: int = 5) -> str:
    """
    Main entry point used by the RAG node.
    Split file text into chunks, attach metadata (file path, summary) per chunk,
    then retrieve top-k most relevant chunks with hybrid search.
    """
    if not parsed_documents:
        return ""

    logger.debug("[rag] Splitting text into Document chunks (metadata separated)")
    chunk_documents = _build_chunk_documents(parsed_documents)
    if not chunk_documents:
        return ""

    retrieval_k = _resolve_retrieval_k(top_k)

    # 1) Build keyword retriever (BM25).
    logger.debug("[rag] Building BM25 retriever")
    bm25_retriever = BM25Retriever.from_documents(chunk_documents)
    bm25_retriever.k = retrieval_k

    # If embeddings are unavailable, run BM25-only retrieval.
    if not embeddings:
        logger.warning("[rag] Missing OpenAI key for embeddings; using ONLY BM25 lexical search")
        try:
            docs = bm25_retriever.invoke(query)
            candidate_docs = docs[:retrieval_k]
            if _should_rerank():
                candidate_docs = _rerank_documents(query, candidate_docs)
            final_docs = candidate_docs[:top_k]
            return _render_documents_for_prompt(final_docs)
        except Exception as e:
            logger.error("[rag] BM25 retrieval failed: %s", e, exc_info=True)
            return ""

    # 2) Build semantic retriever (FAISS vector search).
    try:
        logger.debug("[rag] Building FAISS index with OpenAI embeddings")
        vectorstore = FAISS.from_documents(chunk_documents, embeddings)
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})
        
        # 3) Combine BM25 and FAISS via EnsembleRetriever (hybrid search).
        logger.debug("[rag] Executing Hybrid Search (BM25 + FAISS)")
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5],  # Balanced lexical and semantic weighting.
        )
        
        # invoke() executes both retrievers and fuses results (RRF-style ranking).
        docs = ensemble_retriever.invoke(query)

        candidate_docs = docs[:retrieval_k]
        if _should_rerank():
            candidate_docs = _rerank_documents(query, candidate_docs)

        final_docs = candidate_docs[:top_k]

        logger.info(
            "[rag] Retrieved %s chunks from hybrid search (candidates=%s reranker=%s)",
            len(final_docs),
            len(candidate_docs),
            _should_rerank(),
        )
        return _render_documents_for_prompt(final_docs)
        
    except Exception as e:
        logger.error("[rag] Hybrid retrieval failed; falling back to BM25 ONLY: %s", e, exc_info=True)
        # Safe fallback: if FAISS/ensemble fails, still attempt BM25 output.
        try:
            docs = bm25_retriever.invoke(query)
            candidate_docs = docs[:retrieval_k]
            if _should_rerank():
                candidate_docs = _rerank_documents(query, candidate_docs)
            final_docs = candidate_docs[:top_k]
            return _render_documents_for_prompt(final_docs)
        except Exception as fallback_err:
            logger.error("[rag] Fallback BM25 also failed: %s", fallback_err)
            return ""
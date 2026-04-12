import unittest
from unittest import mock

from langchain_core.documents import Document

from devday_agent.tools import rag_engine


class _FakeBM25Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.k = 0

    def invoke(self, query):
        _ = query
        return list(self.docs)


class RagEngineUpgradeTests(unittest.TestCase):
    def setUp(self):
        rag_engine._reranker_model = None
        rag_engine._reranker_failed = False

    def test_build_chunk_documents_keeps_metadata_separated(self):
        parsed_documents = [
            {
                "file_path": "Public/a.pdf",
                "summary": "Summary A",
                "text": "This is content A.",
            }
        ]

        chunk_docs = rag_engine._build_chunk_documents(parsed_documents)

        self.assertEqual(len(chunk_docs), 1)
        self.assertEqual(chunk_docs[0].metadata.get("file_path"), "Public/a.pdf")
        self.assertEqual(chunk_docs[0].metadata.get("summary"), "Summary A")
        self.assertIn("This is content A.", chunk_docs[0].page_content)
        self.assertNotIn("[Source:", chunk_docs[0].page_content)

    def test_render_documents_for_prompt_includes_metadata_prefix(self):
        docs = [
            Document(
                page_content="chunk body",
                metadata={"file_path": "Public/b.pdf", "summary": "Summary B"},
            )
        ]

        rendered = rag_engine._render_documents_for_prompt(docs)

        self.assertIn("[Source: Public/b.pdf - Summary: Summary B]", rendered)
        self.assertIn("chunk body", rendered)

    def test_build_and_retrieve_context_reranks_candidates_before_top_k_slice(self):
        parsed_documents = [
            {
                "file_path": "Public/file-1.pdf",
                "summary": "Summary 1",
                "text": "alpha",
            },
            {
                "file_path": "Public/file-2.pdf",
                "summary": "Summary 2",
                "text": "beta",
            },
        ]
        captured = {}

        def _fake_from_documents(documents):
            captured["documents"] = list(documents)
            return _FakeBM25Retriever(documents)

        with mock.patch.object(rag_engine, "embeddings", None), mock.patch.object(
            rag_engine.BM25Retriever,
            "from_documents",
            side_effect=_fake_from_documents,
        ), mock.patch.object(rag_engine, "_should_rerank", return_value=True), mock.patch.object(
            rag_engine,
            "_rerank_documents",
            side_effect=lambda query, docs: list(reversed(docs)),
        ):
            context = rag_engine.build_and_retrieve_context(parsed_documents, query="find schedule", top_k=1)

        self.assertIn("documents", captured)
        self.assertEqual(len(captured["documents"]), 2)
        self.assertNotIn("[Source:", captured["documents"][0].page_content)
        self.assertIn("Public/file-2.pdf", context)
        self.assertNotIn("Public/file-1.pdf", context)

    def test_rerank_documents_returns_original_order_when_model_unavailable(self):
        docs = [
            Document(page_content="d1", metadata={"file_path": "f1"}),
            Document(page_content="d2", metadata={"file_path": "f2"}),
        ]

        with mock.patch.object(rag_engine, "_get_reranker_model", return_value=None):
            reranked = rag_engine._rerank_documents("query", docs)

        self.assertEqual(reranked, docs)


if __name__ == "__main__":
    unittest.main()

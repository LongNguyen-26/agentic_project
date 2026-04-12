import unittest
from unittest import mock

from models.llm_schemas import FileSummaryResponse
from tools import context_manager as cm


class _FakeLLMService:
    def __init__(self, summary: str):
        self.summary = summary
        self.calls = 0

    def generate_structured(self, **kwargs):
        self.calls += 1
        return FileSummaryResponse(summary=self.summary)


class ContextManagerCacheTests(unittest.TestCase):
    def setUp(self):
        cm._file_summary_cache.clear()
        cm._cache_loaded = False

    def test_get_or_create_uses_cache_on_second_call(self):
        fake_llm = _FakeLLMService(summary="Summary A")

        with mock.patch.object(cm, "load_file_summary_cache", return_value={}), mock.patch.object(
            cm, "save_file_summary_cache", return_value=True
        ):
            summary1, hit1 = cm.get_or_create_file_summary(
                file_path="docs/a.pdf",
                raw_text="A long enough source text for summary generation.",
                llm_service=fake_llm,
            )
            summary2, hit2 = cm.get_or_create_file_summary(
                file_path="docs/a.pdf",
                raw_text="A long enough source text for summary generation.",
                llm_service=fake_llm,
            )

        self.assertEqual(summary1, "Summary A")
        self.assertEqual(summary2, "Summary A")
        self.assertFalse(hit1)
        self.assertTrue(hit2)
        self.assertEqual(fake_llm.calls, 1)

    def test_get_cached_file_summary_loads_persisted_cache_once(self):
        text = "Persisted text content"
        key = cm._cache_key("docs/b.pdf", text)
        persisted = {
            key: {
                "file_path": "docs/b.pdf",
                "summary": "Persisted Summary",
                "updated_at": 123,
            }
        }

        with mock.patch.object(cm, "load_file_summary_cache", return_value=persisted):
            found = cm.get_cached_file_summary("docs/b.pdf", text)
            found_again = cm.get_cached_file_summary("docs/b.pdf", text)

        self.assertEqual(found, "Persisted Summary")
        self.assertEqual(found_again, "Persisted Summary")


if __name__ == "__main__":
    unittest.main()

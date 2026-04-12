import unittest
from unittest import mock

from devday_agent.agent.nodes import inner_loop
from config import config


class FolderVerifiabilityTests(unittest.TestCase):
    def test_folder_invalid_answer_requests_retry(self):
        fake_llm = mock.Mock()
        fake_llm.generate_verification_response.return_value = mock.Mock(
            confidence=0.1,
            thought_log="No confident folder mapping.",
            used_tools=["verifier"],
        )

        state = {
            "task_id": "task-1",
            "task_type": "folder-organisation",
            "draft_answer": {"answers": [], "confidence": 0.1},
            "confidence_score": 0.1,
            "attempts": 0,
            "retrieved_context": "",
            "prompt_template": "",
        }

        with mock.patch.object(inner_loop, "_get_llm_service", return_value=fake_llm):
            update = inner_loop.verifiability_node(state)

        self.assertFalse(update["is_verified"])
        self.assertEqual(update["attempts"], 1)
        self.assertIn("Folder verification failed", update["verification_feedback"])

    def test_folder_valid_answer_passes_verification(self):
        confidence = max(config.VERIFIER_MIN_CONFIDENCE, 0.8)
        valid_folder = next(iter(inner_loop.VALID_FOLDERS))

        fake_llm = mock.Mock()
        fake_llm.generate_verification_response.return_value = mock.Mock(
            confidence=confidence,
            thought_log="",
            used_tools=["verifier"],
        )

        state = {
            "task_id": "task-2",
            "task_type": "folder-organisation",
            "draft_answer": {
                "answers": ["fileA -> folder1"],
                "thought_log": (
                    "Sorting Details:\n"
                    "- File: Public/a.pdf\n"
                    f"  Folder: {valid_folder}\n"
                    "  Reasoning: matched by summary"
                ),
                "confidence": confidence,
            },
            "confidence_score": confidence,
            "attempts": 1,
            "retrieved_context": "",
            "prompt_template": "",
            "parsed_documents": [{"file_path": "Public/a.pdf", "summary": "", "text": ""}],
        }

        with mock.patch.object(inner_loop, "_get_llm_service", return_value=fake_llm):
            update = inner_loop.verifiability_node(state)

        self.assertTrue(update["is_verified"])
        self.assertEqual(update["attempts"], 2)
        self.assertEqual(update["verification_feedback"], "")


if __name__ == "__main__":
    unittest.main()

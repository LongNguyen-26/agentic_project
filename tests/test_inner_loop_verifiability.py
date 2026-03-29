import unittest

from agent.nodes.inner_loop import verifiability_node
from config import config


class FolderVerifiabilityTests(unittest.TestCase):
    def test_folder_invalid_answer_requests_retry(self):
        state = {
            "task_id": "task-1",
            "task_type": "folder-organisation",
            "draft_answer": {"answers": [], "confidence": 0.1},
            "confidence_score": 0.1,
            "attempts": 0,
            "retrieved_context": "",
            "prompt_template": "",
        }

        update = verifiability_node(state)

        self.assertFalse(update["is_verified"])
        self.assertEqual(update["attempts"], 1)
        self.assertIn("Folder verification failed", update["verification_feedback"])

    def test_folder_valid_answer_passes_verification(self):
        confidence = max(config.VERIFIER_MIN_CONFIDENCE, 0.8)
        state = {
            "task_id": "task-2",
            "task_type": "folder-organisation",
            "draft_answer": {"answers": ["fileA -> folder1"], "confidence": confidence},
            "confidence_score": confidence,
            "attempts": 1,
            "retrieved_context": "",
            "prompt_template": "",
        }

        update = verifiability_node(state)

        self.assertTrue(update["is_verified"])
        self.assertEqual(update["attempts"], 2)
        self.assertEqual(update["verification_feedback"], "")


if __name__ == "__main__":
    unittest.main()

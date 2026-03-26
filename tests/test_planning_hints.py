import unittest
from unittest import mock

from agent.nodes import outer_loop
from agent.prompts.user_prompt import build_action_prompt


class PlanningHintsTests(unittest.TestCase):
    def test_planning_node_extracts_hints(self):
        fake_llm = mock.Mock()
        fake_llm.extract_planning_hints.return_value = "1. Check format"

        state = {
            "current_task": {
                "id": "task-1",
                "prompt_template": "Extract date and keep YYYY-MM-DD",
            }
        }

        with mock.patch.object(outer_loop, "_get_llm_service", return_value=fake_llm):
            update = outer_loop.planning_node(state)

        self.assertEqual(update["planning_hints"], "1. Check format")

    def test_action_prompt_contains_planning_hints(self):
        prompt = build_action_prompt(
            task_type="question-answering",
            prompt_template="Find commissioning date",
            context="Document text",
            planning_hints="1. Keep output format strict",
        )
        self.assertIn("[CẢNH BÁO / HINTS TRƯỚC KHI GIẢI]", prompt)
        self.assertIn("Keep output format strict", prompt)


if __name__ == "__main__":
    unittest.main()

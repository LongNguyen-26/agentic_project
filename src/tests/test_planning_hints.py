import unittest
from unittest import mock

from devday_agent.agent.nodes import outer_loop
from devday_agent.agent.prompts.user_prompt import build_qa_action_prompt, build_sort_action_prompt


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

    def test_qa_action_prompt_contains_planning_hints(self):
        prompt = build_qa_action_prompt(
            prompt_template="Find commissioning date",
            context="Document text",
            planning_hints="1. Keep output format strict",
        )
        self.assertIn("<planning_hints>", prompt)
        self.assertIn("Keep output format strict", prompt)

    def test_sort_action_prompt_contains_planning_hints(self):
        prompt = build_sort_action_prompt(
            prompt_template="Sort files into valid folders",
            file_summaries={"a.pdf": "Manual for PCS"},
            planning_hints="1. Match only valid folders",
        )
        self.assertIn("<planning_hints>", prompt)
        self.assertIn("Match only valid folders", prompt)


if __name__ == "__main__":
    unittest.main()

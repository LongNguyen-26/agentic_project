import unittest
from unittest import mock

from agent.nodes import inner_loop, router


class ForcedVisionKeywordTests(unittest.TestCase):
    def _build_state(self):
        return {
            "task_id": "task-vision-keyword-1",
            "task_type": "question-answering",
            "prompt_template": "Please inspect this image and answer the question.",
            "retrieved_context": "[IMAGE_PLACEHOLDER | ID: IMG_001 | SIZE: 1200x800]",
            "parsed_documents": [
                {
                    "file_path": "Public/sample.pdf",
                    "text": "[IMAGE_PLACEHOLDER | ID: IMG_001 | SIZE: 1200x800]",
                    "summary": "Sample summary",
                }
            ],
            "verification_feedback": "",
            "planning_hints": "",
            "tool_observations": [],
            "tool_calls": [],
            "vision_prompt": "",
            "used_tools": [],
            "confidence_score": 0.0,
        }

    def test_keyword_forces_vision_tool_usage_with_logs(self):
        state = self._build_state()

        with self.assertLogs(inner_loop.logger, level="INFO") as action_logs:
            action_update = inner_loop.action_generation_node(state)

        print("\n[ACTION_LOGS]")
        for line in action_logs.output:
            print(line)

        self.assertTrue(action_update["tool_calls"])
        self.assertEqual(router.route_after_action(action_update), "call_vision_tool")

        vision_state = {**state, **action_update}

        with mock.patch.object(
            inner_loop,
            "analyze_images_from_cache",
            return_value="[Vision Tool] Visual evidence extracted from IMG_001",
        ) as mocked_vision:
            with self.assertLogs(inner_loop.logger, level="INFO") as vision_logs:
                vision_update = inner_loop.vision_tool_node(vision_state)

        print("\n[VISION_LOGS]")
        for line in vision_logs.output:
            print(line)

        mocked_vision.assert_called_once()
        self.assertEqual(vision_update["tool_calls"], [])
        self.assertIn("vision_tool", vision_update["used_tools"])
        self.assertTrue(vision_update["tool_observations"])

        merged_logs = "\n".join(action_logs.output + vision_logs.output)
        self.assertIn("Forcing vision route", merged_logs)
        self.assertIn("VisionToolNode", merged_logs)


if __name__ == "__main__":
    unittest.main()

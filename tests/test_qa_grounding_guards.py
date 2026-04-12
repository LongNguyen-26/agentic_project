import unittest

from agent.nodes import inner_loop


class QAGroundingGuardTests(unittest.TestCase):
    def test_fallback_answer_keeps_clean_schema_content(self):
        draft_answer = {
            "answers": ["The commissioning date is 2024-09-01."],
            "thought_log": "Initial draft based on retrieved context.",
            "used_tools": ["llm_client"],
            "confidence": 0.92,
        }

        fallback = inner_loop._build_grounding_fallback_answer(
            draft_answer=draft_answer,
            grounded_count=0,
            detected_count=2,
            required_count=1,
        )

        self.assertEqual(fallback["answers"], ["The commissioning date is 2024-09-01."])
        self.assertNotIn("Grounding markers found", fallback["answers"][0])
        self.assertIn("[Grounding Diagnostics]", fallback["thought_log"])

    def test_fallback_answer_without_tentative_uses_safe_default(self):
        draft_answer = {
            "answers": [],
            "thought_log": "",
            "used_tools": [],
            "confidence": 0.15,
        }

        fallback = inner_loop._build_grounding_fallback_answer(
            draft_answer=draft_answer,
            grounded_count=0,
            detected_count=0,
            required_count=1,
        )

        self.assertEqual(fallback["answers"], ["Insufficient evidence in provided context."])
        self.assertIn("[Grounding Diagnostics]", fallback["thought_log"])

    def test_grounding_uses_parsed_document_anchors(self):
        output_text = "Use Public/sample.pdf and image abcdef1234567890 as evidence."
        context_text = "Context body has no explicit source anchors."
        parsed_documents = [
            {
                "file_path": "Public/sample.pdf",
                "text": "[IMAGE_PLACEHOLDER | ID: abcdef1234567890 | Size: 1200x800]",
                "summary": "Sample summary",
            }
        ]

        grounded, detected = inner_loop._count_grounded_markers(
            output_text=output_text,
            context_text=context_text,
            parsed_documents=parsed_documents,
        )

        self.assertEqual(detected, 2)
        self.assertEqual(grounded, 2)


if __name__ == "__main__":
    unittest.main()

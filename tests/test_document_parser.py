import unittest
from unittest import mock

from tools import document_parser as parser


class TieredParserTests(unittest.TestCase):
    def test_pdf_tier1_short_circuits_other_tiers(self):
        tier1_text = "X" * 200
        with mock.patch.object(parser, "_parse_pdf_text_fallback", return_value=tier1_text) as m1, mock.patch.object(
            parser, "_parse_pdf_with_ollama", return_value=""
        ) as m2, mock.patch.object(parser, "_parse_pdf_with_openai", return_value="") as m3:
            out = parser.parse_resource_bytes("sample.pdf", b"pdf-bytes", "application/pdf")

        self.assertEqual(out, tier1_text)
        self.assertEqual(m1.call_count, 1)
        self.assertEqual(m2.call_count, 0)
        self.assertEqual(m3.call_count, 0)

    def test_pdf_tier2_used_when_tier1_insufficient(self):
        tier2_text = "Y" * 200
        with mock.patch.object(parser, "_parse_pdf_text_fallback", return_value="") as m1, mock.patch.object(
            parser, "_parse_pdf_with_ollama", return_value=tier2_text
        ) as m2, mock.patch.object(parser, "_parse_pdf_with_openai", return_value="vision") as m3:
            out = parser.parse_resource_bytes("sample.pdf", b"pdf-bytes", "application/pdf")

        self.assertEqual(out, tier2_text)
        self.assertEqual(m1.call_count, 1)
        self.assertEqual(m2.call_count, 1)
        self.assertEqual(m3.call_count, 0)

    def test_image_escalates_to_tier3_when_tier2_insufficient(self):
        with mock.patch.object(
            parser,
            "_normalize_image_for_vision",
            return_value=(b"norm", "image/jpeg"),
        ) as m_norm, mock.patch.object(
            parser, "_ocr_image_with_ollama", return_value=""
        ) as m_tier2, mock.patch.object(
            parser, "_parse_image_with_openai", return_value="vision-output"
        ) as m_tier3:
            out = parser.parse_resource_bytes("sample.png", b"img-bytes", "image/png")

        self.assertEqual(out, "vision-output")
        self.assertEqual(m_norm.call_count, 1)
        self.assertEqual(m_tier2.call_count, 1)
        self.assertEqual(m_tier3.call_count, 1)


if __name__ == "__main__":
    unittest.main()

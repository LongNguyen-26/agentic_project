import unittest
from unittest import mock

from devday_agent.tools import document_parser as parser


class TieredParserTests(unittest.TestCase):
    def test_pdf_delegates_to_robust_pdf_parser(self):
        parsed_text = "X" * 200
        with mock.patch.object(parser, "_parse_pdf_robust", return_value=parsed_text) as m_pdf:
            out = parser.parse_resource_bytes("sample.pdf", b"pdf-bytes", "application/pdf")

        self.assertEqual(out, parsed_text)
        self.assertEqual(m_pdf.call_count, 1)

    def test_text_file_decodes_directly(self):
        out = parser.parse_resource_bytes("sample.txt", b"hello", "text/plain")
        self.assertEqual(out, "hello")

    def test_image_uses_openai_ocr_after_normalization(self):
        with mock.patch.object(
            parser,
            "_normalize_image_for_vision",
            return_value=(b"norm", "image/jpeg"),
        ) as m_norm, mock.patch.object(
            parser, "_parse_single_image_with_openai", return_value="vision-output"
        ) as m_tier3:
            out = parser.parse_resource_bytes("sample.png", b"img-bytes", "image/png")

        self.assertEqual(out, "vision-output")
        self.assertEqual(m_norm.call_count, 1)
        self.assertEqual(m_tier3.call_count, 1)


if __name__ == "__main__":
    unittest.main()

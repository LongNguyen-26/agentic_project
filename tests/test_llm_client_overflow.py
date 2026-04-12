import unittest
from unittest import mock

from instructor.core.exceptions import InstructorRetryException
from pydantic import BaseModel

from clients.llm_client import LLMService


class _DummyResponse(BaseModel):
    value: str = "ok"


class _FakeCreate:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise InstructorRetryException(
                "maximum context length",
                n_attempts=1,
                total_usage=0,
            )
        return _DummyResponse(value="ok")


class ContextOverflowRetryTests(unittest.TestCase):
    def test_trims_messages_and_retries_on_context_overflow(self):
        fake_create = _FakeCreate()
        fake_client = mock.Mock()
        fake_client.chat.completions.create.side_effect = fake_create

        with mock.patch("clients.llm_client.instructor.from_openai", return_value=fake_client), mock.patch(
            "clients.llm_client.OpenAI"
        ), mock.patch("clients.llm_client.time.sleep", return_value=None):
            service = LLMService()
            result = service._chat_with_retries(
                response_model=_DummyResponse,
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "older user"},
                    {"role": "assistant", "content": "older assistant"},
                    {"role": "user", "content": "latest user"},
                ],
                max_completion_tokens=256,
                retry_max_output_tokens=1024,
                temperature=0.1,
            )

        self.assertEqual(result.value, "ok")
        self.assertEqual(len(fake_create.calls), 2)
        first_messages = fake_create.calls[0]["messages"]
        second_messages = fake_create.calls[1]["messages"]
        self.assertGreater(len(first_messages), len(second_messages))
        self.assertEqual(second_messages[0]["role"], "system")
        self.assertEqual(second_messages[-1]["content"], "latest user")


if __name__ == "__main__":
    unittest.main()

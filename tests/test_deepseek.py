from __future__ import annotations

import json
import unittest
from pathlib import Path

from refundpilot.deepseek_client import DeepSeekStructuredClient
from refundpilot.provider import DeepSeekProvider


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self._payload


class DeepSeekClientTest(unittest.TestCase):
    def test_posts_json_output_request_and_extracts_content(self) -> None:
        captured = {}

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["authorization"] = request.headers["Authorization"]
            captured["timeout"] = timeout
            return _FakeResponse(
                {"choices": [{"message": {"content": '{"type":"final"}'}}]}
            )

        schema = (
            Path(__file__).resolve().parents[1]
            / "refundpilot"
            / "data"
            / "agent_action.schema.json"
        )
        client = DeepSeekStructuredClient(
            model="deepseek-chat",
            api_key="test-key",
            base_url="https://example.test/v1",
            timeout_seconds=17,
            opener=opener,
        )

        result = client.complete("return JSON", schema)

        self.assertEqual(result, '{"type":"final"}')
        self.assertEqual(captured["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(captured["authorization"], "Bearer test-key")
        self.assertEqual(captured["timeout"], 17)
        self.assertEqual(captured["body"]["model"], "deepseek-chat")
        self.assertEqual(
            captured["body"]["response_format"], {"type": "json_object"}
        )

    def test_provider_uses_shared_action_contract(self) -> None:
        provider = DeepSeekProvider(
            model="deepseek-chat",
            api_key="test-key",
            base_url="https://example.test",
        )

        action = provider.decode_action(
            json.dumps(
                {
                    "type": "tool",
                    "tool_name": "get_order_details",
                    "arguments_json": '{"order_id":"#W1"}',
                    "outcome": "",
                    "message": "",
                }
            )
        )

        self.assertEqual(provider.name, "deepseek:deepseek-chat")
        self.assertEqual(action.tool_name, "get_order_details")
        self.assertEqual(action.arguments, {"order_id": "#W1"})


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import unittest
from pathlib import Path

from refundpilot.contracts import (
    AgentContext,
    FinalAction,
    ToolAction,
    ToolObservation,
)
from refundpilot.deepseek_client import DeepSeekStructuredClient
from refundpilot.provider import DeepSeekProvider
from refundpilot.user_simulator import DeepSeekUserSimulator


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

    def test_strict_schema_uses_function_call_arguments(self) -> None:
        captured = {}

        def opener(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "tool_calls": [
                                    {
                                        "function": {
                                            "arguments": '{"type":"final"}'
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            )

        schema = (
            Path(__file__).resolve().parents[1]
            / "refundpilot"
            / "data"
            / "agent_action.schema.json"
        )
        client = DeepSeekStructuredClient(
            api_key="test-key",
            base_url="https://example.test",
            strict_schema=True,
            opener=opener,
        )

        self.assertEqual(client.complete("return JSON", schema), '{"type":"final"}')
        self.assertEqual(captured["body"]["tool_choice"]["function"]["name"], "emit_agent_action")
        self.assertTrue(captured["body"]["tools"][0]["function"]["strict"])
        self.assertNotIn("response_format", captured["body"])

    def test_provider_rejects_premature_completion_placeholder(self) -> None:
        context = AgentContext(
            session_id="test",
            user_request="exchange an item",
            tool_schemas=(),
            observations=(),
            step=1,
            max_steps=30,
        )
        error = DeepSeekProvider._validation_error(
            context,
            FinalAction(outcome="needs_information", message="Task completed."),
        )

        self.assertIsNotNone(error)

    def test_provider_rejects_empty_required_tool_arguments(self) -> None:
        context = AgentContext(
            session_id="test",
            user_request="exchange an item",
            tool_schemas=(
                {
                    "type": "function",
                    "function": {
                        "name": "find_user_id_by_name_zip",
                        "parameters": {
                            "type": "object",
                            "required": ["first_name", "last_name", "zip"],
                        },
                    },
                },
            ),
            observations=(),
            step=1,
            max_steps=30,
        )
        error = DeepSeekProvider._validation_error(
            context,
            ToolAction(
                "find_user_id_by_name_zip",
                {"first_name": "", "last_name": "", "zip": "28236"},
            ),
        )

        self.assertIn("first_name", error or "")
        self.assertIn("last_name", error or "")

    def test_provider_rejects_cross_payment_refund_before_environment(self) -> None:
        context = AgentContext(
            session_id="test",
            user_request="return the item and refund to PayPal",
            tool_schemas=(),
            observations=(
                ToolObservation(
                    tool_name="get_order_details",
                    arguments={"order_id": "#W1"},
                    result={
                        "output": {
                            "observation": json.dumps(
                                {
                                    "order_id": "#W1",
                                    "payment_history": [
                                        {"payment_method_id": "credit_card_1"}
                                    ],
                                }
                            )
                        }
                    },
                ),
            ),
            step=2,
            max_steps=30,
        )
        error = DeepSeekProvider._validation_error(
            context,
            ToolAction(
                "return_delivered_order_items",
                {
                    "order_id": "#W1",
                    "item_ids": ["item-1"],
                    "payment_method_id": "paypal_1",
                },
            ),
        )

        self.assertIn("credit_card_1", error or "")
        self.assertIn("paypal_1", error or "")

    def test_user_simulator_retries_empty_opening_turn(self) -> None:
        simulator = DeepSeekUserSimulator(
            api_key="test-key",
            base_url="https://example.test",
            max_retries=1,
        )

        class FakeUserClient:
            def __init__(self):
                self.responses = [
                    '{"message":"","stop":false}',
                    '{"message":"I need help with my order.","stop":false}',
                ]

            def complete(self, prompt, schema_path):
                return self.responses.pop(0)

        simulator._client = FakeUserClient()
        self.assertEqual(
            simulator.reset("You are a customer who needs help."),
            "I need help with my order.",
        )
        self.assertEqual(simulator.call_count, 2)


if __name__ == "__main__":
    unittest.main()

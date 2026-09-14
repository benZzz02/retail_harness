from __future__ import annotations

import unittest

from refundpilot.contracts import FinalAction, ToolAction
from refundpilot.provider import CodexCliProvider


class CodexCliProviderTest(unittest.TestCase):
    def test_decodes_structured_tool_action(self) -> None:
        action = CodexCliProvider.decode_action(
            """{
                "type": "tool",
                "tool_name": "get_order_details",
                "arguments_json": "{\\\"order_id\\\": \\\"#W1\\\"}",
                "outcome": "",
                "message": ""
            }"""
        )
        self.assertIsInstance(action, ToolAction)
        self.assertEqual(action.tool_name, "get_order_details")
        self.assertEqual(action.arguments, {"order_id": "#W1"})

    def test_decodes_structured_final_action(self) -> None:
        action = CodexCliProvider.decode_action(
            """{
                "type": "final",
                "tool_name": "",
                "arguments_json": "{}",
                "outcome": "completed",
                "message": "The exchange was requested."
            }"""
        )
        self.assertIsInstance(action, FinalAction)
        self.assertEqual(action.outcome, "completed")

    def test_rejects_invalid_arguments_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid arguments_json"):
            CodexCliProvider.decode_action(
                """{
                    "type": "tool",
                    "tool_name": "get_order_details",
                    "arguments_json": "not-json",
                    "outcome": "",
                    "message": ""
                }"""
            )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from refundpilot.budget import RunBudget
from refundpilot.contracts import AgentContext, FinalAction, ToolAction, ToolObservation
from refundpilot.context_view import build_context_payload
from refundpilot.events import load_events
from refundpilot.policy import GATE_TOOL_NAME, RetailActionGate
from refundpilot.tau_runtime import TauHarnessRuntime


def _order_observation(order_id: str = "#W1") -> ToolObservation:
    return ToolObservation(
        tool_name="get_order_details",
        arguments={"order_id": order_id},
        result={
            "ok": True,
            "output": {
                "observation": json.dumps(
                    {
                        "order_id": order_id,
                        "payment_history": [
                            {"payment_method_id": "credit_card_1"}
                        ],
                    }
                )
            },
            "done": False,
        },
    )


def _context(
    *,
    request: str = "Cancel order #W1",
    observations=(),
    conversation=(),
    schemas=(),
    step: int = 2,
) -> AgentContext:
    return AgentContext(
        session_id="v2-test",
        user_request=request,
        tool_schemas=tuple(schemas),
        observations=tuple(observations),
        step=step,
        max_steps=10,
        system_instructions="Retail policy",
        conversation=tuple(conversation),
    )


class HarnessV2Test(unittest.TestCase):
    def test_compact_context_removes_duplicate_tool_payload(self) -> None:
        observation = _order_observation()
        context = _context(
            observations=(observation,),
            conversation=(
                {"role": "user", "content": "Cancel order #W1."},
                {
                    "role": "assistant",
                    "type": "tool_call",
                    "tool_name": "get_order_details",
                    "arguments": {"order_id": "#W1"},
                },
                {
                    "role": "tool",
                    "tool_name": "get_order_details",
                    "content": {"large": "duplicate payload"},
                },
            ),
        )

        compact = build_context_payload(context, compact=True)

        self.assertEqual(
            [item["role"] for item in compact["conversation"]],
            ["user", "assistant"],
        )
        self.assertEqual(
            compact["observations"][0]["observation"]["order_id"], "#W1"
        )

    def test_gate_requires_read_before_mutation(self) -> None:
        gate = RetailActionGate()
        context = _context()

        decision = gate.check(
            context, ToolAction("cancel_pending_order", {"order_id": "#W1"})
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "missing_order_snapshot")

    def test_gate_accepts_confirmed_current_scope(self) -> None:
        gate = RetailActionGate(require_confirmation=True)
        context = _context(
            observations=(_order_observation(),),
            conversation=(
                {"role": "assistant", "content": "Please confirm canceling order #W1."},
                {"role": "user", "content": "Yes, please do it."},
            ),
        )

        decision = gate.check(
            context, ToolAction("cancel_pending_order", {"order_id": "#W1"})
        )

        self.assertTrue(decision.allowed)

    def test_gate_invalidates_confirmation_after_scope_change(self) -> None:
        gate = RetailActionGate(require_confirmation=True)
        context = _context(
            observations=(_order_observation(),),
            conversation=(
                {"role": "assistant", "content": "Please confirm canceling order #W1."},
                {"role": "user", "content": "Yes, but cancel only one item."},
            ),
        )

        decision = gate.check(
            context, ToolAction("cancel_pending_order", {"order_id": "#W1"})
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.code, "missing_confirmation")

    def test_budget_stops_before_third_model_call(self) -> None:
        budget = RunBudget(max_model_calls=2)
        budget.mark_attempt()
        budget.record(
            {
                "prompt_chars": 400,
                "output_chars": 80,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
            "agent",
        )
        budget.mark_attempt()
        budget.record({"prompt_chars": 200, "output_chars": 40}, "user")

        self.assertFalse(budget.can_start_model_call())
        self.assertEqual(budget.to_dict()["attempted_model_calls"], 2)
        self.assertEqual(budget.to_dict()["total_tokens"], 120)
        self.assertEqual(budget.to_dict()["estimated_input_tokens"], 150)

    def test_guarded_runtime_blocks_bad_action_then_recovers(self) -> None:
        class SequenceProvider:
            name = "scripted-v2-test"

            def __init__(self):
                self.actions = [
                    ToolAction("cancel_pending_order", {"order_id": "#W1"}),
                    ToolAction("get_order_details", {"order_id": "#W1"}),
                    ToolAction("cancel_pending_order", {"order_id": "#W1"}),
                ]
                self.index = 0

            def next_action(self, context):
                action = self.actions[self.index]
                self.index += 1
                return action

        class FakeEnvironment:
            task_split = "test"
            task_index = 0
            instruction = "Cancel order #W1."
            policy = "Read before write."

            def __init__(self):
                self.mutations = 0

            def describe(self):
                return {"environment": "fake-retail", "task": {"index": 0}}

            def tool_schemas(self):
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "cancel_pending_order",
                            "parameters": {
                                "type": "object",
                                "required": ["order_id"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_order_details",
                            "parameters": {
                                "type": "object",
                                "required": ["order_id"],
                            },
                        },
                    },
                ]

            def data_hash(self):
                return "fake-hash-%d" % self.mutations

            def step(self, action):
                if action.tool_name == "cancel_pending_order":
                    self.mutations += 1
                    done = True
                    payload = {"status": "cancelled"}
                else:
                    done = False
                    payload = {"order_id": "#W1"}
                return ToolObservation(
                    tool_name=action.tool_name,
                    arguments=dict(action.arguments),
                    result={
                        "ok": True,
                        "output": {"observation": json.dumps(payload)},
                        "done": done,
                        "reward": 1.0 if done else 0.0,
                    },
                )

            def official_reward(self):
                return {
                    "reward": 1.0 if self.mutations == 1 else 0.0,
                    "actual_data_hash": self.data_hash(),
                    "info": {"r_actions": self.mutations == 1},
                }

        environment = FakeEnvironment()
        with tempfile.TemporaryDirectory() as directory:
            result = TauHarnessRuntime(
                SequenceProvider(),
                Path(directory),
                max_steps=5,
                gate_mode="guarded",
            ).run(environment)
            events = load_events(Path(directory), result.session_id)

        self.assertEqual(result.official_reward, 1.0)
        self.assertEqual(result.gate_rejections, 1)
        self.assertEqual(environment.mutations, 1)
        self.assertEqual(result.observations[0].tool_name, GATE_TOOL_NAME)
        self.assertTrue(any(event["type"] == "gate_rejected" for event in events))
        self.assertEqual(result.budget["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()

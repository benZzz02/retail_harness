from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from refundpilot.cases import find_case, load_cases
from refundpilot.evaluate import aggregate_evaluations, evaluate_run
from refundpilot.environment import RetailEnvironment
from refundpilot.events import load_events
from refundpilot.provider import RuleBasedProvider
from refundpilot.runtime import HarnessRuntime
from refundpilot.contracts import ToolAction


class RefundPilotHarnessTest(unittest.TestCase):
    def test_all_visible_samples_reach_expected_terminal_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime_dir = Path(directory)
            runtime = HarnessRuntime(RuleBasedProvider(), runtime_dir, max_steps=6)
            rows = []
            for case in load_cases():
                result = runtime.run(case)
                rows.append(evaluate_run(case, result))
            summary = aggregate_evaluations(rows)
            self.assertEqual(summary["tasks"], 5)
            self.assertEqual(summary["task_success_rate"], 1.0)
            self.assertEqual(summary["invalid_tool_session_rate"], 0.0)
            self.assertEqual(summary["policy_violation_session_rate"], 0.0)

    def test_trace_is_persisted_and_replayable(self) -> None:
        case = find_case("refund_happy_path")
        with tempfile.TemporaryDirectory() as directory:
            runtime_dir = Path(directory)
            result = HarnessRuntime(
                RuleBasedProvider(), runtime_dir, max_steps=6
            ).run(case)
            events = load_events(runtime_dir, result.session_id)
            self.assertEqual(events[0]["type"], "session_started")
            self.assertEqual(events[-1]["type"], "session_finished")
            self.assertEqual(
                [event["sequence"] for event in events],
                list(range(1, len(events) + 1)),
            )
            self.assertNotIn("expected", events[0]["payload"])

    def test_policy_guard_blocks_direct_expired_refund(self) -> None:
        case = find_case("refund_expired")
        with tempfile.TemporaryDirectory() as directory:
            environment = RetailEnvironment.load(
                case, Path(directory) / "state.sqlite"
            )
            observation = environment.step(
                ToolAction(
                    "create_refund",
                    {"order_id": "ORD-1002", "reason": "attempted bypass"},
                )
            )
            self.assertFalse(observation.result["ok"])
            self.assertEqual(observation.result["error_code"], "policy_violation")
            self.assertEqual(environment.snapshot()["refund_count"], 0)
            environment.close()

    def test_registry_rejects_unknown_tool_and_bad_arguments(self) -> None:
        case = find_case("refund_happy_path")
        with tempfile.TemporaryDirectory() as directory:
            environment = RetailEnvironment.load(
                case, Path(directory) / "state.sqlite"
            )
            unknown = environment.step(
                ToolAction("delete_order", {"order_id": "ORD-1001"})
            )
            missing = environment.step(ToolAction("get_order", {}))
            self.assertEqual(unknown.result["error_code"], "unknown_tool")
            self.assertEqual(missing.result["error_code"], "invalid_arguments")
            environment.close()

    def test_environment_load_step_snapshot_and_reset(self) -> None:
        case = find_case("refund_happy_path")
        with tempfile.TemporaryDirectory() as directory:
            environment = RetailEnvironment.load(
                case, Path(directory) / "state.sqlite"
            )
            description = environment.describe()
            self.assertEqual(description["environment"], "refund-retail-v1")
            self.assertEqual(description["initial_state"]["refund_count"], 0)
            self.assertNotIn("expected", description)
            environment.step(ToolAction("get_order", {"order_id": "ORD-1001"}))
            environment.step(
                ToolAction("check_refund_policy", {"order_id": "ORD-1001"})
            )
            refund = environment.step(
                ToolAction(
                    "create_refund",
                    {"order_id": "ORD-1001", "reason": "test refund"},
                )
            )
            self.assertTrue(refund.result["ok"])
            self.assertEqual(environment.snapshot()["refund_count"], 1)
            reset_state = environment.reset()
            self.assertEqual(reset_state["refund_count"], 0)
            self.assertEqual(reset_state["refunded_order_ids"], [])
            environment.close()


if __name__ == "__main__":
    unittest.main()

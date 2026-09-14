from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

try:
    import tau_bench  # noqa: F401

    TAU_BENCH_AVAILABLE = True
except ImportError:
    TAU_BENCH_AVAILABLE = False

from refundpilot.contracts import FinalAction, ToolAction
from refundpilot.events import load_events
from refundpilot.provider import OracleReplayProvider
from refundpilot.tau_adapter import TauRetailEnvironment
from refundpilot.tau_runtime import TauDialogueRuntime, TauHarnessRuntime


@unittest.skipUnless(
    TAU_BENCH_AVAILABLE,
    "external tau-bench is available after scripts/setup_tau_bench.sh",
)
class TauRetailAdapterTest(unittest.TestCase):
    def test_inspection_comes_from_external_retail_data_without_gold(self) -> None:
        environment = TauRetailEnvironment("test", 0)
        description = environment.describe()

        self.assertEqual(description["environment"], "tau-bench-retail-legacy")
        self.assertEqual(description["source"]["ownership"], "external")
        self.assertEqual(description["dataset"]["users"], 500)
        self.assertEqual(description["dataset"]["orders"], 1000)
        self.assertEqual(description["dataset"]["products"], 50)
        self.assertEqual(description["tools"]["count"], 16)
        self.assertNotIn("actions", json.dumps(description))

    def test_official_task_actions_receive_official_full_reward(self) -> None:
        environment = TauRetailEnvironment("test", 0)
        initial_hash = environment.data_hash()

        observations = [
            environment.step(action) for action in environment.oracle_actions()
        ]
        post_action_hash = environment.data_hash()
        reward = environment.official_reward()

        self.assertTrue(all(item.result["ok"] for item in observations))
        self.assertNotEqual(initial_hash, post_action_hash)
        self.assertEqual(reward["reward"], 1.0)
        self.assertTrue(reward["info"]["r_actions"])

    def test_unknown_tool_is_rejected_by_external_environment(self) -> None:
        environment = TauRetailEnvironment("test", 0)
        observation = environment.step(ToolAction("invent_order", {}))

        self.assertFalse(observation.result["ok"])
        self.assertEqual(observation.result["error_code"], "unknown_tool")

    def test_terminal_response_is_recorded_without_mutating_retail_data(self) -> None:
        environment = TauRetailEnvironment("test", 2)
        initial_hash = environment.data_hash()
        environment.record_agent_response("A tool-grounded final answer.")

        self.assertEqual(environment.data_hash(), initial_hash)
        self.assertEqual(environment._env.actions[-1].name, "respond")
        self.assertEqual(
            environment._env.actions[-1].kwargs["content"],
            "A tool-grounded final answer.",
        )

    def test_harness_persists_external_trace_and_reward(self) -> None:
        environment = TauRetailEnvironment("test", 5)
        provider = OracleReplayProvider(environment.oracle_actions())
        with tempfile.TemporaryDirectory() as directory:
            result = TauHarnessRuntime(
                provider,
                Path(directory),
                max_steps=len(environment.oracle_actions()) + 1,
            ).run(environment)
            events = load_events(Path(directory), result.session_id)

        self.assertTrue(result.wiring_ok)
        self.assertEqual(result.official_reward, 1.0)
        self.assertEqual(events[0]["payload"]["evaluation_label"], "wiring_only")
        self.assertEqual(events[-2]["type"], "environment_reward")
        self.assertEqual(events[-1]["type"], "session_finished")
        self.assertNotIn("oracle_actions", json.dumps(events[0]))

    def test_dialogue_runtime_routes_agent_responses_through_user(self) -> None:
        class SequenceProvider:
            name = "scripted-dialogue-test"

            def __init__(self, actions):
                self.actions = list(actions)
                self.index = 0

            def next_action(self, context):
                action = self.actions[self.index]
                self.index += 1
                return action

        class ScriptedUser:
            name = "scripted-user-test"

            def __init__(self):
                self.responses = 0

            def reset(self, instruction=None):
                return (
                    "I'm Yusuf Rossi in 19122. Please exchange the keyboard "
                    "and thermostat from order #W2378156."
                )

            def step(self, content):
                self.responses += 1
                if self.responses == 1:
                    return "Yes, I confirm both exchanges."
                return "###STOP###"

            def get_total_cost(self):
                return 0.0

        environment = TauRetailEnvironment("test", 0)
        gold = environment.oracle_actions()
        provider = SequenceProvider(
            gold[:4]
            + [
                FinalAction(
                    "needs_information",
                    "Please confirm both exchange item choices and payment method.",
                ),
                gold[4],
                FinalAction("completed", "Both exchanges were requested."),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = TauDialogueRuntime(
                provider,
                ScriptedUser(),
                Path(directory),
                max_steps=10,
            ).run(environment)

        self.assertTrue(result.task_success)
        self.assertEqual(result.official_reward, 1.0)
        self.assertEqual(result.user_turns, 3)
        self.assertEqual(result.premature_completion_continuations, 0)
        self.assertEqual(result.conversation[-1]["content"], "###STOP###")

    def test_dialogue_runtime_flags_continuation_after_completed_claim(self) -> None:
        class SequenceProvider:
            name = "scripted-premature-completion-test"

            def __init__(self, actions):
                self.actions = list(actions)
                self.index = 0

            def next_action(self, context):
                action = self.actions[self.index]
                self.index += 1
                return action

        class ScriptedUser:
            name = "scripted-user-continuation-test"

            def __init__(self):
                self.responses = 0

            def reset(self, instruction=None):
                return "Please make the requested exchanges."

            def step(self, content):
                self.responses += 1
                if self.responses == 1:
                    return "That is not finished yet."
                return "###STOP###"

            def get_total_cost(self):
                return 0.0

        environment = TauRetailEnvironment("test", 0)
        gold = environment.oracle_actions()
        provider = SequenceProvider(
            gold[:-1]
            + [
                FinalAction("completed", "The request is complete."),
                gold[-1],
                FinalAction("completed", "The request is now complete."),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            result = TauDialogueRuntime(
                provider,
                ScriptedUser(),
                Path(directory),
                max_steps=10,
            ).run(environment)

        self.assertEqual(result.official_reward, 1.0)
        self.assertEqual(result.premature_completion_continuations, 1)
        self.assertFalse(result.task_success)


if __name__ == "__main__":
    unittest.main()

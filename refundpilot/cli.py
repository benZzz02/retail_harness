"""Command-line demo for samples, runs, evaluation and replay."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
from uuid import uuid4

from .cases import find_case, load_cases
from .contracts import RunResult, TaskCase
from .evaluate import aggregate_evaluations, evaluate_run
from .environment import RetailEnvironment
from .events import load_events
from .provider import (
    CodexCliProvider,
    DeepSeekProvider,
    OracleReplayProvider,
    RuleBasedProvider,
)
from .runtime import HarnessRuntime
from .tau_adapter import TauBenchUnavailable, TauRetailEnvironment
from .tau_runtime import (
    TauDialogueRuntime,
    TauHarnessRuntime,
    TauRunResult,
)
from .user_simulator import CodexCliUserSimulator, DeepSeekUserSimulator


def _agent_provider(
    provider_name: str,
    model: str,
    reasoning_effort: str,
    multi_turn: bool = False,
    additional_instructions: str = "",
):
    if provider_name == "deepseek":
        return DeepSeekProvider(
            model=model,
            reasoning_effort=reasoning_effort,
            multi_turn=multi_turn,
            additional_instructions=additional_instructions,
        )
    return CodexCliProvider(
        model=model,
        reasoning_effort=reasoning_effort,
        multi_turn=multi_turn,
        additional_instructions=additional_instructions,
    )


def _user_simulator(provider_name: str, model: str, reasoning_effort: str):
    if provider_name == "deepseek":
        return DeepSeekUserSimulator(
            model=model,
            reasoning_effort=reasoning_effort,
        )
    return CodexCliUserSimulator(
        model=model,
        reasoning_effort=reasoning_effort,
    )


def _compact_output(tool_name: str, result: Dict[str, Any]) -> str:
    if not result.get("ok"):
        return "ERROR %s: %s" % (
            result.get("error_code"),
            result.get("error_message"),
        )
    output = result.get("output", {})
    if tool_name == "get_order":
        order = output.get("order", {})
        return "%s | ¥%s | %s | refunded=%s" % (
            order.get("item"),
            order.get("amount"),
            order.get("status"),
            order.get("refunded"),
        )
    if tool_name == "check_refund_policy":
        return "%s | %s" % (output.get("decision"), output.get("reason"))
    if tool_name == "create_refund":
        return "refund_id=%s | %s" % (
            output.get("refund_id"),
            output.get("status"),
        )
    if tool_name == "escalate_to_human":
        return "escalation_id=%s | %s" % (
            output.get("escalation_id"),
            output.get("status"),
        )
    return json.dumps(output, ensure_ascii=False, sort_keys=True)


def _print_result(case: TaskCase, result: RunResult, show_request: bool = True) -> None:
    evaluation = evaluate_run(case, result)
    if show_request:
        print("用户：%s" % case.user_request)
    for index, observation in enumerate(result.observations, start=1):
        print(
            "  %d. %s(%s) -> %s"
            % (
                index,
                observation.tool_name,
                json.dumps(observation.arguments, ensure_ascii=False, sort_keys=True),
                _compact_output(observation.tool_name, observation.result),
            )
        )
    print("结果：%s | %s" % (result.outcome, result.message))
    print(
        "评测：%s | steps=%d | session=%s"
        % ("PASS" if evaluation["task_success"] else "FAIL", result.steps, result.session_id)
    )


def _runtime(runtime_dir: Path) -> HarnessRuntime:
    return HarnessRuntime(
        provider=RuleBasedProvider(), runtime_dir=runtime_dir, max_steps=6
    )


def _environment_id(task_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return "env-%s-%s-%s" % (timestamp, task_id, uuid4().hex[:8])


def _load_environment(runtime_dir: Path, task_id: str) -> int:
    try:
        case = find_case(task_id)
    except KeyError as exc:
        print(str(exc))
        return 2
    environment_id = _environment_id(case.task_id)
    database_path = runtime_dir / environment_id / "state.sqlite"
    environment = RetailEnvironment.load(case, database_path)
    description = environment.describe()
    description["environment_id"] = environment_id
    print(json.dumps(description, ensure_ascii=False, indent=2, sort_keys=True))
    environment.close()
    return 0


def _samples() -> int:
    for case in load_cases():
        expected = case.expected.get("outcome", "unknown")
        print("%-28s %-18s %s" % (case.task_id, "expected=" + expected, case.description))
    return 0


def _run_one(runtime_dir: Path, task_id: str) -> int:
    try:
        case = find_case(task_id)
    except KeyError as exc:
        print(str(exc))
        return 2
    result = _runtime(runtime_dir).run(case)
    _print_result(case, result)
    return 0 if evaluate_run(case, result)["task_success"] else 1


def _demo(runtime_dir: Path) -> int:
    rows: List[Dict[str, Any]] = []
    cases = load_cases()
    runtime = _runtime(runtime_dir)
    for index, case in enumerate(cases, start=1):
        print("\n[%d/%d] %s — %s" % (index, len(cases), case.task_id, case.description))
        result = runtime.run(case)
        _print_result(case, result)
        rows.append(evaluate_run(case, result))
    summary = aggregate_evaluations(rows)
    print("\n=== Aggregate ===")
    print("tasks: %d" % summary["tasks"])
    print("task_success_rate: %.1f%%" % (summary["task_success_rate"] * 100))
    print(
        "invalid_tool_session_rate: %.1f%%"
        % (summary["invalid_tool_session_rate"] * 100)
    )
    print(
        "policy_violation_session_rate: %.1f%%"
        % (summary["policy_violation_session_rate"] * 100)
    )
    print("average_steps: %.2f" % summary["average_steps"])
    print("outcomes: %s" % json.dumps(summary["outcomes"], ensure_ascii=False))
    return 0 if summary["task_success_rate"] == 1.0 else 1


def _replay(runtime_dir: Path, session_id: str) -> int:
    try:
        events = load_events(runtime_dir, session_id)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        return 2
    for event in events:
        print(
            "[%02d] %-16s %s"
            % (
                event["sequence"],
                event["type"],
                json.dumps(event["payload"], ensure_ascii=False, sort_keys=True),
            )
        )
    return 0


def _tau_inspect(task_split: str, task_index: int) -> int:
    try:
        environment = TauRetailEnvironment(task_split, task_index)
    except (TauBenchUnavailable, ValueError) as exc:
        print(str(exc))
        return 2
    print(json.dumps(environment.describe(), ensure_ascii=False, indent=2))
    return 0


def _compact_tau_observation(result: Dict[str, Any], limit: int = 220) -> str:
    if not result.get("ok"):
        text = "%s: %s" % (
            result.get("error_code"),
            result.get("error_message"),
        )
    else:
        text = str(result.get("output", {}).get("observation", ""))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _print_tau_result(
    environment: TauRetailEnvironment, result: TauRunResult
) -> None:
    print("\n[%s/%d] %s" % (
        environment.task_split,
        environment.task_index,
        environment.instruction,
    ))
    for index, observation in enumerate(result.observations, start=1):
        print(
            "  %d. %s(%s) -> %s"
            % (
                index,
                observation.tool_name,
                json.dumps(
                    observation.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                _compact_tau_observation(observation.result),
            )
        )
    print(
        "Official reward: %.1f | wiring=%s | session=%s"
        % (
            result.official_reward,
            "PASS" if result.wiring_ok else "FAIL",
            result.session_id,
        )
    )


def _tau_smoke(runtime_dir: Path, task_split: str, task_indices: List[int]) -> int:
    print("=== ORACLE WIRING CHECK — NOT A MODEL EVALUATION ===")
    print(
        "Gold actions are used only to verify the external environment, tools, "
        "state mutation, reward and trace path."
    )
    passed = 0
    for task_index in task_indices:
        try:
            environment = TauRetailEnvironment(task_split, task_index)
        except (TauBenchUnavailable, ValueError) as exc:
            print(str(exc))
            return 2
        actions = environment.oracle_actions()
        runtime = TauHarnessRuntime(
            provider=OracleReplayProvider(actions),
            runtime_dir=runtime_dir,
            max_steps=len(actions) + 1,
        )
        result = runtime.run(environment)
        _print_tau_result(environment, result)
        passed += int(result.wiring_ok)
    print("\nWiring summary: %d/%d passed" % (passed, len(task_indices)))
    return 0 if passed == len(task_indices) else 1


def _tau_llm(
    runtime_dir: Path,
    task_split: str,
    task_index: int,
    provider_name: str,
    model: str,
    reasoning_effort: str,
    max_steps: int,
) -> int:
    print("=== BLIND ONE-TURN LLM RUN — GOLD ACTIONS HIDDEN ===")
    print(
        "No live user simulator is attached; the initial explicit request is "
        "treated as authorization. This is not an official tau-bench score."
    )
    try:
        environment = TauRetailEnvironment(task_split, task_index)
        provider = _agent_provider(
            provider_name=provider_name,
            model=model,
            reasoning_effort=reasoning_effort,
            additional_instructions=(
                "This is a one-turn offline LLM smoke test with no live user "
                "simulator. Treat the user's initial explicit request to make a "
                "change as their final authorization after you have gathered all "
                "required details. Do not stop to ask for confirmation. Gold "
                "actions and expected state are not available."
            ),
        )
    except (TauBenchUnavailable, RuntimeError, ValueError) as exc:
        print(str(exc))
        return 2

    result = TauHarnessRuntime(
        provider=provider,
        runtime_dir=runtime_dir,
        max_steps=max_steps,
    ).run(environment)
    print("\n[%s/%d] %s" % (task_split, task_index, environment.instruction))
    for index, observation in enumerate(result.observations, start=1):
        print(
            "  %d. %s(%s) -> %s"
            % (
                index,
                observation.tool_name,
                json.dumps(
                    observation.arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                _compact_tau_observation(observation.result),
            )
        )
    task_success = (
        result.official_reward == 1.0
        and all(item.result.get("ok") for item in result.observations)
        and result.outcome != "runtime_error"
    )
    print("Final: %s | %s" % (result.outcome, result.message))
    print(
        "Official state reward: %.1f | one-turn LLM=%s | session=%s"
        % (
            result.official_reward,
            "PASS" if task_success else "FAIL",
            result.session_id,
        )
    )
    return 0 if task_success else 1


def _tau_dialogue(
    runtime_dir: Path,
    task_split: str,
    task_index: int,
    agent_provider_name: str,
    user_provider_name: str,
    agent_model: str,
    user_model: str,
    reasoning_effort: str,
    max_steps: int,
) -> int:
    print("=== MULTI-TURN LLM RUN — HIDDEN USER INSTRUCTION ===")
    print(
        "The agent sees only simulated user utterances; confirmations and "
        "request changes are handled through tau-bench Env.step(respond)."
    )
    try:
        environment = TauRetailEnvironment(task_split, task_index)
        provider = _agent_provider(
            provider_name=agent_provider_name,
            model=agent_model,
            reasoning_effort=reasoning_effort,
            multi_turn=True,
            additional_instructions=(
                "This is a multi-turn retail conversation. You cannot see the "
                "user simulator's hidden instruction. Follow the Retail policy "
                "exactly: authenticate first, gather necessary details, and ask "
                "for explicit confirmation before every consequential mutation. "
                "Honor the user's latest scope if they change their mind. Use a "
                "final action to send one message to the customer; the simulator "
                "will then reply or stop."
            ),
        )
        user_simulator = _user_simulator(
            provider_name=user_provider_name,
            model=user_model,
            reasoning_effort=reasoning_effort,
        )
    except (TauBenchUnavailable, RuntimeError, ValueError) as exc:
        print(str(exc))
        return 2

    result = TauDialogueRuntime(
        provider=provider,
        user_simulator=user_simulator,
        runtime_dir=runtime_dir,
        max_steps=max_steps,
    ).run(environment)
    print("\n[%s/%d] visible conversation" % (task_split, task_index))
    for item in result.conversation:
        role = item.get("role")
        if role == "user":
            print("  Customer: %s" % item.get("content", ""))
        elif role == "assistant" and item.get("type") == "tool_call":
            print(
                "  Agent tool: %s(%s)"
                % (
                    item.get("tool_name"),
                    json.dumps(
                        item.get("arguments", {}),
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            )
        elif role == "assistant":
            print("  Agent: %s" % item.get("content", ""))
        elif role == "tool":
            content = dict(item.get("content", {}))
            print(
                "  Tool: %s -> %s"
                % (item.get("tool_name"), _compact_tau_observation(content))
            )
    print(
        "Official reward: %.1f | dialogue=%s | steps=%d | user_turns=%d | "
        "completion_continuations=%d | session=%s"
        % (
            result.official_reward,
            "PASS" if result.task_success else "FAIL",
            result.steps,
            result.user_turns,
            result.premature_completion_continuations,
            result.session_id,
        )
    )
    return 0 if result.task_success else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refundpilot",
        description="Agent Harness with an external tau-bench Retail backend",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path.cwd() / ".refundpilot_runs",
        help="directory used for local fixture state and append-only traces",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("samples", help="list the bundled visible cases")
    env_parser = subparsers.add_parser(
        "env", help="inspect one bundled local fixture (not tau-bench)"
    )
    env_parser.add_argument("--case", required=True, dest="task_id")
    run_parser = subparsers.add_parser("run", help="run one sample case")
    run_parser.add_argument("--case", required=True, dest="task_id")
    subparsers.add_parser("demo", help="run and evaluate every sample case")
    replay_parser = subparsers.add_parser("replay", help="print one saved trace")
    replay_parser.add_argument("session_id")
    tau_inspect_parser = subparsers.add_parser(
        "tau-inspect", help="load and inspect the external tau-bench Retail env"
    )
    tau_inspect_parser.add_argument(
        "--split", choices=("train", "dev", "test"), default="test"
    )
    tau_inspect_parser.add_argument("--task", type=int, default=0)
    tau_smoke_parser = subparsers.add_parser(
        "tau-smoke", help="replay gold actions as an external wiring check"
    )
    tau_smoke_parser.add_argument(
        "--split", choices=("train", "dev", "test"), default="test"
    )
    tau_smoke_parser.add_argument(
        "--tasks", nargs="+", type=int, default=[0, 5, 18]
    )
    tau_llm_parser = subparsers.add_parser(
        "tau-llm", help="run a blind one-turn task with a real Codex LLM"
    )
    tau_llm_parser.add_argument(
        "--split", choices=("train", "dev", "test"), default="test"
    )
    tau_llm_parser.add_argument("--task", type=int, default=0)
    tau_llm_parser.add_argument(
        "--provider", choices=("codex", "deepseek"), default="codex"
    )
    tau_llm_parser.add_argument("--model", default="gpt-5.6-luna")
    tau_llm_parser.add_argument(
        "--reasoning", choices=("low", "medium", "high"), default="low"
    )
    tau_llm_parser.add_argument("--max-steps", type=int, default=10)
    tau_dialogue_parser = subparsers.add_parser(
        "tau-dialogue", help="run agent and hidden-instruction user simulator"
    )
    tau_dialogue_parser.add_argument(
        "--split", choices=("train", "dev", "test"), default="test"
    )
    tau_dialogue_parser.add_argument("--task", type=int, default=0)
    tau_dialogue_parser.add_argument(
        "--agent-provider", choices=("codex", "deepseek"), default="codex"
    )
    tau_dialogue_parser.add_argument(
        "--user-provider", choices=("codex", "deepseek"), default="codex"
    )
    tau_dialogue_parser.add_argument("--agent-model", default="gpt-5.6-luna")
    tau_dialogue_parser.add_argument("--user-model", default="gpt-5.6-luna")
    tau_dialogue_parser.add_argument(
        "--reasoning", choices=("low", "medium", "high"), default="low"
    )
    tau_dialogue_parser.add_argument("--max-steps", type=int, default=30)
    return parser


def main(argv: Iterable[str] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "samples":
        return _samples()
    if args.command == "env":
        return _load_environment(args.runtime_dir, args.task_id)
    if args.command == "run":
        return _run_one(args.runtime_dir, args.task_id)
    if args.command == "demo":
        return _demo(args.runtime_dir)
    if args.command == "replay":
        return _replay(args.runtime_dir, args.session_id)
    if args.command == "tau-inspect":
        return _tau_inspect(args.split, args.task)
    if args.command == "tau-smoke":
        return _tau_smoke(args.runtime_dir, args.split, args.tasks)
    if args.command == "tau-llm":
        return _tau_llm(
            args.runtime_dir,
            args.split,
            args.task,
            args.provider,
            args.model,
            args.reasoning,
            args.max_steps,
        )
    if args.command == "tau-dialogue":
        return _tau_dialogue(
            args.runtime_dir,
            args.split,
            args.task,
            args.agent_provider,
            args.user_provider,
            args.agent_model,
            args.user_model,
            args.reasoning,
            args.max_steps,
        )
    return 2

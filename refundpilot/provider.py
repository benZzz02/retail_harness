"""Provider abstractions and deterministic offline demonstration providers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Protocol, Sequence

from .contracts import (
    AgentAction,
    AgentContext,
    FinalAction,
    ToolAction,
    ToolObservation,
)
from .codex_client import CodexStructuredClient
from .deepseek_client import DeepSeekStructuredClient


class AgentProvider(Protocol):
    name: str

    def next_action(self, context: AgentContext) -> AgentAction:
        ...


class RuleBasedProvider:
    """Offline provider that exercises the exact same action boundary as an LLM."""

    name = "rule-based-offline"
    _order_pattern = re.compile(r"\bORD-\d+\b", re.IGNORECASE)

    @staticmethod
    def _latest(
        context: AgentContext, tool_name: str
    ) -> Optional[ToolObservation]:
        for observation in reversed(context.observations):
            if observation.tool_name == tool_name:
                return observation
        return None

    def next_action(self, context: AgentContext) -> AgentAction:
        match = self._order_pattern.search(context.user_request)
        if match is None:
            return FinalAction(
                outcome="needs_information",
                message="请提供格式为 ORD-数字 的订单号。",
            )
        order_id = match.group(0).upper()

        order_observation = self._latest(context, "get_order")
        if order_observation is None:
            return ToolAction("get_order", {"order_id": order_id})
        if not order_observation.result.get("ok"):
            return FinalAction(
                outcome="needs_information",
                message="未找到该订单，请核对订单号后重试。",
            )

        policy_observation = self._latest(context, "check_refund_policy")
        if policy_observation is None:
            return ToolAction("check_refund_policy", {"order_id": order_id})
        if not policy_observation.result.get("ok"):
            return FinalAction(
                outcome="system_error",
                message="退款策略检查失败，请稍后重试。",
            )

        decision = policy_observation.result.get("output", {}).get("decision")
        reason = policy_observation.result.get("output", {}).get("reason", "")
        if decision == "deny":
            return FinalAction(outcome="denied", message=reason)
        if decision == "manual_review":
            escalation = self._latest(context, "escalate_to_human")
            if escalation is None:
                return ToolAction(
                    "escalate_to_human", {"order_id": order_id, "reason": reason}
                )
            if escalation.result.get("ok"):
                return FinalAction(
                    outcome="escalated", message="退款请求已提交人工复核。"
                )
            return FinalAction(
                outcome="system_error", message="人工复核队列写入失败。"
            )
        if decision == "auto_refund":
            refund = self._latest(context, "create_refund")
            if refund is None:
                return ToolAction(
                    "create_refund",
                    {"order_id": order_id, "reason": "用户申请七天内退款"},
                )
            if refund.result.get("ok"):
                return FinalAction(outcome="refunded", message="退款已创建。")
            return FinalAction(
                outcome="system_error", message="退款创建失败，请稍后重试。"
            )
        return FinalAction(
            outcome="system_error", message="策略工具返回了未知决策。"
        )


class OracleReplayProvider:
    """Replay benchmark gold actions to verify environment wiring only.

    This provider must never be presented as an agent evaluation. It consumes
    task answers that are intentionally hidden from normal providers.
    """

    name = "tau-oracle-wiring-smoke"

    def __init__(self, actions: Sequence[ToolAction]) -> None:
        self._actions = tuple(actions)

    def next_action(self, context: AgentContext) -> AgentAction:
        action_index = len(context.observations)
        if action_index < len(self._actions):
            action = self._actions[action_index]
            return ToolAction(action.tool_name, dict(action.arguments))
        return FinalAction(
            outcome="oracle_replay_complete",
            message="Official task actions were replayed through the adapter.",
        )


class CodexCliProvider:
    """Use an authenticated local Codex CLI as a real LLM action selector."""

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        reasoning_effort: str = "low",
        timeout_seconds: int = 180,
        additional_instructions: str = "",
        multi_turn: bool = False,
        executable: Optional[str] = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.additional_instructions = additional_instructions
        self.multi_turn = multi_turn
        self.name = "codex-cli:%s" % model
        self._client = CodexStructuredClient(
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            executable=executable,
        )
        self.executable = self._client.executable
        self._schema_path = (
            Path(__file__).resolve().parent / "data" / "agent_action.schema.json"
        )

    @staticmethod
    def decode_action(raw: str) -> AgentAction:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("LLM action must be a JSON object")

        action_type = payload.get("type")
        if action_type == "final":
            return FinalAction(
                outcome=str(payload.get("outcome") or "completed"),
                message=str(payload.get("message") or "Task completed."),
            )
        if action_type != "tool":
            raise ValueError("LLM action type must be 'tool' or 'final'")

        tool_name = str(payload.get("tool_name") or "")
        if not tool_name:
            raise ValueError("LLM tool action omitted tool_name")
        try:
            arguments = json.loads(str(payload.get("arguments_json") or "{}"))
        except json.JSONDecodeError as exc:
            raise ValueError("LLM returned invalid arguments_json") from exc
        if not isinstance(arguments, dict):
            raise ValueError("LLM tool arguments must decode to an object")

        # Preserve unknown names as actions so the environment boundary, trace,
        # and invalid-tool metrics can observe the model error.
        return ToolAction(tool_name=tool_name, arguments=arguments)

    def _build_prompt(self, context: AgentContext) -> str:
        history = [item.to_dict() for item in context.observations]
        payload = {
            "step": context.step,
            "max_steps": context.max_steps,
            "retail_policy": context.system_instructions,
            "runtime_note": self.additional_instructions,
            "user_task": context.user_request,
            "conversation": list(context.conversation),
            "available_tools": list(context.tool_schemas),
            "observations": history,
        }
        action_guidance = (
            "Use type=final to send exactly one natural-language turn to the "
            "customer. A final action does not necessarily end the session: use "
            "it to ask questions or request explicit confirmation, and the user "
            "simulator may reply. Before any state-changing tool, summarize the "
            "exact current items, options, payment/refund method, and amount when "
            "known, then obtain explicit confirmation for that unchanged scope. "
            "If the customer changes any part of the request while responding to "
            "a confirmation, the old confirmation is invalid: incorporate the "
            "change and ask for fresh confirmation before acting. Treat a response "
            "that changes scope as a change, not authorization, even if it also "
            "contains words such as yes or proceed. Set outcome=needs_information "
            "for a question and outcome=completed only when reporting completion."
            if self.multi_turn
            else (
                "Use type=final when the task is complete. A final message must "
                "directly and fully answer the user's request using all relevant "
                "facts learned from tools."
            )
        )
        return (
            "You are the decision engine inside an e-commerce Agent Harness. "
            "Choose exactly one next action from the supplied state. Do not use "
            "shell commands, files, web browsing, or any tool outside the listed "
            "retail tools. Never invent IDs or facts. Use type=tool with one "
            "tool_name and a JSON-encoded object in arguments_json. "
            + action_guidance
            + " For unused string fields return an empty string. Return only the "
            "object required by the output schema.\n\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )

    def next_action(self, context: AgentContext) -> AgentAction:
        raw = self._client.complete(self._build_prompt(context), self._schema_path)
        return self.decode_action(raw)


class DeepSeekProvider(CodexCliProvider):
    """Use DeepSeek's JSON-output API at the same action boundary as Codex."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        reasoning_effort: str = "low",
        timeout_seconds: int = 180,
        additional_instructions: str = "",
        multi_turn: bool = False,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout_seconds = timeout_seconds
        self.additional_instructions = additional_instructions
        self.multi_turn = multi_turn
        self.max_retries = max(0, max_retries)
        self.name = "deepseek:%s" % model
        self._client = DeepSeekStructuredClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            strict_schema=True,
        )
        self._schema_path = (
            Path(__file__).resolve().parent / "data" / "agent_action.schema.json"
        )

    @staticmethod
    def _validation_error(context: AgentContext, action: AgentAction) -> Optional[str]:
        if isinstance(action, ToolAction):
            for schema in context.tool_schemas:
                function = schema.get("function", {})
                if function.get("name") != action.tool_name:
                    continue
                parameters = function.get("parameters", {})
                required = parameters.get("required", [])
                missing = [
                    name
                    for name in required
                    if name not in action.arguments
                    or action.arguments.get(name) is None
                    or (
                        isinstance(action.arguments.get(name), str)
                        and not action.arguments.get(name).strip()
                    )
                ]
                if missing:
                    return (
                        "Do not call %s with empty required arguments: %s. "
                        "Ask the customer for the missing information first."
                        % (action.tool_name, ", ".join(missing))
                    )
                break
        if not isinstance(action, FinalAction):
            return None
        normalized = " ".join(action.message.lower().split()).strip(".!?")
        if normalized in {"task completed", "completed", "done"}:
            if action.outcome == "needs_information":
                return (
                    "This is not a valid information request. Ask one concrete "
                    "question or request confirmation; never use a completion "
                    "placeholder for needs_information."
                )
            if context.step == 1 or not context.observations:
                return (
                    "Do not claim completion before inspecting the customer "
                    "request and taking any required tool actions."
                )
        return None

    def _strict_prompt(self, prompt: str, feedback: str = "") -> str:
        instruction = (
            "DeepSeek action contract: output one schema-valid action. Never use "
            "'Task completed.' as a placeholder. For needs_information, the "
            "message must ask a concrete question or request explicit approval. "
            "Only report completion after the required state-changing tool has "
            "successfully executed; otherwise continue querying or ask the user."
        )
        if feedback:
            instruction += " Previous output was rejected: " + feedback
        return prompt + "\n\n" + instruction

    def next_action(self, context: AgentContext) -> AgentAction:
        feedback = ""
        for _ in range(self.max_retries + 1):
            raw = self._client.complete(
                self._strict_prompt(self._build_prompt(context), feedback),
                self._schema_path,
            )
            action = self.decode_action(raw)
            feedback = self._validation_error(context, action) or ""
            if not feedback:
                return action
        raise ValueError("DeepSeek action failed semantic validation: %s" % feedback)

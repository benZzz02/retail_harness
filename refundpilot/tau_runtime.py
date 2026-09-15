"""Bounded harness loop for the external tau-bench Retail adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from .contracts import AgentContext, FinalAction, ToolAction, ToolObservation
from .budget import RunBudget
from .events import EventLogger
from .policy import GATE_TOOL_NAME, RetailActionGate, gate_observation
from .provider import AgentProvider
from .tau_adapter import TauRetailEnvironment


def _model_backed(owner: Any) -> bool:
    return hasattr(owner, "_client")


def _client_metrics(owner: Any) -> Dict[str, Any]:
    client = getattr(owner, "_client", None)
    metrics = getattr(client, "last_call_metrics", {})
    return dict(metrics) if isinstance(metrics, dict) else {}


def _actual_tool_calls(observations: List[ToolObservation]) -> List[ToolObservation]:
    return [item for item in observations if item.tool_name != GATE_TOOL_NAME]


@dataclass
class TauRunResult:
    session_id: str
    task_split: str
    task_index: int
    provider: str
    outcome: str
    message: str
    steps: int
    observations: List[ToolObservation]
    initial_data_hash: str
    post_action_data_hash: str
    official_reward: float
    reward_info: Dict[str, Any]
    event_path: Path
    gate_rejections: int = 0
    budget: Dict[str, Any] = field(default_factory=dict)

    @property
    def wiring_ok(self) -> bool:
        return (
            self.provider == "tau-oracle-wiring-smoke"
            and self.outcome in {"oracle_replay_complete", "environment_done"}
            and all(item.result.get("ok") for item in self.observations)
            and self.official_reward == 1.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_split": self.task_split,
            "task_index": self.task_index,
            "provider": self.provider,
            "outcome": self.outcome,
            "message": self.message,
            "steps": self.steps,
            "observations": [item.to_dict() for item in self.observations],
            "initial_data_hash": self.initial_data_hash,
            "post_action_data_hash": self.post_action_data_hash,
            "official_reward": self.official_reward,
            "reward_info": dict(self.reward_info),
            "gate_rejections": self.gate_rejections,
            "budget": dict(self.budget),
            "wiring_ok": self.wiring_ok,
            "event_path": str(self.event_path),
        }


class TauHarnessRuntime:
    """Run any AgentProvider against an already loaded external environment."""

    def __init__(
        self,
        provider: AgentProvider,
        runtime_dir: Path,
        max_steps: int = 30,
        gate_mode: str = "off",
        max_model_calls: int | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if gate_mode not in {"off", "audit", "guarded"}:
            raise ValueError("gate_mode must be 'off', 'audit' or 'guarded'")
        self.provider = provider
        self.runtime_dir = runtime_dir
        self.max_steps = max_steps
        self.gate_mode = gate_mode
        self.action_gate = RetailActionGate(require_confirmation=False)
        self.max_model_calls = max_model_calls

    @staticmethod
    def _new_session_id(task_split: str, task_index: int) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return "tau-%s-%03d-%s-%s" % (
            task_split,
            task_index,
            timestamp,
            uuid4().hex[:8],
        )

    def run(self, environment: TauRetailEnvironment) -> TauRunResult:
        session_id = self._new_session_id(
            environment.task_split, environment.task_index
        )
        event_path = self.runtime_dir / session_id / "events.jsonl"
        logger = EventLogger(event_path)
        observations: List[ToolObservation] = []
        outcome = "max_steps_exceeded"
        message = "Agent reached the maximum number of steps."
        steps = 0
        gate_rejections = 0
        budget = RunBudget(max_model_calls=self.max_model_calls)
        description = environment.describe()
        initial_hash = environment.data_hash()

        logger.append(
            "session_started",
            {
                "session_id": session_id,
                "provider": self.provider.name,
                "environment": description,
                "max_steps": self.max_steps,
                "gate_mode": self.gate_mode,
                "max_model_calls": self.max_model_calls,
                "evaluation_label": "wiring_only"
                if self.provider.name == "tau-oracle-wiring-smoke"
                else "agent_run",
            },
        )

        try:
            for step in range(1, self.max_steps + 1):
                steps = step
                if _model_backed(self.provider):
                    if not budget.can_start_model_call():
                        outcome = "budget_exhausted"
                        message = "Model-call budget exhausted before the next Agent turn."
                        logger.append("budget_exhausted", budget.to_dict())
                        break
                    budget.mark_attempt()
                context = AgentContext(
                    session_id=session_id,
                    user_request=environment.instruction,
                    tool_schemas=tuple(environment.tool_schemas()),
                    observations=tuple(observations),
                    step=step,
                    max_steps=self.max_steps,
                    system_instructions=environment.policy,
                )
                try:
                    action = self.provider.next_action(context)
                finally:
                    if _model_backed(self.provider):
                        budget.record(_client_metrics(self.provider), "agent")
                logger.append("agent_action", action.to_dict())

                decision = self.action_gate.check(context, action)
                if not decision.allowed:
                    gate_rejections += 1
                    gate_payload = {
                        "action": action.to_dict(),
                        **decision.to_dict(),
                    }
                    if self.gate_mode == "audit":
                        logger.append("policy_violation", gate_payload)
                    elif self.gate_mode == "guarded":
                        logger.append("gate_rejected", gate_payload)
                        observations.append(gate_observation(action, decision))
                        logger.append(
                            "harness_observation",
                            observations[-1].to_dict(),
                        )
                        continue

                if isinstance(action, FinalAction):
                    environment.record_agent_response(action.message)
                    outcome = action.outcome
                    message = action.message
                    break
                if not isinstance(action, ToolAction):
                    outcome = "runtime_error"
                    message = "Provider returned an unsupported action."
                    logger.append("runtime_error", {"message": message})
                    break

                observation = environment.step(action)
                observations.append(observation)
                logger.append("tool_result", observation.to_dict())
                if observation.result.get("done"):
                    outcome = "environment_done"
                    message = "The external environment reached a terminal tool."
                    break
        except Exception as exc:
            outcome = "runtime_error"
            message = "Runtime boundary caught an error: %s" % exc
            logger.append("runtime_error", {"message": message})

        post_action_hash = environment.data_hash()
        reward_result = environment.official_reward()
        official_reward = float(reward_result["reward"])
        reward_info = dict(reward_result.get("info", {}))
        logger.append("environment_reward", reward_result)
        logger.append(
            "session_finished",
            {
                "outcome": outcome,
                "message": message,
                "steps": steps,
                "tool_calls": len(_actual_tool_calls(observations)),
                "all_tool_calls_ok": all(
                    item.result.get("ok") for item in _actual_tool_calls(observations)
                ),
                "gate_rejections": gate_rejections,
                "budget": budget.to_dict(),
                "post_action_data_hash": post_action_hash,
                "official_reward": official_reward,
            },
        )
        return TauRunResult(
            session_id=session_id,
            task_split=environment.task_split,
            task_index=environment.task_index,
            provider=self.provider.name,
            outcome=outcome,
            message=message,
            steps=steps,
            observations=observations,
            initial_data_hash=initial_hash,
            post_action_data_hash=post_action_hash,
            official_reward=official_reward,
            reward_info=reward_info,
            event_path=event_path,
            gate_rejections=gate_rejections,
            budget=budget.to_dict(),
        )


@dataclass
class TauDialogueResult:
    session_id: str
    task_split: str
    task_index: int
    agent_provider: str
    user_simulator: str
    outcome: str
    message: str
    steps: int
    user_turns: int
    premature_completion_continuations: int
    observations: List[ToolObservation]
    conversation: List[Dict[str, Any]]
    official_reward: float
    reward_info: Dict[str, Any]
    actual_data_hash: str | None
    event_path: Path
    gate_rejections: int = 0
    budget: Dict[str, Any] = field(default_factory=dict)

    @property
    def task_success(self) -> bool:
        return (
            self.outcome == "environment_done"
            and self.official_reward == 1.0
            and all(
                item.result.get("ok")
                for item in _actual_tool_calls(self.observations)
            )
            and self.premature_completion_continuations == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_split": self.task_split,
            "task_index": self.task_index,
            "agent_provider": self.agent_provider,
            "user_simulator": self.user_simulator,
            "outcome": self.outcome,
            "message": self.message,
            "steps": self.steps,
            "user_turns": self.user_turns,
            "premature_completion_continuations": (
                self.premature_completion_continuations
            ),
            "observations": [item.to_dict() for item in self.observations],
            "conversation": list(self.conversation),
            "official_reward": self.official_reward,
            "reward_info": dict(self.reward_info),
            "actual_data_hash": self.actual_data_hash,
            "gate_rejections": self.gate_rejections,
            "budget": dict(self.budget),
            "task_success": self.task_success,
            "event_path": str(self.event_path),
        }


class TauDialogueRuntime:
    """Alternate agent actions with a hidden-instruction user simulator."""

    def __init__(
        self,
        provider: AgentProvider,
        user_simulator: Any,
        runtime_dir: Path,
        max_steps: int = 30,
        gate_mode: str = "off",
        max_model_calls: int | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if gate_mode not in {"off", "audit", "guarded"}:
            raise ValueError("gate_mode must be 'off', 'audit' or 'guarded'")
        self.provider = provider
        self.user_simulator = user_simulator
        self.runtime_dir = runtime_dir
        self.max_steps = max_steps
        self.gate_mode = gate_mode
        self.action_gate = RetailActionGate(require_confirmation=True)
        self.max_model_calls = max_model_calls

    def run(self, environment: TauRetailEnvironment) -> TauDialogueResult:
        session_id = TauHarnessRuntime._new_session_id(
            environment.task_split, environment.task_index
        )
        event_path = self.runtime_dir / session_id / "events.jsonl"
        logger = EventLogger(event_path)
        observations: List[ToolObservation] = []
        conversation: List[Dict[str, Any]] = []
        outcome = "max_steps_exceeded"
        message = "Agent reached the maximum number of dialogue steps."
        steps = 0
        user_turns = 0
        premature_completion_continuations = 0
        gate_rejections = 0
        budget = RunBudget(max_model_calls=self.max_model_calls)

        logger.append(
            "session_started",
            {
                "session_id": session_id,
                "provider": self.provider.name,
                "user_simulator": self.user_simulator.name,
                "environment": environment.describe(),
                "max_steps": self.max_steps,
                "gate_mode": self.gate_mode,
                "max_model_calls": self.max_model_calls,
                "evaluation_label": "multi_turn_agent_run",
                "agent_context": "current user utterance and visible transcript only",
            },
        )

        try:
            initial_user_call = _model_backed(self.user_simulator)
            if initial_user_call and not budget.can_start_model_call():
                outcome = "budget_exhausted"
                message = "Model-call budget exhausted before the opening user turn."
                logger.append("budget_exhausted", budget.to_dict())
                current_user_message = ""
            else:
                if initial_user_call:
                    budget.mark_attempt()
                try:
                    current_user_message = environment.start_conversation(
                        self.user_simulator
                    )
                finally:
                    if initial_user_call:
                        budget.record(_client_metrics(self.user_simulator), "user")
            user_turns = 1
            if current_user_message:
                conversation.append(
                    {"role": "user", "content": current_user_message}
                )
                logger.append(
                    "user_message",
                    {"content": current_user_message, "initial": True},
                )

            for step in range(1, self.max_steps + 1):
                if outcome == "budget_exhausted":
                    break
                steps = step
                if _model_backed(self.provider):
                    if not budget.can_start_model_call():
                        outcome = "budget_exhausted"
                        message = "Model-call budget exhausted before the next Agent turn."
                        logger.append("budget_exhausted", budget.to_dict())
                        break
                    budget.mark_attempt()
                context = AgentContext(
                    session_id=session_id,
                    user_request=current_user_message,
                    tool_schemas=tuple(environment.tool_schemas()),
                    observations=tuple(observations),
                    step=step,
                    max_steps=self.max_steps,
                    system_instructions=environment.policy,
                    conversation=tuple(dict(item) for item in conversation),
                )
                try:
                    action = self.provider.next_action(context)
                finally:
                    if _model_backed(self.provider):
                        budget.record(_client_metrics(self.provider), "agent")
                logger.append("agent_action", action.to_dict())

                decision = self.action_gate.check(context, action)
                if not decision.allowed:
                    gate_rejections += 1
                    gate_payload = {
                        "action": action.to_dict(),
                        **decision.to_dict(),
                    }
                    if self.gate_mode == "audit":
                        logger.append("policy_violation", gate_payload)
                    elif self.gate_mode == "guarded":
                        logger.append("gate_rejected", gate_payload)
                        observations.append(gate_observation(action, decision))
                        logger.append(
                            "harness_observation",
                            observations[-1].to_dict(),
                        )
                        conversation.append(
                            {
                                "role": "harness",
                                "type": "gate_rejection",
                                "code": decision.code,
                                "content": decision.message,
                            }
                        )
                        continue

                if isinstance(action, ToolAction):
                    observation = environment.step(action)
                    observations.append(observation)
                    logger.append("tool_result", observation.to_dict())
                    conversation.extend(
                        [
                            {
                                "role": "assistant",
                                "type": "tool_call",
                                "tool_name": action.tool_name,
                                "arguments": dict(action.arguments),
                            },
                            {
                                "role": "tool",
                                "tool_name": observation.tool_name,
                                "content": dict(observation.result),
                            },
                        ]
                    )
                    if observation.result.get("done"):
                        outcome = "environment_done"
                        message = "The external environment reached a terminal tool."
                        break
                    continue

                if not isinstance(action, FinalAction):
                    raise TypeError("Provider returned an unsupported action")

                conversation.append(
                    {"role": "assistant", "content": action.message}
                )
                user_call_required = _model_backed(self.user_simulator)
                if user_call_required and hasattr(
                    self.user_simulator, "will_call_model"
                ):
                    user_call_required = bool(
                        self.user_simulator.will_call_model(action.message)
                    )
                if user_call_required and not budget.can_start_model_call():
                    outcome = "budget_exhausted"
                    message = "Model-call budget exhausted before the next user turn."
                    logger.append("budget_exhausted", budget.to_dict())
                    break
                if user_call_required:
                    budget.mark_attempt()
                try:
                    user_response = environment.respond(action.message)
                finally:
                    if user_call_required:
                        budget.record(_client_metrics(self.user_simulator), "user")
                current_user_message = str(user_response["message"])
                user_turns += 1
                conversation.append(
                    {"role": "user", "content": current_user_message}
                )
                logger.append(
                    "user_message",
                    {
                        "content": current_user_message,
                        "initial": False,
                        "done": bool(user_response["done"]),
                    },
                )
                if action.outcome == "completed" and not user_response["done"]:
                    premature_completion_continuations += 1
                    logger.append(
                        "dialogue_violation",
                        {
                            "code": "continued_after_agent_completed",
                            "agent_message": action.message,
                            "user_message": current_user_message,
                        },
                    )
                if user_response["done"]:
                    outcome = "environment_done"
                    message = action.message
                    break
        except Exception as exc:
            outcome = "runtime_error"
            message = "Dialogue runtime caught an error: %s" % exc
            logger.append("runtime_error", {"message": message})

        reward_result = environment.official_reward()
        official_reward = float(reward_result["reward"])
        reward_info = dict(reward_result.get("info", {}))
        actual_data_hash = reward_result.get("actual_data_hash")
        logger.append("environment_reward", reward_result)
        logger.append(
            "session_finished",
            {
                "outcome": outcome,
                "message": message,
                "steps": steps,
                "user_turns": user_turns,
                "premature_completion_continuations": (
                    premature_completion_continuations
                ),
                "tool_calls": len(_actual_tool_calls(observations)),
                "all_tool_calls_ok": all(
                    item.result.get("ok") for item in _actual_tool_calls(observations)
                ),
                "gate_rejections": gate_rejections,
                "budget": budget.to_dict(),
                "official_reward": official_reward,
            },
        )
        return TauDialogueResult(
            session_id=session_id,
            task_split=environment.task_split,
            task_index=environment.task_index,
            agent_provider=self.provider.name,
            user_simulator=self.user_simulator.name,
            outcome=outcome,
            message=message,
            steps=steps,
            user_turns=user_turns,
            premature_completion_continuations=(
                premature_completion_continuations
            ),
            observations=observations,
            conversation=conversation,
            official_reward=official_reward,
            reward_info=reward_info,
            actual_data_hash=actual_data_hash,
            event_path=event_path,
            gate_rejections=gate_rejections,
            budget=budget.to_dict(),
        )

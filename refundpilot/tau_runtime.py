"""Bounded harness loop for the external tau-bench Retail adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from .contracts import AgentContext, FinalAction, ToolAction, ToolObservation
from .events import EventLogger
from .provider import AgentProvider
from .tau_adapter import TauRetailEnvironment


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
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.provider = provider
        self.runtime_dir = runtime_dir
        self.max_steps = max_steps

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
        description = environment.describe()
        initial_hash = environment.data_hash()

        logger.append(
            "session_started",
            {
                "session_id": session_id,
                "provider": self.provider.name,
                "environment": description,
                "max_steps": self.max_steps,
                "evaluation_label": "wiring_only"
                if self.provider.name == "tau-oracle-wiring-smoke"
                else "agent_run",
            },
        )

        try:
            for step in range(1, self.max_steps + 1):
                steps = step
                context = AgentContext(
                    session_id=session_id,
                    user_request=environment.instruction,
                    tool_schemas=tuple(environment.tool_schemas()),
                    observations=tuple(observations),
                    step=step,
                    max_steps=self.max_steps,
                    system_instructions=environment.policy,
                )
                action = self.provider.next_action(context)
                logger.append("agent_action", action.to_dict())

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
                "tool_calls": len(observations),
                "all_tool_calls_ok": all(
                    item.result.get("ok") for item in observations
                ),
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

    @property
    def task_success(self) -> bool:
        return (
            self.outcome == "environment_done"
            and self.official_reward == 1.0
            and all(item.result.get("ok") for item in self.observations)
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
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.provider = provider
        self.user_simulator = user_simulator
        self.runtime_dir = runtime_dir
        self.max_steps = max_steps

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

        logger.append(
            "session_started",
            {
                "session_id": session_id,
                "provider": self.provider.name,
                "user_simulator": self.user_simulator.name,
                "environment": environment.describe(),
                "max_steps": self.max_steps,
                "evaluation_label": "multi_turn_agent_run",
                "agent_context": "current user utterance and visible transcript only",
            },
        )

        try:
            current_user_message = environment.start_conversation(
                self.user_simulator
            )
            user_turns = 1
            conversation.append(
                {"role": "user", "content": current_user_message}
            )
            logger.append(
                "user_message",
                {"content": current_user_message, "initial": True},
            )

            for step in range(1, self.max_steps + 1):
                steps = step
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
                action = self.provider.next_action(context)
                logger.append("agent_action", action.to_dict())

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
                user_response = environment.respond(action.message)
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
                "tool_calls": len(observations),
                "all_tool_calls_ok": all(
                    item.result.get("ok") for item in observations
                ),
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
        )

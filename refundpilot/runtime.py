"""Session lifecycle and bounded Agent loop."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import uuid4

from .contracts import (
    AgentContext,
    FinalAction,
    RunResult,
    TaskCase,
    ToolAction,
    ToolObservation,
)
from .events import EventLogger
from .environment import RetailEnvironment
from .provider import AgentProvider


class HarnessRuntime:
    def __init__(
        self,
        provider: AgentProvider,
        runtime_dir: Path,
        max_steps: int = 6,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.provider = provider
        self.runtime_dir = runtime_dir
        self.max_steps = max_steps

    @staticmethod
    def _new_session_id(task_id: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_task = "".join(
            character if character.isalnum() or character in "-_" else "-"
            for character in task_id
        )
        return "%s-%s-%s" % (timestamp, safe_task, uuid4().hex[:8])

    def run(self, case: TaskCase) -> RunResult:
        session_id = self._new_session_id(case.task_id)
        session_dir = self.runtime_dir / session_id
        event_path = session_dir / "events.jsonl"
        logger = EventLogger(event_path)
        environment = RetailEnvironment.load(case, session_dir / "state.sqlite")
        observations: List[ToolObservation] = []
        invalid_tool_calls = 0
        policy_violations = 0
        outcome = "max_steps_exceeded"
        message = "Agent reached the maximum number of steps."
        steps = 0

        logger.append(
            "session_started",
            {
                "session_id": session_id,
                "task_id": case.task_id,
                "provider": self.provider.name,
                "user_request": case.user_request,
                "max_steps": self.max_steps,
                "environment": environment.name,
                "tool_names": [
                    schema["name"] for schema in environment.tool_schemas()
                ],
            },
        )

        try:
            for step in range(1, self.max_steps + 1):
                steps = step
                context = AgentContext(
                    session_id=session_id,
                    user_request=case.user_request,
                    tool_schemas=tuple(environment.tool_schemas()),
                    observations=tuple(observations),
                    step=step,
                    max_steps=self.max_steps,
                )
                action = self.provider.next_action(context)
                logger.append("agent_action", action.to_dict())

                if isinstance(action, FinalAction):
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
                error_code = observation.result.get("error_code")
                if error_code in {"unknown_tool", "invalid_arguments"}:
                    invalid_tool_calls += 1
                if error_code == "policy_violation":
                    policy_violations += 1
        except Exception as exc:  # Provider boundary may wrap a remote service.
            outcome = "runtime_error"
            message = "Runtime boundary caught an error: %s" % exc
            logger.append("runtime_error", {"message": message})

        state_summary = environment.snapshot()
        logger.append(
            "session_finished",
            {
                "outcome": outcome,
                "message": message,
                "steps": steps,
                "tool_calls": len(observations),
                "invalid_tool_calls": invalid_tool_calls,
                "policy_violations": policy_violations,
                "state_summary": state_summary,
            },
        )
        environment.close()
        return RunResult(
            session_id=session_id,
            task_id=case.task_id,
            provider=self.provider.name,
            outcome=outcome,
            message=message,
            steps=steps,
            observations=observations,
            invalid_tool_calls=invalid_tool_calls,
            policy_violations=policy_violations,
            state_summary=state_summary,
            event_path=event_path,
        )

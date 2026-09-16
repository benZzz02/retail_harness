"""Shared contracts between providers, tools, the runtime and evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class TaskCase:
    task_id: str
    description: str
    user_request: str
    as_of: str
    orders: Tuple[Dict[str, Any], ...]
    initial_refunds: Tuple[Dict[str, Any], ...]
    policy: Dict[str, Any]
    expected: Dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TaskCase":
        return cls(
            task_id=str(raw["task_id"]),
            description=str(raw.get("description", "")),
            user_request=str(raw["user_request"]),
            as_of=str(raw["as_of"]),
            orders=tuple(dict(item) for item in raw.get("orders", [])),
            initial_refunds=tuple(
                dict(item) for item in raw.get("initial_refunds", [])
            ),
            policy=dict(
                raw.get(
                    "policy",
                    {"refund_window_days": 7, "manual_review_amount": 2000.0},
                )
            ),
            expected=dict(raw.get("expected", {})),
        )


@dataclass(frozen=True)
class ToolAction:
    tool_name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "tool",
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
        }


@dataclass(frozen=True)
class FinalAction:
    outcome: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "final", "outcome": self.outcome, "message": self.message}


AgentAction = Union[ToolAction, FinalAction]


@dataclass(frozen=True)
class ToolObservation:
    tool_name: str
    arguments: Dict[str, Any]
    result: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "result": dict(self.result),
        }


@dataclass(frozen=True)
class AgentContext:
    session_id: str
    user_request: str
    tool_schemas: Tuple[Dict[str, Any], ...]
    observations: Tuple[ToolObservation, ...]
    step: int
    max_steps: int
    system_instructions: str = ""
    conversation: Tuple[Dict[str, Any], ...] = ()
    memory: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    session_id: str
    task_id: str
    provider: str
    outcome: str
    message: str
    steps: int
    observations: List[ToolObservation]
    invalid_tool_calls: int
    policy_violations: int
    state_summary: Dict[str, Any]
    event_path: Path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "provider": self.provider,
            "outcome": self.outcome,
            "message": self.message,
            "steps": self.steps,
            "observations": [item.to_dict() for item in self.observations],
            "invalid_tool_calls": self.invalid_tool_calls,
            "policy_violations": self.policy_violations,
            "state_summary": self.state_summary,
            "event_path": str(self.event_path),
        }

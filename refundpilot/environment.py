"""Explicit retail environment boundary used by the Agent Harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .contracts import TaskCase, ToolAction, ToolObservation
from .state import WorldState
from .tools import ToolRegistry, build_tool_registry


class RetailEnvironment:
    """A loadable, resettable and inspectable e-commerce task world.

    The environment owns business state and tool execution. Providers never
    receive the task's expected result; they see only the user request, tool
    schemas and observations returned by ``step``.
    """

    name = "refund-retail-v1"

    def __init__(self, case: TaskCase, database_path: Path) -> None:
        self.case = case
        self.database_path = database_path
        self.state = WorldState(database_path, case)
        self.registry: ToolRegistry = build_tool_registry(self.state)

    @classmethod
    def load(cls, case: TaskCase, database_path: Path) -> "RetailEnvironment":
        return cls(case=case, database_path=database_path)

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return self.registry.schemas()

    def step(self, action: ToolAction) -> ToolObservation:
        result = self.registry.execute(action.tool_name, action.arguments)
        return ToolObservation(
            tool_name=action.tool_name,
            arguments=dict(action.arguments),
            result=result.to_dict(),
        )

    def snapshot(self) -> Dict[str, Any]:
        return self.state.summary()

    def describe(self) -> Dict[str, Any]:
        """Return inspectable environment metadata without evaluator answers."""

        return {
            "environment": self.name,
            "task_id": self.case.task_id,
            "user_request": self.case.user_request,
            "as_of": self.case.as_of,
            "policy": dict(self.case.policy),
            "tools": self.tool_schemas(),
            "initial_state": self.snapshot(),
            "database_path": str(self.database_path),
        }

    def reset(self) -> Dict[str, Any]:
        self.state.reset(self.case)
        self.registry = build_tool_registry(self.state)
        return self.snapshot()

    def close(self) -> None:
        self.state.close()


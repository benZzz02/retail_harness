"""Adapter around Sierra's external tau-bench Retail environment.

The tau-bench package is imported lazily so the small local fixture remains
usable without third-party dependencies. The adapter never recreates Retail
state: data, tool execution and reward all come from the external package.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Tuple

from .contracts import ToolAction, ToolObservation


TAU_BENCH_REPOSITORY = "https://github.com/sierra-research/tau-bench.git"
TAU_BENCH_COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"
TAU_ENVIRONMENT_NAME = "tau-bench-retail-legacy"


class TauBenchUnavailable(RuntimeError):
    """Raised when the pinned external environment is not installed."""


def _load_tau_types() -> Tuple[Any, Any, str]:
    # LiteLLM otherwise performs an unrelated network fetch during import.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    try:
        from tau_bench.envs.retail.env import MockRetailDomainEnv
        from tau_bench.types import Action, RESPOND_ACTION_NAME
    except (ImportError, ModuleNotFoundError) as exc:
        raise TauBenchUnavailable(
            "tau-bench is not installed in this interpreter. Run "
            "./scripts/setup_tau_bench.sh, then use .venv/bin/python."
        ) from exc
    return MockRetailDomainEnv, Action, RESPOND_ACTION_NAME


class TauRetailEnvironment:
    """Thin lifecycle adapter over the external tau-bench Retail domain."""

    name = TAU_ENVIRONMENT_NAME

    def __init__(self, task_split: str = "test", task_index: int = 0) -> None:
        self.task_split = task_split
        self.task_index = task_index
        self._env_type, self._action_type, self._respond_action_name = (
            _load_tau_types()
        )
        self._terminal_reward: Dict[str, Any] | None = None
        self._load_external_environment()

    def _load_external_environment(self) -> None:
        try:
            self._env = self._env_type(
                user_strategy="human",
                user_model="unused",
                task_split=self.task_split,
                task_index=self.task_index,
            )
        except (IndexError, ValueError) as exc:
            raise ValueError(
                "invalid tau-bench Retail task %s/%s"
                % (self.task_split, self.task_index)
            ) from exc
        self._initial_data_hash = self._env.get_data_hash()
        self._terminal_reward = None

    @property
    def instruction(self) -> str:
        return str(self._env.task.instruction)

    @property
    def task_count(self) -> int:
        return len(self._env.tasks)

    @property
    def policy(self) -> str:
        return str(self._env.wiki)

    def tool_schemas(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._env.tools_info)

    def data_hash(self) -> str:
        return str(self._env.get_data_hash())

    def describe(self) -> Dict[str, Any]:
        tool_names = [
            schema["function"]["name"] for schema in self._env.tools_info
        ]
        return {
            "environment": self.name,
            "source": {
                "repository": TAU_BENCH_REPOSITORY,
                "pinned_commit": TAU_BENCH_COMMIT,
                "ownership": "external",
            },
            "task": {
                "split": self.task_split,
                "index": self.task_index,
                "available_in_split": self.task_count,
                "instruction": self.instruction,
            },
            "dataset": {
                "users": len(self._env.data.get("users", {})),
                "orders": len(self._env.data.get("orders", {})),
                "products": len(self._env.data.get("products", {})),
            },
            "tools": {"count": len(tool_names), "names": tool_names},
            "initial_data_hash": self._initial_data_hash,
        }

    def oracle_actions(self) -> List[ToolAction]:
        """Return gold tool calls for an explicitly labelled wiring smoke test."""
        return [
            ToolAction(action.name, dict(action.kwargs))
            for action in self._env.task.actions
            if action.name != self._respond_action_name
        ]

    @staticmethod
    def _is_error(observation: str) -> Tuple[bool, str | None]:
        normalized = observation.strip()
        if normalized.startswith("Unknown action"):
            return True, "unknown_tool"
        if normalized.startswith("Error:"):
            return True, "environment_error"
        return False, None

    def step(self, action: ToolAction) -> ToolObservation:
        # Calling `respond` with the HUMAN simulator would read stdin. The MVP
        # deliberately covers tool execution only and fails safely here.
        if action.tool_name == self._respond_action_name:
            return ToolObservation(
                tool_name=action.tool_name,
                arguments=dict(action.arguments),
                result={
                    "ok": False,
                    "output": {},
                    "error_code": "interactive_user_not_enabled",
                    "error_message": (
                        "The offline adapter does not forward agent messages to "
                        "the interactive user simulator."
                    ),
                    "done": False,
                    "reward": 0.0,
                },
            )

        external_action = self._action_type(
            name=action.tool_name, kwargs=dict(action.arguments)
        )
        response = self._env.step(external_action)
        observation = str(response.observation)
        is_error, error_code = self._is_error(observation)
        source = response.info.source

        if response.done and response.info.reward_info is not None:
            reward_info = response.info.reward_info.info.model_dump()
            self._terminal_reward = {
                "reward": float(response.reward),
                "actual_data_hash": None,
                "info": reward_info,
                "source": "tau_bench.Env.step",
            }

        return ToolObservation(
            tool_name=action.tool_name,
            arguments=dict(action.arguments),
            result={
                "ok": not is_error,
                "output": {
                    "observation": observation,
                    "source": source,
                },
                "error_code": error_code,
                "error_message": observation if is_error else None,
                "done": bool(response.done),
                "reward": float(response.reward),
            },
        )

    def start_conversation(self, user_simulator: Any) -> str:
        """Reset upstream state and return the simulator's opening utterance."""
        self._env.user = user_simulator
        response = self._env.reset(task_index=self.task_index)
        self._initial_data_hash = self._env.get_data_hash()
        self._terminal_reward = None
        return str(response.observation)

    def respond(self, message: str) -> Dict[str, Any]:
        """Send an agent message through upstream ``Env.step(respond)``."""
        actual_data_hash = self.data_hash()
        response = self._env.step(
            self._action_type(
                name=self._respond_action_name,
                kwargs={"content": str(message)},
            )
        )
        if response.done and response.info.reward_info is not None:
            self._terminal_reward = {
                "reward": float(response.reward),
                "actual_data_hash": actual_data_hash,
                "info": response.info.reward_info.info.model_dump(),
                "source": "tau_bench.Env.step(respond)",
            }
        return {
            "message": str(response.observation),
            "done": bool(response.done),
            "reward": float(response.reward),
            "source": response.info.source,
        }

    def record_agent_response(self, message: str) -> None:
        """Record a terminal assistant reply for upstream output evaluation.

        Calling ``Env.step(respond)`` with the HUMAN strategy would block on
        stdin. The upstream reward only inspects the action transcript, so the
        one-turn adapter records the same external Action without advancing a
        nonexistent live user.
        """
        self._env.actions.append(
            self._action_type(
                name=self._respond_action_name,
                kwargs={"content": str(message)},
            )
        )

    def official_reward(self) -> Dict[str, Any]:
        """Run the external environment's own terminal-state reward function."""
        if self._terminal_reward is not None:
            return copy.deepcopy(self._terminal_reward)

        actual_data_hash = self.data_hash()
        result = self._env.calculate_reward()
        return {
            "reward": float(result.reward),
            "actual_data_hash": actual_data_hash,
            "info": result.info.model_dump(),
            "source": "tau_bench.Env.calculate_reward",
        }

    def reset(self) -> Dict[str, Any]:
        """Reload pristine external data without invoking the HUMAN simulator."""
        self._load_external_environment()
        return self.describe()

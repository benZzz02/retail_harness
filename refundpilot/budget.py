"""Per-run usage accounting and hard execution budgets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class UsageLedger:
    """Record provider-independent usage signals without requiring a provider SDK."""

    agent_calls: int = 0
    user_calls: int = 0
    prompt_chars: int = 0
    output_chars: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    provider_usage_available: bool = False

    def record(self, metrics: Optional[Dict[str, Any]], role: str) -> None:
        metrics = metrics or {}
        if role == "agent":
            self.agent_calls += 1
        elif role == "user":
            self.user_calls += 1
        self.prompt_chars += int(metrics.get("prompt_chars") or 0)
        self.output_chars += int(metrics.get("output_chars") or 0)

        usage = metrics.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or 0)
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens or prompt_tokens + completion_tokens
            self.provider_usage_available = True

    def to_dict(self) -> Dict[str, Any]:
        estimated_input_tokens = (self.prompt_chars + 3) // 4
        estimated_output_tokens = (self.output_chars + 3) // 4
        return {
            "agent_calls": self.agent_calls,
            "user_calls": self.user_calls,
            "model_calls": self.agent_calls + self.user_calls,
            "prompt_chars": self.prompt_chars,
            "output_chars": self.output_chars,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "provider_usage_available": self.provider_usage_available,
        }


@dataclass
class RunBudget:
    """Stop a long-running dialogue before it becomes an unbounded API bill."""

    max_model_calls: Optional[int] = None
    usage: UsageLedger = field(default_factory=UsageLedger)
    attempted_model_calls: int = 0
    exhausted: bool = False

    def can_start_model_call(self) -> bool:
        limit = self.max_model_calls
        if limit is None:
            return True
        current = self.attempted_model_calls
        if current >= limit:
            self.exhausted = True
            return False
        return True

    def mark_attempt(self) -> None:
        self.attempted_model_calls += 1
        limit = self.max_model_calls
        if limit is not None and self.attempted_model_calls >= limit:
            self.exhausted = True

    def record(self, metrics: Optional[Dict[str, Any]], role: str) -> None:
        self.usage.record(metrics, role)
        limit = self.max_model_calls
        if limit is not None and self.usage.agent_calls + self.usage.user_calls >= limit:
            self.exhausted = True

    def to_dict(self) -> Dict[str, Any]:
        result = self.usage.to_dict()
        result["attempted_model_calls"] = self.attempted_model_calls
        result["max_model_calls"] = self.max_model_calls
        result["budget_exhausted"] = self.exhausted
        return result

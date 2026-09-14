"""Evaluate outcomes against terminal business state, not model wording."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence

from .contracts import RunResult, TaskCase


def evaluate_run(case: TaskCase, result: RunResult) -> Dict[str, Any]:
    expected = case.expected
    tool_sequence = [item.tool_name for item in result.observations]
    checks: Dict[str, bool] = {}
    if "outcome" in expected:
        checks["outcome"] = result.outcome == expected["outcome"]
    if "refund_count" in expected:
        checks["refund_count"] = (
            result.state_summary["refund_count"] == expected["refund_count"]
        )
    if "escalation_count" in expected:
        checks["escalation_count"] = (
            result.state_summary["escalation_count"]
            == expected["escalation_count"]
        )
    if "refunded_order_ids" in expected:
        checks["refunded_order_ids"] = (
            result.state_summary["refunded_order_ids"]
            == expected["refunded_order_ids"]
        )
    if "tool_sequence" in expected:
        checks["tool_sequence"] = tool_sequence == expected["tool_sequence"]
    return {
        "task_id": case.task_id,
        "task_success": bool(checks) and all(checks.values()),
        "checks": checks,
        "outcome": result.outcome,
        "tool_sequence": tool_sequence,
        "steps": result.steps,
        "invalid_tool_calls": result.invalid_tool_calls,
        "policy_violations": result.policy_violations,
        "session_id": result.session_id,
    }


def aggregate_evaluations(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    outcomes = Counter(str(row["outcome"]) for row in rows)
    return {
        "tasks": count,
        "task_success_rate": (
            sum(bool(row["task_success"]) for row in rows) / count if count else 0.0
        ),
        "invalid_tool_session_rate": (
            sum(int(row["invalid_tool_calls"] > 0) for row in rows) / count
            if count
            else 0.0
        ),
        "policy_violation_session_rate": (
            sum(int(row["policy_violations"] > 0) for row in rows) / count
            if count
            else 0.0
        ),
        "average_steps": (
            sum(int(row["steps"]) for row in rows) / count if count else 0.0
        ),
        "outcomes": dict(sorted(outcomes.items())),
        "details": list(rows),
    }


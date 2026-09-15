"""Deterministic context views for long-running tool-agent turns."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .contracts import AgentContext, ToolObservation


def _parsed_observation(observation: ToolObservation) -> Any:
    output = observation.result.get("output", {})
    if not isinstance(output, dict):
        return output
    raw = output.get("observation")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return " ".join(raw.split())
    return output


def _compact_observation(observation: ToolObservation) -> Dict[str, Any]:
    result = observation.result
    if not result.get("ok"):
        return {
            "tool_name": observation.tool_name,
            "arguments": dict(observation.arguments),
            "ok": False,
            "error_code": result.get("error_code"),
            "error_message": result.get("error_message"),
        }
    return {
        "tool_name": observation.tool_name,
        "arguments": dict(observation.arguments),
        "ok": True,
        "done": bool(result.get("done")),
        "observation": _parsed_observation(observation),
    }


def _compact_conversation(context: AgentContext) -> List[Dict[str, Any]]:
    # Tool payloads are already represented in observations. Keeping them in
    # conversation as well needlessly duplicates the largest part of a prompt.
    visible: List[Dict[str, Any]] = []
    for item in context.conversation:
        role = item.get("role")
        if role == "tool":
            continue
        if role == "assistant" and item.get("type") == "tool_call":
            visible.append(
                {
                    "role": "assistant",
                    "type": "tool_call",
                    "tool_name": item.get("tool_name"),
                    "arguments": dict(item.get("arguments", {})),
                }
            )
            continue
        visible.append(dict(item))
    if len(visible) <= 12:
        return visible
    return [
        {"role": "harness", "content": "Earlier conversation turns omitted; observations remain authoritative."},
        *visible[-11:],
    ]


def build_context_payload(context: AgentContext, compact: bool = True) -> Dict[str, Any]:
    if not compact:
        observations = [item.to_dict() for item in context.observations]
        conversation = list(context.conversation)
    else:
        observations = [_compact_observation(item) for item in context.observations]
        conversation = _compact_conversation(context)
    return {
        "step": context.step,
        "max_steps": context.max_steps,
        "retail_policy": context.system_instructions,
        "user_task": context.user_request,
        "conversation": conversation,
        "available_tools": list(context.tool_schemas),
        "observations": observations,
    }

"""Session-scoped structured memory for long-running Retail tasks."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .contracts import AgentAction, FinalAction, ToolAction, ToolObservation
from .policy import GateDecision, MUTATING_TOOLS


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _normalize_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def _parse_observation(observation: ToolObservation) -> Any:
    output = observation.result.get("output", {})
    if not isinstance(output, dict):
        return output
    raw = output.get("observation")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return _normalize_text(raw)
    return output


def _is_confirmation_request(message: str) -> bool:
    return bool(
        re.search(
            r"\b(confirm|confirmation|proceed|go ahead|approve|authorize|shall i|would you like me to|please confirm)\b",
            message.lower(),
        )
    )


def _is_scope_change(message: str) -> bool:
    return bool(
        re.search(
            r"\b(actually|instead|only|skip|change|rather|not anymore|no longer|different|remove|add|but)\b",
            _normalize_text(message).lower(),
        )
    )


@dataclass
class TaskMemory:
    """A bounded, serializable state ledger for one task/session.

    The external environment remains authoritative. This object stores facts
    already observed from it and the control state needed to reconstruct a
    useful prompt after compaction or process restart; it does not invent facts.
    """

    session_id: str
    original_request: str = ""
    current_user_turn: str = ""
    goal_hash: str = ""
    current_turn_hash: str = ""
    identity: Dict[str, Any] = field(default_factory=dict)
    orders: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    products: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    payment_methods: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    action_ledger: list[Dict[str, Any]] = field(default_factory=list)
    scope_changes: list[Dict[str, Any]] = field(default_factory=list)
    confirmation: Dict[str, Any] = field(
        default_factory=lambda: {"status": "not_requested"}
    )
    last_gate_rejection: Dict[str, Any] = field(default_factory=dict)
    version: int = 0
    _max_history: int = 24
    _max_scope_changes: int = 8

    def start(self, request: str) -> None:
        request = _normalize_text(request)
        if not self.original_request and request:
            self.original_request = request
            self.goal_hash = _fingerprint(request)
        if request:
            self.current_user_turn = request
            self.current_turn_hash = _fingerprint(request)
        self.version += 1

    def record_user_turn(self, message: str) -> None:
        message = _normalize_text(message)
        if not message:
            return
        previous = self.current_user_turn
        self.current_user_turn = message
        self.current_turn_hash = _fingerprint(message)
        if previous and previous != message and _is_scope_change(message):
            self.scope_changes.append(
                {
                    "from": previous,
                    "to": message,
                    "turn_hash": self.current_turn_hash,
                }
            )
            self.scope_changes = self.scope_changes[-self._max_scope_changes :]
            self.confirmation = {
                "status": "invalidated",
                "reason": "user_scope_changed",
                "scope_hash": self.current_turn_hash,
            }
        self.version += 1

    def record_agent_message(self, message: str) -> None:
        message = _normalize_text(message)
        if message and _is_confirmation_request(message):
            self.confirmation = {
                "status": "pending",
                "request": message,
                "scope_hash": self.current_turn_hash,
            }
        self.version += 1

    def record_action(self, action: AgentAction) -> None:
        if isinstance(action, ToolAction):
            entry = {
                "kind": "tool",
                "tool_name": action.tool_name,
                "arguments": copy.deepcopy(action.arguments),
                "status": "proposed",
            }
        else:
            entry = {
                "kind": "final",
                "outcome": action.outcome,
                "message": action.message,
                "status": "proposed",
            }
        self.action_ledger.append(entry)
        self.action_ledger = self.action_ledger[-self._max_history :]
        self.version += 1

    def record_gate(
        self,
        action: AgentAction,
        decision: GateDecision,
        require_confirmation: bool,
    ) -> None:
        if isinstance(action, ToolAction):
            for entry in reversed(self.action_ledger):
                if (
                    entry.get("kind") == "tool"
                    and entry.get("tool_name") == action.tool_name
                    and entry.get("arguments") == action.arguments
                    and entry.get("status") == "proposed"
                ):
                    entry["gate_allowed"] = decision.allowed
                    entry["gate_code"] = decision.code
                    break
            if action.tool_name in MUTATING_TOOLS and require_confirmation:
                if decision.allowed:
                    self.confirmation = {
                        "status": "consumed",
                        "scope_hash": _fingerprint(action.to_dict()),
                    }
                elif decision.code == "missing_confirmation":
                    self.confirmation = {
                        "status": "missing",
                        "scope_hash": _fingerprint(action.to_dict()),
                    }
        if not decision.allowed:
            self.last_gate_rejection = {
                "code": decision.code,
                "message": decision.message,
                "action": action.to_dict(),
            }
        self.version += 1

    def record_observation(self, observation: ToolObservation) -> None:
        parsed = _parse_observation(observation)
        ok = bool(observation.result.get("ok"))
        for entry in reversed(self.action_ledger):
            if (
                entry.get("kind") == "tool"
                and entry.get("tool_name") == observation.tool_name
                and entry.get("arguments") == observation.arguments
                and entry.get("status") == "proposed"
            ):
                entry["status"] = "executed" if ok else "failed"
                entry["ok"] = ok
                entry["done"] = bool(observation.result.get("done"))
                if not ok:
                    entry["error_code"] = observation.result.get("error_code")
                break

        if not ok:
            self.version += 1
            return

        if observation.tool_name == "get_order_details" and isinstance(parsed, dict):
            order_id = str(parsed.get("order_id") or observation.arguments.get("order_id") or "")
            if order_id:
                self.orders[order_id] = copy.deepcopy(parsed)
        elif observation.tool_name == "get_product_details" and isinstance(parsed, dict):
            product_id = str(parsed.get("product_id") or observation.arguments.get("product_id") or "")
            if product_id:
                self.products[product_id] = copy.deepcopy(parsed)
        elif observation.tool_name == "get_user_details" and isinstance(parsed, dict):
            user_id = parsed.get("user_id") or observation.arguments.get("user_id")
            if user_id:
                self.identity["user_id"] = user_id
            methods = parsed.get("payment_methods")
            if isinstance(methods, dict):
                self.payment_methods.update(copy.deepcopy(methods))
        elif observation.tool_name in {
            "find_user_id_by_email",
            "find_user_id_by_name_zip",
        }:
            if isinstance(parsed, str) and parsed:
                self.identity["user_id"] = parsed
        self.version += 1

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "session_id": self.session_id,
            "goal": {
                "original_request": self.original_request,
                "goal_hash": self.goal_hash,
                "current_user_turn": self.current_user_turn,
                "current_turn_hash": self.current_turn_hash,
                "scope_changes": copy.deepcopy(self.scope_changes[-4:]),
            },
            "identity": copy.deepcopy(self.identity),
            "orders": copy.deepcopy(dict(list(self.orders.items())[-4:])),
            "products": copy.deepcopy(dict(list(self.products.items())[-8:])),
            "payment_methods": copy.deepcopy(self.payment_methods),
            "confirmation": copy.deepcopy(self.confirmation),
            "action_ledger": copy.deepcopy(self.action_ledger[-12:]),
            "last_gate_rejection": copy.deepcopy(self.last_gate_rejection),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.to_prompt_dict()


class MemoryStore:
    """Persist the latest structured memory as an atomic session checkpoint."""

    def __init__(self, path: Path):
        self.path = path

    def save(self, memory: TaskMemory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(memory.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

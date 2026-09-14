"""Codex-backed user simulator compatible with tau-bench's user interface."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .codex_client import CodexStructuredClient


class CodexCliUserSimulator:
    """Generate one customer turn at a time from a hidden task instruction."""

    metadata = {"provider": "codex-cli", "strategy": "llm+grounded-auth"}
    _account_id_pattern = re.compile(
        r"\b(?P<first>[A-Za-z][A-Za-z'-]*)_"
        r"(?P<last>[A-Za-z][A-Za-z'-]*)_(?P<suffix>\d+)\b"
    )
    _named_identity_pattern = re.compile(
        r"\b(?:you are|your name is|you name is)\s+"
        r"(?P<first>[A-Za-z][A-Za-z'-]*)\s+"
        r"(?P<last>[A-Za-z][A-Za-z'-]*)\b",
        re.IGNORECASE,
    )
    _email_pattern = re.compile(
        r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
        r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b"
    )
    _zip_pattern = re.compile(r"\b\d{5}\b")

    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        reasoning_effort: str = "low",
        timeout_seconds: int = 180,
        executable: Optional[str] = None,
    ) -> None:
        self.model = model
        self.name = "codex-user-simulator:%s+grounded-auth" % model
        self._client = CodexStructuredClient(
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
            executable=executable,
        )
        self._schema_path = (
            Path(__file__).resolve().parent / "data" / "user_turn.schema.json"
        )
        self._instruction = ""
        self.transcript: List[Dict[str, str]] = []
        self.call_count = 0

    def _safe_identity_derivation(self) -> Dict[str, str]:
        """Extract only identity facts stated or encoded in the instruction."""
        facts: Dict[str, str] = {}
        account_match = self._account_id_pattern.search(self._instruction)
        name_match = self._named_identity_pattern.search(self._instruction)
        identity_match = account_match or name_match
        if identity_match is not None:
            facts.update(
                {
                    "first_name": identity_match.group("first").title(),
                    "last_name": identity_match.group("last").title(),
                }
            )
        if account_match is not None:
            facts["account_user_id"] = account_match.group(0)
        email_match = self._email_pattern.search(self._instruction)
        if email_match is not None:
            facts["email"] = email_match.group(0)
        zip_match = self._zip_pattern.search(self._instruction)
        if zip_match is not None:
            facts["zip"] = zip_match.group(0)
        return facts

    def _grounded_auth_response(self, agent_message: str) -> Optional[str]:
        """Answer routine authentication prompts without allowing invented PII."""
        lowered = agent_message.lower()
        asks_for_auth = any(
            token in lowered
            for token in (
                "provide",
                "share",
                "tell me",
                "what is",
                "what's",
                "authenticate",
                "verify",
                "need either",
                "requires either",
            )
        )
        if not asks_for_auth:
            return None

        asks_email = "email" in lowered
        asks_name = (
            "first" in lowered and "last" in lowered and "name" in lowered
        ) or "name on" in lowered
        asks_zip = "zip" in lowered or "postal code" in lowered
        if not (asks_email or asks_name or asks_zip):
            return None

        facts = self._safe_identity_derivation()
        if asks_email and facts.get("email"):
            return "The email address is %s." % facts["email"]

        full_name = " ".join(
            value
            for value in (facts.get("first_name"), facts.get("last_name"))
            if value
        )
        zip_code = facts.get("zip")
        if full_name and zip_code:
            prefix = "I don't have the email address handy. " if asks_email else ""
            return "%sMy name is %s, and my ZIP code is %s." % (
                prefix,
                full_name,
                zip_code,
            )
        if full_name and asks_name:
            return "My name is %s." % full_name
        if zip_code and asks_zip:
            return "My ZIP code is %s." % zip_code
        return None

    def _build_prompt(self, initial: bool) -> str:
        payload: Dict[str, Any] = {
            "hidden_instruction": self._instruction,
            "phase": "opening_customer_turn" if initial else "next_customer_turn",
            "conversation": list(self.transcript),
            "safe_identity_derivation": self._safe_identity_derivation(),
        }
        return (
            "You are the CUSTOMER simulator in a retail support benchmark. "
            "The hidden instruction is private: never quote it, mention it, or "
            "reveal future conditional behavior to the agent. Generate only the "
            "customer's next turn. On the opening turn, naturally state the "
            "current request and only information a customer would volunteer. "
            "On later turns, answer the latest agent message directly. Follow "
            "every conditional instruction exactly, including first/second "
            "confirmation changes; infer how many confirmation requests have "
            "occurred from the transcript. Count an agent turn as a confirmation "
            "request when it asks the customer to authorize a concrete action, "
            "but do not count ordinary questions that only gather missing choices. "
            "A conditional change triggered by a confirmation request is NOT an "
            "approval: state the changed request without saying yes, confirm, "
            "proceed, or otherwise authorizing execution. This lets the agent "
            "restate the new scope and request fresh confirmation. Approve only "
            "when the latest confirmation request triggers no still-unconsumed "
            "change in the hidden instruction. A triggered request change replaces "
            "the earlier scope permanently: do not revive an item or goal that was "
            "removed unless a later explicit condition in the hidden instruction "
            "reintroduces it. After the agent reports completing the latest "
            "confirmed scope, stop instead of returning to the opening request. "
            "Do not invent facts absent from the "
            "instruction. The safe_identity_derivation, when non-empty, contains "
            "the first and last name literally encoded by a benchmark account ID; "
            "provide those fields if the agent requests identity verification, "
            "and never claim they are unavailable. Set stop=true only after the "
            "agent has explicitly "
            "reported that the customer's current goal is completed, denied, or "
            "cannot proceed and no further customer response is needed. When "
            "stop=false, message must contain one concise natural customer turn. "
            "When stop=true, message may be empty. Return only the required JSON "
            "object.\n\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )

    def _generate(self, initial: bool) -> str:
        raw = self._client.complete(
            self._build_prompt(initial=initial), self._schema_path
        )
        self.call_count += 1
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("user simulator returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("user simulator response must be an object")
        if bool(payload.get("stop")):
            return "###STOP###"
        message = str(payload.get("message") or "").strip()
        if not message:
            raise ValueError("user simulator returned an empty active turn")
        return message

    def reset(self, instruction: Optional[str] = None) -> str:
        if instruction is None:
            raise ValueError("user simulator requires a hidden task instruction")
        self._instruction = str(instruction)
        self.transcript = []
        self.call_count = 0
        message = self._generate(initial=True)
        self.transcript.append({"role": "customer", "content": message})
        return message

    def step(self, content: str) -> str:
        self.transcript.append({"role": "agent", "content": str(content)})
        message = self._grounded_auth_response(str(content))
        if message is None:
            message = self._generate(initial=False)
        self.transcript.append({"role": "customer", "content": message})
        return message

    def get_total_cost(self) -> float:
        # Codex account usage is not exposed as a per-call currency amount.
        return 0.0

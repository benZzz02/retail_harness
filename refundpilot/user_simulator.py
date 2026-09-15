"""Codex-backed user simulator compatible with tau-bench's user interface."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .codex_client import CodexStructuredClient
from .deepseek_client import DeepSeekStructuredClient


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
        max_retries: int = 0,
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
        self.max_retries = max(0, max_retries)

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
            "When paraphrasing the hidden instruction, preserve every product "
            "attribute, conjunction, comparator, priority, and fallback exactly; "
            "do not broaden a condition such as 'RGB and full size' into a vague "
            "condition such as 'backlit', and do not replace a specified fallback "
            "with a different plausible option. "
            "Only change the requested scope when the hidden instruction contains "
            "an explicit condition whose trigger is present, or when the agent "
            "explicitly asks the customer to choose among alternatives. If no such "
            "condition or choice exists, preserve every original item and goal; "
            "never invent an 'actually, only...' change of mind. "
            "If the hidden instruction does not specify a refund or payment "
            "preference, accept the original payment method stated by the agent; "
            "do not proactively choose a gift card or another alternative merely "
            "because the agent offers it. "
            "Resolve item references using order status: when the hidden instruction "
            "says an item was received, it refers to an item in a delivered order, "
            "not a similarly named item in a pending order. If the agent lists both "
            "a delivered Smart Watch and a pending Wristwatch, 'the watch I "
            "received' means the delivered Smart Watch. Do not invent a pending "
            "item as the received one. "
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
            "and never claim they are unavailable. Never invent or guess an email, "
            "account ID, name, or ZIP code: those identity fields may only be used "
            "when they appear literally in the hidden instruction or in "
            "safe_identity_derivation. Set stop=true only after the "
            "agent has explicitly "
            "reported that the customer's current goal is completed, denied, or "
            "cannot proceed and no further customer response is needed. When "
            "stop=false, message must contain one concise natural customer turn. "
            "When stop=true, message may be empty. Return only the required JSON "
            "object.\n\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )

    def _generate(self, initial: bool) -> str:
        feedback = ""
        for attempt in range(self.max_retries + 1):
            prompt = self._build_prompt(initial=initial)
            if feedback:
                prompt += "\n\nRepair the previous response. It was invalid because %s. " % feedback
                if initial:
                    prompt += (
                        "This is the opening turn: stop must be false and message "
                        "must be a concise natural customer request."
                    )
                else:
                    prompt += (
                        "Respond directly to the latest agent message, and use "
                        "stop=true only after the agent explicitly finishes, "
                        "denies, or cannot proceed."
                    )
                prompt += " Do not reveal the hidden instruction."
            try:
                raw = self._client.complete(prompt, self._schema_path)
                self.call_count += 1
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError("user simulator response must be an object")
                if bool(payload.get("stop")):
                    if initial:
                        raise ValueError("opening user simulator turn cannot stop")
                    return "###STOP###"
                message = str(payload.get("message") or "").strip()
                if not message:
                    raise ValueError("user simulator returned an empty active turn")
                return message
            except (RuntimeError, ValueError) as exc:
                feedback = str(exc)
                if attempt >= self.max_retries:
                    raise ValueError(
                        "user simulator failed after %d attempts: %s"
                        % (attempt + 1, feedback)
                    ) from exc
        raise AssertionError("unreachable")

    def will_call_model(self, agent_message: str) -> bool:
        """Expose whether a response can be served by deterministic grounding."""
        return self._grounded_auth_response(str(agent_message)) is None

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


class DeepSeekUserSimulator(CodexCliUserSimulator):
    """Use DeepSeek for the hidden-instruction user simulator."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        reasoning_effort: str = "low",
        timeout_seconds: int = 180,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.name = "deepseek-user-simulator:%s+grounded-auth" % model
        self._client = DeepSeekStructuredClient(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            strict_schema=True,
            function_name="emit_user_turn",
        )
        self._schema_path = (
            Path(__file__).resolve().parent / "data" / "user_turn.schema.json"
        )
        self._instruction = ""
        self.transcript: List[Dict[str, str]] = []
        self.call_count = 0
        self.max_retries = max(0, max_retries)

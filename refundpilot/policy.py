"""Provider-neutral business action gates for Retail mutations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .contracts import AgentAction, AgentContext, FinalAction, ToolAction, ToolObservation


GATE_TOOL_NAME = "__harness_action_gate__"
MUTATING_TOOLS = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}
DELIVERED_OPERATIONS = {
    "return_delivered_order_items",
    "exchange_delivered_order_items",
}


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    code: str = ""
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "code": self.code,
            "message": self.message,
        }


class RetailActionGate:
    """Check observable Retail invariants before a model action reaches the env."""

    def __init__(self, require_confirmation: bool = False) -> None:
        self.require_confirmation = require_confirmation

    @staticmethod
    def _allow() -> GateDecision:
        return GateDecision(True)

    @staticmethod
    def _observed_json(observation: ToolObservation) -> Optional[dict]:
        raw = observation.result.get("output", {}).get("observation")
        if not isinstance(raw, str):
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _has_successful_observation(
        context: AgentContext, tool_name: str, order_id: Optional[str] = None
    ) -> bool:
        for observation in context.observations:
            if observation.tool_name != tool_name or not observation.result.get("ok"):
                continue
            if order_id is None or observation.arguments.get("order_id") == order_id:
                return True
        return False

    @staticmethod
    def _is_confirmation_request(message: str) -> bool:
        lowered = message.lower()
        return bool(
            re.search(
                r"\b(confirm|confirmation|proceed|go ahead|approve|authorize|shall i|would you like me to|please confirm)\b",
                lowered,
            )
        )

    @staticmethod
    def _is_affirmation(message: str) -> bool:
        lowered = " ".join(message.lower().split())
        return bool(
            re.search(
                r"\b(yes|yep|yeah|confirmed|confirm|proceed|go ahead|do it|please do|sounds good|that works|okay|ok)\b",
                lowered,
            )
        )

    @staticmethod
    def _is_scope_change(message: str) -> bool:
        lowered = " ".join(message.lower().split())
        return bool(
            re.search(
                r"\b(actually|instead|only|skip|change|rather|not anymore|no longer|different|remove|add|but)\b",
                lowered,
            )
        )

    def _has_current_confirmation(self, context: AgentContext) -> bool:
        pending = False
        confirmed = False
        for item in context.conversation:
            role = item.get("role")
            content = str(item.get("content", ""))
            if role == "assistant" and self._is_confirmation_request(content):
                pending = True
                confirmed = False
            elif role == "user":
                if pending:
                    confirmed = self._is_affirmation(content) and not self._is_scope_change(content)
                    pending = False
                elif confirmed and self._is_scope_change(content):
                    confirmed = False
        return confirmed

    def _required_argument_error(
        self, context: AgentContext, action: ToolAction
    ) -> Optional[GateDecision]:
        for schema in context.tool_schemas:
            function = schema.get("function", {})
            if function.get("name") != action.tool_name:
                continue
            parameters = function.get("parameters", {})
            required = parameters.get("required", [])
            missing = [
                name
                for name in required
                if action.arguments.get(name) in (None, "", [], {})
            ]
            if missing:
                return GateDecision(
                    False,
                    "empty_required_argument",
                    "Do not call %s with empty required arguments: %s. Ask for the missing information first."
                    % (action.tool_name, ", ".join(missing)),
                )
            break
        return None

    def _read_before_write_error(
        self, context: AgentContext, action: ToolAction
    ) -> Optional[GateDecision]:
        if action.tool_name not in MUTATING_TOOLS:
            return None
        order_id = action.arguments.get("order_id")
        if action.tool_name != "modify_user_address" and not self._has_successful_observation(
            context, "get_order_details", order_id if isinstance(order_id, str) else None
        ):
            return GateDecision(
                False,
                "missing_order_snapshot",
                "Read the current order details successfully before performing %s."
                % action.tool_name,
            )
        if (
            action.tool_name == "exchange_delivered_order_items"
            and not self._has_successful_observation(context, "get_product_details")
        ):
            return GateDecision(
                False,
                "missing_product_snapshot",
                "Read the candidate product details successfully before exchanging delivered items.",
            )
        return None

    def _payment_error(
        self, context: AgentContext, action: ToolAction
    ) -> Optional[GateDecision]:
        if action.tool_name not in {
            "return_delivered_order_items",
            "exchange_delivered_order_items",
            "modify_pending_order_items",
            "modify_pending_order_payment",
        }:
            return None
        payment_method_id = action.arguments.get("payment_method_id")
        order_id = action.arguments.get("order_id")
        if not isinstance(payment_method_id, str) or not payment_method_id:
            return None
        order_details = None
        user_details = None
        for observation in context.observations:
            if observation.tool_name not in {"get_order_details", "get_user_details"}:
                continue
            payload = self._observed_json(observation)
            if not payload:
                continue
            if observation.tool_name == "get_order_details":
                if order_id is None or payload.get("order_id") == order_id:
                    order_details = payload
            else:
                user_details = payload
        if order_details is None:
            return None
        original_ids = {
            entry.get("payment_method_id")
            for entry in order_details.get("payment_history", [])
            if isinstance(entry, dict) and entry.get("payment_method_id")
        }
        if payment_method_id in original_ids:
            return None
        known_gift_cards = set()
        if user_details:
            for method_id, method in user_details.get("payment_methods", {}).items():
                if isinstance(method, dict) and method.get("source") == "gift_card":
                    known_gift_cards.add(method_id)
        if payment_method_id in known_gift_cards:
            return None
        return GateDecision(
            False,
            "invalid_payment_method",
            "The payment method %s is not the order's original method(s) %s and is not an observed gift card. Ask the customer to choose a valid method first."
            % (
                payment_method_id,
                ", ".join(sorted(original_ids)) or "unknown",
            ),
        )

    def _delivered_operation_error(
        self, context: AgentContext, action: ToolAction
    ) -> Optional[GateDecision]:
        if action.tool_name not in DELIVERED_OPERATIONS:
            return None
        order_id = action.arguments.get("order_id")
        if not isinstance(order_id, str) or not order_id:
            return None
        for observation in context.observations:
            if (
                observation.tool_name in DELIVERED_OPERATIONS
                and observation.tool_name != action.tool_name
                and observation.arguments.get("order_id") == order_id
                and observation.result.get("ok")
            ):
                return GateDecision(
                    False,
                    "conflicting_delivered_operation",
                    "Do not perform both a return and an exchange on delivered order %s; ask the customer to choose one."
                    % order_id,
                )
        latest_user_message = ""
        for item in reversed(context.conversation):
            if item.get("role") == "user":
                latest_user_message = str(item.get("content", "")).lower()
                break
        if not latest_user_message:
            latest_user_message = context.user_request.lower()
        mentions_return = bool(re.search(r"\breturn(?:ed)?\b", latest_user_message))
        mentions_exchange = bool(re.search(r"\b(?:exchange|swap|replace)\w*\b", latest_user_message))
        if mentions_return and mentions_exchange:
            return GateDecision(
                False,
                "conflicting_requested_operations",
                "The customer mentions both a return and an exchange for this delivered-order request; these operations are mutually exclusive. Ask them to choose exactly one before mutating.",
            )
        return None

    def check(self, context: AgentContext, action: AgentAction) -> GateDecision:
        if isinstance(action, ToolAction):
            for checker in (
                self._required_argument_error,
                self._delivered_operation_error,
                self._payment_error,
                self._read_before_write_error,
            ):
                decision = checker(context, action)
                if decision is not None:
                    return decision
            if self.require_confirmation and action.tool_name in MUTATING_TOOLS:
                if not self._has_current_confirmation(context):
                    return GateDecision(
                        False,
                        "missing_confirmation",
                        "Ask the customer for explicit confirmation of the current unchanged operation scope before mutating the order.",
                    )
            return self._allow()
        if isinstance(action, FinalAction):
            normalized = " ".join(action.message.lower().split()).strip(".!?")
            if normalized in {"task completed", "completed", "done"}:
                if action.outcome == "needs_information":
                    return GateDecision(
                        False,
                        "invalid_completion_placeholder",
                        "Ask one concrete question or request confirmation; do not use a completion placeholder.",
                    )
                if context.step == 1 or not context.observations:
                    return GateDecision(
                        False,
                        "premature_completion",
                        "Do not claim completion before inspecting the request and taking required tool actions.",
                    )
        return self._allow()


def gate_observation(action: AgentAction, decision: GateDecision) -> ToolObservation:
    arguments = action.arguments if isinstance(action, ToolAction) else {}
    return ToolObservation(
        tool_name=GATE_TOOL_NAME,
        arguments=dict(arguments),
        result={
            "ok": False,
            "output": {
                "observation": decision.message,
                "source": "refundpilot.action_gate",
            },
            "error_code": decision.code,
            "error_message": decision.message,
            "done": False,
            "reward": 0.0,
        },
    )

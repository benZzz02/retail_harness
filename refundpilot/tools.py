"""Tool registry and refund-policy enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from time import perf_counter
from typing import Any, Callable, Dict, List, Mapping, Optional

from .state import WorldState


class ToolExecutionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: Dict[str, Any]
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "output": dict(self.output),
            "error_code": self.error_code,
            "error_message": self.error_message,
            "duration_ms": round(self.duration_ms, 3),
        }


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[..., Dict[str, Any]]

    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._tools:
            raise ValueError("duplicate tool: %s" % definition.name)
        self._tools[definition.name] = definition

    def schemas(self) -> List[Dict[str, Any]]:
        return [self._tools[name].schema() for name in sorted(self._tools)]

    def execute(self, name: str, arguments: Mapping[str, Any]) -> ToolResult:
        started = perf_counter()
        definition = self._tools.get(name)
        if definition is None:
            return ToolResult(
                ok=False,
                output={},
                error_code="unknown_tool",
                error_message="tool is not registered: %s" % name,
                duration_ms=(perf_counter() - started) * 1000,
            )
        validation_error = self._validate(definition, arguments)
        if validation_error:
            return ToolResult(
                ok=False,
                output={},
                error_code="invalid_arguments",
                error_message=validation_error,
                duration_ms=(perf_counter() - started) * 1000,
            )
        try:
            output = definition.handler(**dict(arguments))
            return ToolResult(
                ok=True,
                output=output,
                duration_ms=(perf_counter() - started) * 1000,
            )
        except ToolExecutionError as exc:
            return ToolResult(
                ok=False,
                output={},
                error_code=exc.code,
                error_message=exc.message,
                duration_ms=(perf_counter() - started) * 1000,
            )
        except Exception as exc:  # Boundary: tools may be external in real use.
            return ToolResult(
                ok=False,
                output={},
                error_code="tool_error",
                error_message=str(exc),
                duration_ms=(perf_counter() - started) * 1000,
            )

    @staticmethod
    def _validate(
        definition: ToolDefinition, arguments: Mapping[str, Any]
    ) -> Optional[str]:
        if not isinstance(arguments, Mapping):
            return "arguments must be an object"
        parameters = definition.parameters
        properties = parameters.get("properties", {})
        for required in parameters.get("required", []):
            if required not in arguments:
                return "missing required argument: %s" % required
        if parameters.get("additionalProperties") is False:
            unknown = sorted(set(arguments) - set(properties))
            if unknown:
                return "unknown arguments: %s" % ", ".join(unknown)
        for key, value in arguments.items():
            expected = properties.get(key, {}).get("type")
            if expected == "string" and not isinstance(value, str):
                return "%s must be a string" % key
        return None


def _policy_decision(state: WorldState, order_id: str) -> Dict[str, Any]:
    order = state.get_order(order_id)
    if order is None:
        raise ToolExecutionError("order_not_found", "未找到订单 %s" % order_id)
    if order["refunded"] or state.has_refund(order_id):
        return {"decision": "deny", "reason": "订单已退款，不能重复退款"}
    if order["status"] != "delivered":
        return {
            "decision": "deny",
            "reason": "只有已送达订单可以进入当前退款流程",
        }
    if not order.get("delivered_at"):
        return {"decision": "manual_review", "reason": "缺少送达时间"}

    as_of = date.fromisoformat(state.metadata("as_of"))
    delivered_at = date.fromisoformat(order["delivered_at"])
    age_days = (as_of - delivered_at).days
    policy = state.metadata("policy")
    window = int(policy.get("refund_window_days", 7))
    threshold = float(policy.get("manual_review_amount", 2000.0))
    if age_days < 0:
        return {"decision": "manual_review", "reason": "送达日期异常"}
    if age_days > window:
        return {
            "decision": "deny",
            "reason": "已超过%d天退款期限" % window,
            "days_since_delivery": age_days,
        }
    if float(order["amount"]) >= threshold:
        return {
            "decision": "manual_review",
            "reason": "高金额订单需要人工复核",
            "days_since_delivery": age_days,
        }
    return {
        "decision": "auto_refund",
        "reason": "符合自动退款政策",
        "days_since_delivery": age_days,
    }


def build_tool_registry(state: WorldState) -> ToolRegistry:
    registry = ToolRegistry()

    def get_order(order_id: str) -> Dict[str, Any]:
        order = state.get_order(order_id)
        if order is None:
            raise ToolExecutionError("order_not_found", "未找到订单 %s" % order_id)
        return {"order": order}

    def check_refund_policy(order_id: str) -> Dict[str, Any]:
        return _policy_decision(state, order_id)

    def create_refund(order_id: str, reason: str) -> Dict[str, Any]:
        decision = _policy_decision(state, order_id)
        if decision["decision"] != "auto_refund":
            raise ToolExecutionError(
                "policy_violation",
                "退款写入被策略层阻止：%s" % decision["reason"],
            )
        refund_id = state.create_refund(order_id, reason)
        return {"refund_id": refund_id, "order_id": order_id, "status": "created"}

    def escalate_to_human(order_id: str, reason: str) -> Dict[str, Any]:
        decision = _policy_decision(state, order_id)
        if decision["decision"] != "manual_review":
            raise ToolExecutionError(
                "policy_violation",
                "人工升级与当前策略不匹配：%s" % decision["reason"],
            )
        escalation_id = state.create_escalation(order_id, reason)
        return {
            "escalation_id": escalation_id,
            "order_id": order_id,
            "status": "queued",
        }

    string_object = lambda properties, required: {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    registry.register(
        ToolDefinition(
            name="get_order",
            description="读取订单状态、金额、商品和退款标记。",
            parameters=string_object(
                {"order_id": {"type": "string"}}, ["order_id"]
            ),
            handler=get_order,
        )
    )
    registry.register(
        ToolDefinition(
            name="check_refund_policy",
            description="根据订单状态、退款期限、金额和历史退款检查策略。",
            parameters=string_object(
                {"order_id": {"type": "string"}}, ["order_id"]
            ),
            handler=check_refund_policy,
        )
    )
    registry.register(
        ToolDefinition(
            name="create_refund",
            description="为符合自动退款策略的订单创建退款。",
            parameters=string_object(
                {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                ["order_id", "reason"],
            ),
            handler=create_refund,
        )
    )
    registry.register(
        ToolDefinition(
            name="escalate_to_human",
            description="将必须人工复核的退款请求加入审核队列。",
            parameters=string_object(
                {
                    "order_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                ["order_id", "reason"],
            ),
            handler=escalate_to_human,
        )
    )
    return registry


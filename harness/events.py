"""Harness 事件流。

主循环对外暴露 ``Harness.astream()`` 返回 ``AsyncIterator[HarnessEvent]``。
每个语义节点 yield 一个事件，让 FastAPI 路由能转成 SSE 输出，
让前端在 agent 思考过程中实时看到 token / 工具调用 / 工具结果。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .turn import ToolCall, ToolResult

EventType = Literal[
    "run_start",
    "thinking_token",
    "tool_call",
    "tool_result",
    "final_text",
    "run_end",
    "error",
]


@dataclass
class HarnessEvent:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)


# ----- 工厂函数 -----


def run_start(
    trace_id: str,
    parent_trace_id: str | None,
    user_query: str,
    started_at: str,
) -> HarnessEvent:
    return HarnessEvent(
        type="run_start",
        data={
            "trace_id": trace_id,
            "parent_trace_id": parent_trace_id,
            "user_query": user_query,
            "started_at": started_at,
        },
    )


def thinking_token(turn_index: int, delta: str) -> HarnessEvent:
    return HarnessEvent(
        type="thinking_token",
        data={"turn_index": turn_index, "delta": delta},
    )


def tool_call(turn_index: int, call: ToolCall) -> HarnessEvent:
    return HarnessEvent(
        type="tool_call",
        data={"turn_index": turn_index, "call": asdict(call)},
    )


def tool_result(turn_index: int, result: ToolResult) -> HarnessEvent:
    return HarnessEvent(
        type="tool_result",
        data={"turn_index": turn_index, "result": asdict(result)},
    )


def final_text(text: str) -> HarnessEvent:
    return HarnessEvent(type="final_text", data={"final_text": text})


def run_end(
    stopped_reason: str,
    total_tool_calls: int,
    finished_at: str,
) -> HarnessEvent:
    return HarnessEvent(
        type="run_end",
        data={
            "stopped_reason": stopped_reason,
            "total_tool_calls": total_tool_calls,
            "finished_at": finished_at,
        },
    )


def error(message: str) -> HarnessEvent:
    return HarnessEvent(type="error", data={"message": message})


__all__ = [
    "EventType",
    "HarnessEvent",
    "run_start",
    "thinking_token",
    "tool_call",
    "tool_result",
    "final_text",
    "run_end",
    "error",
]

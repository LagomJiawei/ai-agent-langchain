"""Harness 执行轮次的结构化记录。

替代 `agents/state.py` 里那种动态造类 + 用 `tool_call_id` 冒充 `tool_name`
的反模式。每一轮 LLM 推理产生一个 Turn，里面包含本轮思考、本轮决定调用的
工具列表，以及对应的执行结果。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """LLM 决策出的一次工具调用请求。"""

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """工具执行后的结构化结果。

    ``stop_loop`` 是 hook 体系预留的"立即停止主循环"信号：
    PreToolUse hook 拦截返回 ToolResult 时把它设为 True，
    Harness 主循环会在本轮追加 ToolMessage 后停止，并把 stopped_reason
    设为 hook 指定的语义（如 ``loop_guard``）。
    """

    call_id: str
    name: str
    ok: bool
    content: str
    error: str | None = None
    elapsed_ms: float = 0.0
    stop_loop: bool = False
    stop_reason: str | None = None


@dataclass
class Turn:
    """一轮 think -> (act -> observe) 的完整记录。"""

    index: int
    thought: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    final_text: str | None = None

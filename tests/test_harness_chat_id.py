"""Harness.chat_id 贯穿测试：contextvar / HookContext / subagent 继承。"""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool

from harness import (
    Harness,
    HookBus,
    HookContext,
    ToolCall,
    ToolRegistry,
    ToolResult,
    current_chat_id,
    current_trace_id,
)
from harness.loop import _current_chat_id_var


class _Scripted:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    def bind_tools(self, tools):
        return self

    async def astream(self, messages, *args, **kwargs):
        msg = self._scripted.pop(0)
        yield AIMessageChunk(
            content=msg.content,
            tool_calls=getattr(msg, "tool_calls", []) or [],
        )


@tool
def _probe(x: str = "") -> str:
    """探测 hook ctx 的 chat_id"""
    _probe._observed_chat_id = current_chat_id()  # type: ignore[attr-defined]
    return "ok"


def _drain_run(harness: Harness, query: str):
    return harness.run(query)


# ---------- ContextVar 发布 / reset ----------


def test_contextvar_initial_value_is_none():
    assert current_chat_id() is None


def test_harness_publishes_chat_id_during_run():
    reg = ToolRegistry()
    reg.register(_probe, scope="test")

    llm = _Scripted(
        [
            AIMessage(
                content="探测",
                tool_calls=[{"id": "c1", "name": "_probe", "args": {"x": "1"}}],
            ),
            AIMessage(content="完成。"),
        ]
    )
    harness = Harness(
        llm=llm, registry=reg, hooks=HookBus(), max_iterations=5, chat_id="alice"
    )
    _probe._observed_chat_id = None  # type: ignore[attr-defined]
    harness.run("问题")
    assert _probe._observed_chat_id == "alice"  # type: ignore[attr-defined]
    # run 结束后 reset
    assert current_chat_id() is None


# ---------- HookContext 注入 chat_id ----------


def test_hook_context_carries_chat_id():
    captured: dict[str, HookContext] = {}

    def capture_pre(call: ToolCall, ctx: HookContext):
        captured["ctx"] = ctx
        return None  # 放行

    reg = ToolRegistry()
    reg.register(_probe, scope="test")
    bus = HookBus()
    bus.register_pre(capture_pre)

    llm = _Scripted(
        [
            AIMessage(
                content="t",
                tool_calls=[{"id": "c1", "name": "_probe", "args": {"x": ""}}],
            ),
            AIMessage(content="done."),
        ]
    )
    harness = Harness(
        llm=llm, registry=reg, hooks=bus, max_iterations=3, chat_id="zhangsan"
    )
    harness.run("问题")
    assert captured["ctx"].chat_id == "zhangsan"


def test_hook_context_chat_id_none_when_not_provided():
    captured: dict[str, HookContext] = {}

    def capture(call, ctx):
        captured["ctx"] = ctx
        return None

    reg = ToolRegistry()
    reg.register(_probe, scope="test")
    bus = HookBus()
    bus.register_pre(capture)
    llm = _Scripted(
        [
            AIMessage(
                content="t",
                tool_calls=[{"id": "c1", "name": "_probe", "args": {"x": ""}}],
            ),
            AIMessage(content="done."),
        ]
    )
    harness = Harness(llm=llm, registry=reg, hooks=bus, max_iterations=3)
    harness.run("问题")
    assert captured["ctx"].chat_id is None


# ---------- subagent 继承 ----------


def test_subagent_inherits_chat_id(monkeypatch):
    """dispatch_subagent 工具体读 current_chat_id() 并传给子 Harness。"""
    import asyncio as _asyncio

    from harness.subagent import dispatch_subagent

    captured = {}

    async def fake_arun(self, q):
        captured["chat_id"] = self.chat_id
        captured["parent_trace_id"] = self.parent_trace_id
        from harness import HarnessResult

        return HarnessResult(final_text="子 ok", turns=[], stopped_reason="final_text")

    monkeypatch.setattr(Harness, "arun", fake_arun)

    token = _current_chat_id_var.set("bob")
    try:
        _asyncio.run(dispatch_subagent.ainvoke({"task": "子任务", "allowed_scopes": ["kb"]}))
    finally:
        _current_chat_id_var.reset(token)

    assert captured["chat_id"] == "bob"

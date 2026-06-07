"""Harness.arun 公开异步入口测试。"""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool

from harness import Harness, HookBus, ToolRegistry


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
def _noop(x: str = "") -> str:
    """noop"""
    return "ok"


@pytest.fixture
def harness_with_final_text() -> Harness:
    reg = ToolRegistry()
    reg.register(_noop, scope="test")
    llm = _Scripted([AIMessage(content="直接答案")])
    return Harness(llm=llm, registry=reg, hooks=HookBus(), max_iterations=3)


def test_arun_returns_harness_result(harness_with_final_text: Harness):
    result = asyncio.run(harness_with_final_text.arun("问题"))
    assert result.stopped_reason == "final_text"
    assert result.final_text == "直接答案"


def test_run_delegates_to_arun(harness_with_final_text: Harness):
    """同步 run() 应通过 asyncio.run 走 arun，结果一致。"""
    result = harness_with_final_text.run("问题")
    assert result.stopped_reason == "final_text"
    assert result.final_text == "直接答案"


def test_arun_can_be_awaited_in_existing_event_loop():
    """asyncio.gather 场景下：N 个 arun 并发可在同一 loop 内 await。"""
    reg = ToolRegistry()
    reg.register(_noop, scope="test")

    async def _run_many():
        outcomes = []
        for delta in ["A", "B", "C"]:
            llm = _Scripted([AIMessage(content=delta)])
            h = Harness(llm=llm, registry=reg, hooks=HookBus(), max_iterations=2)
            outcomes.append(await h.arun("q"))
        return outcomes

    results = asyncio.run(_run_many())
    assert [r.final_text for r in results] == ["A", "B", "C"]

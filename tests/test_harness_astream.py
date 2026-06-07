"""Harness.astream 异步事件流测试。

覆盖：
- 单轮无工具调用的事件序列（含 thinking_token）。
- 单轮含工具调用的事件序列。
- 4 种 stopped_reason 在 astream 路径下的正确触发。
- Harness.run() 走 astream 后返回 HarnessResult 与原 run 等价。
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool

from harness import Harness, HarnessEvent, HookBus, ToolRegistry
from harness.builtin_hooks import LoopGuardHook


# ---------- 测试桩 ----------


class FakeStreamLLM:
    """异步流式桩：每次 astream 把脚本里的 AIMessage 拆成若干 chunk yield。

    若脚本项是单个 AIMessage（无 chunks 元信息），整条作为一个 chunk yield。
    若脚本项是 list[AIMessageChunk]，逐 chunk yield —— 用于测试多 token 增量场景。
    """

    def __init__(self, scripted: list):
        self._scripted = list(scripted)

    def bind_tools(self, tools):
        return self

    async def astream(self, messages, *args: Any, **kwargs: Any):
        if not self._scripted:
            raise AssertionError("FakeStreamLLM 脚本耗尽")
        item = self._scripted.pop(0)
        if isinstance(item, list):
            for chunk in item:
                yield chunk
        else:
            yield AIMessageChunk(
                content=item.content,
                tool_calls=getattr(item, "tool_calls", []) or [],
            )


def _drain(harness: Harness, query: str) -> list[HarnessEvent]:
    async def _collect():
        out: list[HarnessEvent] = []
        async for ev in harness.astream(query):
            out.append(ev)
        return out

    return asyncio.run(_collect())


# ---------- 工具桩 ----------


@tool
def echo_tool(text: str) -> str:
    """echo"""
    return f"echo:{text}"


@tool
def do_terminate(final_answer: str) -> str:
    """terminate"""
    return final_answer


@pytest.fixture
def registry_with_echo() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(echo_tool, scope="test")
    reg.register(do_terminate, scope="control")
    return reg


@pytest.fixture
def bare_hooks() -> HookBus:
    return HookBus()


# ---------- 单轮无工具调用 ----------


def test_astream_single_turn_final_text(registry_with_echo, bare_hooks):
    llm = FakeStreamLLM([AIMessage(content="直接答案。")])
    harness = Harness(
        llm=llm, registry=registry_with_echo, hooks=bare_hooks, max_iterations=3
    )
    events = _drain(harness, "问题")
    types = [e.type for e in events]
    assert types[0] == "run_start"
    assert "thinking_token" in types
    assert types[-2:] == ["final_text", "run_end"]
    # 最后 final_text 内容正确
    assert events[-2].data["final_text"] == "直接答案。"
    # run_end 携带 stopped_reason
    assert events[-1].data["stopped_reason"] == "final_text"


def test_astream_emits_token_per_chunk(registry_with_echo, bare_hooks):
    """多 chunk 增量都应转成 thinking_token 事件。"""
    llm = FakeStreamLLM(
        [
            [
                AIMessageChunk(content="你好"),
                AIMessageChunk(content="，"),
                AIMessageChunk(content="世界"),
            ]
        ]
    )
    harness = Harness(
        llm=llm, registry=registry_with_echo, hooks=bare_hooks, max_iterations=3
    )
    events = _drain(harness, "问题")
    token_events = [e for e in events if e.type == "thinking_token"]
    assert [e.data["delta"] for e in token_events] == ["你好", "，", "世界"]


# ---------- 含工具调用 ----------


def test_astream_tool_call_event_sequence(registry_with_echo, bare_hooks):
    llm = FakeStreamLLM(
        [
            AIMessage(
                content="先调工具",
                tool_calls=[{"id": "c1", "name": "echo_tool", "args": {"text": "hi"}}],
            ),
            AIMessage(content="完成。"),
        ]
    )
    harness = Harness(
        llm=llm, registry=registry_with_echo, hooks=bare_hooks, max_iterations=3
    )
    events = _drain(harness, "问题")
    types = [e.type for e in events]
    # 期望顺序：run_start → thinking_token → tool_call → tool_result → thinking_token → final_text → run_end
    assert types[0] == "run_start"
    assert "tool_call" in types
    assert "tool_result" in types
    assert types[-2:] == ["final_text", "run_end"]
    # tool_call 携带的工具名
    tc = next(e for e in events if e.type == "tool_call")
    assert tc.data["call"]["name"] == "echo_tool"
    tr = next(e for e in events if e.type == "tool_result")
    assert tr.data["result"]["content"] == "echo:hi"


# ---------- run / astream 等价 ----------


def test_run_returns_same_result_as_astream(registry_with_echo, bare_hooks):
    """同步 run() 委托 astream() 后语义一致。"""
    llm = FakeStreamLLM([AIMessage(content="只回答。")])
    harness = Harness(
        llm=llm, registry=registry_with_echo, hooks=bare_hooks, max_iterations=3
    )
    result = harness.run("问题")
    assert result.stopped_reason == "final_text"
    assert result.final_text == "只回答。"
    assert len(result.turns) == 1


# ---------- 4 种 stopped_reason ----------


def test_astream_loop_guard_stop(registry_with_echo):
    same = [{"id": "c1", "name": "echo_tool", "args": {"text": "x"}}]
    same_again = [{"id": "c2", "name": "echo_tool", "args": {"text": "x"}}]
    llm = FakeStreamLLM(
        [
            AIMessage(content="第一次", tool_calls=same),
            AIMessage(content="第二次", tool_calls=same_again),
        ]
    )
    hooks = HookBus()
    hooks.register_pre(LoopGuardHook(threshold=2))
    harness = Harness(
        llm=llm, registry=registry_with_echo, hooks=hooks, max_iterations=5
    )
    events = _drain(harness, "问题")
    run_end_evt = events[-1]
    assert run_end_evt.type == "run_end"
    assert run_end_evt.data["stopped_reason"] == "loop_guard"


def test_astream_terminate_tool_stop(registry_with_echo, bare_hooks):
    llm = FakeStreamLLM(
        [
            AIMessage(
                content="决定结束",
                tool_calls=[
                    {
                        "id": "t1",
                        "name": "do_terminate",
                        "args": {"final_answer": "最终答案"},
                    }
                ],
            )
        ]
    )
    harness = Harness(
        llm=llm, registry=registry_with_echo, hooks=bare_hooks, max_iterations=5
    )
    events = _drain(harness, "问题")
    assert events[-1].data["stopped_reason"] == "terminate_tool"
    final_evt = next(e for e in events if e.type == "final_text")
    assert final_evt.data["final_text"] == "最终答案"


def test_astream_max_iterations_stop(registry_with_echo, bare_hooks):
    llm = FakeStreamLLM(
        [
            AIMessage(
                content=f"轮 {i}",
                tool_calls=[
                    {"id": f"r{i}", "name": "echo_tool", "args": {"text": f"x{i}"}}
                ],
            )
            for i in range(2)
        ]
    )
    harness = Harness(
        llm=llm, registry=registry_with_echo, hooks=bare_hooks, max_iterations=2
    )
    events = _drain(harness, "问题")
    assert events[-1].data["stopped_reason"] == "max_iterations"


# ---------- 异常路径 ----------


def test_astream_emits_error_event_on_exception(registry_with_echo, bare_hooks):
    class BoomLLM:
        def bind_tools(self, tools):
            return self

        async def astream(self, messages, *args, **kwargs):
            raise RuntimeError("llm 炸了")
            yield  # pragma: no cover  让函数成为 async generator

    harness = Harness(
        llm=BoomLLM(), registry=registry_with_echo, hooks=bare_hooks, max_iterations=3
    )
    events = _drain(harness, "问题")
    types = [e.type for e in events]
    # 期望：run_start -> error -> run_end（异常路径不发 final_text）
    assert types[0] == "run_start"
    assert types[-2] == "error"
    assert types[-1] == "run_end"
    assert "final_text" not in types
    # error 携带异常信息
    err_evt = events[-2]
    assert "llm 炸了" in err_evt.data["message"]
    # run_end 携带 stopped_reason="error"
    run_end_evt = events[-1]
    assert run_end_evt.data["stopped_reason"] == "error"
    assert "finished_at" in run_end_evt.data


def test_astream_invokes_onstop_hook_on_exception(registry_with_echo):
    """异常路径下 OnStop 仍要触发，确保 trace 写盘。"""
    captured: list = []

    def capture_stop(result, ctx):
        captured.append({"final_text": result.final_text, "stopped_reason": result.stopped_reason})

    hooks = HookBus()
    hooks.register_stop(capture_stop)

    class BoomLLM:
        def bind_tools(self, tools):
            return self

        async def astream(self, messages, *args, **kwargs):
            raise RuntimeError("network down")
            yield  # pragma: no cover

    harness = Harness(
        llm=BoomLLM(), registry=registry_with_echo, hooks=hooks, max_iterations=3
    )
    _drain(harness, "问题")

    assert len(captured) == 1
    assert captured[0]["stopped_reason"] == "error"
    assert "network down" in captured[0]["final_text"]


def test_astream_run_end_sent_even_if_onstop_hook_fails(registry_with_echo):
    """OnStop 钩子在异常路径下自己再炸，不能阻塞 run_end 发射。"""

    def broken_stop(result, ctx):
        raise RuntimeError("trace 盘写满了")

    hooks = HookBus()
    hooks.register_stop(broken_stop)

    class BoomLLM:
        def bind_tools(self, tools):
            return self

        async def astream(self, messages, *args, **kwargs):
            raise RuntimeError("llm 也炸了")
            yield  # pragma: no cover

    harness = Harness(
        llm=BoomLLM(), registry=registry_with_echo, hooks=hooks, max_iterations=3
    )
    events = _drain(harness, "问题")
    types = [e.type for e in events]
    assert types[-1] == "run_end"
    assert events[-1].data["stopped_reason"] == "error"


def test_arun_returns_error_result_on_exception(registry_with_echo, bare_hooks):
    """同步 run() / arun() 在 LLM 异常时仍返回 HarnessResult（stopped_reason=error）。"""

    class BoomLLM:
        def bind_tools(self, tools):
            return self

        async def astream(self, messages, *args, **kwargs):
            raise RuntimeError("oops")
            yield  # pragma: no cover

    harness = Harness(
        llm=BoomLLM(), registry=registry_with_echo, hooks=bare_hooks, max_iterations=3
    )
    result = harness.run("问题")
    assert result.stopped_reason == "error"
    assert "oops" in result.final_text

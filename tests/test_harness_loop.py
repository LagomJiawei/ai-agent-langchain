"""Harness 主循环测试：用 FakeLLM 覆盖 4 种 stopped_reason 分支。"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from harness import Harness, HookBus, ToolRegistry, current_trace_id
from harness.builtin_hooks import LoopGuardHook


# ---------- 测试桩 ----------


class FakeLLM:
    """按预设脚本顺序返回 AIMessage。

    每次 invoke/astream 弹出 scripted 列表第一项；列表用完后报错（防止意外多调用）。
    bind_tools 返回自身，不做实际绑定（fake 已知道返回什么）。
    """

    def __init__(self, scripted: list[AIMessage]):
        self._scripted: list[AIMessage] = list(scripted)
        self.calls: list[list] = []

    def bind_tools(self, tools):  # noqa: ARG002
        return self

    def invoke(self, messages, *args: Any, **kwargs: Any) -> AIMessage:  # noqa: ARG002
        if not self._scripted:
            raise AssertionError("FakeLLM 脚本已耗尽，loop 多调用了一次")
        self.calls.append(list(messages))
        return self._scripted.pop(0)

    async def astream(self, messages, *args: Any, **kwargs: Any):  # noqa: ARG002
        """把脚本里的 AIMessage 整条作为单个 chunk yield。

        Harness 内部用 ``accumulated = chunk if accumulated is None else accumulated + chunk``
        累加；单个 AIMessage 直接拿来当 chunk 也能工作（AIMessageChunk 是 AIMessage 的子类
        重写了 ``+``；这里测试桩只 yield 一次，不依赖 ``+`` 行为）。
        """
        from langchain_core.messages import AIMessageChunk

        if not self._scripted:
            raise AssertionError("FakeLLM 脚本已耗尽，loop 多调用了一次")
        self.calls.append(list(messages))
        msg = self._scripted.pop(0)
        # 转成 chunk，保留 content / tool_calls
        chunk = AIMessageChunk(
            content=msg.content,
            tool_calls=getattr(msg, "tool_calls", []) or [],
        )
        yield chunk


def _ai_with_tool_calls(tool_calls: list[dict], text: str = "") -> AIMessage:
    return AIMessage(content=text, tool_calls=tool_calls)


@tool
def echo_tool(text: str) -> str:
    """echo back"""
    return f"echo:{text}"


@tool
def do_terminate(final_answer: str) -> str:
    """终止"""
    return final_answer


@pytest.fixture
def registry_with_echo() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(echo_tool, scope="test")
    reg.register(do_terminate, scope="control")
    return reg


@pytest.fixture
def bare_hooks() -> HookBus:
    """裸 hook 总线，避免默认装配把限流/白名单卷入主循环测试。"""
    return HookBus()


# ---------- 4 种 stopped_reason ----------


def test_stopped_reason_final_text(registry_with_echo: ToolRegistry, bare_hooks: HookBus):
    llm = FakeLLM([AIMessage(content="直接回答：这是答案。")])
    harness = Harness(
        llm=llm, registry=registry_with_echo, hooks=bare_hooks, max_iterations=5
    )

    result = harness.run("问题")

    assert result.stopped_reason == "final_text"
    assert result.final_text == "直接回答：这是答案。"
    assert len(result.turns) == 1
    assert result.total_tool_calls == 0


def test_stopped_reason_loop_guard(registry_with_echo: ToolRegistry):
    same_call = [{"id": "1", "name": "echo_tool", "args": {"text": "hi"}}]
    same_call_round_two = [{"id": "2", "name": "echo_tool", "args": {"text": "hi"}}]
    llm = FakeLLM(
        [
            _ai_with_tool_calls(same_call, text="第一次"),
            _ai_with_tool_calls(same_call_round_two, text="再来一次"),
        ]
    )
    hooks = HookBus()
    hooks.register_pre(LoopGuardHook(threshold=2))
    harness = Harness(
        llm=llm, registry=registry_with_echo, hooks=hooks, max_iterations=5
    )

    result = harness.run("问题")

    assert result.stopped_reason == "loop_guard"
    assert "echo_tool" in result.final_text
    assert len(result.turns) == 2


def test_stopped_reason_terminate_tool(registry_with_echo: ToolRegistry, bare_hooks: HookBus):
    llm = FakeLLM(
        [
            _ai_with_tool_calls(
                [
                    {
                        "id": "t1",
                        "name": "do_terminate",
                        "args": {"final_answer": "终止答案"},
                    }
                ],
                text="决定终止",
            )
        ]
    )
    harness = Harness(
        llm=llm, registry=registry_with_echo, hooks=bare_hooks, max_iterations=5
    )

    result = harness.run("问题")

    assert result.stopped_reason == "terminate_tool"
    assert result.final_text == "终止答案"
    assert result.total_tool_calls == 1


def test_stopped_reason_max_iterations(registry_with_echo: ToolRegistry, bare_hooks: HookBus):
    # 每轮发不同参数避免触发 loop_guard，凑满 max_iterations=2
    llm = FakeLLM(
        [
            _ai_with_tool_calls(
                [{"id": f"r{i}", "name": "echo_tool", "args": {"text": f"x{i}"}}],
                text=f"轮 {i}",
            )
            for i in range(2)
        ]
    )
    harness = Harness(
        llm=llm, registry=registry_with_echo, hooks=bare_hooks, max_iterations=2
    )

    result = harness.run("问题")

    assert result.stopped_reason == "max_iterations"
    assert "最大迭代次数" in result.final_text
    assert len(result.turns) == 2


# ---------- Turn / ToolResult 元数据完整 ----------


def test_tool_call_metadata_preserved(registry_with_echo: ToolRegistry, bare_hooks: HookBus):
    llm = FakeLLM(
        [
            _ai_with_tool_calls(
                [{"id": "c1", "name": "echo_tool", "args": {"text": "hello"}}],
                text="调用一下",
            ),
            AIMessage(content="完成。"),
        ]
    )
    harness = Harness(
        llm=llm, registry=registry_with_echo, hooks=bare_hooks, max_iterations=5
    )

    result = harness.run("问题")

    assert result.stopped_reason == "final_text"
    assert len(result.turns) == 2
    first = result.turns[0]
    assert first.tool_calls[0].name == "echo_tool"
    assert first.tool_calls[0].args == {"text": "hello"}
    assert first.tool_results[0].ok is True
    assert first.tool_results[0].content == "echo:hello"
    assert first.tool_results[0].call_id == "c1"


def test_unknown_tool_returns_failed_result(registry_with_echo: ToolRegistry, bare_hooks: HookBus):
    llm = FakeLLM(
        [
            _ai_with_tool_calls(
                [{"id": "x1", "name": "ghost_tool", "args": {}}],
                text="试试不存在的",
            ),
            AIMessage(content="放弃。"),
        ]
    )
    harness = Harness(
        llm=llm, registry=registry_with_echo, hooks=bare_hooks, max_iterations=5
    )

    result = harness.run("问题")

    failed = result.turns[0].tool_results[0]
    assert failed.ok is False
    assert "未知工具" in failed.content


# ---------- contextvar 发布 ----------


@tool
def _trace_probe(_x: str = "") -> str:
    """在工具执行期捕获 current_trace_id。"""
    _trace_probe._captured = current_trace_id()  # type: ignore[attr-defined]
    return "ok"


def test_harness_publishes_current_trace_id_to_contextvar(bare_hooks: HookBus):
    assert current_trace_id() is None

    reg = ToolRegistry()
    reg.register(_trace_probe, scope="test")

    llm = FakeLLM(
        [
            _ai_with_tool_calls(
                [{"id": "c1", "name": "_trace_probe", "args": {"_x": "y"}}],
                text="探测",
            ),
            AIMessage(content="完成。"),
        ]
    )
    harness = Harness(
        llm=llm, registry=reg, hooks=bare_hooks, max_iterations=5
    )
    _trace_probe._captured = None  # type: ignore[attr-defined]
    harness.run("问题")

    # 工具执行时 contextvar 已设置
    assert _trace_probe._captured is not None  # type: ignore[attr-defined]
    # run 结束后 contextvar 被 reset
    assert current_trace_id() is None

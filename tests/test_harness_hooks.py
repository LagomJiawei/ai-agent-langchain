"""Hook 总线协议测试：拦截、短路、链式、OnStop、frozen ctx。"""
from __future__ import annotations

import pytest
from dataclasses import FrozenInstanceError

from harness import HookBus, HookContext, ToolCall, ToolResult


def _ctx(turn_index: int = 0) -> HookContext:
    return HookContext(user_query="q", turn_index=turn_index, call_counts={})


def _call(name: str = "t", args: dict | None = None) -> ToolCall:
    return ToolCall(id="c1", name=name, args=args or {})


def _result(name: str = "t", content: str = "ok") -> ToolResult:
    return ToolResult(call_id="c1", name=name, ok=True, content=content)


# ---------- HookContext ----------


def test_hook_context_is_frozen():
    ctx = _ctx()
    with pytest.raises(FrozenInstanceError):
        ctx.user_query = "tampered"  # type: ignore[misc]


def test_hook_context_call_counts_is_mutable_inside():
    """frozen=True 锁的是字段绑定，字段值（dict）本身仍可变。"""
    ctx = _ctx()
    ctx.call_counts["foo"] = 1
    assert ctx.call_counts["foo"] == 1


# ---------- Pre 短路 ----------


def test_pre_first_non_none_short_circuits():
    bus = HookBus()
    called: list[str] = []

    def hook_a(call, ctx):
        called.append("a")
        return _result(content="from-a")

    def hook_b(call, ctx):
        called.append("b")
        return _result(content="from-b")

    bus.register_pre(hook_a)
    bus.register_pre(hook_b)

    outcome = bus.run_pre(_call(), _ctx())
    assert outcome is not None
    assert outcome.content == "from-a"
    assert called == ["a"]


def test_pre_all_return_none_means_pass():
    bus = HookBus()
    bus.register_pre(lambda call, ctx: None)
    bus.register_pre(lambda call, ctx: None)
    assert bus.run_pre(_call(), _ctx()) is None


# ---------- Post 链式 ----------


def test_post_hooks_chain_outputs():
    bus = HookBus()

    def append_a(call, result, ctx):
        return ToolResult(
            call_id=result.call_id,
            name=result.name,
            ok=result.ok,
            content=result.content + "+a",
        )

    def append_b(call, result, ctx):
        return ToolResult(
            call_id=result.call_id,
            name=result.name,
            ok=result.ok,
            content=result.content + "+b",
        )

    bus.register_post(append_a)
    bus.register_post(append_b)

    final = bus.run_post(_call(), _result(content="raw"), _ctx())
    assert final.content == "raw+a+b"


def test_post_with_no_hooks_returns_input():
    bus = HookBus()
    r = _result(content="x")
    assert bus.run_post(_call(), r, _ctx()) is r


# ---------- OnStop ----------


def test_on_stop_runs_all_hooks():
    bus = HookBus()
    fired: list[int] = []
    bus.register_stop(lambda result, ctx: fired.append(1))
    bus.register_stop(lambda result, ctx: fired.append(2))

    # 假 HarnessResult；OnStop hook 只需要拿到对象，不强检字段
    class FakeResult:
        pass

    bus.run_stop(FakeResult(), _ctx())
    assert fired == [1, 2]


# ---------- bus 内省 ----------


def test_bus_counters():
    bus = HookBus()
    bus.register_pre(lambda c, x: None)
    bus.register_pre(lambda c, x: None)
    bus.register_post(lambda c, r, x: r)
    bus.register_stop(lambda r, x: None)
    assert bus.pre_count == 2
    assert bus.post_count == 1
    assert bus.stop_count == 1

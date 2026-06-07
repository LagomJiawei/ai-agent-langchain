"""内置 hook 行为测试：Allowlist / LoopGuard / RateLimit + 端到端集成。"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from harness import Harness, HookBus, HookContext, ToolCall, ToolRegistry
from harness.builtin_hooks import (
    FilePathAllowlistHook,
    LoopGuardHook,
    RateLimitPostHook,
    RateLimitPreHook,
    TerminalAllowlistHook,
    default_hooks,
)
from tools.rate_limiter import RateLimiter


def _ctx(counts: dict | None = None) -> HookContext:
    return HookContext(user_query="q", turn_index=0, call_counts=counts if counts is not None else {})


def _call(name: str, args: dict | None = None) -> ToolCall:
    return ToolCall(id="c1", name=name, args=args or {})


# ---------- TerminalAllowlistHook ----------


def test_terminal_allowlist_passes_safe_command():
    hook = TerminalAllowlistHook()
    assert hook(_call("terminal_exec", {"command": "ls"}), _ctx()) is None


def test_terminal_allowlist_blocks_unsafe_command():
    hook = TerminalAllowlistHook()
    blocked = hook(_call("terminal_exec", {"command": "rm -rf /"}), _ctx())
    assert blocked is not None
    assert blocked.ok is False
    assert "安全检查失败" in blocked.content
    assert blocked.error == "terminal_allowlist_denied"


def test_terminal_allowlist_ignores_other_tools():
    hook = TerminalAllowlistHook()
    assert hook(_call("search_web", {"query": "x"}), _ctx()) is None


# ---------- FilePathAllowlistHook ----------


def test_file_allowlist_blocks_out_of_scope_path():
    hook = FilePathAllowlistHook()
    blocked = hook(_call("file_read", {"path": "../../etc/passwd"}), _ctx())
    assert blocked is not None
    assert blocked.error == "file_path_denied"


def test_file_allowlist_ignores_unrelated_tool():
    hook = FilePathAllowlistHook()
    assert hook(_call("search_web", {"query": "x"}), _ctx()) is None


def test_file_allowlist_handles_list_files_directory_arg():
    hook = FilePathAllowlistHook()
    # 越界目录
    blocked = hook(_call("list_files", {"directory": "/etc"}), _ctx())
    assert blocked is not None
    assert blocked.error == "file_path_denied"


# ---------- LoopGuardHook ----------


def test_loop_guard_passes_first_call():
    hook = LoopGuardHook(threshold=2)
    counts: dict[str, int] = {}
    assert hook(_call("search_web", {"q": "a"}), _ctx(counts)) is None
    assert counts  # 计数已写入


def test_loop_guard_blocks_at_threshold():
    hook = LoopGuardHook(threshold=2)
    counts: dict[str, int] = {}
    call = _call("search_web", {"q": "a"})
    assert hook(call, _ctx(counts)) is None
    blocked = hook(call, _ctx(counts))
    assert blocked is not None
    assert blocked.stop_loop is True
    assert blocked.stop_reason == "loop_guard"
    assert blocked.error == "loop_guard"


def test_loop_guard_distinguishes_args():
    hook = LoopGuardHook(threshold=2)
    counts: dict[str, int] = {}
    assert hook(_call("t", {"x": 1}), _ctx(counts)) is None
    assert hook(_call("t", {"x": 2}), _ctx(counts)) is None  # 不同 args，不算重复


# ---------- RateLimit Pre/Post ----------


def test_rate_limit_pre_blocks_when_qps_exhausted():
    """构造一个 QPS=0 的极端 limiter，第一次 acquire 就拒绝。"""
    limiter = RateLimiter(qps=0, max_concurrent=10)
    # token_bucket 容量 = qps*2 = 0
    pre = RateLimitPreHook(limiter=limiter)
    blocked = pre(_call("search_web", {"q": "a"}), _ctx())
    assert blocked is not None
    assert blocked.error == "rate_limited"


def test_rate_limit_pre_post_round_trip():
    limiter = RateLimiter(qps=100, max_concurrent=5)
    pre = RateLimitPreHook(limiter=limiter)
    post = RateLimitPostHook(pre)

    call = _call("search_web", {"q": "a"})
    assert pre(call, _ctx()) is None  # 放行
    assert call.id in pre._inflight

    # 模拟工具成功
    from harness import ToolResult

    success = ToolResult(call_id=call.id, name=call.name, ok=True, content="ok")
    post(call, success, _ctx())
    assert call.id not in pre._inflight


# ---------- 回归 #2：线程安全 + 重发幂等 + 兜底清扫 ----------


def test_rate_limit_pre_is_idempotent_on_duplicate_call_id():
    """同一 call.id 被 Pre 看到第二次时直接放行，不重复占 semaphore。"""
    limiter = RateLimiter(qps=100, max_concurrent=2)
    pre = RateLimitPreHook(limiter=limiter)

    call = _call("search_web", {"q": "a"})
    # 第一次正常占用
    assert pre(call, _ctx()) is None
    # 同 call.id 第二次：幂等放行
    assert pre(call, _ctx()) is None
    # 第三次也是
    assert pre(call, _ctx()) is None

    # semaphore 只占了 1 个，剩余 1 个还能拿
    assert limiter.semaphore.acquire(block=False) is True
    # 再拿就拿不到了（容量 2，已占 2）
    assert limiter.semaphore.acquire(block=False) is False


def test_rate_limit_post_releases_only_when_pre_acquired():
    """Post 没在 inflight 找到 call.id 时直接透传（防误 release 别人占的位置）。"""
    limiter = RateLimiter(qps=100, max_concurrent=2)
    pre = RateLimitPreHook(limiter=limiter)
    post = RateLimitPostHook(pre)

    from harness import ToolResult

    call = _call("unknown_call", {"q": "x"})
    result = ToolResult(call_id=call.id, name=call.name, ok=True, content="ok")
    # Pre 从未占过这个 call.id
    out = post(call, result, _ctx())
    assert out is result  # 透传

    # semaphore 完整无损
    assert limiter.semaphore.acquire(block=False) is True
    assert limiter.semaphore.acquire(block=False) is True


def test_rate_limit_sweeper_returns_leaked_semaphores():
    """主循环异常吞掉 Post 时，Sweeper StopHook 兜底归还所有残留 semaphore。"""
    from harness.builtin_hooks import RateLimitSweeperStopHook

    limiter = RateLimiter(qps=100, max_concurrent=3)
    pre = RateLimitPreHook(limiter=limiter)
    sweeper = RateLimitSweeperStopHook(pre)

    # 模拟两次工具调用 Pre 占用但 Post 没跑
    pre(ToolCall(id="a", name="search_web", args={"q": "a"}), _ctx())
    pre(ToolCall(id="b", name="search_web", args={"q": "b"}), _ctx())

    # 此时 semaphore 占了 2 个，剩 1 个
    assert limiter.semaphore.acquire(block=False) is True
    assert limiter.semaphore.acquire(block=False) is False
    limiter.semaphore.release()  # 还回探测占用的那 1 个

    # 触发 sweeper（OnStop 参数兼容签名即可）
    from harness import HarnessResult

    fake_result = HarnessResult(final_text="x", turns=[], stopped_reason="error")
    sweeper(fake_result, _ctx())

    # 残留 2 个都被归还，semaphore 完全恢复
    assert pre._inflight == {}
    for _ in range(3):
        assert limiter.semaphore.acquire(block=False) is True
    assert limiter.semaphore.acquire(block=False) is False


def test_rate_limit_pre_thread_safe_under_concurrent_acquire():
    """N 个线程并发对同一 Pre hook 调用不同 call.id；inflight 数量必须精确。"""
    import threading

    # 容量给足，避免被 semaphore 阻塞掩盖真问题
    limiter = RateLimiter(qps=10_000, max_concurrent=128)
    pre = RateLimitPreHook(limiter=limiter)

    n_threads = 64
    barrier = threading.Barrier(n_threads)

    def worker(i: int):
        barrier.wait()
        pre(ToolCall(id=f"c-{i}", name="search_web", args={"i": i}), _ctx())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(pre._inflight) == n_threads
    assert set(pre._inflight.keys()) == {f"c-{i}" for i in range(n_threads)}


# ---------- 回归 #7：sync hook 通过 HookBus.arun_pre 不阻塞 event loop ----------


def test_arun_pre_does_not_block_event_loop_when_sync_hook_blocks():
    """sync hook 内部阻塞 acquire 必须被 to_thread 隔离，
    event loop 上的其他协程要能继续推进。
    """
    import asyncio
    import time

    from harness import HookBus

    # 一个会阻塞 0.3s 的 sync hook
    def slow_blocking_hook(call, ctx):
        time.sleep(0.3)
        return None

    bus = HookBus()
    bus.register_pre(slow_blocking_hook)

    async def _drive():
        # 在 arun_pre 跑的同时，让 event loop 有别的协程要推进
        ticks = []

        async def ticker():
            for _ in range(5):
                await asyncio.sleep(0.05)
                ticks.append(time.monotonic())

        async def fire_hook():
            await bus.arun_pre(_call("x", {}), _ctx())

        start = time.monotonic()
        await asyncio.gather(ticker(), fire_hook())
        return ticks, time.monotonic() - start

    ticks, elapsed = asyncio.run(_drive())
    # 如果 sync hook 冻结 loop，ticker 5 次 sleep 总耗时 0.25s 应被 hook 的 0.3s 串行卡死
    # 现在并发执行，总耗时应 ≈ max(0.3, 0.25) + 调度抖动 < 0.5s
    assert elapsed < 0.5, f"event loop 被阻塞了，总耗时 {elapsed:.2f}s"
    assert len(ticks) == 5


# ---------- 回归 #6：async hook 被 arun_pre / arun_post / arun_stop 原生 await ----------


def test_arun_pre_awaits_async_hook_natively():
    """async pre hook 被 await，不走 to_thread；返回 ToolResult 正确短路。"""
    import asyncio

    from harness import HookBus, ToolResult

    captured = {}

    async def async_pre(call, ctx):
        await asyncio.sleep(0.01)
        captured["called"] = True
        return ToolResult(
            call_id=call.id, name=call.name, ok=False, content="async blocked"
        )

    bus = HookBus()
    bus.register_pre(async_pre)

    out = asyncio.run(bus.arun_pre(_call("x", {}), _ctx()))
    assert captured["called"] is True
    assert out is not None
    assert out.content == "async blocked"


def test_arun_post_chains_async_and_sync_hooks():
    """链式 post：async 和 sync 混着排，每个 hook 拿到上一个的输出。"""
    import asyncio

    from harness import HookBus, ToolResult

    async def async_post(call, result, ctx):
        await asyncio.sleep(0.01)
        return ToolResult(
            call_id=result.call_id,
            name=result.name,
            ok=result.ok,
            content=result.content + "-async",
        )

    def sync_post(call, result, ctx):
        return ToolResult(
            call_id=result.call_id,
            name=result.name,
            ok=result.ok,
            content=result.content + "-sync",
        )

    bus = HookBus()
    bus.register_post(async_post)
    bus.register_post(sync_post)

    initial = ToolResult(call_id="c1", name="x", ok=True, content="base")
    out = asyncio.run(bus.arun_post(_call("x", {}), initial, _ctx()))
    assert out.content == "base-async-sync"


def test_arun_stop_awaits_async_hook():
    """async stop hook 也被 await，不走 to_thread。"""
    import asyncio

    from harness import HarnessResult, HookBus

    captured = {}

    async def async_stop(result, ctx):
        await asyncio.sleep(0.01)
        captured["final_text"] = result.final_text

    bus = HookBus()
    bus.register_stop(async_stop)

    fake = HarnessResult(final_text="done", turns=[], stopped_reason="final_text")
    asyncio.run(bus.arun_stop(fake, _ctx()))
    assert captured["final_text"] == "done"


# ---------- 默认装配 ----------


def test_default_hooks_has_expected_order():
    bus = default_hooks()
    assert bus.pre_count == 4
    assert bus.post_count == 1
    # stop: RateLimitSweeperStopHook + TraceWriterHook
    assert bus.stop_count == 2


# ---------- 端到端 ----------


@tool
def echo_tool(text: str) -> str:
    """echo"""
    return f"echo:{text}"


@tool
def do_terminate(final_answer: str) -> str:
    """terminate"""
    return final_answer


class FakeLLM:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, *args, **kwargs):
        if not self._scripted:
            raise AssertionError("FakeLLM 脚本耗尽")
        return self._scripted.pop(0)

    async def astream(self, messages, *args, **kwargs):
        from langchain_core.messages import AIMessageChunk

        if not self._scripted:
            raise AssertionError("FakeLLM 脚本耗尽")
        msg = self._scripted.pop(0)
        yield AIMessageChunk(
            content=msg.content,
            tool_calls=getattr(msg, "tool_calls", []) or [],
        )


def test_loop_guard_via_default_hooks_end_to_end():
    """同 (name, args) 第二次发起时 LoopGuardHook 拦截，主循环停在 loop_guard。"""
    registry = ToolRegistry()
    registry.register(echo_tool, scope="test")
    registry.register(do_terminate, scope="control")

    same = [{"id": "c1", "name": "echo_tool", "args": {"text": "hi"}}]
    same_again = [{"id": "c2", "name": "echo_tool", "args": {"text": "hi"}}]
    llm = FakeLLM(
        [
            AIMessage(content="第一次", tool_calls=same),
            AIMessage(content="第二次", tool_calls=same_again),
        ]
    )

    bus = HookBus()
    bus.register_pre(LoopGuardHook(threshold=2))

    harness = Harness(llm=llm, registry=registry, hooks=bus, max_iterations=5)
    result = harness.run("问题")

    assert result.stopped_reason == "loop_guard"
    assert "echo_tool" in result.final_text
    # 工具结果里能看到拦截 content
    assert any(r.error == "loop_guard" for t in result.turns for r in t.tool_results)

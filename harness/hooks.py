"""Harness 钩子体系。

定义三类钩子和总线，让横切关注点（限流、权限、循环防御、审计）
以可插拔方式接入主循环，工具体内回归纯执行逻辑。

钩子语义：
- PreToolUseHook：返回 ToolResult 即拦截，返回 None 即放行。
- PostToolUseHook：必须返回 ToolResult，可改写或脱敏。
- OnStopHook：只读回调，用于审计、统计、产物落盘。

同步 / 异步：
- 每类钩子既支持普通 ``def``，也支持 ``async def``。
- HookBus 同时暴露 ``run_*`` 同步与 ``arun_*`` 异步入口。Harness 主循环
  走 ``arun_*``，sync hook 自动用 ``asyncio.to_thread`` 包装放到工作线程，
  避免阻塞 event loop（典型场景：RateLimit 的 semaphore 阻塞 acquire）。
"""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable, Protocol, Union

from .turn import ToolCall, ToolResult

if TYPE_CHECKING:
    from .loop import HarnessResult


@dataclass(frozen=True)
class HookContext:
    """单次工具调用上下文，frozen 防止 hook 间互相串改。

    ``call_counts`` 字段值（dict）本身可变，用于 LoopGuardHook 累计调用次数；
    frozen 锁的是字段绑定而非内部对象。OnStop hook 看到的 ctx 会通过
    ``dataclasses.replace`` 重建出带 ``started_at`` / ``finished_at``
    的新实例。

    ``trace_id`` 由 Harness 在 ``run()`` 开始时生成并贯穿整次调用；
    ``parent_trace_id`` 仅当本 Harness 是子 agent 时由 ``dispatch_subagent``
    工具设置，指向调用方主 Harness 的 trace_id。

    ``chat_id`` 是会话维度的审计标签，由调用方（FastAPI 路由层）传入；
    Harness 主循环不消费它，只透传给 trace 分桶用。
    """

    user_query: str
    turn_index: int
    call_counts: dict[str, int] = field(default_factory=dict)
    started_at: str | None = None
    finished_at: str | None = None
    trace_id: str | None = None
    parent_trace_id: str | None = None
    chat_id: str | None = None


# 三类 hook 都支持 sync 或 async。Protocol 表达起来繁琐，统一用 Callable 别名。
PreReturn = Union[ToolResult, None, Awaitable[Union[ToolResult, None]]]
PostReturn = Union[ToolResult, Awaitable[ToolResult]]
StopReturn = Union[None, Awaitable[None]]

PreToolUseHook = Callable[[ToolCall, HookContext], PreReturn]
PostToolUseHook = Callable[[ToolCall, ToolResult, HookContext], PostReturn]
OnStopHook = Callable[["HarnessResult", HookContext], StopReturn]


async def _maybe_await(value):
    """如果 value 是协程就 await 它，否则原样返回。

    支持 hook 写成 ``def`` 或 ``async def`` 两种形式。
    """
    if inspect.isawaitable(value):
        return await value
    return value


class HookBus:
    """有序 hook 容器，按注册顺序执行。"""

    def __init__(self) -> None:
        self._pre: list[PreToolUseHook] = []
        self._post: list[PostToolUseHook] = []
        self._stop: list[OnStopHook] = []

    def register_pre(self, hook: PreToolUseHook) -> None:
        self._pre.append(hook)

    def register_post(self, hook: PostToolUseHook) -> None:
        self._post.append(hook)

    def register_stop(self, hook: OnStopHook) -> None:
        self._stop.append(hook)

    # ---------------- 同步入口（向后兼容） ----------------

    def run_pre(self, call: ToolCall, ctx: HookContext) -> ToolResult | None:
        """按顺序执行 pre hooks，首个返回非 None 即短路。

        ⚠️ 只能用于 sync hook；async hook 在这里会返回一个 coroutine 对象，
        被当作非 None 短路掉。生产路径请走 ``arun_pre``。
        """
        for hook in self._pre:
            outcome = hook(call, ctx)
            if outcome is not None:
                return outcome
        return None

    def run_post(
        self, call: ToolCall, result: ToolResult, ctx: HookContext
    ) -> ToolResult:
        """链式执行 post hooks，每个 hook 看到的是前一个的输出。

        ⚠️ 同 ``run_pre``，只能用于 sync hook。
        """
        current = result
        for hook in self._post:
            current = hook(call, current, ctx)
        return current

    def run_stop(self, result: "HarnessResult", ctx: HookContext) -> None:
        for hook in self._stop:
            hook(result, ctx)

    # ---------------- 异步入口（Harness 主循环用） ----------------

    async def arun_pre(self, call: ToolCall, ctx: HookContext) -> ToolResult | None:
        """异步执行 pre hooks。

        - async hook：直接 await。
        - sync hook：用 ``asyncio.to_thread`` 派到工作线程，避免 RateLimit
          的 blocking semaphore acquire 冻结 event loop。
        """
        for hook in self._pre:
            if inspect.iscoroutinefunction(hook) or inspect.iscoroutinefunction(
                getattr(hook, "__call__", None)
            ):
                outcome = await _maybe_await(hook(call, ctx))
            else:
                outcome = await asyncio.to_thread(hook, call, ctx)
                # sync hook 也可能返回 awaitable（比如 functools.partial 包了 async）
                outcome = await _maybe_await(outcome)
            if outcome is not None:
                return outcome
        return None

    async def arun_post(
        self, call: ToolCall, result: ToolResult, ctx: HookContext
    ) -> ToolResult:
        current = result
        for hook in self._post:
            if inspect.iscoroutinefunction(hook) or inspect.iscoroutinefunction(
                getattr(hook, "__call__", None)
            ):
                current = await _maybe_await(hook(call, current, ctx))
            else:
                current = await asyncio.to_thread(hook, call, current, ctx)
                current = await _maybe_await(current)
        return current

    async def arun_stop(self, result: "HarnessResult", ctx: HookContext) -> None:
        for hook in self._stop:
            if inspect.iscoroutinefunction(hook) or inspect.iscoroutinefunction(
                getattr(hook, "__call__", None)
            ):
                await _maybe_await(hook(result, ctx))
            else:
                await asyncio.to_thread(hook, result, ctx)

    # ----- 给测试用的内省接口 -----

    @property
    def pre_count(self) -> int:
        return len(self._pre)

    @property
    def post_count(self) -> int:
        return len(self._post)

    @property
    def stop_count(self) -> int:
        return len(self._stop)


__all__ = [
    "HookContext",
    "HookBus",
    "PreToolUseHook",
    "PostToolUseHook",
    "OnStopHook",
]

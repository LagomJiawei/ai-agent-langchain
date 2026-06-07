"""内置 hook 集合：限流、终端白名单、文件路径白名单、循环防御、trace 写盘。

把工具体内的横切逻辑收归到此处。``default_hooks()`` 返回生产配置下的
默认 hook 总线，供 ``Harness`` 默认装配。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from loguru import logger

from config.settings import settings
from tools.file_ops import FileSecurity
from tools.rate_limiter import RateLimiter, get_rate_limiter
from tools.terminal import TerminalSecurity

from .hooks import HookBus, HookContext
from .trace import HarnessTrace
from .turn import ToolCall, ToolResult

if TYPE_CHECKING:
    from .loop import HarnessResult


# ---------------------------------------------------------------------------
# 限流：Pre 占令牌，Post 反馈成败给熔断器
# ---------------------------------------------------------------------------


class RateLimitPreHook:
    """占令牌、查熔断；不真实执行工具。

    线程安全 + 同 call.id 重发幂等：
    - ``_inflight`` 的 add / contains / discard 全部走 ``_inflight_lock``，
      避免并发场景下 set 的非原子组合（add 后 release 漏配对）导致 semaphore 泄漏。
    - 同一 ``call.id`` 第二次被 Pre 看到时直接放行，不重复占 token / 并发 / 熔断查询
      （LLM 流式重发、hook 链异常重试等场景下避免双占）。
    - 配套 ``RateLimitSweeperStopHook``：在主循环异常吞掉 Post 时兜底归还残留信号量。
    """

    def __init__(self, limiter: RateLimiter | None = None) -> None:
        self._limiter = limiter
        # call.id -> tool.name 的映射：Post 凭 id 找回工具名（也可避免重发误判）
        self._inflight: dict[str, str] = {}
        self._inflight_lock = threading.Lock()

    def _resolve_limiter(self) -> RateLimiter:
        if self._limiter is not None:
            return self._limiter
        return get_rate_limiter()

    def __call__(self, call: ToolCall, ctx: HookContext) -> ToolResult | None:
        if not settings.tool_rate_limit.enabled:
            return None

        # 幂等：同 call.id 已被占过，直接放行不再占资源
        with self._inflight_lock:
            if call.id and call.id in self._inflight:
                logger.warning(
                    f"RateLimitPreHook 收到同 call.id={call.id} 的重发，跳过重复占用"
                )
                return None

        limiter = self._resolve_limiter()

        # 占令牌
        if not limiter.token_bucket.acquire(block=False):
            logger.warning(f"工具调用被 QPS 限流: {call.name}")
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content=f"调用被限流: {call.name} 超出 QPS 上限",
                error="rate_limited",
            )

        # 占并发（阻塞拿；这一层不应该长时间阻塞，因为 semaphore 数量等于配置）
        if not limiter.semaphore.acquire(block=True):
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content=f"调用并发超限: {call.name}",
                error="concurrency_limited",
            )

        # 查熔断
        breaker = limiter._get_circuit_breaker(call.name)
        if breaker.state == breaker.STATE_OPEN:
            # 给 semaphore 还回去，因为我们不会执行
            limiter.semaphore.release()
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content=f"工具 {call.name} 熔断中，请稍后再试",
                error="circuit_open",
            )

        with self._inflight_lock:
            self._inflight[call.id] = call.name
        return None

    def _release_one(self, call_id: str) -> str | None:
        """原子地从 inflight 移除并返回原 tool 名；不存在返回 None。"""
        with self._inflight_lock:
            return self._inflight.pop(call_id, None)

    def _drain_inflight(self) -> dict[str, str]:
        """原子地清空并返回所有残留 (call_id -> name)，用于 OnStop 兜底。"""
        with self._inflight_lock:
            snapshot = dict(self._inflight)
            self._inflight.clear()
            return snapshot


class RateLimitPostHook:
    """反馈成败给熔断器并归还信号量。需与 RateLimitPreHook 成对使用。"""

    def __init__(self, pre: RateLimitPreHook) -> None:
        self._pre = pre

    def __call__(
        self, call: ToolCall, result: ToolResult, ctx: HookContext
    ) -> ToolResult:
        original_name = self._pre._release_one(call.id)
        if original_name is None:
            # 没被 Pre 占用（被前面的 hook 拦截 / 重发幂等跳过等），直接透传
            return result

        limiter = self._pre._resolve_limiter()
        breaker = limiter._get_circuit_breaker(original_name)
        try:
            if result.ok:
                breaker._on_success()
            else:
                breaker._on_failure()
        finally:
            limiter.semaphore.release()
        return result


class RateLimitSweeperStopHook:
    """OnStop 兜底：归还所有未配对释放的 semaphore，防止异常路径泄漏。

    正常路径下 Post hook 已经在 ``_release_one`` 后 release；这里只清扫
    残留（例如主循环异常吞掉 Post、hook 链中途抛异常等）。
    """

    def __init__(self, pre: RateLimitPreHook) -> None:
        self._pre = pre

    def __call__(self, result: "HarnessResult", ctx: HookContext) -> None:
        leftovers = self._pre._drain_inflight()
        if not leftovers:
            return
        limiter = self._pre._resolve_limiter()
        for call_id, name in leftovers.items():
            try:
                limiter.semaphore.release()
                logger.warning(
                    f"RateLimitSweeperStopHook 归还泄漏的 semaphore: "
                    f"call_id={call_id} tool={name}"
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"sweeper 归还 semaphore 失败: {exc}")


# ---------------------------------------------------------------------------
# 权限：终端命令白名单 / 文件路径白名单
# ---------------------------------------------------------------------------


class TerminalAllowlistHook:
    """拦截不在白名单内的 terminal_exec 调用。"""

    def __call__(self, call: ToolCall, ctx: HookContext) -> ToolResult | None:
        if call.name != "terminal_exec":
            return None
        command = call.args.get("command", "")
        is_safe, reason = TerminalSecurity.is_safe_command(command)
        if is_safe:
            return None
        logger.warning(f"终端命令被白名单拦截: {command} ({reason})")
        return ToolResult(
            call_id=call.id,
            name=call.name,
            ok=False,
            content=f"安全检查失败: {reason}",
            error="terminal_allowlist_denied",
        )


_FILE_TOOL_PATH_ARG = {
    "file_read": "path",
    "file_write": "path",
    "list_files": "directory",
}


class FilePathAllowlistHook:
    """拦截越界的文件读写。"""

    def __call__(self, call: ToolCall, ctx: HookContext) -> ToolResult | None:
        arg_name = _FILE_TOOL_PATH_ARG.get(call.name)
        if arg_name is None:
            return None

        raw_path = call.args.get(arg_name, "")
        # list_files 是目录粒度，复用 is_safe_path 时需要加哨兵
        check_path = raw_path if call.name != "list_files" else f"{raw_path}/dummy"
        if FileSecurity.is_safe_path(check_path):
            return None

        logger.warning(f"文件路径被白名单拦截: {call.name}({arg_name}={raw_path})")
        return ToolResult(
            call_id=call.id,
            name=call.name,
            ok=False,
            content=(
                f"不允许访问路径 {raw_path}，请使用允许的目录: {FileSecurity.ALLOWED_DIRS}"
            ),
            error="file_path_denied",
        )


# ---------------------------------------------------------------------------
# 循环防御
# ---------------------------------------------------------------------------


class LoopGuardHook:
    """同 (name, args) 累计达到阈值即拒绝并请求停止主循环。"""

    def __init__(self, threshold: int = 2) -> None:
        self.threshold = threshold

    def __call__(self, call: ToolCall, ctx: HookContext) -> ToolResult | None:
        args_key = json.dumps(call.args or {}, sort_keys=True, ensure_ascii=False)
        key = f"{call.name}:{args_key}"
        # ctx.call_counts 是 frozen dataclass 里的字典字段 —— 字典本身可变
        ctx.call_counts[key] = ctx.call_counts.get(key, 0) + 1
        if ctx.call_counts[key] < self.threshold:
            return None

        logger.warning(f"循环防御拦截: {call.name}")
        return ToolResult(
            call_id=call.id,
            name=call.name,
            ok=False,
            content=(
                f"检测到工具 '{call.name}' 被重复调用，已停止继续调用以避免循环。"
                "请基于已有信息重新提问，或换一种更具体的问法。"
            ),
            error="loop_guard",
            stop_loop=True,
            stop_reason="loop_guard",
        )


# ---------------------------------------------------------------------------
# OnStop: trace 写盘
# ---------------------------------------------------------------------------


class TraceWriterHook:
    """把每次 Harness.run 的完整 trace 序列化为 JSON 落盘。

    ``settings.agent.trace_enabled=False`` 时整体跳过。
    任何 I/O 异常吞掉只记 warning，绝不影响主循环。
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._explicit_base_dir = base_dir

    def _resolve_base_dir(self) -> Path:
        if self._explicit_base_dir is not None:
            return Path(self._explicit_base_dir)
        return Path(settings.agent.trace_dir)

    def __call__(self, result: "HarnessResult", ctx: HookContext) -> None:
        if not settings.agent.trace_enabled:
            return
        try:
            # ctx.trace_id 由 Harness.run() 注入；老调用者若没传 trace_id 走兜底
            from .trace import new_trace_id

            trace = HarnessTrace.from_harness_result(
                result=result,
                user_query=ctx.user_query,
                started_at=ctx.started_at or "",
                finished_at=ctx.finished_at or "",
                trace_id=ctx.trace_id or new_trace_id(),
                parent_trace_id=ctx.parent_trace_id,
                chat_id=ctx.chat_id,
            )
            path = trace.write(self._resolve_base_dir())
            logger.info(f"Harness trace 已写入 {path}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Harness trace 写入失败（已忽略）: {exc}")


# ---------------------------------------------------------------------------
# 默认装配
# ---------------------------------------------------------------------------


def default_hooks(
    pre_extras: Iterable | None = None,
    post_extras: Iterable | None = None,
    stop_extras: Iterable | None = None,
) -> HookBus:
    """生产配置下的默认 hook 总线。"""
    bus = HookBus()

    rate_pre = RateLimitPreHook()
    bus.register_pre(rate_pre)
    bus.register_pre(TerminalAllowlistHook())
    bus.register_pre(FilePathAllowlistHook())
    bus.register_pre(LoopGuardHook())

    bus.register_post(RateLimitPostHook(rate_pre))

    # Sweeper 必须先于 TraceWriterHook 注册：异常路径下先归还泄漏的 semaphore，
    # 再写 trace；OnStop 按注册顺序执行，TraceWriter 慢一拍不影响 sweeper。
    bus.register_stop(RateLimitSweeperStopHook(rate_pre))
    bus.register_stop(TraceWriterHook())

    for extra in pre_extras or ():
        bus.register_pre(extra)
    for extra in post_extras or ():
        bus.register_post(extra)
    for extra in stop_extras or ():
        bus.register_stop(extra)

    return bus


__all__ = [
    "RateLimitPreHook",
    "RateLimitPostHook",
    "TerminalAllowlistHook",
    "FilePathAllowlistHook",
    "LoopGuardHook",
    "TraceWriterHook",
    "default_hooks",
]

"""Harness 主循环。

单一 think -> act -> observe 循环，替代旧 agents/ 目录里 ReAct + Plan-Execute
两套实现。规划、子 agent 调度等高阶能力以"工具"形式注入（见 harness/subagent.py）。

横切关注点（限流、权限、循环防御、审计）通过 ``harness.hooks.HookBus``
挂载，不再硬编码在 loop 内或工具体内。

主循环原生异步：``astream()`` 是头等接口，同步 ``run()`` 内部委托给
``astream()``（``asyncio.run``），保证两条路径行为完全一致。
"""
from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.tools import BaseTool
from loguru import logger

from config import create_chat_model, settings

from . import events as _events
from ._message_utils import extract_chunk_text, extract_message_text
from .context import ConversationContext
from .events import HarnessEvent
from .hooks import HookBus, HookContext
from .registry import ToolRegistry, default_registry
from .trace import new_trace_id
from .turn import ToolCall, ToolResult, Turn

StoppedReason = Literal["final_text", "terminate_tool", "loop_guard", "max_iterations", "error"]

_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"

# 暴露给 dispatch_subagent 工具：让子 Harness 知道父 trace_id
_current_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "harness_current_trace_id", default=None
)
# 暴露给 dispatch_subagent 工具：让子 Harness 继承父 chat_id
_current_chat_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "harness_current_chat_id", default=None
)


def current_trace_id() -> str | None:
    """返回当前运行栈最内层 Harness 的 trace_id；不在 run 上下文中返回 None。"""
    return _current_trace_id_var.get()


def current_chat_id() -> str | None:
    """返回当前运行栈最内层 Harness 的 chat_id；不在 run 上下文中返回 None。"""
    return _current_chat_id_var.get()


@lru_cache(maxsize=1)
def _load_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


# llm 实例级缓存属性名：把 cache 挂在 llm 实例上而不是模块级 dict，
# 好处：(1) llm 被 GC 回收时缓存跟着回收，零泄漏；
#       (2) 不同 llm 实例的缓存天然隔离，零跨实例污染；
#       (3) 测试桩 llm 通常不可 setattr 或不可 bind_tools，自动绕开缓存。
_BIND_CACHE_ATTR = "_harness_bind_cache"
_BIND_CACHE_LOCK_ATTR = "_harness_bind_cache_lock"


def _with_bind_tools_cached(llm, tool_list: list[BaseTool]):
    """对 ``llm.bind_tools(tool_list)`` 加进程级缓存。

    缓存粒度按 ``frozenset(tool.name for tool in tool_list)``；
    相同 llm 实例 + 相同 tool 名集合 → 返回上次 bind 出来的 Runnable。

    无 ``bind_tools`` 或工具列表为空时不走缓存，按原行为返回 llm。
    若 llm 不允许 ``setattr``（某些 frozen 测试桩），同样退化为每次重 bind。
    """
    if not tool_list:
        return llm
    if not hasattr(llm, "bind_tools"):
        return llm

    tool_names = frozenset(t.name for t in tool_list)

    try:
        cache = getattr(llm, _BIND_CACHE_ATTR, None)
        if cache is None:
            lock = threading.Lock()
            cache = {}
            # 先挂 lock 再挂 cache，避免另一线程看到 cache 但拿不到 lock
            setattr(llm, _BIND_CACHE_LOCK_ATTR, lock)
            setattr(llm, _BIND_CACHE_ATTR, cache)
        else:
            lock = getattr(llm, _BIND_CACHE_LOCK_ATTR)
    except (AttributeError, TypeError):
        # llm 不允许 setattr（frozen pydantic / 某些桩）→ 退化
        return llm.bind_tools(tool_list)

    with lock:
        bound = cache.get(tool_names)
        if bound is None:
            bound = llm.bind_tools(tool_list)
            cache[tool_names] = bound
        return bound


@dataclass
class HarnessResult:
    """一次 Harness.run() 的完整产出。"""

    final_text: str
    turns: list[Turn] = field(default_factory=list)
    stopped_reason: StoppedReason = "final_text"

    @property
    def total_tool_calls(self) -> int:
        return sum(len(t.tool_calls) for t in self.turns)


class Harness:
    """LiCaiManus Agent 主循环。"""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        registry: ToolRegistry | None = None,
        max_iterations: int | None = None,
        hooks: HookBus | None = None,
        parent_trace_id: str | None = None,
        chat_id: str | None = None,
    ) -> None:
        self.llm = llm if llm is not None else create_chat_model(temperature=0.7)
        self.registry = registry if registry is not None else default_registry
        self.max_iterations = (
            max_iterations
            if max_iterations is not None
            else settings.agent.max_iterations
        )
        if hooks is None:
            # 延迟导入，避免 harness 启动期循环依赖（builtin_hooks 反向 import tools）
            from .builtin_hooks import default_hooks

            hooks = default_hooks()
        self.hooks = hooks
        self.parent_trace_id = parent_trace_id
        self.chat_id = chat_id

        tools = self.registry.list()
        # llm 可能是 fake 测试桩，没有 bind_tools；helper 内部 hasattr 检查后兜底返回 llm。
        # 相同 llm + 相同 tool 名集合 → 复用上次 bind 出来的 Runnable，避免每次构造重做 ~25ms 的 bind。
        self.llm_with_tools = _with_bind_tools_cached(self.llm, tools)

    # ---------- 公开入口 ----------

    def run(self, user_query: str) -> HarnessResult:
        """同步入口：内部委托给 arun()，保证单一事实源。"""
        return asyncio.run(self.arun(user_query))

    async def arun(self, user_query: str) -> HarnessResult:
        """异步入口：组装 HarnessResult 而不暴露事件流。

        外部如果需要事件流，直接调 ``astream()``；如果需要异步等结果，
        直接 ``await sub.arun(...)``（dispatch_subagents 工具就是这么用的）。
        """
        holder: dict[str, HarnessResult] = {}
        async for event in self.astream(user_query, _result_holder=holder):
            # 仅用于推进迭代器；run_end 事件触发时 holder["result"] 已填好
            pass
        return holder["result"]

    async def astream(
        self,
        user_query: str,
        *,
        _result_holder: dict | None = None,
    ) -> AsyncIterator[HarnessEvent]:
        """主异步迭代器：发出 run_start / thinking_token / tool_call /
        tool_result / final_text / run_end / error 事件。

        ``_result_holder`` 是给 ``run()`` 同步路径回填 ``HarnessResult`` 的内部钩子，
        外部调用方不应该传它。
        """
        logger.info(f"Harness 开始执行: {user_query[:50]}...")
        started_at = datetime.now(timezone.utc).isoformat()
        trace_id = new_trace_id()
        token = _current_trace_id_var.set(trace_id)
        chat_token = _current_chat_id_var.set(self.chat_id)

        yield _events.run_start(
            trace_id=trace_id,
            parent_trace_id=self.parent_trace_id,
            user_query=user_query,
            started_at=started_at,
        )

        try:
            context = ConversationContext(
                system_prompt=_load_system_prompt(),
                initial_user_message=user_query,
            )
            turns: list[Turn] = []
            call_counts: dict[str, int] = {}

            result: HarnessResult
            async for evt_or_done in self._iterate_loop(
                user_query, context, turns, call_counts, trace_id
            ):
                if isinstance(evt_or_done, HarnessEvent):
                    yield evt_or_done
                else:
                    result = evt_or_done  # 内部 sentinel：HarnessResult
                    break

            # 走完没出 result 应该不会发生；兜底
            if "result" not in locals():
                result = HarnessResult(
                    final_text="达到最大迭代次数，任务终止。",
                    turns=turns,
                    stopped_reason="max_iterations",
                )

            finished_at = datetime.now(timezone.utc).isoformat()
            base_ctx = HookContext(
                user_query=user_query,
                turn_index=len(turns),
                call_counts=call_counts,
                trace_id=trace_id,
                parent_trace_id=self.parent_trace_id,
                chat_id=self.chat_id,
            )
            final_ctx = dataclasses.replace(
                base_ctx, started_at=started_at, finished_at=finished_at
            )
            # OnStop 在 run_end 之前调用，确保 trace 文件已落地
            await self.hooks.arun_stop(result, final_ctx)

            if _result_holder is not None:
                _result_holder["result"] = result

            yield _events.final_text(result.final_text)
            yield _events.run_end(
                stopped_reason=result.stopped_reason,
                total_tool_calls=result.total_tool_calls,
                finished_at=finished_at,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Harness astream 异常: {exc}")
            # 异常路径同样要：1) 发 error 事件 2) 触发 OnStop 写 trace
            # 3) 发 run_end 让 SSE 客户端能确定结束。
            # 不发 final_text 事件——避免客户端把"执行异常: xxx"当真答案。
            yield _events.error(str(exc))

            finished_at = datetime.now(timezone.utc).isoformat()
            # 异常时 turns / call_counts 可能未初始化（_iterate_loop 之前就炸），兜底空值
            error_result = HarnessResult(
                final_text=f"执行异常: {exc}",
                turns=locals().get("turns", []),
                stopped_reason="error",
            )
            error_ctx = HookContext(
                user_query=user_query,
                turn_index=len(error_result.turns),
                call_counts=locals().get("call_counts", {}),
                started_at=started_at,
                finished_at=finished_at,
                trace_id=trace_id,
                parent_trace_id=self.parent_trace_id,
                chat_id=self.chat_id,
            )
            try:
                await self.hooks.arun_stop(error_result, error_ctx)
            except Exception as hook_exc:  # noqa: BLE001
                # OnStop 自己再炸也不能影响 run_end 发射
                logger.warning(f"OnStop 钩子在异常路径下再次失败（已忽略）: {hook_exc}")

            if _result_holder is not None and "result" not in _result_holder:
                _result_holder["result"] = error_result

            yield _events.run_end(
                stopped_reason="error",
                total_tool_calls=error_result.total_tool_calls,
                finished_at=finished_at,
            )
        finally:
            _current_trace_id_var.reset(token)
            _current_chat_id_var.reset(chat_token)

    # ---------- 内部 ----------

    async def _iterate_loop(
        self,
        user_query: str,
        context: ConversationContext,
        turns: list[Turn],
        call_counts: dict[str, int],
        trace_id: str,
    ):
        """yield 事件序列；遇到 stop 条件时 yield 一个 HarnessResult sentinel。"""
        for turn_index in range(self.max_iterations):
            # ---- LLM astream，累加 chunk 拿完整 tool_calls ----
            accumulated: AIMessageChunk | None = None
            async for chunk in self.llm_with_tools.astream(context.snapshot()):
                accumulated = chunk if accumulated is None else accumulated + chunk
                text = extract_chunk_text(chunk)
                if text:
                    yield _events.thinking_token(turn_index, text)

            response = accumulated  # AIMessageChunk 是 AIMessage 子类，可直接当 AIMessage 用
            turn = Turn(index=turn_index, thought=extract_message_text(response) if response else "")
            if response is not None:
                context.append(response)

            raw_tool_calls = getattr(response, "tool_calls", None) or []

            if not raw_tool_calls:
                turn.final_text = turn.thought
                turns.append(turn)
                logger.info(f"Harness 完成: final_text (轮次 {turn_index + 1})")
                yield HarnessResult(
                    final_text=turn.thought,
                    turns=turns,
                    stopped_reason="final_text",
                )
                return

            ctx = HookContext(
                user_query=user_query,
                turn_index=turn_index,
                call_counts=call_counts,
                trace_id=trace_id,
                parent_trace_id=self.parent_trace_id,
                chat_id=self.chat_id,
            )

            stop_after_turn: tuple[StoppedReason, str] | None = None
            terminate_payload: str | None = None

            for raw in raw_tool_calls:
                call = ToolCall(
                    id=raw.get("id", ""),
                    name=raw.get("name", ""),
                    args=raw.get("args", {}) or {},
                )
                turn.tool_calls.append(call)
                yield _events.tool_call(turn_index, call)

                intercepted = await self.hooks.arun_pre(call, ctx)
                if intercepted is not None:
                    final_result = await self.hooks.arun_post(call, intercepted, ctx)
                else:
                    exec_result = await self._ainvoke_tool(call)
                    final_result = await self.hooks.arun_post(call, exec_result, ctx)

                turn.tool_results.append(final_result)
                context.append(
                    ToolMessage(content=final_result.content, tool_call_id=call.id)
                )
                yield _events.tool_result(turn_index, final_result)

                if final_result.stop_loop:
                    reason: StoppedReason = (
                        final_result.stop_reason  # type: ignore[assignment]
                        if final_result.stop_reason in ("loop_guard",)
                        else "loop_guard"
                    )
                    stop_after_turn = (reason, final_result.content)
                    break

                if call.name == "do_terminate":
                    terminate_payload = call.args.get("final_answer", "") or ""

            turns.append(turn)

            if stop_after_turn is not None:
                reason, final_text_value = stop_after_turn
                logger.warning(f"Harness 触发 hook 停止: {reason}")
                yield HarnessResult(
                    final_text=final_text_value,
                    turns=turns,
                    stopped_reason=reason,
                )
                return

            if terminate_payload is not None:
                logger.info("Harness 完成: terminate_tool")
                yield HarnessResult(
                    final_text=terminate_payload,
                    turns=turns,
                    stopped_reason="terminate_tool",
                )
                return

        logger.warning(f"Harness 达到 max_iterations={self.max_iterations}")
        yield HarnessResult(
            final_text="达到最大迭代次数，任务终止。",
            turns=turns,
            stopped_reason="max_iterations",
        )

    async def _ainvoke_tool(self, call: ToolCall) -> ToolResult:
        """异步执行单个工具调用。

        统一走 ``tool.ainvoke``：sync 工具会被 LangChain 自动用 ``to_thread`` 包成 async；
        async 工具（如 ``dispatch_subagents``）原生 await，避免 ``asyncio.run`` 嵌套陷阱。
        """
        start = time.perf_counter()
        try:
            tool = self.registry.get(call.name)
        except KeyError as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"未知工具: {call.name}")
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content=f"未知工具: {call.name}",
                error=str(exc),
                elapsed_ms=elapsed,
            )

        try:
            output = await tool.ainvoke(call.args)
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=True,
                content=str(output),
                elapsed_ms=elapsed,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"工具 {call.name} 执行失败: {exc}")
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                content=f"工具执行失败: {exc}",
                error=str(exc),
                elapsed_ms=elapsed,
            )


__all__ = ["Harness", "HarnessResult", "StoppedReason", "current_trace_id", "current_chat_id"]

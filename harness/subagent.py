"""dispatch_subagent / dispatch_subagents 工具：派发子任务到隔离子 Harness。

设计要点：
- 子 Harness 用受限 registry（按 ``allowed_scopes`` 并集过滤）。
- ``dispatch_subagent`` 与 ``dispatch_subagents`` 自身永远不进子 registry，防递归 / 嵌套并发爆栈。
- ``do_terminate`` 强制注入到子 registry，保证子 agent 有显式收尾能力。
- 主 Harness 的 trace_id / chat_id 通过 ``loop.current_trace_id()`` / ``current_chat_id()``
  读取并作为子 Harness 的 ``parent_trace_id`` / ``chat_id``，让事后能从
  ``./traces/<chat_id>/`` 拼调用树。
- 子任务执行失败不上抛，返回字符串说明，主 agent 自行决定下一步。
- 并发执行（``dispatch_subagents``）用 ``asyncio.gather`` + ``Semaphore``，
  不引入线程池；并发上限默认 4，硬上限 16。
"""
from __future__ import annotations

import asyncio
import json

from langchain_core.tools import tool
from loguru import logger

from .loop import Harness, current_chat_id, current_trace_id
from .registry import ToolRegistry, default_registry

_SELF_NAMES = {"dispatch_subagent", "dispatch_subagents"}
_MAX_CONCURRENCY_CAP = 16
_MAX_TASKS = 16


def _build_sub_registry(allowed_scopes: list[str]) -> ToolRegistry:
    """根据 scope 白名单构造隔离 registry。

    - 排除 ``dispatch_subagent`` / ``dispatch_subagents``（防递归 / 嵌套并发）。
    - 强制注入 ``do_terminate``，即便 ``allowed_scopes`` 不含 ``control``。
    """
    sub = ToolRegistry()
    for tool_obj in default_registry.list(scopes=allowed_scopes):
        if tool_obj.name in _SELF_NAMES:
            continue
        sub.register(tool_obj, scope=default_registry.scope_of(tool_obj.name))

    if "do_terminate" not in sub.names() and "do_terminate" in default_registry.names():
        sub.register(default_registry.get("do_terminate"), scope="control")
    return sub


@tool
async def dispatch_subagent(
    task: str,
    allowed_scopes: list[str],
    max_iterations: int = 6,
) -> str:
    """把单个独立子任务交给一个隔离的子 Harness 执行，只回传最终答案。

    Args:
        task: 子任务的完整中文描述
        allowed_scopes: 子 agent 可访问的工具 scope 列表，例如 ["kb"] 只检索知识库
        max_iterations: 子 Harness 最多轮次，默认 6（小于主 agent 默认 10）

    Returns:
        子 agent 的 final_text；失败时返回友好错误说明
    """
    parent_id = current_trace_id()
    parent_chat_id = current_chat_id()
    logger.info(
        f"dispatch_subagent: scopes={allowed_scopes} parent_trace_id={parent_id} "
        f"chat_id={parent_chat_id} task={task[:60]}..."
    )
    try:
        sub_registry = _build_sub_registry(allowed_scopes)
        sub_harness = Harness(
            registry=sub_registry,
            max_iterations=max_iterations,
            parent_trace_id=parent_id,
            chat_id=parent_chat_id,
        )
        sub_result = await sub_harness.arun(task)
        return sub_result.final_text or "(子 agent 未给出答案)"
    except Exception as exc:  # noqa: BLE001
        logger.error(f"dispatch_subagent 异常: {exc}")
        return f"子 agent 执行失败: {exc}"


def _validate_tasks(tasks) -> str | None:
    """校验 tasks 列表；不通过返回 JSON 错误字符串，通过返回 None。"""
    if not isinstance(tasks, list) or not tasks:
        return json.dumps({"error": "tasks 必须是非空列表"}, ensure_ascii=False)
    if len(tasks) > _MAX_TASKS:
        return json.dumps(
            {"error": f"tasks 数量上限 {_MAX_TASKS}，当前 {len(tasks)}"},
            ensure_ascii=False,
        )
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            return json.dumps(
                {"error": f"tasks[{i}] 必须是 dict"}, ensure_ascii=False
            )
        if "task" not in t or "allowed_scopes" not in t:
            return json.dumps(
                {"error": f"tasks[{i}] 必须含 task 与 allowed_scopes"},
                ensure_ascii=False,
            )
    return None


@tool
async def dispatch_subagents(
    tasks: list[dict],
    max_concurrency: int = 4,
) -> str:
    """并发派发多个独立子任务到隔离子 Harness 执行，收集所有 final_text。

    Args:
        tasks: 子任务列表。每个元素是 dict：
            {
                "task": "子任务的完整中文描述",
                "allowed_scopes": ["kb"],   # 必填
                "max_iterations": 6,         # 可选，默认 6
            }
        max_concurrency: 同时执行的子 agent 数上限，默认 4，硬上限 16

    Returns:
        JSON 字符串数组，每项含 ``index`` / ``task`` / ``final_text`` 或 ``error``，
        主 agent 拿来综合后再回答用户。
    """
    err = _validate_tasks(tasks)
    if err is not None:
        return err

    effective = max(1, min(int(max_concurrency or 4), _MAX_CONCURRENCY_CAP))
    parent_trace = current_trace_id()
    parent_chat = current_chat_id()
    logger.info(
        f"dispatch_subagents: n={len(tasks)} concurrency={effective} "
        f"parent_trace_id={parent_trace} chat_id={parent_chat}"
    )

    sem = asyncio.Semaphore(effective)

    async def _one(idx: int, t: dict) -> dict:
        async with sem:
            try:
                sub_registry = _build_sub_registry(list(t["allowed_scopes"]))
                sub_harness = Harness(
                    registry=sub_registry,
                    max_iterations=int(t.get("max_iterations", 6)),
                    parent_trace_id=parent_trace,
                    chat_id=parent_chat,
                )
                result = await sub_harness.arun(t["task"])
                return {
                    "index": idx,
                    "task": t["task"],
                    "final_text": result.final_text or "(子 agent 未给出答案)",
                }
            except Exception as exc:  # noqa: BLE001
                logger.error(f"dispatch_subagents[{idx}] 异常: {exc}")
                return {
                    "index": idx,
                    "task": t["task"],
                    "error": str(exc),
                }

    try:
        results = await asyncio.gather(*[_one(i, t) for i, t in enumerate(tasks)])
    except Exception as exc:  # noqa: BLE001
        logger.error(f"dispatch_subagents 整体调度异常: {exc}")
        return json.dumps({"error": f"整体调度异常: {exc}"}, ensure_ascii=False)

    return json.dumps(results, ensure_ascii=False, indent=2)


__all__ = ["dispatch_subagent", "dispatch_subagents"]

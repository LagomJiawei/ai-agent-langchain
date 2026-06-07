"""dispatch_subagent 工具与 _build_sub_registry 行为测试。"""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import tools  # noqa: F401  触发 default_registry 注册
from harness import (
    Harness,
    HarnessResult,
    HookBus,
    ToolRegistry,
    current_trace_id,
    default_registry,
    dispatch_subagent,
)
from harness.loop import _current_trace_id_var
from harness.subagent import _build_sub_registry


def _ainvoke(tool_obj, args: dict) -> str:
    """async 工具的同步执行 helper：测试场景从同步上下文调 ainvoke。"""
    return asyncio.run(tool_obj.ainvoke(args))


# ---------- _build_sub_registry ----------


def test_sub_registry_filters_by_allowed_scopes():
    sub = _build_sub_registry(["kb"])
    assert "search_knowledge_base" in sub.names()
    # web/fs/system 范围工具不在
    assert "search_web" not in sub.names()
    assert "file_read" not in sub.names()
    assert "terminal_exec" not in sub.names()


def test_sub_registry_always_injects_do_terminate():
    sub = _build_sub_registry(["kb"])
    assert "do_terminate" in sub.names()


def test_sub_registry_never_includes_dispatch_subagent_itself():
    """即使 allowed_scopes=['control']，dispatch_subagent 也被排除以防递归。"""
    sub = _build_sub_registry(["control"])
    assert "dispatch_subagent" not in sub.names()
    # 但 do_terminate（也在 control）仍然在
    assert "do_terminate" in sub.names()


def test_sub_registry_never_includes_dispatch_subagents():
    """dispatch_subagents 也必须被排除，防嵌套并发爆栈。"""
    sub = _build_sub_registry(["control"])
    assert "dispatch_subagents" not in sub.names()


def test_sub_registry_empty_scope_still_has_do_terminate():
    sub = _build_sub_registry([])
    # 即便没有任何 scope，强制注入 do_terminate 兜底
    assert sub.names() == ["do_terminate"]


# ---------- dispatch_subagent 工具体 ----------


def _fake_result(text: str = "subagent 答案") -> HarnessResult:
    return HarnessResult(final_text=text, turns=[], stopped_reason="final_text")


def test_dispatch_returns_subagent_final_text(monkeypatch):
    captured = {}

    async def fake_arun(self, query):
        captured["query"] = query
        captured["parent_trace_id"] = self.parent_trace_id
        return _fake_result("hello-from-sub")

    monkeypatch.setattr(Harness, "arun", fake_arun)

    out = _ainvoke(dispatch_subagent,
        {"task": "子任务", "allowed_scopes": ["kb"], "max_iterations": 3}
    )
    assert out == "hello-from-sub"
    assert captured["query"] == "子任务"


def test_dispatch_returns_placeholder_when_subagent_empty(monkeypatch):
    async def fake_arun(self, q):
        return _fake_result("")
    monkeypatch.setattr(Harness, "arun", fake_arun)
    out = _ainvoke(dispatch_subagent, {"task": "x", "allowed_scopes": ["kb"]})
    assert "未给出答案" in out


def test_dispatch_swallows_subagent_exception(monkeypatch):
    async def boom(self, q):
        raise RuntimeError("子任务炸了")

    monkeypatch.setattr(Harness, "arun", boom)

    out = _ainvoke(dispatch_subagent, {"task": "x", "allowed_scopes": ["kb"]})
    assert "子 agent 执行失败" in out
    assert "子任务炸了" in out


# ---------- parent_trace_id 通过 contextvar 传递 ----------


def test_parent_trace_id_picked_from_contextvar(monkeypatch):
    captured = {}

    async def fake_arun(self, q):
        captured["parent_trace_id"] = self.parent_trace_id
        return _fake_result("ok")

    monkeypatch.setattr(Harness, "arun", fake_arun)

    token = _current_trace_id_var.set("trace-parent-xyz")
    try:
        _ainvoke(dispatch_subagent, {"task": "x", "allowed_scopes": ["kb"]})
    finally:
        _current_trace_id_var.reset(token)

    assert captured["parent_trace_id"] == "trace-parent-xyz"


def test_parent_trace_id_is_none_when_outside_run(monkeypatch):
    captured = {}

    async def fake_arun(self, q):
        captured["parent_trace_id"] = self.parent_trace_id
        return _fake_result("ok")

    monkeypatch.setattr(Harness, "arun", fake_arun)
    _ainvoke(dispatch_subagent, {"task": "x", "allowed_scopes": ["kb"]})
    assert captured["parent_trace_id"] is None


# ---------- 端到端：主 Harness 调子 agent ----------


class _Scripted:
    def __init__(self, scripted):
        self._scripted = list(scripted)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages, *args, **kwargs):
        return self._scripted.pop(0)

    async def astream(self, messages, *args, **kwargs):
        from langchain_core.messages import AIMessageChunk

        msg = self._scripted.pop(0)
        chunk = AIMessageChunk(
            content=msg.content,
            tool_calls=getattr(msg, "tool_calls", []) or [],
        )
        yield chunk


@tool
def _echo(text: str) -> str:
    """echo"""
    return f"echo:{text}"


def test_main_harness_consumes_subagent_final_text(monkeypatch):
    """主 LLM 调 dispatch_subagent，子 LLM 直接给 final_text，主 LLM 看到 ToolMessage 后收尾。"""
    # 子 Harness 的 LLM 是 monkeypatch 替换 create_chat_model 注入的
    sub_llm = _Scripted([AIMessage(content="子答案：基金定投适合波动市场。")])

    def fake_create_chat_model(*args, **kwargs):
        return sub_llm

    monkeypatch.setattr("harness.loop.create_chat_model", fake_create_chat_model)

    main_registry = ToolRegistry()
    main_registry.register(dispatch_subagent, scope="control")

    main_llm = _Scripted(
        [
            AIMessage(
                content="先拆任务。",
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "dispatch_subagent",
                        "args": {"task": "请回答基金定投是否适合波动市场", "allowed_scopes": ["kb"]},
                    }
                ],
            ),
            AIMessage(content="综合答案：是的，基金定投适合波动市场。"),
        ]
    )

    main = Harness(
        llm=main_llm,
        registry=main_registry,
        hooks=HookBus(),
        max_iterations=5,
    )
    result = main.run("帮我评估基金定投")

    assert result.stopped_reason == "final_text"
    assert "波动市场" in result.final_text
    # 主 trace 里能看到工具调用
    tool_results = result.turns[0].tool_results
    assert tool_results[0].name == "dispatch_subagent"
    assert "子答案" in tool_results[0].content


# ============================================================================
# dispatch_subagents 并发派发
# ============================================================================


import asyncio
import json
import time

from harness import dispatch_subagents
from harness.subagent import _MAX_CONCURRENCY_CAP, _MAX_TASKS


# ---------- schema 校验 ----------


def test_dispatch_subagents_rejects_empty_list():
    out = _ainvoke(dispatch_subagents, {"tasks": []})
    assert json.loads(out) == {"error": "tasks 必须是非空列表"}


def test_dispatch_subagents_rejects_oversized_batch():
    tasks = [
        {"task": f"t{i}", "allowed_scopes": ["kb"]}
        for i in range(_MAX_TASKS + 1)
    ]
    out = _ainvoke(dispatch_subagents, {"tasks": tasks})
    assert "上限" in json.loads(out)["error"]


def test_dispatch_subagents_rejects_missing_fields():
    out = _ainvoke(dispatch_subagents, {"tasks": [{"task": "x"}]})
    assert "task 与 allowed_scopes" in json.loads(out)["error"]


# ---------- 并发执行 ----------


def test_dispatch_subagents_runs_in_parallel(monkeypatch):
    """N=3、每个 fake arun sleep 0.2s；并发应在 ~0.2s 完成而非 0.6s。"""

    async def fake_arun(self, q):
        await asyncio.sleep(0.2)
        return HarnessResult(
            final_text=f"done:{q}", turns=[], stopped_reason="final_text"
        )

    monkeypatch.setattr(Harness, "arun", fake_arun)

    tasks = [
        {"task": "a", "allowed_scopes": ["kb"]},
        {"task": "b", "allowed_scopes": ["kb"]},
        {"task": "c", "allowed_scopes": ["kb"]},
    ]
    start = time.perf_counter()
    out = _ainvoke(dispatch_subagents, {"tasks": tasks, "max_concurrency": 3})
    elapsed = time.perf_counter() - start

    parsed = json.loads(out)
    assert len(parsed) == 3
    # 并发 3，每个 0.2s，总耗时应 < 0.5s（含调度开销留余量）
    assert elapsed < 0.5
    assert [p["final_text"] for p in parsed] == ["done:a", "done:b", "done:c"]


def test_dispatch_subagents_isolates_per_task_failure(monkeypatch):
    """单个 task 抛异常，其余兄弟仍返回 final_text。"""

    async def fake_arun(self, q):
        if q == "boom":
            raise RuntimeError("子任务炸了")
        return HarnessResult(
            final_text=f"done:{q}", turns=[], stopped_reason="final_text"
        )

    monkeypatch.setattr(Harness, "arun", fake_arun)

    tasks = [
        {"task": "a", "allowed_scopes": ["kb"]},
        {"task": "boom", "allowed_scopes": ["kb"]},
        {"task": "c", "allowed_scopes": ["kb"]},
    ]
    out = _ainvoke(dispatch_subagents, {"tasks": tasks})
    parsed = json.loads(out)
    assert len(parsed) == 3

    by_task = {p["task"]: p for p in parsed}
    assert by_task["a"]["final_text"] == "done:a"
    assert by_task["c"]["final_text"] == "done:c"
    assert "error" in by_task["boom"]
    assert "子任务炸了" in by_task["boom"]["error"]


def test_dispatch_subagents_propagates_parent_ids(monkeypatch):
    """parent_trace_id / chat_id 在工具体外层一次性捕获，传给所有子 Harness。"""
    captured: list[dict] = []

    async def fake_arun(self, q):
        captured.append(
            {
                "parent_trace_id": self.parent_trace_id,
                "chat_id": self.chat_id,
            }
        )
        return HarnessResult(
            final_text="x", turns=[], stopped_reason="final_text"
        )

    monkeypatch.setattr(Harness, "arun", fake_arun)

    from harness.loop import _current_chat_id_var

    trace_token = _current_trace_id_var.set("trace-parent")
    chat_token = _current_chat_id_var.set("alice")
    try:
        _ainvoke(dispatch_subagents, 
            {
                "tasks": [
                    {"task": "a", "allowed_scopes": ["kb"]},
                    {"task": "b", "allowed_scopes": ["kb"]},
                ]
            }
        )
    finally:
        _current_trace_id_var.reset(trace_token)
        _current_chat_id_var.reset(chat_token)

    assert len(captured) == 2
    for c in captured:
        assert c["parent_trace_id"] == "trace-parent"
        assert c["chat_id"] == "alice"


def test_dispatch_subagents_caps_max_concurrency(monkeypatch):
    """传 max_concurrency=999 实际只用 _MAX_CONCURRENCY_CAP。"""
    peak = {"value": 0, "active": 0}

    async def fake_arun(self, q):
        peak["active"] += 1
        peak["value"] = max(peak["value"], peak["active"])
        await asyncio.sleep(0.05)
        peak["active"] -= 1
        return HarnessResult(
            final_text="x", turns=[], stopped_reason="final_text"
        )

    monkeypatch.setattr(Harness, "arun", fake_arun)

    n_tasks = _MAX_CONCURRENCY_CAP + 4
    tasks = [{"task": f"t{i}", "allowed_scopes": ["kb"]} for i in range(n_tasks)]
    _ainvoke(dispatch_subagents, {"tasks": tasks, "max_concurrency": 999})

    assert peak["value"] <= _MAX_CONCURRENCY_CAP


# ---------- 回归 #1：在已运行 loop 内调用主 Harness 触发 dispatch_subagents 不崩溃 ----------


def test_dispatch_tools_are_async_coroutine_funcs():
    """dispatch_subagent / dispatch_subagents 必须是 async 工具。

    历史 bug：sync 工具体内 ``asyncio.run(_run_all())`` 在已有 running loop
    的协程上下文里直接调 ``.invoke(...)`` 会触发
    ``RuntimeError: asyncio.run() cannot be called from a running event loop``。
    修复后改 async 工具，由 Harness 主循环走 ``tool.ainvoke`` 原生 await。
    """
    import inspect

    # StructuredTool.coroutine 是 async 实现；func 是 sync 实现。
    # async 工具：coroutine 存在且是 coroutinefunction，func 为 None
    assert dispatch_subagent.coroutine is not None
    assert inspect.iscoroutinefunction(dispatch_subagent.coroutine)
    assert dispatch_subagent.func is None

    assert dispatch_subagents.coroutine is not None
    assert inspect.iscoroutinefunction(dispatch_subagents.coroutine)
    assert dispatch_subagents.func is None


def test_main_harness_calls_dispatch_subagents_inside_running_loop(monkeypatch):
    """端到端：主 Harness 在 async 上下文里跑完整 astream，
    其中一个工具调用是 dispatch_subagents，必须不抛 asyncio.run 重入异常。

    这里直接验证"在已运行 loop 内主循环调 async 工具"路径全链路通畅。
    """
    # 子 Harness 的 arun 直接返回固定结果，跳过真实 LLM
    async def fake_arun(self, q):
        return HarnessResult(
            final_text=f"sub-done:{q}", turns=[], stopped_reason="final_text"
        )

    monkeypatch.setattr(Harness, "arun", fake_arun)

    from langchain_core.messages import AIMessageChunk

    class _MainLLM:
        def __init__(self):
            self._calls = 0

        def bind_tools(self, tools):
            return self

        async def astream(self, messages, *args, **kwargs):
            self._calls += 1
            if self._calls == 1:
                yield AIMessageChunk(
                    content="并发派两个子任务",
                    tool_calls=[
                        {
                            "id": "c1",
                            "name": "dispatch_subagents",
                            "args": {
                                "tasks": [
                                    {"task": "a", "allowed_scopes": ["kb"]},
                                    {"task": "b", "allowed_scopes": ["kb"]},
                                ]
                            },
                        }
                    ],
                )
            else:
                yield AIMessageChunk(content="综合完成")

    main_registry = ToolRegistry()
    main_registry.register(dispatch_subagents, scope="control")

    main = Harness(
        llm=_MainLLM(),
        registry=main_registry,
        hooks=HookBus(),
        max_iterations=5,
    )

    async def _drive():
        events = []
        async for ev in main.astream("帮我并发拆分"):
            events.append(ev)
        return events

    events = asyncio.run(_drive())

    types = [e.type for e in events]
    assert "error" not in types, f"主循环在已运行 loop 内调 dispatch_subagents 出错: {types}"
    assert types[-1] == "run_end"
    tr_events = [e for e in events if e.type == "tool_result"]
    assert len(tr_events) == 1
    payload = json.loads(tr_events[0].data["result"]["content"])
    assert len(payload) == 2
    assert {p["task"] for p in payload} == {"a", "b"}
    assert all("sub-done:" in p["final_text"] for p in payload)

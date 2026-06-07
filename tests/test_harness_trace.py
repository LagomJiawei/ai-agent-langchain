"""HarnessTrace + TraceWriterHook 测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness import (
    HarnessResult,
    HarnessTrace,
    HookContext,
    ToolCall,
    ToolResult,
    Turn,
    TraceWriterHook,
)
from harness.trace import TRACE_SCHEMA_VERSION
from config.settings import settings


def _result_with_one_turn() -> HarnessResult:
    turn = Turn(index=0, thought="thinking")
    turn.tool_calls.append(ToolCall(id="c1", name="echo", args={"x": 1}))
    turn.tool_results.append(
        ToolResult(call_id="c1", name="echo", ok=True, content="ok", elapsed_ms=12.3)
    )
    turn.final_text = None
    return HarnessResult(
        final_text="done",
        turns=[turn],
        stopped_reason="final_text",
    )


# ---------- HarnessTrace ----------


def test_trace_from_harness_result_serializes_turns():
    result = _result_with_one_turn()
    trace = HarnessTrace.from_harness_result(
        result=result,
        user_query="q",
        started_at="2026-06-06T00:00:00+00:00",
        finished_at="2026-06-06T00:00:01+00:00",
        trace_id="trace-xyz",
    )
    assert trace.trace_id == "trace-xyz"
    assert trace.user_query == "q"
    assert trace.stopped_reason == "final_text"
    assert trace.total_tool_calls == 1
    assert len(trace.turns) == 1
    assert trace.turns[0]["tool_calls"][0]["name"] == "echo"
    assert trace.turns[0]["tool_results"][0]["ok"] is True
    assert trace.parent_trace_id is None


def test_trace_carries_parent_id():
    trace = HarnessTrace.from_harness_result(
        result=_result_with_one_turn(),
        user_query="q",
        started_at="x",
        finished_at="y",
        trace_id="child",
        parent_trace_id="parent-abc",
    )
    assert trace.parent_trace_id == "parent-abc"


def test_trace_write_creates_json_file(tmp_path: Path):
    trace = HarnessTrace.from_harness_result(
        result=_result_with_one_turn(),
        user_query="q",
        started_at="2026-06-06T00:00:00+00:00",
        finished_at="2026-06-06T00:00:01+00:00",
        trace_id="trace-write-1",
        parent_trace_id="parent-1",
    )
    written = trace.write(tmp_path)
    assert written.exists()
    assert written.suffix == ".json"
    # chat_id 未提供时落到 _default 子目录
    assert written.parent.name == "_default"

    reloaded = json.loads(written.read_text(encoding="utf-8"))
    assert reloaded["trace_id"] == "trace-write-1"
    assert reloaded["user_query"] == "q"
    assert reloaded["parent_trace_id"] == "parent-1"
    assert reloaded["turns"][0]["tool_calls"][0]["name"] == "echo"


def test_trace_writes_to_chat_id_subdir(tmp_path: Path):
    trace = HarnessTrace.from_harness_result(
        result=_result_with_one_turn(),
        user_query="q",
        started_at="x",
        finished_at="y",
        trace_id="t-2",
        chat_id="alice",
    )
    written = trace.write(tmp_path)
    assert written.parent.name == "alice"
    assert written.parent.parent == tmp_path


def test_trace_sanitizes_chat_id(tmp_path: Path):
    """含路径分隔符 / .. 的 chat_id 不能写穿目录。"""
    trace = HarnessTrace.from_harness_result(
        result=_result_with_one_turn(),
        user_query="q",
        started_at="x",
        finished_at="y",
        trace_id="t-3",
        chat_id="../bad/path",
    )
    written = trace.write(tmp_path)
    # 子目录应被消毒为不含 / 或 ..
    assert written.parent.parent == tmp_path
    assert "/" not in written.parent.name
    assert ".." not in written.parent.name


def test_trace_default_subdir_when_chat_id_blank(tmp_path: Path):
    trace = HarnessTrace.from_harness_result(
        result=_result_with_one_turn(),
        user_query="q",
        started_at="x",
        finished_at="y",
        trace_id="t-4",
        chat_id="",
    )
    written = trace.write(tmp_path)
    assert written.parent.name == "_default"


# ---------- TraceWriterHook ----------


def test_trace_writer_hook_writes_when_enabled(tmp_path: Path):
    hook = TraceWriterHook(base_dir=tmp_path)
    ctx = HookContext(
        user_query="q",
        turn_index=1,
        call_counts={},
        started_at="2026-06-06T00:00:00+00:00",
        finished_at="2026-06-06T00:00:01+00:00",
    )

    original_enabled = settings.agent.trace_enabled
    settings.agent.trace_enabled = True
    try:
        hook(_result_with_one_turn(), ctx)
    finally:
        settings.agent.trace_enabled = original_enabled

    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1


def test_trace_writer_hook_skips_when_disabled(tmp_path: Path):
    hook = TraceWriterHook(base_dir=tmp_path)
    ctx = HookContext(
        user_query="q",
        turn_index=0,
        call_counts={},
        started_at="2026-06-06T00:00:00+00:00",
        finished_at="2026-06-06T00:00:01+00:00",
    )

    original_enabled = settings.agent.trace_enabled
    settings.agent.trace_enabled = False
    try:
        hook(_result_with_one_turn(), ctx)
    finally:
        settings.agent.trace_enabled = original_enabled

    assert list(tmp_path.rglob("*.json")) == []


def test_trace_writer_hook_swallows_io_errors(monkeypatch, tmp_path: Path):
    hook = TraceWriterHook(base_dir=tmp_path)
    ctx = HookContext(
        user_query="q",
        turn_index=0,
        call_counts={},
        started_at="x",
        finished_at="y",
    )

    def boom(self, base_dir):
        raise IOError("disk full")

    monkeypatch.setattr(HarnessTrace, "write", boom)

    original_enabled = settings.agent.trace_enabled
    settings.agent.trace_enabled = True
    try:
        # 不应抛
        hook(_result_with_one_turn(), ctx)
    finally:
        settings.agent.trace_enabled = original_enabled


# ---------- schema_version (#15) ----------


def test_trace_carries_current_schema_version():
    """from_harness_result 构造的 trace 自动带当前版本号。"""
    trace = HarnessTrace.from_harness_result(
        result=_result_with_one_turn(),
        user_query="q",
        started_at="x",
        finished_at="y",
        trace_id="t-sv-1",
    )
    assert trace.schema_version == TRACE_SCHEMA_VERSION
    # 当前版本必须是正整数，未来递增也不能回 0
    assert isinstance(TRACE_SCHEMA_VERSION, int)
    assert TRACE_SCHEMA_VERSION >= 1


def test_trace_json_includes_schema_version_at_top(tmp_path: Path):
    """落盘 JSON 顶层必须含 schema_version 字段；离线分析脚本依赖它做兼容性切换。"""
    trace = HarnessTrace.from_harness_result(
        result=_result_with_one_turn(),
        user_query="q",
        started_at="x",
        finished_at="y",
        trace_id="t-sv-2",
    )
    written = trace.write(tmp_path)
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert "schema_version" in payload
    assert payload["schema_version"] == TRACE_SCHEMA_VERSION

    # 字段顺序：schema_version 应出现在所有字段之前，让 cat / head -c 一眼能看到
    keys = list(payload.keys())
    assert keys[0] == "schema_version"


def test_trace_legacy_payload_without_schema_version_can_be_detected():
    """模拟读取无版本号的旧 trace（v0），按 ``payload.get("schema_version", 0)`` 处理。

    这是离线分析脚本应当遵循的兼容模式——本测试固化这一契约，防止
    后续误改成 KeyError 风格的强校验。
    """
    legacy_payload = {
        "trace_id": "old-trace",
        "started_at": "x",
        "finished_at": "y",
        "user_query": "q",
        "stopped_reason": "final_text",
        "final_text": "ok",
        "total_tool_calls": 0,
        "turns": [],
        # 故意没有 schema_version
    }
    version = legacy_payload.get("schema_version", 0)
    assert version == 0  # 缺失即 v0

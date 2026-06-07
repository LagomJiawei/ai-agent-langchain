"""HarnessEvent 数据类与工厂函数测试。"""
from __future__ import annotations

from harness import HarnessEvent, ToolCall, ToolResult
from harness.events import (
    error,
    final_text,
    run_end,
    run_start,
    thinking_token,
    tool_call,
    tool_result,
)


def test_run_start_factory():
    evt = run_start(
        trace_id="t1",
        parent_trace_id=None,
        user_query="q",
        started_at="2026-06-06T00:00:00+00:00",
    )
    assert evt.type == "run_start"
    assert evt.data == {
        "trace_id": "t1",
        "parent_trace_id": None,
        "user_query": "q",
        "started_at": "2026-06-06T00:00:00+00:00",
    }


def test_thinking_token_factory():
    evt = thinking_token(turn_index=2, delta="hello")
    assert evt.type == "thinking_token"
    assert evt.data == {"turn_index": 2, "delta": "hello"}


def test_tool_call_factory_serializes_call():
    call = ToolCall(id="c1", name="echo", args={"x": 1})
    evt = tool_call(turn_index=0, call=call)
    assert evt.type == "tool_call"
    assert evt.data["turn_index"] == 0
    assert evt.data["call"]["name"] == "echo"
    assert evt.data["call"]["args"] == {"x": 1}


def test_tool_result_factory_serializes_result():
    res = ToolResult(call_id="c1", name="echo", ok=True, content="ok")
    evt = tool_result(turn_index=0, result=res)
    assert evt.type == "tool_result"
    assert evt.data["result"]["ok"] is True
    assert evt.data["result"]["content"] == "ok"


def test_final_text_and_run_end_factories():
    f = final_text("done")
    assert f.type == "final_text" and f.data == {"final_text": "done"}

    e = run_end(stopped_reason="final_text", total_tool_calls=2, finished_at="2026")
    assert e.type == "run_end"
    assert e.data["stopped_reason"] == "final_text"
    assert e.data["total_tool_calls"] == 2


def test_error_factory():
    evt = error("boom")
    assert evt.type == "error"
    assert evt.data == {"message": "boom"}


def test_harness_event_default_data():
    """直接构造 HarnessEvent 不传 data 应得到空 dict。"""
    evt = HarnessEvent(type="run_end")
    assert evt.data == {}

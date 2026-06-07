"""Harness 地基模块测试：Turn / ToolRegistry / ConversationContext / system prompt。"""
from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from harness import (
    ConversationContext,
    ToolCall,
    ToolRegistry,
    ToolResult,
    Turn,
    default_registry,
    register_tool,
)


# ---------- Turn / ToolCall / ToolResult ----------


def test_turn_default_collections_are_independent():
    a = Turn(index=0)
    b = Turn(index=1)
    a.tool_calls.append(ToolCall(id="x", name="t", args={}))
    assert b.tool_calls == []


def test_tool_call_args_default_empty_dict():
    call = ToolCall(id="1", name="search")
    assert call.args == {}


def test_tool_result_holds_failure_info():
    result = ToolResult(call_id="1", name="t", ok=False, content="", error="boom")
    assert result.ok is False
    assert result.error == "boom"


# ---------- ToolRegistry ----------


@tool
def fake_alpha(query: str) -> str:
    """fake alpha tool"""
    return f"alpha:{query}"


@tool
def fake_beta(query: str) -> str:
    """fake beta tool"""
    return f"beta:{query}"


@pytest.fixture
def fresh_registry() -> ToolRegistry:
    return ToolRegistry()


def test_registry_register_and_get(fresh_registry: ToolRegistry):
    fresh_registry.register(fake_alpha)
    assert fresh_registry.get("fake_alpha") is fake_alpha


def test_registry_rejects_non_tool(fresh_registry: ToolRegistry):
    with pytest.raises(TypeError):
        fresh_registry.register("not a tool")  # type: ignore[arg-type]


def test_registry_rejects_duplicate_name(fresh_registry: ToolRegistry):
    fresh_registry.register(fake_alpha)
    with pytest.raises(ValueError):
        fresh_registry.register(fake_alpha)


def test_registry_get_missing_raises(fresh_registry: ToolRegistry):
    with pytest.raises(KeyError):
        fresh_registry.get("ghost")


def test_registry_scope_filter(fresh_registry: ToolRegistry):
    fresh_registry.register(fake_alpha, scope="web")
    fresh_registry.register(fake_beta, scope="kb")
    assert fresh_registry.names(scope="web") == ["fake_alpha"]
    assert fresh_registry.names(scope="kb") == ["fake_beta"]
    assert set(fresh_registry.names()) == {"fake_alpha", "fake_beta"}


def test_registry_list_supports_multiple_scopes(fresh_registry: ToolRegistry):
    fresh_registry.register(fake_alpha, scope="web")
    fresh_registry.register(fake_beta, scope="kb")
    names = sorted(fresh_registry.names(scopes=["web", "kb"]))
    assert names == ["fake_alpha", "fake_beta"]
    # 不存在的 scope 不会爆，只是空集
    assert fresh_registry.names(scopes=["ghost"]) == []


def test_registry_scope_and_scopes_mutually_exclusive(fresh_registry: ToolRegistry):
    fresh_registry.register(fake_alpha, scope="web")
    with pytest.raises(ValueError):
        fresh_registry.list(scope="web", scopes=["web"])


def test_registry_scope_of(fresh_registry: ToolRegistry):
    fresh_registry.register(fake_alpha, scope="web")
    assert fresh_registry.scope_of("fake_alpha") == "web"
    with pytest.raises(KeyError):
        fresh_registry.scope_of("ghost")


def test_default_registry_register_tool_helper():
    snapshot_before = set(default_registry.names())
    try:

        @tool
        def temp_tool(x: str) -> str:
            """temp"""
            return x

        register_tool(temp_tool, scope="temp")
        assert "temp_tool" in default_registry.names()
        assert default_registry.get("temp_tool") is temp_tool
    finally:
        # 清掉本测试注入的工具，避免污染其他测试
        if "temp_tool" in default_registry.names():
            default_registry._tools.pop("temp_tool")
            default_registry._scopes.pop("temp_tool")
    assert set(default_registry.names()) == snapshot_before


# ---------- ConversationContext ----------


def test_context_initial_messages():
    ctx = ConversationContext(system_prompt="sys", initial_user_message="hi")
    msgs = ctx.snapshot()
    assert len(msgs) == 2
    assert isinstance(msgs[0], SystemMessage) and msgs[0].content == "sys"
    assert isinstance(msgs[1], HumanMessage) and msgs[1].content == "hi"


def test_context_append_and_snapshot_isolation():
    ctx = ConversationContext(system_prompt="sys", initial_user_message="hi")
    snap = ctx.snapshot()
    snap.append(AIMessage(content="leak"))
    # 外部 mutate snapshot 不应影响内部状态
    assert len(ctx) == 2
    ctx.append(AIMessage(content="ok"))
    assert len(ctx) == 3


def test_context_append_rejects_non_message():
    ctx = ConversationContext(system_prompt="sys", initial_user_message="hi")
    with pytest.raises(TypeError):
        ctx.append("not a message")  # type: ignore[arg-type]


# ---------- system prompt 资源 ----------


def test_system_prompt_file_exists_and_non_empty():
    path = Path(__file__).resolve().parents[1] / "harness" / "prompts" / "system.md"
    assert path.exists(), f"system prompt 文件不存在: {path}"
    content = path.read_text(encoding="utf-8").strip()
    assert content, "system prompt 文件为空"
    assert "LiCaiManus" in content

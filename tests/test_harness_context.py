"""ConversationContext 压缩行为测试。"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from harness import ConversationContext, count_tokens


# ---------- token_counter ----------


def test_count_tokens_monotonic():
    short = [HumanMessage(content="hi")]
    long = [HumanMessage(content="hi" * 1000)]
    assert count_tokens(long) > count_tokens(short)


def test_count_tokens_handles_empty_content():
    assert count_tokens([HumanMessage(content="")]) > 0  # role 开销 ≈ 4


# ---------- compress_if_needed: no-op ----------


def test_compress_skipped_when_below_budget():
    ctx = ConversationContext(
        system_prompt="sys",
        initial_user_message="hi",
        max_tokens=10_000,
    )
    ctx.append(AIMessage(content="ok"))
    triggered = ctx.compress_if_needed()
    assert triggered is False
    assert ctx.compressed is False
    assert len(ctx) == 3


# ---------- compress_if_needed: 截短中间 ToolMessage ----------


def _build_long_context(tool_payload_chars: int = 8000) -> ConversationContext:
    """构造一个超阈值的上下文：system + user + 3 轮 (ai+tool) + 最近一轮 (ai+tool)。"""
    ctx = ConversationContext(
        system_prompt="sys-prompt",
        initial_user_message="user-query",
        max_tokens=500,
        keep_last_turns=2,
        tool_message_keep_chars=100,
    )
    payload = "x" * tool_payload_chars
    for i in range(3):
        ctx.append(AIMessage(content=f"think-{i}", tool_calls=[]))
        ctx.append(ToolMessage(content=payload, tool_call_id=f"call-{i}"))
    # 最近 2 轮（受保护）
    ctx.append(AIMessage(content="think-last-2", tool_calls=[]))
    ctx.append(ToolMessage(content=payload, tool_call_id="call-last-2"))
    ctx.append(AIMessage(content="think-last-1", tool_calls=[]))
    ctx.append(ToolMessage(content=payload, tool_call_id="call-last-1"))
    return ctx


def test_compress_protects_system_and_first_user():
    ctx = _build_long_context()
    ctx.compress_if_needed()
    msgs = ctx.snapshot()
    # 头两条始终是 system + 首条 user
    assert isinstance(msgs[0], SystemMessage)
    assert msgs[0].content == "sys-prompt"
    assert isinstance(msgs[1], HumanMessage)
    assert msgs[1].content == "user-query"


def test_compress_protects_last_two_turns():
    ctx = _build_long_context()
    ctx.compress_if_needed()
    msgs = ctx.snapshot()

    # 最近两轮的 ToolMessage 内容不被截短
    last_tools = [
        m for m in msgs if isinstance(m, ToolMessage) and m.tool_call_id.startswith("call-last")
    ]
    assert len(last_tools) == 2
    for tm in last_tools:
        # 不含截短占位
        assert "[truncated:" not in tm.content


def test_compress_truncates_middle_tool_messages():
    ctx = _build_long_context()
    ctx.compress_if_needed()
    msgs = ctx.snapshot()

    middle_tools = [
        m
        for m in msgs
        if isinstance(m, ToolMessage) and not m.tool_call_id.startswith("call-last")
    ]
    # 至少有一条中间 ToolMessage 被截短或被删
    assert len(middle_tools) < 3 or any("[truncated:" in m.content for m in middle_tools)


def test_compress_preserves_all_ai_messages():
    ctx = _build_long_context()
    ctx.compress_if_needed()
    msgs = ctx.snapshot()
    ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
    # 5 轮 AI（3 中间 + 2 末尾）应全部保留
    assert len(ai_msgs) == 5


def test_compress_extreme_case_only_warns():
    """单轮巨型 ToolMessage 受最后 2 轮保护，无法压缩到阈值，但不能死循环。"""
    ctx = ConversationContext(
        system_prompt="s",
        initial_user_message="u",
        max_tokens=50,
        keep_last_turns=2,
    )
    huge = "y" * 50_000
    ctx.append(AIMessage(content="t"))
    ctx.append(ToolMessage(content=huge, tool_call_id="solo"))

    triggered = ctx.compress_if_needed()
    assert triggered is True
    # 不抛异常，不死循环
    assert ctx.compressed is True


def test_snapshot_triggers_compression():
    ctx = _build_long_context()
    assert ctx.compressed is False
    ctx.snapshot()
    assert ctx.compressed is True


# ---------- 第二轮删除：多条 ToolMessage 必须全部被处理（回归 #4） ----------


def test_compress_deletes_all_middle_tool_messages_when_needed():
    """回归测试：第二轮删除阶段，多条中间 ToolMessage 必须都进入删除候选。

    历史 bug：循环里用 ``self._messages[idx]`` 取消息，删一条后下标整体前移，
    后续 idx 会取到 AIMessage 并 continue 漏删，导致只删了第 1 条剩余 5 条
    全部跳过。修复后改为先按 idx 收集对象引用，再按引用删除。
    """
    ctx = ConversationContext(
        system_prompt="s",
        initial_user_message="u",
        max_tokens=200,
        keep_last_turns=1,
        tool_message_keep_chars=50,
    )
    huge = "z" * 5_000
    for i in range(7):
        ctx.append(AIMessage(content=f"a{i}"))
        ctx.append(ToolMessage(content=huge, tool_call_id=f"t{i}"))

    ctx.compress_if_needed()
    msgs = ctx.snapshot()

    # 7 条 AIMessage 全部保留（最后一对 AI+Tool 受 protected_tail 保护）
    ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
    assert len(ai_msgs) == 7

    # 中间 6 条 ToolMessage（t0..t5）应全部被删除，只保留最后受保护的 t6
    remaining_tools = [m for m in msgs if isinstance(m, ToolMessage)]
    assert len(remaining_tools) == 1
    assert remaining_tools[0].tool_call_id == "t6"


def test_compress_deletes_subset_when_threshold_reached_midway():
    """删除过程中一旦达标立刻 return，剩余 ToolMessage 不再继续删。"""
    ctx = ConversationContext(
        system_prompt="s",
        initial_user_message="u",
        max_tokens=1500,  # 删 2~3 条即可达标
        keep_last_turns=1,
        tool_message_keep_chars=50,
    )
    huge = "z" * 5_000
    for i in range(5):
        ctx.append(AIMessage(content=f"a{i}"))
        ctx.append(ToolMessage(content=huge, tool_call_id=f"t{i}"))

    ctx.compress_if_needed()
    msgs = ctx.snapshot()

    ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]
    assert len(ai_msgs) == 5  # 所有 AIMessage 仍在

    remaining_tools = [m for m in msgs if isinstance(m, ToolMessage)]
    # 至少删了一部分 middle ToolMessage，t4 仍受保护必然保留
    assert any(t.tool_call_id == "t4" for t in remaining_tools)
    assert len(remaining_tools) < 5

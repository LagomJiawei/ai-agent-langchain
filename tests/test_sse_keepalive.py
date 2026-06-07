"""SSE keepalive 心跳测试（回归 #14）。

覆盖：
- ``with_keepalive`` helper：快流不插心跳 / 慢流插心跳 / 异常透传 / 上游结束即停 / 取消安全。
- 3 个 SSE 端点端到端：慢上游下用 TestClient.stream() 收到 ``: keepalive`` 行。
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.sse import with_keepalive


# ============================================================================
# 单元测试 with_keepalive
# ============================================================================


async def _collect(it: AsyncIterator[str]) -> list[str]:
    out: list[str] = []
    async for chunk in it:
        out.append(chunk)
    return out


def _is_keepalive(chunk: str) -> bool:
    return chunk.startswith(": keepalive ") and chunk.endswith("\n\n")


def test_no_keepalive_when_data_streams_fast():
    """上游连续 yield，全部在 interval 内 → 不插心跳。"""

    async def fast() -> AsyncIterator[str]:
        for i in range(5):
            yield f"data: {i}\n\n"
            await asyncio.sleep(0.01)

    chunks = asyncio.run(_collect(with_keepalive(fast(), interval=1.0)))
    assert chunks == [f"data: {i}\n\n" for i in range(5)]
    assert all(not _is_keepalive(c) for c in chunks)


def test_emits_keepalive_when_idle():
    """上游 0.5s 才 yield，interval=0.1s → 至少有 3 条 keepalive。"""

    async def slow() -> AsyncIterator[str]:
        await asyncio.sleep(0.5)
        yield "data: done\n\n"

    chunks = asyncio.run(_collect(with_keepalive(slow(), interval=0.1)))
    keepalives = [c for c in chunks if _is_keepalive(c)]
    assert len(keepalives) >= 3, f"心跳不够，实际 chunks={chunks}"
    # 业务事件仍正确
    assert chunks[-1] == "data: done\n\n"


def test_keepalive_format_is_sse_comment():
    """心跳行严格符合 SSE comment 规范：``: keepalive <int>\\n\\n``。"""

    async def slow() -> AsyncIterator[str]:
        await asyncio.sleep(0.25)
        yield "data: x\n\n"

    chunks = asyncio.run(_collect(with_keepalive(slow(), interval=0.1)))
    keepalives = [c for c in chunks if _is_keepalive(c)]
    assert keepalives
    for k in keepalives:
        assert re.match(r"^: keepalive \d+\n\n$", k), f"格式不对: {k!r}"


def test_disabled_when_interval_non_positive():
    """interval<=0 → 退化为透传，零心跳。"""

    async def slow() -> AsyncIterator[str]:
        await asyncio.sleep(0.2)
        yield "data: only\n\n"

    chunks = asyncio.run(_collect(with_keepalive(slow(), interval=0.0)))
    assert chunks == ["data: only\n\n"]
    chunks2 = asyncio.run(_collect(with_keepalive(slow(), interval=-1.0)))
    assert chunks2 == ["data: only\n\n"]


def test_terminates_when_source_completes():
    """上游 yield 3 条后 return → 输出恰好 3 条业务事件，无心跳尾巴。"""

    async def three() -> AsyncIterator[str]:
        for i in range(3):
            yield f"data: {i}\n\n"

    chunks = asyncio.run(_collect(with_keepalive(three(), interval=10.0)))
    assert chunks == ["data: 0\n\n", "data: 1\n\n", "data: 2\n\n"]


def test_passes_through_upstream_exception():
    """上游 raise → with_keepalive 也 raise；之前的事件已经能拿到。"""

    async def boom() -> AsyncIterator[str]:
        yield "data: before\n\n"
        raise RuntimeError("上游炸了")

    async def _run():
        collected: list[str] = []
        with pytest.raises(RuntimeError, match="上游炸了"):
            async for chunk in with_keepalive(boom(), interval=10.0):
                collected.append(chunk)
        return collected

    chunks = asyncio.run(_run())
    assert "data: before\n\n" in chunks


def test_cancel_safe_when_consumer_stops_early():
    """消费方 break 后，内部 pump task 必须被 cancel，不留泄漏。"""

    pump_started = asyncio.Event()
    pump_cancelled = {"value": False}

    async def long_source() -> AsyncIterator[str]:
        pump_started.set()
        try:
            for i in range(1000):
                await asyncio.sleep(0.01)
                yield f"data: {i}\n\n"
        except asyncio.CancelledError:
            pump_cancelled["value"] = True
            raise

    async def _run():
        gen = with_keepalive(long_source(), interval=10.0)
        async for chunk in gen:
            # 消费一条就 break
            assert chunk == "data: 0\n\n"
            break
        await gen.aclose()
        # 给事件循环一个 tick 让 pump 接到取消
        await asyncio.sleep(0.05)

    asyncio.run(_run())
    assert pump_cancelled["value"], "pump 任务没被 cancel，存在 task 泄漏"


def test_yields_in_order_across_data_and_keepalive():
    """data 与 keepalive 交错时，顺序严格按到达时间。"""

    async def mixed() -> AsyncIterator[str]:
        yield "data: first\n\n"
        await asyncio.sleep(0.3)  # 此期间应有 keepalive
        yield "data: second\n\n"

    chunks = asyncio.run(_collect(with_keepalive(mixed(), interval=0.1)))
    # 第一条必须是 data: first
    assert chunks[0] == "data: first\n\n"
    # 中间至少一条 keepalive
    assert any(_is_keepalive(c) for c in chunks[1:-1])
    # 最后是 data: second
    assert chunks[-1] == "data: second\n\n"


# ============================================================================
# 端到端：3 个 SSE 端点（建立 SSE 基础测试架）
# ============================================================================


@pytest.fixture
def client(monkeypatch) -> TestClient:
    # 强制 anonymous 模式 + 不要走真实 LLM
    return TestClient(app)


def test_stream_chat_normal_events_intact(client, monkeypatch):
    """快上游下，业务事件完整传到客户端，无心跳干扰。"""

    async def fake_astream_chat(self, message, chat_id="default"):
        for t in ["你好", "，", "世界"]:
            yield t

    from app.services import FinancialAdvisorService

    monkeypatch.setattr(FinancialAdvisorService, "astream_chat", fake_astream_chat)

    with client.stream("GET", "/api/chat/stream", params={"message": "hi"}) as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes()).decode("utf-8")

    # 3 条 data + 一条 event: end
    assert body.count("data: ") >= 4  # 3 tokens + end
    assert "event: end" in body
    assert "你好" in body
    assert "世界" in body


def test_stream_chat_emits_keepalive_under_slow_llm(client, monkeypatch):
    """慢上游下，客户端能在生成完成前先收到 keepalive 注释行。

    覆盖 #14 的核心场景：长任务期间代理不会因 idle 断连。
    """

    async def slow_astream_chat(self, message, chat_id="default"):
        await asyncio.sleep(0.4)  # 比 keepalive interval 长
        yield "终于"

    from app.services import FinancialAdvisorService

    monkeypatch.setattr(FinancialAdvisorService, "astream_chat", slow_astream_chat)
    # 把 keepalive 间隔调小，加速测试
    from config import settings

    monkeypatch.setattr(settings.app, "sse_keepalive_interval", 0.1)

    with client.stream("GET", "/api/chat/stream", params={"message": "hi"}) as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes()).decode("utf-8")

    assert ": keepalive " in body, f"未发现心跳行，body={body!r}"
    assert "终于" in body  # 业务事件最后仍正确


def test_stream_agent_emits_keepalive_under_slow_harness(client, monkeypatch):
    """Agent 流式端点：慢上游 → 心跳；快事件仍完整。"""
    from app.services import FinancialAdvisorService
    from harness import HarnessEvent

    async def slow_astream_agent(self, message, chat_id="default"):
        await asyncio.sleep(0.4)
        yield HarnessEvent(type="run_end", data={"stopped_reason": "final_text"})

    monkeypatch.setattr(FinancialAdvisorService, "astream_agent", slow_astream_agent)
    from config import settings

    monkeypatch.setattr(settings.app, "sse_keepalive_interval", 0.1)

    with client.stream("GET", "/api/chat/agent/stream", params={"message": "x"}) as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes()).decode("utf-8")

    assert ": keepalive " in body
    assert "event: run_end" in body


def test_stream_rag_emits_keepalive_under_slow_pipeline(client, monkeypatch):
    """RAG 流式端点：慢上游 → 心跳。"""
    from app.services import FinancialAdvisorService
    from rag import RagEvent

    async def slow_astream_rag(self, message, chat_id="default"):
        await asyncio.sleep(0.4)
        yield RagEvent(type="done", data={"finished_at": "2026-06-07T00:00:00Z"})

    monkeypatch.setattr(FinancialAdvisorService, "astream_rag", slow_astream_rag)
    from config import settings

    monkeypatch.setattr(settings.app, "sse_keepalive_interval", 0.1)

    with client.stream("GET", "/api/chat/rag/stream", params={"message": "x"}) as resp:
        assert resp.status_code == 200
        body = b"".join(resp.iter_bytes()).decode("utf-8")

    assert ": keepalive " in body
    assert "event: done" in body

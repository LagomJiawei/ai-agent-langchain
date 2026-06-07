"""RAG 流式事件。

``RagPipeline.astream_execute()`` 发出的结构化事件，让 FastAPI 路由
能转成 SSE 输出，让前端实时看到检索 / 生成过程。

与 ``harness/events.py`` 形态平行但独立 —— RAG 是子系统，
事件语义与 Harness 完全不同（没有 turn / tool_calls 概念）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EventType = Literal[
    "retrieval_started",
    "retrieval_done",
    "generation_token",
    "done",
    "error",
]


@dataclass
class RagEvent:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)


# ----- 工厂函数 -----


def retrieval_started(original_query: str, rewritten_query: str) -> RagEvent:
    return RagEvent(
        type="retrieval_started",
        data={
            "original_query": original_query,
            "rewritten_query": rewritten_query,
        },
    )


def retrieval_done(
    doc_count: int,
    quality_score: float,
    sufficiency: str,
    titles: list[str],
    from_cache: bool = False,
) -> RagEvent:
    return RagEvent(
        type="retrieval_done",
        data={
            "doc_count": doc_count,
            "quality_score": quality_score,
            "sufficiency": sufficiency,
            "titles": titles,
            "from_cache": from_cache,
        },
    )


def generation_token(delta: str) -> RagEvent:
    return RagEvent(type="generation_token", data={"delta": delta})


def done(finished_at: str) -> RagEvent:
    return RagEvent(type="done", data={"finished_at": finished_at})


def error(message: str) -> RagEvent:
    return RagEvent(type="error", data={"message": message})


__all__ = [
    "EventType",
    "RagEvent",
    "retrieval_started",
    "retrieval_done",
    "generation_token",
    "done",
    "error",
]

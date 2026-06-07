"""RagEvent 数据类与工厂函数测试。"""
from __future__ import annotations

from rag import RagEvent
from rag.events import (
    done,
    error,
    generation_token,
    retrieval_done,
    retrieval_started,
)


def test_retrieval_started_factory():
    evt = retrieval_started(original_query="原文", rewritten_query="改写")
    assert evt.type == "retrieval_started"
    assert evt.data == {"original_query": "原文", "rewritten_query": "改写"}


def test_retrieval_done_factory():
    evt = retrieval_done(
        doc_count=3,
        quality_score=0.65,
        sufficiency="adequate",
        titles=["A", "B", "C"],
    )
    assert evt.type == "retrieval_done"
    assert evt.data["doc_count"] == 3
    assert evt.data["quality_score"] == 0.65
    assert evt.data["sufficiency"] == "adequate"
    assert evt.data["titles"] == ["A", "B", "C"]
    # from_cache 缺省为 False
    assert evt.data["from_cache"] is False


def test_retrieval_done_factory_with_from_cache_true():
    evt = retrieval_done(
        doc_count=2,
        quality_score=0.5,
        sufficiency="adequate",
        titles=["X", "Y"],
        from_cache=True,
    )
    assert evt.data["from_cache"] is True


def test_generation_token_factory():
    evt = generation_token("hello")
    assert evt.type == "generation_token"
    assert evt.data == {"delta": "hello"}


def test_done_factory():
    evt = done("2026-06-06T00:00:00+00:00")
    assert evt.type == "done"
    assert evt.data == {"finished_at": "2026-06-06T00:00:00+00:00"}


def test_error_factory():
    evt = error("boom")
    assert evt.type == "error"
    assert evt.data == {"message": "boom"}


def test_rag_event_default_data():
    evt = RagEvent(type="done")
    assert evt.data == {}

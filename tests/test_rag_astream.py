"""RagPipeline.astream_execute 事件序列测试。"""
from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from rag import RagPipeline, RagEvent, RerankStrategy


# ----- 桩 -----


class FakeRetriever:
    def retrieve(self, query: str, *args, **kwargs) -> list[Document]:
        return [
            Document(
                page_content=f"内容 {i}",
                metadata={
                    "title": f"文档{i}",
                    "rerank_score": 0.5,
                    "keyword_score": 0.6,
                },
            )
            for i in range(3)
        ]


class FakeReranker:
    def rerank(self, docs, query, strategy=RerankStrategy.HYBRID_SCORE) -> list[Document]:
        return docs


class FakeChain:
    """模拟 generation_chain.astream 行为。"""

    def __init__(self, tokens: list[str]):
        self._tokens = list(tokens)

    async def astream(self, inputs: dict[str, Any]):
        for token in self._tokens:
            yield AIMessageChunk(content=token)


def _pipeline_with_fakes(
    tokens: list[str], cache=None, retrieval_cache=None
) -> RagPipeline:
    pipe = RagPipeline.__new__(RagPipeline)
    pipe.retriever = FakeRetriever()
    pipe.reranker = FakeReranker()
    fake_chain = FakeChain(tokens)
    pipe.generation_chain = fake_chain
    pipe.cache = cache
    pipe.retrieval_cache = retrieval_cache
    pipe.llm = None  # not used when generation_chain is a fake
    return pipe


def _collect_events(pipe: RagPipeline, query: str) -> list[RagEvent]:
    async def _run():
        events = []
        async for evt in pipe.astream_execute(query):
            events.append(evt)
        return events

    return asyncio.run(_run())


# ----- 事件序列 -----


def test_astream_emits_events_in_correct_order():
    pipe = _pipeline_with_fakes(["完", "成", "。"])
    events = _collect_events(pipe, "查询")
    types = [e.type for e in events]
    assert types[0] == "retrieval_started"
    assert types[1] == "retrieval_done"
    assert "generation_token" in types
    assert types[-1] == "done"

    # retrieval_started 携带 rewritten
    assert events[0].data["original_query"] == "查询"

    # retrieval_done 携带文档质量信息
    rd = events[1]
    assert rd.data["doc_count"] == 3
    assert rd.data["quality_score"] > 0
    assert rd.data["sufficiency"] in ("adequate", "insufficient")
    assert len(rd.data["titles"]) == 3

    # 所有 generation_token 累加
    tokens = [e.data["delta"] for e in events if e.type == "generation_token"]
    assert tokens == ["完", "成", "。"]

    # done 携带时间戳
    assert events[-1].data["finished_at"]


def test_astream_skips_cache():
    """流式路径不走缓存：即使 cache.get 永远返回也是形同虚设。"""

    class CachedCache:
        def get(self, query):
            return "BUG: 缓存应被绕过"

        def put(self, query, answer):
            pass

    pipe = _pipeline_with_fakes(["正确路径"], cache=CachedCache())
    events = _collect_events(pipe, "查询")
    # 确认走了 retrieval 流程
    assert events[0].type == "retrieval_started"
    assert events[1].type == "retrieval_done"
    assert any(e.type == "generation_token" for e in events)


def test_astream_reports_error_on_llm_failure():
    """检索完成后生成阶段抛异常 → error 事件。"""

    class BoomChain:
        async def astream(self, inputs: dict[str, Any]):
            raise RuntimeError("模型炸了")
            yield  # noqa: B018  pragma: no cover

    pipe = _pipeline_with_fakes(["x"])
    pipe.generation_chain = BoomChain()
    events = _collect_events(pipe, "查询")
    # 应该先输出 retrieval 事件，再输出 error
    assert events[0].type == "retrieval_started"
    assert events[1].type == "retrieval_done"
    assert events[-1].type == "error"
    assert "模型炸了" in events[-1].data["message"]


def test_astream_generation_tokens_match_full_output():
    pipe = _pipeline_with_fakes(["基金", "定", "投", "是", "好", "方法。"])
    events = _collect_events(pipe, "查询")
    full = "".join(e.data["delta"] for e in events if e.type == "generation_token")
    assert full == "基金定投是好方法。"
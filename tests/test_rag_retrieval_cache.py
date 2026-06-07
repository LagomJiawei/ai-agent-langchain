"""RAG retrieval 级缓存测试。

策略：
- ``RetrievalCacheEntry`` 序列化 / 反序列化用纯 Python，无 Redis 依赖。
- ``RetrievalCache`` 用 in-memory 的假后端（``_FakeBackend``）测试 round-trip 和异常吞下。
- Pipeline 端到端：用 in-memory ``RetrievalCache`` + 桩 retriever / reranker，
  验证两次相同 rewritten_query 时 retriever 只被调一次。
"""
from __future__ import annotations

import asyncio
import json

from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from rag import RagPipeline, RagEvent, RetrievalCache, RetrievalCacheEntry, RerankStrategy


# ============================================================================
# RetrievalCacheEntry 序列化
# ============================================================================


def test_entry_round_trip_preserves_docs_and_metadata():
    docs = [
        Document(
            page_content="内容 A",
            metadata={
                "title": "标题 A",
                "score": 0.8,
                "rerank_score": 0.75,
                "keyword_score": 0.6,
                "source": "doc-a.md",
            },
        ),
        Document(
            page_content="内容 B",
            metadata={"title": "标题 B", "score": 0.5},
        ),
    ]
    entry = RetrievalCacheEntry(
        docs=docs,
        quality_score=0.732,
        sufficiency="adequate",
        titles=["标题 A", "标题 B"],
    )

    raw = entry.to_json()
    restored = RetrievalCacheEntry.from_json(raw)

    assert len(restored.docs) == 2
    assert restored.docs[0].page_content == "内容 A"
    assert restored.docs[0].metadata["title"] == "标题 A"
    assert restored.docs[0].metadata["score"] == 0.8
    assert restored.docs[0].metadata["source"] == "doc-a.md"
    assert restored.docs[1].page_content == "内容 B"
    assert restored.quality_score == 0.732
    assert restored.sufficiency == "adequate"
    assert restored.titles == ["标题 A", "标题 B"]


def test_entry_round_trip_handles_empty_docs():
    entry = RetrievalCacheEntry(
        docs=[],
        quality_score=0.0,
        sufficiency="insufficient",
        titles=[],
    )
    restored = RetrievalCacheEntry.from_json(entry.to_json())
    assert restored.docs == []
    assert restored.quality_score == 0.0
    assert restored.sufficiency == "insufficient"


# ============================================================================
# RetrievalCache（假后端）
# ============================================================================


class _FakeBackend:
    """模拟 RagResultSemanticCache 的最小 string KV 接口。"""

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self.store: dict[str, str] = {}
        # 模拟 prefix 行为：实际只是不带前缀的 KV
        self.put_calls: list[tuple[str, str]] = []
        self.get_calls: list[str] = []

    def get(self, query):
        self.get_calls.append(query)
        if not self._enabled:
            return None
        return self.store.get(query)

    def put(self, query, value):
        self.put_calls.append((query, value))
        if not self._enabled:
            return
        self.store[query] = value


def test_retrieval_cache_put_then_get():
    backend = _FakeBackend()
    cache = RetrievalCache(backend)
    entry = RetrievalCacheEntry(
        docs=[Document(page_content="x", metadata={"title": "t"})],
        quality_score=0.5,
        sufficiency="adequate",
        titles=["t"],
    )
    cache.put("q1", entry)
    restored = cache.get("q1")

    assert restored is not None
    assert restored.docs[0].page_content == "x"
    assert restored.quality_score == 0.5


def test_retrieval_cache_get_miss_returns_none():
    cache = RetrievalCache(_FakeBackend())
    assert cache.get("never-stored") is None


def test_retrieval_cache_get_corrupt_json_returns_none():
    """缓存里塞了非法 JSON（旧 schema 残留），按未命中处理不抛异常。"""
    backend = _FakeBackend()
    backend.store["bad-key"] = "this is not json {{{"
    cache = RetrievalCache(backend)
    assert cache.get("bad-key") is None


def test_retrieval_cache_disabled_backend_no_op():
    backend = _FakeBackend(enabled=False)
    cache = RetrievalCache(backend)
    assert cache.enabled is False
    cache.put("q", RetrievalCacheEntry(docs=[], quality_score=0.0, sufficiency="insufficient"))
    assert cache.get("q") is None


# ============================================================================
# Pipeline 端到端：retrieval cache 命中跳过 retriever
# ============================================================================


class _CountingRetriever:
    def __init__(self, docs):
        self._docs = docs
        self.call_count = 0
        self.last_query: str | None = None

    def retrieve(self, query, *args, **kwargs):
        self.call_count += 1
        self.last_query = query
        return list(self._docs)


class _PassthroughReranker:
    def __init__(self):
        self.call_count = 0

    def rerank(self, docs, query, strategy=RerankStrategy.HYBRID_SCORE):
        self.call_count += 1
        return list(docs)


class _ScriptedChain:
    """generation_chain stub：astream + invoke 都支持。"""

    def __init__(self, tokens: list[str]):
        self._tokens = list(tokens)
        self.astream_count = 0
        self.invoke_count = 0

    async def astream(self, inputs):
        self.astream_count += 1
        for t in self._tokens:
            yield AIMessageChunk(content=t)

    def invoke(self, inputs):
        self.invoke_count += 1
        return AIMessageChunk(content="".join(self._tokens))


def _docs(n=2):
    return [
        Document(
            page_content=f"内容 {i}",
            metadata={
                "title": f"文档{i}",
                "rerank_score": 0.5,
                "keyword_score": 0.6,
            },
        )
        for i in range(n)
    ]


def _build_pipeline(
    *,
    retriever=None,
    reranker=None,
    chain=None,
    cache=None,
    retrieval_cache=None,
):
    pipe = RagPipeline.__new__(RagPipeline)
    pipe.retriever = retriever or _CountingRetriever(_docs())
    pipe.reranker = reranker or _PassthroughReranker()
    pipe.generation_chain = chain or _ScriptedChain(["ok"])
    pipe.cache = cache
    pipe.retrieval_cache = retrieval_cache
    pipe.llm = None
    return pipe


def test_astream_uses_retrieval_cache_on_hit(monkeypatch):
    """同一 rewritten_query 第二次调 astream，retriever 不被再次调用，from_cache=True。"""
    from rag import pipeline as pipeline_mod
    from rag.transformer import QueryTransformResult

    # 锁定 rewritten_query，避免真调 LLM 重写
    def fake_transform(q):
        return QueryTransformResult(original_query=q, translated_query=None, rewritten_query="ANCHOR")

    monkeypatch.setattr(pipeline_mod, "transform_query_for_retrieval", fake_transform)

    retriever = _CountingRetriever(_docs())
    reranker = _PassthroughReranker()
    chain = _ScriptedChain(["token-A", "token-B"])
    rcache = RetrievalCache(_FakeBackend())
    pipe = _build_pipeline(
        retriever=retriever, reranker=reranker, chain=chain, retrieval_cache=rcache
    )

    async def _collect(q):
        out: list[RagEvent] = []
        async for e in pipe.astream_execute(q):
            out.append(e)
        return out

    # 第一次：miss，retriever / reranker 各被调一次
    ev1 = asyncio.run(_collect("用户问题 1"))
    rd1 = next(e for e in ev1 if e.type == "retrieval_done")
    assert rd1.data["from_cache"] is False
    assert retriever.call_count == 1
    assert reranker.call_count == 1

    # 第二次：hit，retriever / reranker 不再被调
    ev2 = asyncio.run(_collect("用户问题 2"))
    rd2 = next(e for e in ev2 if e.type == "retrieval_done")
    assert rd2.data["from_cache"] is True
    assert retriever.call_count == 1, "retriever 被重复调用了"
    assert reranker.call_count == 1, "reranker 被重复调用了"

    # 生成阶段两次都跑（流式仍正常发 token）
    assert chain.astream_count == 2
    tokens1 = [e.data["delta"] for e in ev1 if e.type == "generation_token"]
    tokens2 = [e.data["delta"] for e in ev2 if e.type == "generation_token"]
    assert tokens1 == ["token-A", "token-B"]
    assert tokens2 == ["token-A", "token-B"]


def test_astream_writes_retrieval_cache_on_miss(monkeypatch):
    """第一次 astream 调用应把 retrieval 结果写入缓存。"""
    from rag import pipeline as pipeline_mod
    from rag.transformer import QueryTransformResult

    def fake_transform(q):
        return QueryTransformResult(original_query=q, translated_query=None, rewritten_query="KEY-WRITE")

    monkeypatch.setattr(pipeline_mod, "transform_query_for_retrieval", fake_transform)

    backend = _FakeBackend()
    rcache = RetrievalCache(backend)
    pipe = _build_pipeline(retrieval_cache=rcache, chain=_ScriptedChain(["x"]))

    async def _drain():
        async for _ in pipe.astream_execute("Q"):
            pass

    asyncio.run(_drain())

    # backend 里有 KEY-WRITE 这个键
    assert "KEY-WRITE" in backend.store
    # 拿回的 entry 至少包含 docs
    restored = RetrievalCacheEntry.from_json(backend.store["KEY-WRITE"])
    assert len(restored.docs) == 2


def test_astream_does_not_write_answer_cache(monkeypatch):
    """流式路径**不**写 answer cache（保持 token 流体验）。"""
    from rag import pipeline as pipeline_mod
    from rag.transformer import QueryTransformResult

    def fake_transform(q):
        return QueryTransformResult(original_query=q, translated_query=None, rewritten_query="K")

    monkeypatch.setattr(pipeline_mod, "transform_query_for_retrieval", fake_transform)

    answer_backend = _FakeBackend()
    # answer cache 直接复用 _FakeBackend 接口（put / get 兼容）
    pipe = _build_pipeline(
        cache=answer_backend,
        retrieval_cache=RetrievalCache(_FakeBackend()),
        chain=_ScriptedChain(["x"]),
    )

    async def _drain():
        async for _ in pipe.astream_execute("Q"):
            pass

    asyncio.run(_drain())

    # answer cache 没有任何写入
    assert answer_backend.put_calls == []
    assert answer_backend.store == {}


def test_execute_uses_both_caches(monkeypatch):
    """同步 execute：answer 缓存命中直接返回；未命中走 retrieval 缓存。"""
    from rag import pipeline as pipeline_mod
    from rag.transformer import QueryTransformResult

    rewritten_map = {"q-a": "REWRITE-X", "q-b": "REWRITE-X"}

    def fake_transform(q):
        return QueryTransformResult(
            original_query=q,
            translated_query=None,
            rewritten_query=rewritten_map.get(q, q),
        )

    monkeypatch.setattr(pipeline_mod, "transform_query_for_retrieval", fake_transform)

    retriever = _CountingRetriever(_docs())
    reranker = _PassthroughReranker()
    chain = _ScriptedChain(["最终答案"])
    answer_backend = _FakeBackend()
    rcache = RetrievalCache(_FakeBackend())

    pipe = _build_pipeline(
        retriever=retriever,
        reranker=reranker,
        chain=chain,
        cache=answer_backend,
        retrieval_cache=rcache,
    )

    # 第一次 q-a：answer miss → retrieval miss → 生成 → 写两层
    out1 = pipe.execute("q-a")
    assert out1 == "最终答案"
    assert retriever.call_count == 1
    assert chain.invoke_count == 1
    assert answer_backend.store  # 写了 answer
    assert "REWRITE-X" in rcache._backend.store  # 写了 retrieval

    # 第二次 q-a：answer hit → 直接返回，retriever / generation 都不调
    out2 = pipe.execute("q-a")
    assert out2 == "最终答案"
    assert retriever.call_count == 1
    assert chain.invoke_count == 1

    # 第三次 q-b：answer miss（key 不同），retrieval hit（rewritten 相同），
    # generation 仍要调
    out3 = pipe.execute("q-b")
    assert out3 == "最终答案"
    assert retriever.call_count == 1, "retrieval cache 应命中"
    assert chain.invoke_count == 2  # 生成又跑了一次

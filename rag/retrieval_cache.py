"""RAG retrieval 级缓存。

与 ``RagResultSemanticCache`` 的关系：
- ``RagResultSemanticCache``：缓存最终 answer（同步路径用，流式跳过）。
- ``RetrievalCache``（本模块）：缓存 retrieval + rerank 结果，
  **流式 / 同步 路径都用**，因为 retrieval 是 RAG 中最贵也最稳定的部分。

Key 策略：用 ``rewritten_query``（已经过 transformer 规范化）做 Redis key，
而不是用户原始 query。Value 是 ``RetrievalCacheEntry`` 的 JSON 序列化形态。

失败兜底：底层 Redis 读写异常全部吞掉，永远不影响主流程。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from langchain_core.documents import Document
from loguru import logger

from .cache import RagResultSemanticCache


@dataclass
class RetrievalCacheEntry:
    """retrieval + rerank 后的全部产物，可 JSON 序列化。"""

    docs: List[Document]
    quality_score: float
    sufficiency: Literal["adequate", "insufficient"]
    titles: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        payload = {
            "docs": [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in self.docs
            ],
            "quality_score": self.quality_score,
            "sufficiency": self.sufficiency,
            "titles": self.titles,
        }
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "RetrievalCacheEntry":
        payload = json.loads(raw)
        docs = [
            Document(page_content=d["page_content"], metadata=d.get("metadata") or {})
            for d in payload.get("docs", [])
        ]
        return cls(
            docs=docs,
            quality_score=float(payload.get("quality_score", 0.0)),
            sufficiency=payload.get("sufficiency", "insufficient"),
            titles=list(payload.get("titles") or []),
        )


class RetrievalCache:
    """retrieval 结果缓存。

    薄包装在一个独立 prefix 的 ``RagResultSemanticCache`` 上，复用同一个 Redis
    连接配置；底层只是 string→string，本类负责 JSON (反)序列化和异常吞下。
    """

    def __init__(self, backend: RagResultSemanticCache) -> None:
        self._backend = backend

    @property
    def enabled(self) -> bool:
        return self._backend._enabled

    def get(self, rewritten_query: str) -> Optional[RetrievalCacheEntry]:
        raw = self._backend.get(rewritten_query)
        if raw is None:
            return None
        try:
            return RetrievalCacheEntry.from_json(raw)
        except Exception as exc:  # noqa: BLE001
            # 缓存里塞了坏 JSON（比如 schema 升级前的旧值），治不好就当未命中
            logger.warning(f"RetrievalCache 反序列化失败，按未命中处理: {exc}")
            return None

    def put(self, rewritten_query: str, entry: RetrievalCacheEntry) -> None:
        try:
            self._backend.put(rewritten_query, entry.to_json())
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"RetrievalCache 写入失败（已忽略）: {exc}")


_retrieval_cache: Optional[RetrievalCache] = None


def get_retrieval_cache() -> Optional[RetrievalCache]:
    """返回全局 retrieval cache 单例；Redis 未启用时返回 None。"""
    global _retrieval_cache
    if _retrieval_cache is not None:
        return _retrieval_cache

    from config.settings import settings

    if not settings.redis.enabled:
        return None

    backend = RagResultSemanticCache(
        host=settings.redis.host,
        port=settings.redis.port,
        db=settings.redis.db,
        password=settings.redis.password,
        prefix="rag_cache:retrieval:",
        ttl=settings.rag.retrieval_cache_ttl,
    )
    _retrieval_cache = RetrievalCache(backend)
    return _retrieval_cache


def reset_retrieval_cache() -> None:
    """仅供测试：清掉单例，下次 get 重新构造。"""
    global _retrieval_cache
    _retrieval_cache = None


__all__ = [
    "RetrievalCacheEntry",
    "RetrievalCache",
    "get_retrieval_cache",
    "reset_retrieval_cache",
]

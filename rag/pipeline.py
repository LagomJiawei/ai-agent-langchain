"""
RAG 流水线
"""
import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, List, Literal, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from config import create_chat_model, settings
from harness._message_utils import extract_chunk_text

from . import events as _events
from .cache import RagResultSemanticCache, get_semantic_cache
from .events import RagEvent
from .reranker import DocumentReranker, RerankStrategy, get_document_reranker
from .retrieval_cache import RetrievalCache, RetrievalCacheEntry, get_retrieval_cache
from .retriever import DocumentRetriever, get_document_retriever
from .transformer import transform_query_for_retrieval


@dataclass
class RagStepResult:
    """RAG 分步执行结果"""

    original_query: str
    translated_query: Optional[str] = None
    rewritten_query: Optional[str] = None
    retrieved_docs: Optional[List[Document]] = None
    final_answer: Optional[str] = None


# ----- 公开质量评估 -----


def calculate_quality_score(docs: List[Document]) -> float:
    """根据 rerank / keyword 分数与命中数估算整体质量分。"""
    if not docs:
        return 0.0

    rerank_scores = [
        float(doc.metadata.get("rerank_score", doc.metadata.get("score", 0.0)))
        for doc in docs
    ]
    keyword_scores = [float(doc.metadata.get("keyword_score", 0.0)) for doc in docs]
    avg_rerank_score = sum(rerank_scores) / len(rerank_scores)
    best_keyword_score = max(keyword_scores) if keyword_scores else 0.0
    count_score = min(len(docs) / settings.rag.top_k, 1.0)
    return round(
        min(avg_rerank_score * 0.65 + best_keyword_score * 0.2 + count_score * 0.15, 1.0),
        3,
    )


def judge_sufficiency(
    docs: List[Document], quality_score: float
) -> Literal["adequate", "insufficient"]:
    """根据 doc 数与质量分判定信息充足性，阈值取自 ``settings.rag.quality_threshold``。"""
    if docs and quality_score >= settings.rag.quality_threshold:
        return "adequate"
    return "insufficient"


class RagPipeline:
    """串行 RAG 流水线"""

    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        reranker: Optional[DocumentReranker] = None,
        cache: Optional[RagResultSemanticCache] = None,
        retrieval_cache: Optional[RetrievalCache] = None,
    ):
        self.retriever = retriever or get_document_retriever()
        self.reranker = reranker or get_document_reranker()
        self.cache = cache or get_semantic_cache()
        # retrieval cache 与 answer cache 独立：流式 / 同步路径都用
        self.retrieval_cache = (
            retrieval_cache if retrieval_cache is not None else get_retrieval_cache()
        )
        self.llm = create_chat_model(temperature=0.7)

        self.generation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一位专业的理财顾问。请基于以下参考资料回答用户的问题。

要求：
1. 只能基于参考资料回答，不要编造信息
2. 如果参考资料中没有相关信息，明确告知用户无法回答
3. 回答要专业、准确、有条理
4. 用中文回答""",
                ),
                (
                    "human",
                    """【参考资料】
{context}

【用户问题】
{query}

请回答：""",
                ),
            ]
        )

        self.generation_chain = self.generation_prompt | self.llm

    # ------------------------------------------------------------------
    # retrieval（含缓存）—— execute / astream_execute / execute_with_steps 复用
    # ------------------------------------------------------------------

    def _run_retrieval(
        self,
        rewritten_query: str,
        *,
        enable_bilingual_fallback: bool = False,
        rerank_strategy: RerankStrategy = RerankStrategy.HYBRID_SCORE,
        use_cache: bool = True,
    ) -> tuple[RetrievalCacheEntry, bool]:
        """执行检索 + 重排，带 retrieval-level 缓存。

        Returns:
            (entry, from_cache)：``entry`` 是 ``RetrievalCacheEntry``；
            ``from_cache=True`` 表示命中缓存，调用方可据此设置事件标记。
        """
        if use_cache and self.retrieval_cache is not None:
            cached = self.retrieval_cache.get(rewritten_query)
            if cached is not None:
                logger.info(
                    f"retrieval cache 命中: {rewritten_query[:50]}... "
                    f"({len(cached.docs)} docs, quality={cached.quality_score})"
                )
                return cached, True

        docs = self.retriever.retrieve(rewritten_query, enable_bilingual_fallback)
        reranked = self.reranker.rerank(docs, rewritten_query, rerank_strategy)
        final_docs = reranked[: settings.rag.top_k]
        quality = calculate_quality_score(final_docs)
        sufficiency = judge_sufficiency(final_docs, quality)
        titles = [(doc.metadata.get("title") or "")[:80] for doc in final_docs]

        entry = RetrievalCacheEntry(
            docs=final_docs,
            quality_score=quality,
            sufficiency=sufficiency,
            titles=titles,
        )

        if use_cache and self.retrieval_cache is not None:
            self.retrieval_cache.put(rewritten_query, entry)

        return entry, False

    # ------------------------------------------------------------------
    # 公开入口
    # ------------------------------------------------------------------

    def execute(
        self,
        query: str,
        enable_bilingual_fallback: bool = False,
        rerank_strategy: RerankStrategy = RerankStrategy.HYBRID_SCORE,
    ) -> str:
        """
        执行完整的 RAG 流程

        Args:
            query: 用户查询
            enable_bilingual_fallback: 是否启用双语 fallback
            rerank_strategy: 重排序策略

        Returns:
            最终答案
        """
        start_time = time.time()
        logger.info(f"开始 RAG 流程: {query}")

        # 1. answer 缓存检查（最优先，命中直接返回）
        if self.cache:
            cached = self.cache.get(query)
            if cached:
                logger.info(f"answer 缓存命中，总耗时: {(time.time() - start_time) * 1000:.0f}ms")
                return cached

        # 2. 查询变换
        transform_result = transform_query_for_retrieval(query)
        rewritten = transform_result.rewritten_query

        # 3. 检索 + 重排（走 retrieval 缓存）
        entry, from_cache = self._run_retrieval(
            rewritten,
            enable_bilingual_fallback=enable_bilingual_fallback,
            rerank_strategy=rerank_strategy,
        )
        final_docs = entry.docs

        # 4. 格式化上下文 + 生成答案
        context = self._format_context(final_docs)
        answer = self.generation_chain.invoke({"context": context, "query": query})
        final_answer = answer.content

        # 5. 写入 answer 缓存
        if self.cache:
            self.cache.put(query, final_answer)

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"RAG 流程完成，总耗时: {elapsed:.0f}ms，上下文: {len(final_docs)} 条"
            f"，retrieval_cache={'hit' if from_cache else 'miss'}"
        )

        return final_answer

    async def astream_execute(
        self, query: str
    ) -> AsyncIterator[RagEvent]:
        """流式执行：发出阶段事件 + 生成 token。

        与 ``execute`` 的区别：
        - **不走 answer 缓存**：流式语义下用户期望看到生成全过程（吐 token）。
        - **走 retrieval 缓存**：retrieval + rerank 是稳定且最贵的部分，
          命中时 ``retrieval_done.from_cache=True``，token 流仍正常发。
        - 检索 / 重排同步实现走 ``asyncio.to_thread``，避免阻塞 event loop。
        - 生成阶段用 ``generation_chain.astream`` 拿原生 token chunk。
        """
        logger.info(f"开始 RAG 流式执行: {query[:50]}...")
        try:
            transform_result = await asyncio.to_thread(
                transform_query_for_retrieval, query
            )
            rewritten = transform_result.rewritten_query
            yield _events.retrieval_started(
                original_query=query, rewritten_query=rewritten
            )

            entry, from_cache = await asyncio.to_thread(
                self._run_retrieval, rewritten
            )
            final_docs = entry.docs

            yield _events.retrieval_done(
                doc_count=len(final_docs),
                quality_score=entry.quality_score,
                sufficiency=entry.sufficiency,
                titles=entry.titles,
                from_cache=from_cache,
            )

            context = self._format_context(final_docs)
            async for chunk in self.generation_chain.astream(
                {"context": context, "query": query}
            ):
                delta = extract_chunk_text(chunk)
                if delta:
                    yield _events.generation_token(delta)

            yield _events.done(
                finished_at=datetime.now(timezone.utc).isoformat()
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"RAG 流式执行异常: {exc}")
            yield _events.error(str(exc))

    def execute_with_steps(self, query: str) -> RagStepResult:
        """分步执行（用于调试）"""
        logger.info(f"RAG 分步执行: {query}")

        result = RagStepResult(original_query=query)

        transform_result = transform_query_for_retrieval(query)
        result.translated_query = transform_result.translated_query
        result.rewritten_query = transform_result.rewritten_query

        # 检索（走 retrieval 缓存）
        entry, _ = self._run_retrieval(result.rewritten_query)
        result.retrieved_docs = entry.docs

        # 生成
        context = self._format_context(result.retrieved_docs)
        answer = self.generation_chain.invoke({"context": context, "query": query})
        result.final_answer = answer.content

        return result

    def _format_context(self, docs: List[Document]) -> str:
        """格式化检索结果"""
        if not docs:
            return "无相关文档"

        parts = []
        for i, doc in enumerate(docs, 1):
            header = f"[文档 {i}]"
            if "title" in doc.metadata:
                header += f" [标题: {doc.metadata['title']}]"
            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    def retrieve_context_only(self, query: str) -> str:
        """仅检索上下文（用于流式生成）"""
        transform_result = transform_query_for_retrieval(query)
        rewritten = transform_result.rewritten_query
        docs = self.retriever.retrieve(rewritten)
        reranked = self.reranker.rerank(docs, rewritten, RerankStrategy.HYBRID_SCORE)
        return self._format_context(reranked[: settings.rag.top_k])


# 全局实例
_rag_pipeline: Optional[RagPipeline] = None


def get_rag_pipeline() -> RagPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RagPipeline()
    return _rag_pipeline

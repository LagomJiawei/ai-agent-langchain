"""
RAG 流水线
"""
import time
from typing import Optional, List
from dataclasses import dataclass
from loguru import logger
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from config import create_chat_model, settings
from .transformer import transform_query_for_retrieval
from .retriever import DocumentRetriever, get_document_retriever
from .reranker import DocumentReranker, get_document_reranker, RerankStrategy
from .cache import RagResultSemanticCache, get_semantic_cache


@dataclass
class RagStepResult:
    """RAG 分步执行结果"""

    original_query: str
    translated_query: Optional[str] = None
    rewritten_query: Optional[str] = None
    retrieved_docs: Optional[List[Document]] = None
    final_answer: Optional[str] = None


class RagPipeline:
    """串行 RAG 流水线"""

    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        reranker: Optional[DocumentReranker] = None,
        cache: Optional[RagResultSemanticCache] = None,
    ):
        self.retriever = retriever or get_document_retriever()
        self.reranker = reranker or get_document_reranker()
        self.cache = cache or get_semantic_cache()
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

        # 1. 缓存检查
        if self.cache:
            cached = self.cache.get(query)
            if cached:
                logger.info(f"缓存命中，总耗时: {(time.time() - start_time) * 1000:.0f}ms")
                return cached

        # 2. 查询变换
        transform_result = transform_query_for_retrieval(query)
        rewritten = transform_result.rewritten_query

        # 3. 文档检索
        docs = self.retriever.retrieve(rewritten, enable_bilingual_fallback)

        # 5. 重排序
        reranked_docs = self.reranker.rerank(docs, rewritten, rerank_strategy)
        final_docs = reranked_docs[: settings.rag.top_k]

        # 6. 格式化上下文
        context = self._format_context(final_docs)

        # 7. 生成答案
        answer = self.generation_chain.invoke({"context": context, "query": query})
        final_answer = answer.content

        # 8. 写入缓存
        if self.cache:
            self.cache.put(query, final_answer)

        elapsed = (time.time() - start_time) * 1000
        logger.info(f"RAG 流程完成，总耗时: {elapsed:.0f}ms，候选命中: {len(docs)} 条，上下文: {len(final_docs)} 条")

        return final_answer

    def execute_with_steps(self, query: str) -> RagStepResult:
        """分步执行（用于调试）"""
        logger.info(f"RAG 分步执行: {query}")

        result = RagStepResult(original_query=query)

        transform_result = transform_query_for_retrieval(query)
        result.translated_query = transform_result.translated_query
        result.rewritten_query = transform_result.rewritten_query

        # 检索
        docs = self.retriever.retrieve(result.rewritten_query)
        result.retrieved_docs = self.reranker.rerank(docs, result.rewritten_query, RerankStrategy.HYBRID_SCORE)[: settings.rag.top_k]

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


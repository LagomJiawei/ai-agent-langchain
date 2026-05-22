"""
知识库检索工具
"""
import json
from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from config import settings
from rag import RerankStrategy, get_document_reranker, get_document_retriever, transform_query_for_retrieval
from .rate_limiter import rate_limited


def _format_context(docs: List[Document]) -> str:
    if not docs:
        return "无相关文档"

    parts = []
    for i, doc in enumerate(docs, 1):
        header = f"[文档 {i}]"
        if "title" in doc.metadata:
            header += f" [标题: {doc.metadata['title']}]"
        if "source" in doc.metadata:
            header += f" [来源: {doc.metadata['source']}]"
        parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


def _calculate_quality_score(docs: List[Document]) -> float:
    if not docs:
        return 0.0

    rerank_scores = [float(doc.metadata.get("rerank_score", doc.metadata.get("score", 0.0))) for doc in docs]
    keyword_scores = [float(doc.metadata.get("keyword_score", 0.0)) for doc in docs]
    avg_rerank_score = sum(rerank_scores) / len(rerank_scores)
    best_keyword_score = max(keyword_scores) if keyword_scores else 0.0
    count_score = min(len(docs) / settings.rag.top_k, 1.0)
    return round(min(avg_rerank_score * 0.65 + best_keyword_score * 0.2 + count_score * 0.15, 1.0), 3)


@tool
@rate_limited("knowledge_base_search")
def search_knowledge_base(query: str) -> str:
    """
    搜索本地理财知识库，获取与用户问题相关的参考资料。

    Args:
        query: 检索查询，复杂问题应拆分成更具体的子查询

    Returns:
        包含检索上下文、文档数量、质量分数和信息充足性判断的 JSON 字符串
    """
    logger.info(f"执行知识库检索: {query}")

    transform_result = transform_query_for_retrieval(query)
    rewritten_query = transform_result.rewritten_query

    retriever = get_document_retriever()
    reranker = get_document_reranker()
    docs = retriever.retrieve(rewritten_query)
    reranked_docs = reranker.rerank(docs, rewritten_query, RerankStrategy.HYBRID_SCORE)
    final_docs = reranked_docs[: settings.rag.top_k]

    quality_score = _calculate_quality_score(final_docs)
    sufficiency = "adequate" if final_docs and quality_score >= settings.rag.quality_threshold else "insufficient"

    return json.dumps(
        {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "doc_count": len(final_docs),
            "quality_score": quality_score,
            "sufficiency": sufficiency,
            "context": _format_context(final_docs),
        },
        ensure_ascii=False,
        indent=2,
    )

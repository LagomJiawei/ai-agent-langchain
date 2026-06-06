"""
知识库检索工具
"""
import json
from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from config import settings
from rag import (
    RerankStrategy,
    calculate_quality_score,
    get_document_reranker,
    get_document_retriever,
    judge_sufficiency,
    transform_query_for_retrieval,
)


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


@tool
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

    quality_score = calculate_quality_score(final_docs)
    sufficiency = judge_sufficiency(final_docs, quality_score)

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

"""
文档检索器
支持双语 fallback、结果去重
"""
import hashlib
from typing import List, Optional, Tuple
from loguru import logger
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from config import create_vector_store, settings


class DocumentRetriever:
    """文档检索器"""

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.vector_store = vector_store or create_vector_store()
        self.top_k = settings.rag.top_k
        self.recall_k = settings.rag.recall_k

    def _hash_text(self, text: str) -> str:
        """计算文本哈希用于去重"""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def retrieve(
        self,
        query: str,
        enable_bilingual_fallback: bool = False,
        enable_rerank: bool = True,
    ) -> List[Document]:
        """
        检索相关文档

        Args:
            query: 查询文本
            enable_bilingual_fallback: 是否启用双语 fallback 检索
            enable_rerank: 是否启用重排序

        Returns:
            检索到的文档列表
        """
        logger.debug(f"检索查询: {query}")

        # 主检索
        main_results = self._retrieve_with_score(query, self.recall_k)
        logger.debug(f"主检索命中: {len(main_results)} 条")

        # 双语 fallback 检索
        if enable_bilingual_fallback and len(main_results) < self.recall_k // 2:
            logger.info("主检索结果较少，尝试 fallback 检索")
            fallback_results = self._retrieve_with_score(query, self.recall_k)
            # 合并去重
            merged = self._merge_deduplicate(main_results, fallback_results)
            logger.debug(f"合并后共: {len(merged)} 条")
        else:
            merged = main_results

        logger.debug(f"去重后候选: {len(merged)} 条")

        return [doc for doc, _ in merged[: self.recall_k]]

    def _retrieve_with_score(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """执行检索并返回带分数的结果"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            # LangChain FAISS 返回 (Document, distance)，需要转换为相似度分数
            # 距离越小越相似，所以转换为 1 / (1 + distance)
            normalized = []
            for rank, (doc, score) in enumerate(results, 1):
                # FAISS 使用 L2 距离，归一化到 [0,1]
                if score < 0:
                    # 某些向量库直接返回相似度
                    normalized_score = score
                else:
                    # 将距离转换为相似度分数
                    normalized_score = 1.0 / (1.0 + score)
                metadata = dict(doc.metadata)
                metadata.update(
                    {
                        "score": normalized_score,
                        "vector_score": normalized_score,
                        "vector_distance": float(score),
                        "retrieval_rank": rank,
                    }
                )
                normalized.append((Document(page_content=doc.page_content, metadata=metadata), normalized_score))
            return normalized
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return []

    def _merge_deduplicate(
        self,
        results1: List[Tuple[Document, float]],
        results2: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """合并两个检索结果并去重，保留较高分数"""
        seen = {}

        # 主检索结果优先级更高
        for doc, score in results1:
            doc_hash = self._hash_text(doc.page_content)
            if doc_hash not in seen or score > seen[doc_hash][1]:
                seen[doc_hash] = (doc, score)

        # fallback 结果
        for doc, score in results2:
            doc_hash = self._hash_text(doc.page_content)
            if doc_hash not in seen:
                seen[doc_hash] = (doc, score)

        # 按分数降序排序
        merged = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        return merged

    def retrieve_context(self, query: str, enable_bilingual_fallback: bool = False) -> str:
        """
        检索并格式化上下文

        Args:
            query: 查询文本
            enable_bilingual_fallback: 是否启用双语 fallback

        Returns:
            格式化的上下文字符串
        """
        docs = self.retrieve(query, enable_bilingual_fallback)

        if not docs:
            return "无相关文档"

        context_parts = []
        for i, doc in enumerate(docs, 1):
            header = f"[文档 {i}]"
            if "title" in doc.metadata:
                header += f" [标题: {doc.metadata['title']}]"
            if "source" in doc.metadata:
                header += f" [来源: {doc.metadata['source']}]"

            context_parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(context_parts)


# 单例实例
_retriever: Optional[DocumentRetriever] = None


def get_document_retriever() -> DocumentRetriever:
    global _retriever
    if _retriever is None:
        _retriever = DocumentRetriever()
    return _retriever

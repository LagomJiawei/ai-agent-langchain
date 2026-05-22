"""
文档重排序器
支持多种重排序策略：余弦相似度、BM25、混合分数、互惠排名融合
"""
from enum import Enum
from typing import List, Optional
import re
from collections import defaultdict
from loguru import logger
from langchain_core.documents import Document


class RerankStrategy(str, Enum):
    """重排序策略"""

    COSINE_SIMILARITY = "cosine"  # 余弦相似度
    BM25 = "bm25"  # BM25 关键词匹配
    HYBRID_SCORE = "hybrid"  # 混合加权
    RRF = "rrf"  # 互惠排名融合


class DocumentReranker:
    """文档重排序器"""

    def __init__(self):
        self.stop_words = self._load_stop_words()

    def _load_stop_words(self) -> set:
        """加载中文停用词（简化版）"""
        return {
            "的", "了", "在", "是", "我", "有", "和", "就",
            "不", "人", "都", "一", "一个", "上", "也", "很",
            "到", "说", "要", "去", "你", "会", "着", "没有",
            "看", "好", "自己", "这", "那", "吗", "吧", "呢",
            "啊", "什么", "怎么", "为什么", "如何", "可以", "需要",
        }

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（按单字 + 两字词语）"""
        # 清理非中文字符
        text = re.sub(r"[^一-龥a-zA-Z0-9]", " ", text)
        # 单字分词
        chars = [c for c in text if c.strip()]
        # 两词组合
        words = []
        for i in range(len(chars) - 1):
            words.append(chars[i] + chars[i + 1])
        words.extend(chars)
        # 去停用词
        return [w for w in words if w not in self.stop_words]

    def _calculate_bm25_score(
        self,
        query: str,
        document: str,
        k1: float = 1.5,
        b: float = 0.75,
        avg_doc_len: float = 200.0,
    ) -> float:
        """计算 BM25 分数"""
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(document)

        if not doc_tokens:
            return 0.0

        doc_len = len(doc_tokens)

        # 计算文档词频
        tf = defaultdict(int)
        for token in doc_tokens:
            tf[token] += 1

        score = 0.0
        for token in query_tokens:
            if token in tf:
                # BM25 公式简化版
                idf = 1.0  # 简化，实际需要文档集合的 IDF
                tf_score = tf[token] * (k1 + 1) / (
                    tf[token] + k1 * (1 - b + b * doc_len / avg_doc_len)
                )
                score += idf * tf_score

        return score / (doc_len + 1e-8)  # 归一化

    def _calculate_keyword_overlap_score(self, query: str, document: str) -> float:
        """计算关键词重叠分数"""
        query_tokens = set(self._tokenize(query))
        doc_tokens = set(self._tokenize(document))

        if not query_tokens:
            return 0.0

        intersection = query_tokens & doc_tokens
        return len(intersection) / len(query_tokens)

    def rerank(
        self,
        documents: List[Document],
        query: str,
        strategy: RerankStrategy = RerankStrategy.HYBRID_SCORE,
    ) -> List[Document]:
        """
        重排序文档

        Args:
            documents: 待排序的文档列表
            query: 查询文本
            strategy: 重排序策略

        Returns:
            重排序后的文档列表
        """
        if not documents:
            return []

        logger.debug(f"重排序文档数: {len(documents)}, 策略: {strategy}")

        # 计算各种分数
        scores: List[tuple[Document, dict]] = []
        for doc in documents:
            vector_score = float(doc.metadata.get("vector_score", doc.metadata.get("score", 0.5)))
            bm25_score = min(self._calculate_bm25_score(query, doc.page_content), 1.0)
            keyword_score = self._calculate_keyword_overlap_score(query, doc.page_content)

            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "score": vector_score,
                    "vector_score": vector_score,
                    "bm25_score": bm25_score,
                    "keyword_score": keyword_score,
                    "rerank_strategy": strategy.value,
                }
            )
            scored_doc = Document(page_content=doc.page_content, metadata=metadata)

            scores.append(
                (
                    scored_doc,
                    {
                        "vector": vector_score,
                        "bm25": bm25_score,
                        "keyword": keyword_score,
                    },
                )
            )

        # 根据策略排序
        if strategy == RerankStrategy.COSINE_SIMILARITY:
            scored = [(doc, s["vector"]) for doc, s in scores]
            scored.sort(key=lambda x: x[1], reverse=True)
            sorted_docs = []
            for doc, rerank_score in scored:
                metadata = dict(doc.metadata)
                metadata["rerank_score"] = rerank_score
                sorted_docs.append(Document(page_content=doc.page_content, metadata=metadata))
        elif strategy == RerankStrategy.BM25:
            scores.sort(key=lambda x: x[1]["bm25"], reverse=True)
            sorted_docs = []
            for doc, s in scores:
                metadata = dict(doc.metadata)
                metadata["rerank_score"] = s["bm25"]
                sorted_docs.append(Document(page_content=doc.page_content, metadata=metadata))
        elif strategy == RerankStrategy.HYBRID_SCORE:
            # 混合加权：向量 60% + BM25 30% + 关键词 10%
            scored = []
            for doc, s in scores:
                hybrid = s["vector"] * 0.6 + s["bm25"] * 0.3 + s["keyword"] * 0.1
                scored.append((doc, hybrid))
            scored.sort(key=lambda x: x[1], reverse=True)
            sorted_docs = []
            for doc, rerank_score in scored:
                metadata = dict(doc.metadata)
                metadata["rerank_score"] = rerank_score
                sorted_docs.append(Document(page_content=doc.page_content, metadata=metadata))
        elif strategy == RerankStrategy.RRF:
            # 互惠排名融合
            # 分别按各指标排序
            sorted_by_vector = sorted(scores, key=lambda x: x[1]["vector"], reverse=True)
            sorted_by_bm25 = sorted(scores, key=lambda x: x[1]["bm25"], reverse=True)
            sorted_by_keyword = sorted(
                scores, key=lambda x: x[1]["keyword"], reverse=True
            )

            # 计算排名
            rank_map = defaultdict(float)
            k = 60  # RRF 常数

            def get_rank_key(doc):
                return doc.page_content[:100]

            for rank, (doc, _) in enumerate(sorted_by_vector):
                rank_map[get_rank_key(doc)] += 1.0 / (k + rank)

            for rank, (doc, _) in enumerate(sorted_by_bm25):
                rank_map[get_rank_key(doc)] += 1.0 / (k + rank)

            for rank, (doc, _) in enumerate(sorted_by_keyword):
                rank_map[get_rank_key(doc)] += 1.0 / (k + rank)

            # 按 RRF 分数排序
            sorted_with_rrf = sorted(
                [(doc, rank_map[get_rank_key(doc)]) for doc, _ in scores],
                key=lambda x: x[1],
                reverse=True,
            )
            sorted_docs = []
            for doc, rerank_score in sorted_with_rrf:
                metadata = dict(doc.metadata)
                metadata["rerank_score"] = rerank_score
                sorted_docs.append(Document(page_content=doc.page_content, metadata=metadata))
        else:
            sorted_docs = []
            for doc, s in scores:
                metadata = dict(doc.metadata)
                metadata["rerank_score"] = s["vector"]
                sorted_docs.append(Document(page_content=doc.page_content, metadata=metadata))

        logger.debug(f"重排序完成")
        return sorted_docs


# 单例实例
_reranker: Optional[DocumentReranker] = None


def get_document_reranker() -> DocumentReranker:
    global _reranker
    if _reranker is None:
        _reranker = DocumentReranker()
    return _reranker

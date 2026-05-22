"""
查询变换模块
包含查询翻译和查询重写
"""
from dataclasses import dataclass
from typing import Optional
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from config import create_chat_model


@dataclass
class QueryTransformResult:
    original_query: str
    translated_query: Optional[str]
    rewritten_query: str


def is_chinese_query(query: str) -> bool:
    if not query:
        return False
    chinese_chars = sum(1 for c in query if "一" <= c <= "鿿")
    return chinese_chars / len(query) > 0.3


def _strip_query_label(query: str) -> str:
    labels = (
        "优化后的检索查询:",
        "优化后的检索查询：",
        "优化后的查询:",
        "优化后的查询：",
        "查询:",
        "查询：",
    )
    cleaned = query.strip()
    for label in labels:
        if cleaned.startswith(label):
            return cleaned[len(label):].strip()
    return cleaned


class QueryTranslationTransformer:
    """查询翻译转换器
    将用户查询优化为更适合检索的形式
    """

    def __init__(self):
        self.llm = create_chat_model(temperature=0.3)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个专业的查询优化助手。
将用户的理财咨询问题转换为更适合知识库检索的查询形式。

要求：
1. 提取核心关键词和概念
2. 扩展相关术语（例如："定投" -> "定期定额投资、基金定投"）
3. 保持语义不变，不要添加新信息
4. 输出中文，简洁明了
5. 如果已经是中文且表达清晰，直接返回原查询优化版本""",
                ),
                ("human", "原查询: {query}\n\n优化后的查询:"),
            ]
        )
        self.chain = self.prompt | self.llm

    def transform(self, query: str) -> str:
        """转换查询"""
        logger.debug(f"翻译查询: {query}")

        try:
            result = self.chain.invoke({"query": query})
            transformed = _strip_query_label(result.content)
            logger.debug(f"翻译结果: {transformed}")
            return transformed
        except Exception as e:
            logger.error(f"查询翻译失败: {e}")
            return query


class QueryRewritingTransformer:
    """查询重写转换器
    针对检索场景优化查询，提高召回率
    """

    def __init__(self):
        self.llm = create_chat_model(temperature=0.2)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个专业的检索查询优化专家。
针对金融理财知识库检索场景，重写用户查询以提高召回效果。

要求：
1. 识别查询中的金融术语，补充同义词和相关术语
2. 分解复杂问题为关键点
3. 保持查询简洁，适合向量检索
4. 用中文输出
5. 格式：输出优化后的查询文本，不要解释

例子：
输入: "我有10万块，应该怎么理财"
输出: "10万元 理财 投资建议 资产配置 理财规划"

输入: "什么是指数基金"
输出: "指数基金 定义 被动投资 指数跟踪基金 ETF 基金投资基础知识" """,
                ),
                ("human", "用户查询: {query}\n\n优化后的检索查询:"),
            ]
        )
        self.chain = self.prompt | self.llm

    def transform(self, query: str) -> str:
        """重写查询"""
        logger.debug(f"重写查询: {query}")

        try:
            result = self.chain.invoke({"query": query})
            rewritten = _strip_query_label(result.content)
            logger.debug(f"重写结果: {rewritten}")
            return rewritten
        except Exception as e:
            logger.error(f"查询重写失败: {e}")
            return query


# 单例实例
_translation_transformer: Optional[QueryTranslationTransformer] = None
_rewriting_transformer: Optional[QueryRewritingTransformer] = None


def get_translation_transformer() -> QueryTranslationTransformer:
    global _translation_transformer
    if _translation_transformer is None:
        _translation_transformer = QueryTranslationTransformer()
    return _translation_transformer


def get_rewriting_transformer() -> QueryRewritingTransformer:
    global _rewriting_transformer
    if _rewriting_transformer is None:
        _rewriting_transformer = QueryRewritingTransformer()
    return _rewriting_transformer


def transform_query_for_retrieval(query: str) -> QueryTransformResult:
    if is_chinese_query(query):
        rewritten = get_rewriting_transformer().transform(query)
        return QueryTransformResult(
            original_query=query,
            translated_query=None,
            rewritten_query=rewritten,
        )

    translated = get_translation_transformer().transform(query)
    rewritten = get_rewriting_transformer().transform(translated)
    return QueryTransformResult(
        original_query=query,
        translated_query=translated,
        rewritten_query=rewritten,
    )

"""应用服务层
提供理财顾问核心服务
"""
from typing import AsyncIterator, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

import tools  # noqa: F401  导入即触发工具注册到 harness.default_registry
from config import create_chat_model
from harness import Harness, HarnessEvent
from memory import ChatMemory, get_memory_store
from rag import RagEvent, get_rag_pipeline, get_semantic_cache


FINANCIAL_ADVISOR_PROMPT = """你是一位资深理财专家，名叫 "LiCaiManus"。

你的职责：
1. 通过提问了解用户的财务状况和需求
2. 提供个性化的理财建议和规划
3. 解释复杂的金融概念
4. 提示投资风险

回答要求：
- 专业、严谨、友好
- 用中文回答
- 回答有条理，分点说明
- 如果不确定，诚实地告知用户
"""

_CHAT_WINDOW_TURNS = 10


class FinancialAdvisorService:
    """理财顾问服务"""

    def __init__(self):
        self.llm = create_chat_model(temperature=0.7)
        self.rag_pipeline = get_rag_pipeline()
        self.cache = get_semantic_cache()

    def chat(self, message: str, chat_id: str, use_memory: bool = True) -> str:
        """普通对话；``use_memory=True`` 时跨请求持久化到 ``memory/`` store。"""
        logger.info(f"普通对话: {chat_id} - {message[:50]}...")

        if not use_memory:
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=FINANCIAL_ADVISOR_PROMPT),
                    HumanMessage(content=message),
                ]
            )
            response = self.llm.invoke(prompt.format_messages())
            return response.content

        store = get_memory_store()
        history = store.get_messages(chat_id)

        # 把历史装进窗口对象，沿用 ChatMemory 的 k 截断
        window = ChatMemory(session_id=chat_id, k=_CHAT_WINDOW_TURNS)
        for past in history[-_CHAT_WINDOW_TURNS * 2 :]:
            window._messages.append(past)
        window.add_user_message(message)

        messages = [SystemMessage(content=FINANCIAL_ADVISOR_PROMPT)] + window.messages
        response = self.llm.invoke(messages)
        answer = response.content

        # 写回 store：只写当前轮的两条，避免重复历史
        store.add_message(chat_id, HumanMessage(content=message))
        store.add_message(chat_id, AIMessage(content=answer))

        logger.info(f"对话 {chat_id} 完成，历史长度 {len(history) + 2}")
        return answer

    def chat_with_rag(self, message: str, chat_id: str) -> str:
        """使用 RAG 知识库对话（不接跨请求记忆）。"""
        logger.info(f"RAG 对话: {chat_id} - {message[:50]}...")
        return self.rag_pipeline.execute(message)

    async def astream_rag(
        self, message: str, chat_id: str = "default"
    ) -> AsyncIterator[RagEvent]:
        """流式 RAG：发出 retrieval / generation 阶段事件。不走缓存。"""
        logger.info(f"流式 RAG: {chat_id} - {message[:50]}...")
        async for event in self.rag_pipeline.astream_execute(message):
            yield event

    def chat_with_agent(self, message: str, chat_id: str = "default") -> dict:
        """使用 Harness 主循环处理任务（同步）。"""
        logger.info(f"Agent 处理任务: {chat_id} - {message[:50]}...")

        result = Harness(chat_id=chat_id).run(message)
        return {
            "success": result.stopped_reason != "max_iterations",
            "answer": result.final_text,
            "steps": len(result.turns),
            "tool_calls": result.total_tool_calls,
            "final_answer": result.final_text,
            "stopped_reason": result.stopped_reason,
        }

    async def astream_agent(
        self, message: str, chat_id: str = "default"
    ) -> AsyncIterator[HarnessEvent]:
        """使用 Harness 主循环处理任务（异步事件流）。"""
        logger.info(f"Agent 流式处理任务: {chat_id} - {message[:50]}...")
        async for event in Harness(chat_id=chat_id).astream(message):
            yield event

    async def astream_chat(
        self, message: str, chat_id: str = "default"
    ) -> AsyncIterator[str]:
        """轻量对话的真 token 流。不接 RAG、不接记忆。"""
        logger.info(f"流式对话: {chat_id} - {message[:50]}...")
        messages = [
            SystemMessage(content=FINANCIAL_ADVISOR_PROMPT),
            HumanMessage(content=message),
        ]
        async for chunk in self.llm.astream(messages):
            content = getattr(chunk, "content", "")
            if isinstance(content, str) and content:
                yield content

    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        if self.cache:
            return self.cache.stats()
        return {"enabled": False}

    def warm_up_cache(self, common_queries: list) -> None:
        """预热缓存（占位）。"""
        if self.cache:
            logger.info(f"预热缓存，共 {len(common_queries)} 个查询")

    def clear_chat_memory(self, chat_id: str) -> None:
        """清空对话记忆：实际从 store 删除该会话。"""
        logger.info(f"清空对话记忆: {chat_id}")
        get_memory_store().clear(chat_id)


# 全局服务实例
_financial_service: Optional[FinancialAdvisorService] = None


def get_financial_service() -> FinancialAdvisorService:
    global _financial_service
    if _financial_service is None:
        _financial_service = FinancialAdvisorService()
    return _financial_service

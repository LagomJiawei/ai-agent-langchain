"""
应用服务层
提供理财顾问核心服务
"""
from typing import Optional
from loguru import logger

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

from config import create_chat_model
from memory import create_chat_memory, ChatMemory
from rag import get_rag_pipeline, get_semantic_cache
from agents import AgentSelector, AgentState


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


class FinancialAdvisorService:
    """理财顾问服务"""

    def __init__(self):
        self.llm = create_chat_model(temperature=0.7)
        self.rag_pipeline = get_rag_pipeline()
        self.agent_selector = AgentSelector()
        self.cache = get_semantic_cache()

    def chat(self, message: str, chat_id: str, use_memory: bool = True) -> str:
        """
        普通对话（带记忆）

        Args:
            message: 用户消息
            chat_id: 对话 ID
            use_memory: 是否使用记忆

        Returns:
            AI 回复
        """
        logger.info(f"普通对话: {chat_id} - {message[:50]}...")

        if not use_memory:
            # 无记忆对话
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=FINANCIAL_ADVISOR_PROMPT),
                    HumanMessage(content=message),
                ]
            )
            response = self.llm.invoke(prompt.format_messages())
            return response.content

        # 使用记忆
        chat_memory = ChatMemory(
            session_id=chat_id,
            k=10,
        )
        chat_memory.add_user_message(message)

        # 添加系统提示（首次对话）
        if len(chat_memory.messages) <= 1:
            # 创建一个带系统提示的 prompt
            messages = [SystemMessage(content=FINANCIAL_ADVISOR_PROMPT)] + chat_memory.messages
            response = self.llm.invoke(messages)
        else:
            response = self.llm.invoke(chat_memory.messages)

        answer = response.content
        chat_memory.add_ai_message(answer)

        logger.info(f"对话 {chat_id} 完成")
        return answer

    def chat_with_rag(self, message: str, chat_id: str) -> str:
        """
        使用 RAG 知识库对话

        Args:
            message: 用户消息
            chat_id: 对话 ID

        Returns:
            AI 回复
        """
        logger.info(f"RAG 对话: {chat_id} - {message[:50]}...")
        return self.rag_pipeline.execute(message)

    def chat_with_agent(
        self,
        message: str,
        force_plan_execute: bool = False,
    ) -> dict:
        """
        使用 Agent 处理任务

        Args:
            message: 用户消息
            force_plan_execute: 是否强制使用 Plan-and-Execute 模式

        Returns:
            Agent 执行结果
        """
        logger.info(f"Agent 处理任务: {message[:50]}...")

        if force_plan_execute:
            # 强制 Plan-and-Execute
            from agents import PlanAndExecuteAgent

            agent = PlanAndExecuteAgent()
            result = agent.execute(message)
        else:
            # 自动选择
            result = self.agent_selector.execute(message)

        return {
            "success": result.is_finished,
            "answer": result.final_answer,
            "steps": result.current_step,
            "tool_calls": len(result.tool_call_history),
            "final_answer": result.final_answer,
        }

    def get_cache_stats(self) -> dict:
        """获取缓存统计信息"""
        if self.cache:
            return self.cache.stats()
        return {"enabled": False}

    def warm_up_cache(self, common_queries: list) -> None:
        """预热缓存"""
        if self.cache:
            logger.info(f"预热缓存，共 {len(common_queries)} 个查询")
            # 这里可以实现缓存预热逻辑
        pass

    def clear_chat_memory(self, chat_id: str) -> None:
        """清空对话记忆"""
        logger.info(f"清空对话记忆: {chat_id}")
        # LangChain 记忆清除逻辑
        pass


# 全局服务实例
_financial_service: Optional[FinancialAdvisorService] = None


def get_financial_service() -> FinancialAdvisorService:
    global _financial_service
    if _financial_service is None:
        _financial_service = FinancialAdvisorService()
    return _financial_service

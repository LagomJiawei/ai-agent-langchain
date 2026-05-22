"""
Agent 选择器
根据任务复杂度自动选择合适的 Agent
"""
from typing import Optional
from loguru import logger

from langchain_core.prompts import ChatPromptTemplate

from config import create_chat_model, settings
from .state import AgentState
from .react import ReActAgent
from .plan_execute import PlanAndExecuteAgent


class AgentSelector:
    """Agent 选择器"""

    def __init__(
        self,
        react_agent: Optional[ReActAgent] = None,
        plan_execute_agent: Optional[PlanAndExecuteAgent] = None,
        force_mode: Optional[str] = None,
    ):
        self.react_agent = react_agent or ReActAgent()
        self.plan_execute_agent = plan_execute_agent or PlanAndExecuteAgent()
        self.force_mode = force_mode or settings.agent.mode
        self.llm = create_chat_model(temperature=0.2)

        # 分类 Prompt
        self.classification_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个任务分类专家。请根据用户查询的复杂度，
选择合适的执行模式。

分类标准：
- SIMPLE: 简单问题，不需要工具或仅需1次工具调用即可回答
- COMPLEX: 复杂任务，需要多步推理、多次工具调用、信息收集和整合

返回 JSON 格式：
{{"mode": "SIMPLE" or "COMPLEX", "reason": "分类理由"}}""",
                ),
                (
                    "human",
                    "用户查询: {query}\n\n分类结果:",
                ),
            ]
        )

    def _classify_task(self, query: str) -> str:
        """
        分类任务复杂度

        Returns:
            "SIMPLE" or "COMPLEX"
        """
        # 快速规则判断
        keywords = [
            "分析",
            "研究",
            "对比",
            "比较",
            "规划",
            "方案",
            "步骤",
            "综合",
            "总结",
            "报告",
        ]
        for keyword in keywords:
            if keyword in query:
                logger.info(f"检测到关键词 '{keyword}'，判定为复杂任务")
                return "COMPLEX"

        # 长度判断
        if len(query) > 100:
            logger.info("查询较长，判定为复杂任务")
            return "COMPLEX"

        # LLM 智能分类
        try:
            result = self.llm.invoke(self.classification_prompt.format(query=query))
            content = result.content.strip()

            # 解析 JSON
            import json
            import re

            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                mode = data.get("mode", "SIMPLE")
                logger.info(f"任务分类: {mode} - {data.get('reason', '')}")
                return mode
        except Exception as e:
            logger.debug(f"分类失败，使用默认策略: {e}")

        return "SIMPLE"

    def execute(self, query: str) -> AgentState:
        """
        选择合适的 Agent 执行任务

        Args:
            query: 用户查询

        Returns:
            Agent 执行状态
        """
        logger.info(f"【Agent 选择器】处理查询: {query[:50]}...")

        # 强制模式
        if self.force_mode == "react":
            logger.info("【Agent 选择器】强制使用 ReAct 模式")
            return self.react_agent.execute(query)
        elif self.force_mode == "plan_execute":
            logger.info("【Agent 选择器】强制使用 Plan-and-Execute 模式")
            return self.plan_execute_agent.execute(query)

        # 自动选择
        task_type = self._classify_task(query)

        if task_type == "COMPLEX":
            logger.info("【Agent 选择器】选择 Plan-and-Execute 模式处理复杂任务")
            return self.plan_execute_agent.execute(query)
        else:
            logger.info("【Agent 选择器】选择 ReAct 模式处理简单任务")
            return self.react_agent.execute(query)

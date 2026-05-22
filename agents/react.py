"""
ReAct Agent - 思考-行动-观察循环
基于 LangGraph 实现
"""
import json
from typing import List, Tuple, Optional, Dict, Any
from loguru import logger

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END

from config import create_chat_model, settings
from tools import get_all_tools
from .state import AgentState


SYSTEM_PROMPT = """你是 LiCaiManus，一个全能的理财 AI 助手，使用 ReAct 推理框架工作。

你的工作流程：
1. Thought: 思考当前问题，分析需要做什么
2. Action: 选择合适的工具执行动作并调用
3. Observation: 查看工具执行结果
4. Repeat or Finish: 重复或给出最终答案

【核心规则】
- 可以多次调用工具来获取信息
- 每次只调用一个工具，不要并行调用
- 当获取足够信息后，使用 do_terminate 工具结束并给出答案
- 用中文思考和回答

【循环防御规则 - 必须严格遵守】
1. 不要重复调用相同工具使用相同参数
2. 如果某个工具连续2次没有返回有效信息，立即换其他工具或方法
3. 如果搜索结果不理想，尝试修改关键词而不是重复相同搜索
4. 如果网页抓取失败，尝试其他链接或使用搜索工具获取摘要
5. 最多尝试2种不同方法，然后基于已有信息给出答案
6. 绝对禁止陷入循环调用，效率优先于完美

【知识库检索规则 - 必须严格遵守】
1. 当用户问题涉及理财知识、基金、股票、债券、保险、资产配置等内容时，优先使用 search_knowledge_base 工具
2. 对比、规划、综合分析类问题要拆成多个具体子查询，分别检索后再综合回答
3. 如果检索结果的 sufficiency 是 "insufficient" 或 quality_score 低于 0.5，换关键词重试，不要重复相同查询
4. 基于检索结果回答；如果知识库资料不足，明确说明资料不足，不要编造
5. 最多尝试2种不同检索查询，仍不足时基于已有信息给出谨慎回答

【检索质量信号解读】
- doc_count: 检索到的文档数量
- quality_score: 0.0-1.0，越高表示越相关
- sufficiency: "adequate" 表示信息较充足，"insufficient" 表示需要换查询或补充信息

【理财专家设定】
- 提供专业的理财建议，包括但不限于：基金、股票、债券、保险、资产配置等
- 提示投资风险，不承诺收益
- 给出实用的理财规划建议
- 如果问题超出你的知识范围，诚实地告诉用户
"""


class ReActAgent:
    """ReAct Agent 实现"""

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = None,
    ):
        self.tools = tools or get_all_tools()
        self.max_iterations = max_iterations or settings.agent.max_iterations
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.llm = create_chat_model(temperature=0.7)

        # 绑定工具到 LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # 创建执行图
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph"""
        # 定义状态（简化版，使用消息列表）
        graph = StateGraph(dict)

        # 定义节点
        graph.add_node("think", self._think_node)
        graph.add_node("act", self._act_node)

        # 定义边
        graph.set_entry_point("think")
        graph.add_conditional_edges(
            "think",
            self._should_act,
            {
                "act": "act",
                "end": END,
            },
        )
        graph.add_edge("act", "think")

        return graph.compile()

    def _think_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        思考节点：调用 LLM 分析当前状态，决定下一步动作
        """
        if state.get("is_finished", False):
            return state

        messages = state["messages"]
        step = state.get("step", 0)

        logger.info(f"【思考】执行第 {step + 1} 步")

        # 调用 LLM
        response = self.llm_with_tools.invoke(messages)

        state["messages"].append(response)
        state["step"] = step + 1
        state["last_thought"] = response.content

        if response.content and "[TERMINATE]" in response.content:
            state["is_finished"] = True
            state["final_answer"] = response.content.replace("[TERMINATE]", "").strip()
        elif not getattr(response, "tool_calls", None):
            state["is_finished"] = True
            state["final_answer"] = response.content

        return state

    def _should_act(self, state: Dict[str, Any]) -> str:
        """
        决定是继续执行工具还是结束
        """
        # 检查是否已完成
        if state.get("is_finished", False):
            logger.info("【结束】任务已完成")
            return "end"

        # 检查最大步数
        if state.get("step", 0) >= self.max_iterations:
            logger.warning(f"【结束】达到最大迭代次数 {self.max_iterations}")
            state["final_answer"] = "达到最大迭代次数，任务终止。"
            state["is_finished"] = True
            return "end"

        # 检查是否有工具调用
        messages = state["messages"]
        last_message = messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            # 检查循环防御
            tool_calls = last_message.tool_calls
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                arguments = json.dumps(tool_call.get("args", {}), sort_keys=True)

                # 记录调用次数
                call_key = f"{tool_name}:{arguments}"
                call_count = state.get("tool_call_counts", {}).get(call_key, 0) + 1

                if "tool_call_counts" not in state:
                    state["tool_call_counts"] = {}
                state["tool_call_counts"][call_key] = call_count

                # 检测重复调用
                if call_count >= 2:
                    logger.warning(f"【循环防御】检测到重复调用: {tool_name}")
                    final_answer = (
                        f"检测到工具 '{tool_name}' 被重复调用，已停止继续调用以避免循环。"
                        "请基于已有信息重新提问，或换一种更具体的问法。"
                    )
                    state["messages"].append(HumanMessage(content=final_answer))
                    state["final_answer"] = final_answer
                    state["is_finished"] = True
                    return "end"

            return "act"

        # 没有工具调用，直接结束
        logger.info("【结束】无工具调用，任务完成")
        return "end"

    def _act_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        行动节点：执行工具调用
        """
        messages = state["messages"]
        last_message = messages[-1]

        tool_calls = last_message.tool_calls
        logger.info(f"【行动】执行工具调用: {[tc['name'] for tc in tool_calls]}")

        # 执行所有工具调用
        tool_messages = []
        for tool_call in tool_calls:
            try:
                # 直接执行工具
                tool = self.tool_map.get(tool_call["name"])
                if tool is None:
                    raise ValueError(f"未知工具: {tool_call['name']}")
                result = tool.invoke(tool_call["args"])

                # 记录工具调用
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"],
                    )
                )

                # 检查是否是终止工具
                if tool_call["name"] == "do_terminate":
                    logger.info("【终止】调用 do_terminate 工具")
                    state["is_finished"] = True
                    state["final_answer"] = tool_call["args"].get("final_answer", "")

            except Exception as e:
                logger.error(f"工具执行失败: {e}")
                tool_messages.append(
                    ToolMessage(
                        content=f"工具执行失败: {str(e)}",
                        tool_call_id=tool_call["id"],
                    )
                )

        # 将工具执行结果添加到消息历史
        state["messages"].extend(tool_messages)

        return state

    def execute(self, user_query: str) -> AgentState:
        """
        执行 Agent 任务

        Args:
            user_query: 用户查询

        Returns:
            Agent 状态（包含执行结果）
        """
        logger.info(f"Agent 开始执行任务: {user_query}")

        # 初始化状态
        initial_state = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_query),
            ],
            "step": 0,
            "is_finished": False,
            "tool_call_counts": {},
        }

        # 执行图
        final_state = self.graph.invoke(initial_state)

        # 转换为 AgentState
        agent_state = AgentState(
            user_query=user_query,
            max_steps=self.max_iterations,
            chat_history=final_state["messages"],
        )
        agent_state.current_step = final_state["step"]
        agent_state.is_finished = final_state.get("is_finished", False)
        agent_state.final_answer = final_state.get("final_answer", "")
        agent_state.current_thought = final_state.get("last_thought", "")

        # 填充工具调用历史
        for message in final_state["messages"]:
            if isinstance(message, ToolMessage):
                agent_state.tool_call_history.append(
                    type(
                        "ToolCallRecord",
                        (),
                        {
                            "tool_name": message.tool_call_id,
                            "arguments": "",
                            "result": message.content,
                        },
                    )()
                )

        logger.info(f"Agent 任务完成，共执行 {agent_state.current_step} 步")

        return agent_state

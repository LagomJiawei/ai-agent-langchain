"""
Agent 状态管理
"""
from typing import List, Dict, Any, Optional, Annotated, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages


@dataclass
class ToolCallRecord:
    """工具调用记录"""

    tool_name: str
    arguments: str
    result: Optional[str] = None
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())


class AgentState:
    """Agent 执行状态"""

    def __init__(
        self,
        user_query: str,
        max_steps: int = 10,
        chat_history: Optional[List[BaseMessage]] = None,
    ):
        self.user_query = user_query
        self.current_step = 0
        self.max_steps = max_steps
        self.chat_history: List[BaseMessage] = chat_history or []
        self.tool_call_history: List[ToolCallRecord] = []
        self.current_thought: Optional[str] = None
        self.final_answer: Optional[str] = None
        self.is_finished = False
        # 循环防御：记录相同参数的调用次数
        self._call_counter: Dict[str, int] = {}

    def increment_step(self) -> None:
        """增加步数"""
        self.current_step += 1

    def is_max_steps_reached(self) -> bool:
        """是否达到最大步数"""
        return self.current_step >= self.max_steps

    def record_tool_call(self, tool_name: str, arguments: str, result: Optional[str] = None) -> None:
        """记录工具调用"""
        record = ToolCallRecord(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )
        self.tool_call_history.append(record)

        # 更新调用计数
        key = f"{tool_name}:{arguments}"
        self._call_counter[key] = self._call_counter.get(key, 0) + 1

    def get_tool_call_count(self, tool_name: str, arguments: str) -> int:
        """获取相同工具相同参数的调用次数"""
        key = f"{tool_name}:{arguments}"
        return self._call_counter.get(key, 0)

    def is_in_loop(self, tool_name: str, arguments: str, threshold: int = 2) -> bool:
        """检测是否存在循环调用"""
        return self.get_tool_call_count(tool_name, arguments) >= threshold

    def is_tool_overused(self, tool_name: str, threshold: int = 3) -> bool:
        """检测工具是否过度使用"""
        count = sum(1 for r in self.tool_call_history if r.tool_name == tool_name)
        return count >= threshold

    def get_loop_suggestion(self, tool_name: str) -> str:
        """获取循环防御建议"""
        suggestions = {
            "search_web": "请尝试修改搜索关键词，或者使用其他工具获取信息。",
            "scrape_web_page": "请尝试访问其他链接，或者使用搜索工具获取摘要。",
            "file_read": "请检查文件路径是否正确，或者尝试其他方式获取信息。",
            "file_write": "请确认是否需要重复写入，考虑合并多次写入操作。",
        }
        return suggestions.get(
            tool_name, "检测到重复调用，请尝试其他方法或直接给出答案。"
        )


# LangGraph 状态定义
class GraphState:
    """LangGraph 执行状态"""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_finished: bool = False
    tool_calls_count: Dict[str, int] = field(default_factory=dict)

    def __init__(self, messages: Sequence[BaseMessage]):
        self.messages = list(messages)
        self.is_finished = False
        self.tool_calls_count = {}

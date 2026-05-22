"""
Agent 测试
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from agents import ReActAgent, PlanAndExecuteAgent, AgentSelector
from tools import get_all_tools


def test_react_agent_initialization():
    """测试 ReAct Agent 初始化"""
    agent = ReActAgent(max_iterations=5)
    assert agent.max_iterations == 5


def test_plan_and_execute_agent_initialization():
    """测试 Plan-and-Execute Agent 初始化"""
    agent = PlanAndExecuteAgent(max_steps=10)
    assert agent.max_steps == 10


def test_agent_selector_initialization():
    """测试 Agent 选择器初始化"""
    selector = AgentSelector()
    assert selector.react_agent is not None
    assert selector.plan_execute_agent is not None


def test_agentic_rag_tool_registered():
    """Agent 工具列表包含知识库检索工具"""
    tool_names = [tool.name for tool in get_all_tools()]
    assert "search_knowledge_base" in tool_names

    agent = ReActAgent(max_iterations=5)
    assert "search_knowledge_base" in agent.tool_map



def test_react_loop_defense_marks_state_finished():
    """循环防御结束时也返回完成状态"""
    agent = ReActAgent.__new__(ReActAgent)
    agent.max_iterations = 5
    call_key = f"search_web:{json.dumps({'query': '基金'}, sort_keys=True)}"
    state = {
        "messages": [
            type(
                "MessageWithToolCalls",
                (),
                {
                    "tool_calls": [
                        {
                            "name": "search_web",
                            "args": {"query": "基金"},
                        }
                    ]
                },
            )()
        ],
        "step": 1,
        "tool_call_counts": {call_key: 1},
        "is_finished": False,
    }

    assert agent._should_act(state) == "end"
    assert state["is_finished"] is True
    assert "search_web" in state["final_answer"]



def test_react_think_node_marks_final_answer_when_no_tool_calls():
    """无工具调用的模型回复在 think 节点持久化为最终答案"""
    agent = ReActAgent.__new__(ReActAgent)
    agent.llm_with_tools = type(
        "FakeLlm",
        (),
        {"invoke": lambda self, messages: AIMessage(content="这是最终理财建议。")},
    )()
    state = {
        "messages": [HumanMessage(content="给我一个理财计划")],
        "step": 1,
        "is_finished": False,
    }

    result = agent._think_node(state)

    assert result["is_finished"] is True
    assert result["final_answer"] == "这是最终理财建议。"
    assert result["step"] == 2



def test_task_classification():
    """测试任务分类"""
    selector = AgentSelector()

    # 简单任务
    simple = selector._classify_task("你好")
    assert simple in ["SIMPLE", "COMPLEX"]

    # 复杂任务（含关键词）
    complex_task = selector._classify_task("请分析一下当前的投资市场并给出理财规划方案")
    assert complex_task == "COMPLEX"


if __name__ == "__main__":
    # 快速测试（不需要配置环境）
    print("测试 Agent 初始化...")
    test_react_agent_initialization()
    test_plan_and_execute_agent_initialization()
    test_agent_selector_initialization()
    print("所有 Agent 测试通过！")

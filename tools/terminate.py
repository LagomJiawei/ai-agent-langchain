"""
任务终止工具
用于优雅地结束 Agent 执行并返回最终答案
"""
from langchain_core.tools import tool


@tool
def do_terminate(final_answer: str) -> str:
    """
    终止任务执行，返回最终答案

    当你已经获取了足够的信息来回答用户的问题时，
    使用此工具来结束执行并提供最终答案。

    Args:
        final_answer: 给用户的最终答案

    Returns:
        终止确认
    """
    return f"[TERMINATE] {final_answer}"

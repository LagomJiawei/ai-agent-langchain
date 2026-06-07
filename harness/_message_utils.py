"""LangChain Message / Chunk 的文本提取辅助函数。

历史上 ``harness/loop.py``、``harness/token_counter.py``、``rag/pipeline.py``
各自维护一份近乎重复的 ``_chunk_text`` / ``_extract_text`` / ``_message_text``，
注释里特意写"故意不跨包复用"——但事实上构成了漂移源，任何一处改了其他不会同步。

这里集中实现，对外暴露两个语义清晰的函数：
- ``extract_chunk_text``：增量 chunk，未知形态返回空串（流式累计场景，空增量更安全）。
- ``extract_message_text``：完整 message，未知形态 ``str(content)`` 兜底
  （token 计数 / trace / 显示等场景，宁可拿到一段类型仓库化的描述，也别静默丢内容）。
"""
from __future__ import annotations

from typing import Any


def _collect_parts(content: Any) -> str | None:
    """处理 list-of-parts 形态（多模态 / 工具调用混合内容）。

    不是 list 返回 None（让调用方走各自的标量兜底）。
    """
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            txt = part.get("text") or part.get("content")
            if txt:
                parts.append(str(txt))
        else:
            parts.append(str(part))
    return "".join(parts)


def extract_chunk_text(chunk: Any) -> str:
    """从 AIMessageChunk 提取增量文本。

    未知形态返回空串，避免把类型描述误当增量 token 推给前端。
    """
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    collected = _collect_parts(content)
    return collected if collected is not None else ""


def extract_message_text(message: Any) -> str:
    """从完整 BaseMessage 提取文本内容。

    未知形态走 ``str(content)`` 兜底，保证 token 计数 / trace 不丢失上下文。
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    collected = _collect_parts(content)
    return collected if collected is not None else str(content)


__all__ = ["extract_chunk_text", "extract_message_text"]

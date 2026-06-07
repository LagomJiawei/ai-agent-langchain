"""Token 计数工具。

优先用 tiktoken `cl100k_base` 编码估算消息 token 数；不可用时
退到 UTF-8 字节数 / 3 的粗估，并打一次 warning。
"""
from __future__ import annotations

from functools import cache
from typing import Iterable

from langchain_core.messages import BaseMessage
from loguru import logger

from ._message_utils import extract_message_text


@cache
def _get_encoding():
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"tiktoken 不可用，退到字节估算: {exc}")
        return None


def count_tokens(messages: Iterable[BaseMessage]) -> int:
    """估算一组消息的总 token 数（含每条 4 token 的 role 开销近似）。"""
    enc = _get_encoding()
    total = 0
    for msg in messages:
        text = extract_message_text(msg)
        if enc is not None:
            total += len(enc.encode(text)) + 4
        else:
            # 兜底：UTF-8 字节 / 3 的粗估（中文每字约 3 字节 ≈ 1 token）
            total += len(text.encode("utf-8")) // 3 + 4
    return total


__all__ = ["count_tokens"]


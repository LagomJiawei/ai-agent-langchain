"""harness._message_utils 文本提取测试。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, AIMessageChunk

from harness._message_utils import extract_chunk_text, extract_message_text


# ---------- extract_chunk_text ----------


def test_chunk_text_str_content():
    chunk = AIMessageChunk(content="hello")
    assert extract_chunk_text(chunk) == "hello"


def test_chunk_text_list_of_dicts():
    # 多模态 chunk：list-of-parts 形态
    chunk = AIMessageChunk(content=[{"type": "text", "text": "hi"}, {"type": "text", "text": " there"}])
    assert extract_chunk_text(chunk) == "hi there"


def test_chunk_text_list_with_content_key():
    chunk = AIMessageChunk(content=[{"content": "ok"}])
    assert extract_chunk_text(chunk) == "ok"


def test_chunk_text_list_with_non_dict_part():
    chunk = AIMessageChunk(content=[{"text": "a"}, "raw-str"])
    assert extract_chunk_text(chunk) == "araw-str"


def test_chunk_text_unknown_form_returns_empty():
    """未知形态返回空串：避免把类型仓库化描述误推到前端 token 流。"""

    class _Weird:
        content = 12345  # 整数，不是 str/list

    assert extract_chunk_text(_Weird()) == ""


def test_chunk_text_missing_content_attr():
    assert extract_chunk_text(object()) == ""


# ---------- extract_message_text ----------


def test_message_text_str_content():
    msg = AIMessage(content="hello")
    assert extract_message_text(msg) == "hello"


def test_message_text_list_of_dicts():
    msg = AIMessage(content=[{"text": "x"}, {"text": "y"}])
    assert extract_message_text(msg) == "xy"


def test_message_text_unknown_form_falls_back_to_str():
    """未知形态走 str(content) 兜底，保证 token 计数 / trace 不丢上下文。"""

    class _Weird:
        content = 12345

    assert extract_message_text(_Weird()) == "12345"


def test_message_text_skips_empty_dict_part():
    msg = AIMessage(content=[{"text": ""}, {"text": "real"}])
    # 空字符串不被收集（``or`` 的真值判断），只保留真实内容
    assert extract_message_text(msg) == "real"

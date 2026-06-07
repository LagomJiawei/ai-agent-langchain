"""FinancialAdvisorService.chat 跨请求记忆测试。"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import memory as memory_pkg
from app.services import FinancialAdvisorService


class _ScriptedLLM:
    """记录 invoke 入参的 AIMessage 脚本桩。"""

    def __init__(self, scripted: list[str]):
        self._answers = list(scripted)
        self.calls: list[list] = []

    def invoke(self, messages, *args, **kwargs):
        self.calls.append(list(messages))
        return AIMessage(content=self._answers.pop(0))


@pytest.fixture
def fresh_store(monkeypatch):
    """每个测试拿一个干净的内存 store。"""
    from memory.memory_store import InMemoryChatMemoryStore

    store = InMemoryChatMemoryStore()
    monkeypatch.setattr(memory_pkg, "_memory_store_singleton", store)
    # services 已经 import memory.get_memory_store；它读 module attr，所以
    # 直接覆盖 module 上的单例字段即可
    import app.services as services_module

    yield store


def _make_service(scripted: list[str]) -> tuple[FinancialAdvisorService, _ScriptedLLM]:
    svc = FinancialAdvisorService.__new__(FinancialAdvisorService)
    llm = _ScriptedLLM(scripted)
    svc.llm = llm
    svc.rag_pipeline = None  # type: ignore[assignment]
    svc.cache = None  # type: ignore[assignment]
    return svc, llm


def test_chat_persists_two_messages_per_request(fresh_store):
    svc, _ = _make_service(["你好！"])
    svc.chat(message="你好", chat_id="alice", use_memory=True)
    msgs = fresh_store.get_messages("alice")
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage) and msgs[0].content == "你好"
    assert isinstance(msgs[1], AIMessage) and msgs[1].content == "你好！"


def test_chat_second_turn_sees_first_turn_in_llm_input(fresh_store):
    svc, llm = _make_service(["你好！", "记得，你叫张三。"])
    svc.chat("我叫张三", chat_id="alice", use_memory=True)
    svc.chat("我叫什么", chat_id="alice", use_memory=True)

    # 第二次调用：messages 应包含第一次对话历史
    second_messages = llm.calls[1]
    contents = [getattr(m, "content", "") for m in second_messages]
    # system + 历史 user "我叫张三" + 历史 ai "你好！" + 当前 user "我叫什么"
    assert "我叫张三" in contents
    assert "你好！" in contents
    assert "我叫什么" in contents
    # store 累计 4 条
    assert len(fresh_store.get_messages("alice")) == 4


def test_chat_use_memory_false_does_not_touch_store(fresh_store):
    svc, _ = _make_service(["回答。"])
    svc.chat("hi", chat_id="alice", use_memory=False)
    assert fresh_store.get_messages("alice") == []


def test_chat_isolates_by_chat_id(fresh_store):
    svc, _ = _make_service(["a 回答", "b 回答"])
    svc.chat("alice 问题", chat_id="alice", use_memory=True)
    svc.chat("bob 问题", chat_id="bob", use_memory=True)
    alice = fresh_store.get_messages("alice")
    bob = fresh_store.get_messages("bob")
    assert len(alice) == 2 and len(bob) == 2
    assert alice[0].content == "alice 问题"
    assert bob[0].content == "bob 问题"


def test_clear_chat_memory_removes_session(fresh_store):
    svc, _ = _make_service(["答案"])
    svc.chat("hi", chat_id="alice", use_memory=True)
    assert fresh_store.get_messages("alice")
    svc.clear_chat_memory("alice")
    assert fresh_store.get_messages("alice") == []

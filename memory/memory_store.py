"""
内存聊天记忆存储
用于临时对话，不持久化
"""
from typing import Dict, List
from langchain_core.messages import BaseMessage
from .base import BaseChatMemoryStore


class InMemoryChatMemoryStore(BaseChatMemoryStore):
    """内存存储实现"""

    def __init__(self):
        self._store: Dict[str, List[BaseMessage]] = {}

    def get_messages(self, conversation_id: str) -> List[BaseMessage]:
        """获取对话历史"""
        return self._store.get(conversation_id, [])

    def add_message(self, conversation_id: str, message: BaseMessage) -> None:
        """添加消息"""
        if conversation_id not in self._store:
            self._store[conversation_id] = []
        self._store[conversation_id].append(message)

    def clear(self, conversation_id: str) -> None:
        """清空对话历史"""
        if conversation_id in self._store:
            del self._store[conversation_id]

    def list_conversations(self) -> List[str]:
        """列出所有对话 ID"""
        return list(self._store.keys())

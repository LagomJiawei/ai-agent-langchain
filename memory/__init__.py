"""
聊天记忆系统
支持多种存储后端：内存、文件、Redis
"""
from typing import Optional, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_classic.memory import ConversationBufferWindowMemory

from .base import BaseChatMemoryStore
from .memory_store import InMemoryChatMemoryStore
from .file_store import FileChatMemoryStore
from .redis_store import RedisChatMemoryStore
from config.settings import settings


class ChatMemory:
    """简单的聊天记忆包装类，用于存储对话历史"""

    def __init__(self, session_id: str, k: int = 10):
        self.session_id = session_id
        self.k = k
        self._messages: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:
        """获取所有消息"""
        return self._messages

    def add_user_message(self, message: str) -> None:
        """添加用户消息"""
        self._messages.append(HumanMessage(content=message))
        # 保持窗口大小
        if len(self._messages) > self.k * 2:
            self._messages = self._messages[-self.k * 2 :]

    def add_ai_message(self, message: str) -> None:
        """添加 AI 消息"""
        self._messages.append(AIMessage(content=message))
        # 保持窗口大小
        if len(self._messages) > self.k * 2:
            self._messages = self._messages[-self.k * 2 :]

    def clear(self) -> None:
        """清空记忆"""
        self._messages = []

    def __len__(self) -> int:
        return len(self._messages)


def create_memory_store() -> BaseChatMemoryStore:
    """根据配置创建记忆存储"""
    store_type = settings.chat_memory.store_type

    if store_type == "memory":
        return InMemoryChatMemoryStore()
    elif store_type == "file":
        return FileChatMemoryStore(base_dir=settings.chat_memory.base_dir)
    elif store_type == "redis":
        if not settings.redis.enabled:
            raise ValueError("Redis is not enabled in settings")
        return RedisChatMemoryStore(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            password=settings.redis.password,
        )
    else:
        raise ValueError(f"Unknown memory store type: {store_type}")


def create_chat_memory(
    conversation_id: str,
    store: Optional[BaseChatMemoryStore] = None,
    max_messages: Optional[int] = None,
) -> ConversationBufferWindowMemory:
    """创建带消息窗口的聊天记忆"""
    if store is None:
        store = create_memory_store()

    if max_messages is None:
        max_messages = settings.chat_memory.max_messages

    return ConversationBufferWindowMemory(
        k=max_messages,
        memory_key="chat_history",
        return_messages=True,
    )


__all__ = [
    "BaseChatMemoryStore",
    "InMemoryChatMemoryStore",
    "FileChatMemoryStore",
    "RedisChatMemoryStore",
    "create_memory_store",
    "create_chat_memory",
    "ChatMemory",
]

"""聊天记忆系统
支持多种存储后端：内存、文件、Redis
"""
from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config.settings import settings

from .base import BaseChatMemoryStore
from .file_store import FileChatMemoryStore
from .memory_store import InMemoryChatMemoryStore
from .redis_store import RedisChatMemoryStore


class ChatMemory:
    """简单的聊天记忆包装类，用于存储对话历史"""

    def __init__(self, session_id: str, k: int = 10):
        self.session_id = session_id
        self.k = k
        self._messages: List[BaseMessage] = []

    @property
    def messages(self) -> List[BaseMessage]:
        return self._messages

    def add_user_message(self, message: str) -> None:
        self._messages.append(HumanMessage(content=message))
        if len(self._messages) > self.k * 2:
            self._messages = self._messages[-self.k * 2 :]

    def add_ai_message(self, message: str) -> None:
        self._messages.append(AIMessage(content=message))
        if len(self._messages) > self.k * 2:
            self._messages = self._messages[-self.k * 2 :]

    def clear(self) -> None:
        self._messages = []

    def __len__(self) -> int:
        return len(self._messages)


def create_memory_store() -> BaseChatMemoryStore:
    """根据配置创建记忆存储"""
    store_type = settings.chat_memory.store_type

    if store_type == "memory":
        return InMemoryChatMemoryStore()
    if store_type == "file":
        return FileChatMemoryStore(base_dir=settings.chat_memory.base_dir)
    if store_type == "redis":
        if not settings.redis.enabled:
            raise ValueError("Redis is not enabled in settings")
        return RedisChatMemoryStore(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db,
            password=settings.redis.password,
        )
    raise ValueError(f"Unknown memory store type: {store_type}")


_memory_store_singleton: BaseChatMemoryStore | None = None


def get_memory_store() -> BaseChatMemoryStore:
    """按 CHAT_MEMORY_STORE_TYPE 配置 lazy 构造单例 store。"""
    global _memory_store_singleton
    if _memory_store_singleton is None:
        _memory_store_singleton = create_memory_store()
    return _memory_store_singleton


def reset_memory_store() -> None:
    """仅供测试：清掉单例，下次 get 重新构造。"""
    global _memory_store_singleton
    _memory_store_singleton = None


__all__ = [
    "BaseChatMemoryStore",
    "InMemoryChatMemoryStore",
    "FileChatMemoryStore",
    "RedisChatMemoryStore",
    "create_memory_store",
    "get_memory_store",
    "reset_memory_store",
    "ChatMemory",
]

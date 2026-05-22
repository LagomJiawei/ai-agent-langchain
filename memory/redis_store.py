"""
Redis 分布式聊天记忆存储
支持集群共享，生产环境推荐
"""
import json
from typing import List, Optional

import redis
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from .base import BaseChatMemoryStore


class RedisChatMemoryStore(BaseChatMemoryStore):
    """Redis 存储实现"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "chat_memory:",
        ttl: Optional[int] = None,  # 过期时间（秒），None 表示永不过期
    ):
        self.key_prefix = key_prefix
        self.ttl = ttl
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
        )

    def _get_key(self, conversation_id: str) -> str:
        """获取 Redis Key"""
        return f"{self.key_prefix}{conversation_id}"

    def _message_to_dict(self, message: BaseMessage) -> dict:
        """消息转字典"""
        msg_type = message.__class__.__name__
        return {
            "type": msg_type,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
        }

    def _dict_to_message(self, data: dict) -> BaseMessage:
        """字典转消息"""
        msg_type = data.get("type")
        content = data.get("content", "")
        kwargs = data.get("additional_kwargs", {})

        msg_class_map = {
            "HumanMessage": HumanMessage,
            "AIMessage": AIMessage,
            "SystemMessage": SystemMessage,
            "ToolMessage": ToolMessage,
        }

        msg_class = msg_class_map.get(msg_type, AIMessage)
        return msg_class(content=content, additional_kwargs=kwargs)

    def get_messages(self, conversation_id: str) -> List[BaseMessage]:
        """获取对话历史"""
        key = self._get_key(conversation_id)
        data = self._client.get(key)
        if not data:
            return []

        try:
            messages_data = json.loads(data)
            return [self._dict_to_message(msg) for msg in messages_data]
        except Exception:
            return []

    def add_message(self, conversation_id: str, message: BaseMessage) -> None:
        """添加消息"""
        messages = self.get_messages(conversation_id)
        messages.append(message)
        self._save_messages(conversation_id, messages)

    def _save_messages(self, conversation_id: str, messages: List[BaseMessage]) -> None:
        """保存消息到 Redis"""
        key = self._get_key(conversation_id)
        data = [self._message_to_dict(msg) for msg in messages]
        value = json.dumps(data, ensure_ascii=False)

        if self.ttl:
            self._client.setex(key, self.ttl, value)
        else:
            self._client.set(key, value)

    def clear(self, conversation_id: str) -> None:
        """清空对话历史"""
        key = self._get_key(conversation_id)
        self._client.delete(key)

    def list_conversations(self) -> List[str]:
        """列出所有对话 ID"""
        pattern = f"{self.key_prefix}*"
        keys = self._client.keys(pattern)
        return [key[len(self.key_prefix):] for key in keys]

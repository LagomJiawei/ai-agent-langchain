"""
聊天记忆存储基类
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_core.messages import BaseMessage


class BaseChatMemoryStore(ABC):
    """聊天记忆存储接口"""

    @abstractmethod
    def get_messages(self, conversation_id: str) -> List[BaseMessage]:
        """获取对话历史"""
        pass

    @abstractmethod
    def add_message(self, conversation_id: str, message: BaseMessage) -> None:
        """添加消息"""
        pass

    @abstractmethod
    def clear(self, conversation_id: str) -> None:
        """清空对话历史"""
        pass

    @abstractmethod
    def list_conversations(self) -> List[str]:
        """列出所有对话 ID"""
        pass

"""
文件持久化聊天记忆存储
将对话历史保存为 JSON 文件
"""
import json
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from .base import BaseChatMemoryStore


class FileChatMemoryStore(BaseChatMemoryStore):
    """文件存储实现"""

    def __init__(self, base_dir: str = "./chat-memory"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, conversation_id: str) -> Path:
        """获取对话文件路径"""
        return self.base_dir / f"{conversation_id}.json"

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
        file_path = self._get_file_path(conversation_id)
        if not file_path.exists():
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [self._dict_to_message(msg) for msg in data]
        except Exception:
            return []

    def add_message(self, conversation_id: str, message: BaseMessage) -> None:
        """添加消息"""
        messages = self.get_messages(conversation_id)
        messages.append(message)
        self._save_messages(conversation_id, messages)

    def _save_messages(self, conversation_id: str, messages: List[BaseMessage]) -> None:
        """保存消息到文件"""
        file_path = self._get_file_path(conversation_id)
        data = [self._message_to_dict(msg) for msg in messages]
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def clear(self, conversation_id: str) -> None:
        """清空对话历史"""
        file_path = self._get_file_path(conversation_id)
        if file_path.exists():
            file_path.unlink()

    def list_conversations(self) -> List[str]:
        """列出所有对话 ID"""
        return [f.stem for f in self.base_dir.glob("*.json")]

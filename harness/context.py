"""Harness 的会话上下文容器。

P3 引入 token 预算感知裁剪：``snapshot()`` 调用时若估算 token 数超过
``max_tokens``，按以下策略原地修剪 ``_messages``。

裁剪契约：
1. 第一条 SystemMessage 和第一条 HumanMessage 永不动。
2. 最近 ``keep_last_turns`` 个 AIMessage 及它们对应的后续 ToolMessage 永不动。
3. AIMessage 永不动（保留 tool_calls 声明，避免后续 ToolMessage 的
   tool_call_id 孤儿）。
4. 中间区域的 ToolMessage 内容截到 ``tool_message_keep_chars`` 字符，
   加占位标记。若仍超阈值则按从旧到新顺序整条删除（含已截短的）。
5. 已达上述极限仍超阈值 → 记 warning 后放行，由 LLM 自行报 context 错误。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from loguru import logger

from config.settings import settings

from .token_counter import count_tokens

_TRUNCATION_NOTICE = "...[truncated: omitted {omitted} chars, tool_call_id={call_id}]"


class ConversationContext:
    """单次 Harness 运行的消息上下文。"""

    def __init__(
        self,
        system_prompt: str,
        initial_user_message: str,
        max_tokens: int | None = None,
        keep_last_turns: int = 2,
        tool_message_keep_chars: int = 200,
    ) -> None:
        self._messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=initial_user_message),
        ]
        self.max_tokens = (
            max_tokens if max_tokens is not None else settings.agent.context_max_tokens
        )
        self.keep_last_turns = keep_last_turns
        self.tool_message_keep_chars = tool_message_keep_chars
        self._compressed = False

    @property
    def compressed(self) -> bool:
        return self._compressed

    def append(self, message: BaseMessage) -> None:
        if not isinstance(message, BaseMessage):
            raise TypeError(f"append 只接受 BaseMessage，得到 {type(message)!r}")
        self._messages.append(message)

    def extend(self, messages: list[BaseMessage]) -> None:
        for msg in messages:
            self.append(msg)

    def snapshot(self) -> list[BaseMessage]:
        """返回当前消息列表的浅拷贝；若超 token 预算则先压缩。"""
        self.compress_if_needed()
        return list(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    # ------------------------------------------------------------------
    # 压缩
    # ------------------------------------------------------------------

    def compress_if_needed(self) -> bool:
        """检查并按需压缩。返回是否触发了压缩。"""
        if count_tokens(self._messages) <= self.max_tokens:
            return False

        protected_head = self._protected_head_indices()
        protected_tail = self._protected_tail_indices()
        middle_indices = [
            i
            for i in range(len(self._messages))
            if i not in protected_head and i not in protected_tail
        ]

        # 第一轮：截短中间的 ToolMessage 内容
        for idx in middle_indices:
            msg = self._messages[idx]
            if not isinstance(msg, ToolMessage):
                continue
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) <= self.tool_message_keep_chars:
                continue
            omitted = len(content) - self.tool_message_keep_chars
            kept = content[: self.tool_message_keep_chars]
            self._messages[idx] = ToolMessage(
                content=kept
                + _TRUNCATION_NOTICE.format(
                    omitted=omitted, call_id=msg.tool_call_id
                ),
                tool_call_id=msg.tool_call_id,
            )

        self._compressed = True

        if count_tokens(self._messages) <= self.max_tokens:
            logger.info("ConversationContext 截短中间 ToolMessage 后达标")
            return True

        # 第二轮：从旧到新整条删除中间的 ToolMessage
        # 关键：先按当前 idx 收集对象引用，再按对象引用删除；
        # 不能在删除循环里再用 idx 取 self._messages[idx] —— 删一条后下标全部前移，
        # 后续 idx 会取到错位的消息（实测：6 条中间 ToolMessage 只删 1 条剩 5 条全部漏删）。
        middle_tool_msgs = [
            self._messages[i]
            for i in middle_indices
            if isinstance(self._messages[i], ToolMessage)
        ]
        for msg in middle_tool_msgs:
            self._messages = [m for m in self._messages if m is not msg]
            if count_tokens(self._messages) <= self.max_tokens:
                logger.info("ConversationContext 删除中间 ToolMessage 后达标")
                return True

        logger.warning(
            f"ConversationContext 压缩后仍超 max_tokens={self.max_tokens}, "
            f"current≈{count_tokens(self._messages)}"
        )
        return True

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _protected_head_indices(self) -> set[int]:
        """返回 SystemMessage + 首条 HumanMessage 的下标集合。"""
        protected: set[int] = set()
        for i, msg in enumerate(self._messages):
            if isinstance(msg, SystemMessage):
                protected.add(i)
                continue
            if isinstance(msg, HumanMessage):
                protected.add(i)
                break  # 只保护第一条 HumanMessage
        return protected

    def _protected_tail_indices(self) -> set[int]:
        """返回最近 keep_last_turns 个 AIMessage 及其后跟随 ToolMessage 的下标。

        所有 AIMessage 本身也始终保护（即便不在最近 N 轮），避免孤儿 tool_call_id。
        """
        protected: set[int] = set()
        for i, msg in enumerate(self._messages):
            if isinstance(msg, AIMessage):
                protected.add(i)

        ai_indices = [i for i in protected]
        ai_indices.sort()
        last_n_ai = set(ai_indices[-self.keep_last_turns :])

        for ai_idx in last_n_ai:
            j = ai_idx + 1
            while j < len(self._messages) and isinstance(
                self._messages[j], ToolMessage
            ):
                protected.add(j)
                j += 1
        return protected


__all__ = ["ConversationContext"]

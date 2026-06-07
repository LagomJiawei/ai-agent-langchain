"""Harness 执行 trace。

一次 ``Harness.run()`` 结束后，TraceWriterHook 把这份结构化记录
写到 ``AGENT_TRACE_DIR/<chat_id or "_default">/{trace_id}.json``，
用于离线回放与故障排查。

trace_id 由 ``Harness`` 在 ``run()`` 开始处生成（``loop._new_trace_id()``），
通过 ``HookContext`` 贯穿；``parent_trace_id`` 仅在子 agent 中由
``dispatch_subagent`` 工具注入；``chat_id`` 由 FastAPI 路由层传入。

JSON 顶层字段 ``schema_version`` 是 trace schema 的版本号，离线分析脚本
应按它做兼容性切换：缺失视为 v0（最初无版本号那批），存在则按写入时的
版本解析。当前版本 = ``TRACE_SCHEMA_VERSION``。改动 schema（增删字段、
改语义、改结构）时必须递增此常量，并在 CLAUDE.md 记录变更。
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .loop import HarnessResult


# trace JSON schema 当前版本。改动 schema 时递增并在 CLAUDE.md 记录。
# v1 (2026-06): 首个带版本号的 schema；字段 = {schema_version, trace_id, started_at,
#               finished_at, user_query, stopped_reason, final_text, total_tool_calls,
#               turns, parent_trace_id, chat_id}。
TRACE_SCHEMA_VERSION = 1

_UNSAFE_CHARS = re.compile(r"[^\w\-]+")


def _sanitize_chat_id(chat_id: str | None) -> str:
    """把 chat_id 转成安全的目录名。空 / None 返回 ``"_default"``。

    所有非字母数字/下划线/连字符都替换为 ``_``，防止 ``..`` / 路径分隔符
    导致 trace 写出 base_dir 之外。
    """
    if not chat_id:
        return "_default"
    cleaned = _UNSAFE_CHARS.sub("_", chat_id).strip("_")
    return cleaned or "_default"


def new_trace_id() -> str:
    """构造 ISO 时间戳前缀 + 8 位随机后缀的 trace_id。"""
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


@dataclass
class HarnessTrace:
    # schema_version 放在 dataclass 字段第一位，JSON dump 时也排在顶层最前
    # 让离线分析脚本第一眼就能看到版本号、决定兼容策略。
    schema_version: int = field(default=TRACE_SCHEMA_VERSION)
    trace_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    user_query: str = ""
    stopped_reason: str = ""
    final_text: str = ""
    total_tool_calls: int = 0
    turns: list[dict[str, Any]] = field(default_factory=list)
    parent_trace_id: str | None = None
    chat_id: str | None = None

    @classmethod
    def from_harness_result(
        cls,
        result: "HarnessResult",
        user_query: str,
        started_at: str,
        finished_at: str,
        trace_id: str,
        parent_trace_id: str | None = None,
        chat_id: str | None = None,
    ) -> "HarnessTrace":
        turns_payload: list[dict[str, Any]] = []
        for turn in result.turns:
            turns_payload.append(
                {
                    "index": turn.index,
                    "thought": turn.thought,
                    "final_text": turn.final_text,
                    "tool_calls": [asdict(c) for c in turn.tool_calls],
                    "tool_results": [asdict(r) for r in turn.tool_results],
                }
            )
        return cls(
            schema_version=TRACE_SCHEMA_VERSION,
            trace_id=trace_id,
            started_at=started_at,
            finished_at=finished_at,
            user_query=user_query,
            stopped_reason=result.stopped_reason,
            final_text=result.final_text,
            total_tool_calls=result.total_tool_calls,
            turns=turns_payload,
            parent_trace_id=parent_trace_id,
            chat_id=chat_id,
        )

    def write(self, base_dir: Path | str) -> Path:
        bucket = _sanitize_chat_id(self.chat_id)
        target_dir = Path(base_dir) / bucket
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{self.trace_id}.json"
        path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


__all__ = ["HarnessTrace", "new_trace_id", "TRACE_SCHEMA_VERSION"]

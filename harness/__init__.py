"""LiCaiManus Harness 执行框架。

替代旧 `agents/` 目录里的 ReAct + Plan-Execute 双实现，
提供单一主 loop + Tool Registry + Context Manager + Hook 总线 + Subagent 派发。
"""
from .context import ConversationContext
from .events import EventType, HarnessEvent
from .hooks import HookBus, HookContext, OnStopHook, PostToolUseHook, PreToolUseHook
from .loop import Harness, HarnessResult, StoppedReason, current_chat_id, current_trace_id
from .registry import ToolRegistry, default_registry, register_tool
from .subagent import dispatch_subagent, dispatch_subagents
from .token_counter import count_tokens
from .trace import HarnessTrace, new_trace_id
from .turn import ToolCall, ToolResult, Turn


def default_hooks(*args, **kwargs) -> HookBus:
    """生产配置下的默认 hook 总线（延迟导入避开循环依赖）。"""
    from .builtin_hooks import default_hooks as _impl

    return _impl(*args, **kwargs)


def TraceWriterHook(*args, **kwargs):
    """转发到 builtin_hooks.TraceWriterHook（延迟导入）。"""
    from .builtin_hooks import TraceWriterHook as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "ConversationContext",
    "EventType",
    "Harness",
    "HarnessEvent",
    "HarnessResult",
    "HarnessTrace",
    "HookBus",
    "HookContext",
    "OnStopHook",
    "PostToolUseHook",
    "PreToolUseHook",
    "StoppedReason",
    "ToolRegistry",
    "TraceWriterHook",
    "count_tokens",
    "current_chat_id",
    "current_trace_id",
    "default_hooks",
    "default_registry",
    "dispatch_subagent",
    "dispatch_subagents",
    "new_trace_id",
    "register_tool",
    "Turn",
    "ToolCall",
    "ToolResult",
]

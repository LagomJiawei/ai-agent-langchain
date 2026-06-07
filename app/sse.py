"""SSE 流式响应辅助工具。

主要目的：在不破坏现有 ``StreamingResponse`` 协议格式的前提下，
给业务事件流加 keepalive 心跳，避免长任务下游 idle 导致浏览器 /
反向代理（Nginx 默认 60s）断开连接。

设计：
- 心跳格式是 **SSE comment**（``: keepalive <unix_ts>\\n\\n``），
  符合 WHATWG SSE 规范；标准 ``EventSource`` 客户端自动忽略 comment 行，
  对业务事件 zero impact。
- 用 ``asyncio.Queue`` + ``asyncio.wait_for`` 实现"idle 触发心跳"，
  上游有数据时不阻塞，无延迟透传业务事件。
- 上游异常通过 ``_ErrorMarker`` 透传，让原 generator 的 try/except 仍能捕获。
- 客户端断连 → FastAPI cancel 外层 generator → 内部 pump 任务被 cancel，
  无 task 泄漏。
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from typing import AsyncIterator


class _ErrorMarker:
    """把上游异常装入 queue，让消费端按序 raise 而不是吃掉它。"""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


_SENTINEL: object = object()


async def with_keepalive(
    source: AsyncIterator[str],
    interval: float = 15.0,
) -> AsyncIterator[str]:
    """把上游 SSE 字符串流包一层 keepalive。

    Args:
        source: 上游 async generator，yield 形如 ``"event: X\\ndata: {...}\\n\\n"`` 的 SSE chunk。
        interval: 心跳间隔（秒）。``<= 0`` 时禁用心跳，退化为透传上游。

    Yields:
        与上游同形态的 SSE chunk；idle 超过 ``interval`` 秒时插入
        ``": keepalive <unix_ts>\\n\\n"`` 注释行。
    """
    if interval <= 0:
        async for chunk in source:
            yield chunk
        return

    queue: asyncio.Queue = asyncio.Queue()

    async def pump() -> None:
        try:
            async for chunk in source:
                await queue.put(chunk)
        except BaseException as exc:  # noqa: BLE001
            # 包含 asyncio.CancelledError —— 上游 cancel 也透传，让外层 finally 能感知
            await queue.put(_ErrorMarker(exc))
        finally:
            await queue.put(_SENTINEL)

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=interval)
            except asyncio.TimeoutError:
                yield f": keepalive {int(time.time())}\n\n"
                continue

            if item is _SENTINEL:
                return
            if isinstance(item, _ErrorMarker):
                raise item.exc
            yield item
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


__all__ = ["with_keepalive"]

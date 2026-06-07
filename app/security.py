"""身份与多租户隔离。

最小可用身份层：HTTP 请求带 ``X-API-Key`` 头，服务启动时从
``settings.security.api_keys`` 加载映射，把 key 解析为 ``Principal``。
路由层用 ``principal.namespace_chat_id(raw)`` 把客户端传入的 chat_id
前缀化为 ``<principal>:<raw>``，Harness / store / trace 层零感知。

设计取舍：
- 不引入用户表 / JWT / OAuth；与项目 demo 规模匹配。
- 命名空间化只发生在路由层，下游接口完全不动。
- 默认 ``ALLOW_ANONYMOUS=true`` 向后兼容；强鉴权一个 env 切换。
"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header, HTTPException, status

from config import settings


@dataclass(frozen=True)
class Principal:
    """已解析的调用方身份。"""

    principal_id: str
    is_anonymous: bool = False

    def namespace_chat_id(self, raw: str | None) -> str:
        """把客户端 chat_id 命名空间化为 ``<principal>:<raw>``。

        ``raw`` 为空或 None 时回退到 ``default``。
        """
        safe_raw = raw if raw else "default"
        return f"{self.principal_id}:{safe_raw}"


_ANONYMOUS_ID = "anonymous"


async def resolve_principal(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
        convert_underscores=False,
    ),
) -> Principal:
    """FastAPI dependency：从 ``X-API-Key`` 头解析 Principal。

    - 带合法 key → 对应 principal。
    - 带未知 key → 401。
    - 不带 key + allow_anonymous=True → ``anonymous`` principal。
    - 不带 key + allow_anonymous=False → 401。
    """
    cfg = settings.security
    if x_api_key:
        principal_id = cfg.api_keys.get(x_api_key)
        if principal_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key",
            )
        return Principal(principal_id=principal_id, is_anonymous=False)

    if cfg.allow_anonymous:
        return Principal(principal_id=_ANONYMOUS_ID, is_anonymous=True)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="API Key required",
    )


__all__ = ["Principal", "resolve_principal"]

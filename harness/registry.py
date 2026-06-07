"""工具注册表。

按名字与 scope 管理工具，支持 scope 单值或 scopes 多值并集过滤。
未来 Harness 主 loop 可按 scope 把工具子集注入到不同的 subagent。
"""
from __future__ import annotations

from langchain_core.tools import BaseTool


class ToolRegistry:
    """按名字与 scope 管理工具的容器。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._scopes: dict[str, str] = {}

    def register(self, tool: BaseTool, *, scope: str = "default") -> None:
        if not isinstance(tool, BaseTool):
            raise TypeError(f"register 只接受 BaseTool 实例，得到 {type(tool)!r}")
        if tool.name in self._tools:
            raise ValueError(f"工具名重复注册: {tool.name}")
        self._tools[tool.name] = tool
        self._scopes[tool.name] = scope

    def get(self, name: str) -> BaseTool:
        if name not in self._tools:
            raise KeyError(f"未注册的工具: {name}")
        return self._tools[name]

    def scope_of(self, name: str) -> str:
        if name not in self._scopes:
            raise KeyError(f"未注册的工具: {name}")
        return self._scopes[name]

    def list(
        self,
        scope: str | None = None,
        scopes: list[str] | None = None,
    ) -> list[BaseTool]:
        if scope is not None and scopes is not None:
            raise ValueError("scope 与 scopes 二选一")
        if scope is not None:
            return [t for n, t in self._tools.items() if self._scopes[n] == scope]
        if scopes is not None:
            scope_set = set(scopes)
            return [t for n, t in self._tools.items() if self._scopes[n] in scope_set]
        return list(self._tools.values())

    def names(
        self,
        scope: str | None = None,
        scopes: list[str] | None = None,
    ) -> list[str]:
        return [t.name for t in self.list(scope=scope, scopes=scopes)]

    def clear(self) -> None:
        """仅供测试使用。"""
        self._tools.clear()
        self._scopes.clear()


default_registry = ToolRegistry()


def register_tool(tool: BaseTool, *, scope: str = "default") -> None:
    """向默认注册表注册一个工具。"""
    default_registry.register(tool, scope=scope)

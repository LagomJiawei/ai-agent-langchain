"""Harness 的 llm_with_tools 进程级缓存测试（回归 #11）。

设计要点：
- 缓存挂在 llm 实例上（``_harness_bind_cache``），同 llm + 同 tool 名集合命中。
- 不同 llm 实例 → 不同缓存（无跨实例污染）。
- 不同 tool 集合 → 不同缓存 entry（按 frozenset key）。
- 没有 ``bind_tools`` 的测试桩 / 工具列表为空 → 退化，不走缓存。
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from harness import Harness, HookBus, ToolRegistry
from harness.loop import _BIND_CACHE_ATTR, _with_bind_tools_cached


# ---------- 桩 ----------


class _CountingLLM:
    """记录 bind_tools 被调用次数的桩；实例化要保留 setattr 能力。"""

    def __init__(self):
        self.bind_calls = 0

    def bind_tools(self, tools):
        self.bind_calls += 1
        # 返回一个能被识别的对象，让测试能 assert is
        return _BoundFor(self, tools)


class _BoundFor:
    """模拟 bind_tools 的返回值。"""

    def __init__(self, llm: _CountingLLM, tools):
        self.llm = llm
        self.tools = list(tools)


class _NoBindLLM:
    """完全没有 bind_tools，模拟 FakeStreamLLM 等测试桩。"""


@tool
def _t1(x: str = "") -> str:
    """t1"""
    return "1"


@tool
def _t2(x: str = "") -> str:
    """t2"""
    return "2"


@tool
def _t3(x: str = "") -> str:
    """t3"""
    return "3"


# ---------- _with_bind_tools_cached 直接测 ----------


def test_cache_returns_same_object_on_repeated_call():
    llm = _CountingLLM()
    bound1 = _with_bind_tools_cached(llm, [_t1, _t2])
    bound2 = _with_bind_tools_cached(llm, [_t1, _t2])
    assert bound1 is bound2
    assert llm.bind_calls == 1


def test_cache_distinguishes_by_tool_name_set():
    llm = _CountingLLM()
    bound_a = _with_bind_tools_cached(llm, [_t1, _t2])
    bound_b = _with_bind_tools_cached(llm, [_t1, _t3])
    assert bound_a is not bound_b
    assert llm.bind_calls == 2


def test_cache_is_order_independent_within_same_set():
    """frozenset key 保证 [t1, t2] 与 [t2, t1] 命中同一 entry。"""
    llm = _CountingLLM()
    b1 = _with_bind_tools_cached(llm, [_t1, _t2])
    b2 = _with_bind_tools_cached(llm, [_t2, _t1])
    assert b1 is b2
    assert llm.bind_calls == 1


def test_cache_isolated_per_llm_instance():
    """不同 llm 实例各有独立缓存，互不污染。"""
    llm_a = _CountingLLM()
    llm_b = _CountingLLM()
    bound_a = _with_bind_tools_cached(llm_a, [_t1])
    bound_b = _with_bind_tools_cached(llm_b, [_t1])
    assert bound_a is not bound_b
    assert bound_a.llm is llm_a
    assert bound_b.llm is llm_b


def test_no_bind_tools_method_returns_llm_as_is():
    """测试桩没有 bind_tools → 原样返回 llm，不挂缓存。"""
    llm = _NoBindLLM()
    out = _with_bind_tools_cached(llm, [_t1])
    assert out is llm
    assert not hasattr(llm, _BIND_CACHE_ATTR)


def test_empty_tool_list_returns_llm_as_is():
    """空 tool 列表 → 原样返回 llm，不调 bind_tools，不挂缓存。"""
    llm = _CountingLLM()
    out = _with_bind_tools_cached(llm, [])
    assert out is llm
    assert llm.bind_calls == 0
    assert not hasattr(llm, _BIND_CACHE_ATTR)


# ---------- 在真实 Harness 上验证 ----------


def test_harness_instances_with_same_llm_share_bound_tools():
    """两个 Harness 共享同一 llm + 同一 registry → llm_with_tools 复用。"""
    llm = _CountingLLM()
    reg = ToolRegistry()
    reg.register(_t1, scope="test")
    reg.register(_t2, scope="test")

    h1 = Harness(llm=llm, registry=reg, hooks=HookBus(), max_iterations=1)
    h2 = Harness(llm=llm, registry=reg, hooks=HookBus(), max_iterations=1)

    assert h1.llm_with_tools is h2.llm_with_tools
    assert llm.bind_calls == 1


def test_harness_instances_with_same_llm_different_scopes_dont_collide():
    """同 llm + 不同 scope → 各自独立 cache entry，不串。"""
    llm = _CountingLLM()
    reg_kb = ToolRegistry()
    reg_kb.register(_t1, scope="kb")
    reg_web = ToolRegistry()
    reg_web.register(_t2, scope="web")

    h_kb = Harness(llm=llm, registry=reg_kb, hooks=HookBus(), max_iterations=1)
    h_web = Harness(llm=llm, registry=reg_web, hooks=HookBus(), max_iterations=1)

    assert h_kb.llm_with_tools is not h_web.llm_with_tools
    # 但 kb 再 new 一次 Harness，仍命中 cache
    h_kb_again = Harness(llm=llm, registry=reg_kb, hooks=HookBus(), max_iterations=1)
    assert h_kb_again.llm_with_tools is h_kb.llm_with_tools
    assert llm.bind_calls == 2  # 只 bind 了 kb / web 各一次


def test_harness_with_fake_stub_llm_bypasses_cache():
    """FakeStreamLLM 这种测试桩没有 bind_tools → llm_with_tools 就是 llm 本身。"""
    llm = _NoBindLLM()
    reg = ToolRegistry()
    reg.register(_t1, scope="test")

    h = Harness(llm=llm, registry=reg, hooks=HookBus(), max_iterations=1)
    assert h.llm_with_tools is llm

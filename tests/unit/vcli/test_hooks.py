"""Tests for ToolHookRegistry — pre/post tool execution hooks."""
from vector_os_nano.vcli.hooks import ToolHookContext, ToolHookRegistry
from vector_os_nano.vcli.tools.base import ToolResult


class TestToolHookRegistry:
    def test_pre_hook_fires(self):
        registry = ToolHookRegistry()
        calls = []
        registry.add_pre_hook(lambda ctx: calls.append(("pre", ctx.tool_name)))
        ctx = ToolHookContext(tool_name="navigate", params={"room": "kitchen"})
        registry.fire_pre(ctx)
        assert calls == [("pre", "navigate")]

    def test_post_hook_fires_with_result(self):
        registry = ToolHookRegistry()
        calls = []
        registry.add_post_hook(lambda ctx: calls.append((ctx.tool_name, ctx.result.content)))
        result = ToolResult(content="ok")
        ctx = ToolHookContext(tool_name="navigate", params={}, result=result, duration=1.5)
        registry.fire_post(ctx)
        assert calls == [("navigate", "ok")]

    def test_multiple_hooks_fire_in_order(self):
        registry = ToolHookRegistry()
        order = []
        registry.add_post_hook(lambda ctx: order.append("first"))
        registry.add_post_hook(lambda ctx: order.append("second"))
        ctx = ToolHookContext(tool_name="test", params={}, result=ToolResult(content=""))
        registry.fire_post(ctx)
        assert order == ["first", "second"]

    def test_hook_exception_swallowed(self):
        registry = ToolHookRegistry()
        registry.add_pre_hook(lambda ctx: 1 / 0)  # ZeroDivisionError
        calls_after = []
        registry.add_pre_hook(lambda ctx: calls_after.append("survived"))
        ctx = ToolHookContext(tool_name="test", params={})
        registry.fire_pre(ctx)  # should not raise
        assert calls_after == ["survived"]

    def test_pre_hook_has_no_result(self):
        registry = ToolHookRegistry()
        results = []
        registry.add_pre_hook(lambda ctx: results.append(ctx.result))
        ctx = ToolHookContext(tool_name="test", params={})
        registry.fire_pre(ctx)
        assert results == [None]

    def test_post_hook_has_duration(self):
        registry = ToolHookRegistry()
        durations = []
        registry.add_post_hook(lambda ctx: durations.append(ctx.duration))
        ctx = ToolHookContext(tool_name="test", params={}, result=ToolResult(content=""), duration=2.5)
        registry.fire_post(ctx)
        assert durations == [2.5]

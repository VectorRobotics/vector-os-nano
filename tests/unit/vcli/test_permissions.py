# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Unit tests for vcli permission system — TDD RED phase.

Covers:
- deny rule blocks tool
- ask rule for motor tool
- allow rule for read-only tool
- always-allow persists in session
- deny overrides allow
- tool-specific check_permissions returning deny
- default is ask for unknown tool
- dangerous path safety (bash)
- check flow order: deny -> tool-specific -> allow -> read-only -> ask
- no_permission mode: everything allowed
"""
from __future__ import annotations

from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal tool stubs
# ---------------------------------------------------------------------------


def _make_tool(
    name: str,
    *,
    is_read_only_result: bool = False,
    check_permissions_result: str = "ask",
    check_permissions_reason: str = "",
) -> Any:
    """Return a minimal tool-like object (not decorated, plain duck-typing)."""
    from vector_os_nano.vcli.tools.base import PermissionResult

    class _Tool:
        def is_read_only(self, params: dict[str, Any]) -> bool:
            return is_read_only_result

        def check_permissions(
            self, params: dict[str, Any], context: Any
        ) -> PermissionResult:
            return PermissionResult(
                behavior=check_permissions_result,
                reason=check_permissions_reason,
            )

    t = _Tool()
    t.name = name  # type: ignore[attr-defined]
    return t


def _make_bare_tool(name: str) -> Any:
    """Return a bare object with only a name attribute (no helper methods)."""

    class _Bare:
        pass

    t = _Bare()
    t.name = name  # type: ignore[attr-defined]
    return t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPermissionContext:
    def test_deny_rule_blocks_tool(self) -> None:
        """Tool in deny set returns PermissionResult('deny')."""
        from vector_os_nano.vcli.permissions import PermissionContext

        ctx = PermissionContext(deny_tools={"bad_tool"})
        tool = _make_bare_tool("bad_tool")
        result = ctx.check(tool, {})
        assert result.behavior == "deny"

    def test_ask_rule_for_motor_tool(self) -> None:
        """Motor tool with check_permissions returning 'ask' propagates ask."""
        from vector_os_nano.vcli.permissions import PermissionContext

        ctx = PermissionContext()
        tool = _make_tool("motor_move", check_permissions_result="ask")
        result = ctx.check(tool, {})
        assert result.behavior == "ask"

    def test_allow_rule_for_read_only(self) -> None:
        """A read-only tool with no check of its own auto-allows without prompting."""
        from vector_os_nano.vcli.permissions import PermissionContext

        class _ReadOnly:
            name = "read_file"

            def is_read_only(self, params: dict[str, Any]) -> bool:
                return True

        assert PermissionContext().check(_ReadOnly(), {}).behavior == "allow"

    def test_read_only_never_overrides_the_tools_own_ask(self) -> None:
        """R3: a read-only tool that ASKS (file_read outside the project, web_fetch)
        must prompt; the read-only flag used to turn that ask into a silent allow."""
        from vector_os_nano.vcli.permissions import PermissionContext

        tool = _make_tool("read_file", is_read_only_result=True, check_permissions_result="ask")
        assert PermissionContext().check(tool, {}).behavior == "ask"

    def test_always_allow_persists_in_session(self) -> None:
        """After add_always_allow('bash'), subsequent check returns allow."""
        from vector_os_nano.vcli.permissions import PermissionContext

        ctx = PermissionContext()
        ctx.add_always_allow("bash")
        tool = _make_tool("bash", check_permissions_result="ask")
        result = ctx.check(tool, {})
        assert result.behavior == "allow"

    def test_deny_overrides_allow(self) -> None:
        """Tool in both deny and session_allow — deny wins."""
        from vector_os_nano.vcli.permissions import PermissionContext

        ctx = PermissionContext(deny_tools={"bash"}, session_allow={"bash"})
        tool = _make_tool("bash", check_permissions_result="allow")
        result = ctx.check(tool, {})
        assert result.behavior == "deny"

    def test_tool_specific_safety_check(self) -> None:
        """Tool.check_permissions returning 'deny' is respected even if allow rule exists."""
        from vector_os_nano.vcli.permissions import PermissionContext

        ctx = PermissionContext(session_allow={"dangerous_tool"})
        tool = _make_tool(
            "dangerous_tool",
            check_permissions_result="deny",
            check_permissions_reason="unsafe path",
        )
        result = ctx.check(tool, {})
        assert result.behavior == "deny"
        assert "unsafe path" in result.reason

    def test_default_is_ask(self) -> None:
        """Unknown tool with no rules and non-read-only returns ask."""
        from vector_os_nano.vcli.permissions import PermissionContext

        ctx = PermissionContext()
        # Bare tool has no check_permissions or is_read_only
        tool = _make_bare_tool("unknown_tool")
        result = ctx.check(tool, {})
        assert result.behavior == "ask"

    def test_dangerous_path_safety(self) -> None:
        """Bash tool invoked with a dangerous path is denied regardless of allow rules."""
        from vector_os_nano.vcli.permissions import PermissionContext

        ctx = PermissionContext(session_allow={"bash"})

        class BashTool:
            name = "bash"

            def is_read_only(self, params: dict[str, Any]) -> bool:
                return False

            def check_permissions(
                self, params: dict[str, Any], context: Any
            ) -> "PermissionResult":
                from vector_os_nano.vcli.tools.base import PermissionResult

                command = params.get("command", "")
                danger_paths = ("rm -rf /", "rm -rf /*", "> /etc/passwd", "/dev/sda")
                for danger in danger_paths:
                    if danger in command:
                        return PermissionResult("deny", f"Dangerous path: {danger}")
                return PermissionResult("ask")

        tool = BashTool()
        result = ctx.check(tool, {"command": "rm -rf /"})
        assert result.behavior == "deny"

    def test_check_flow_order(self) -> None:
        """Check flow: deny -> tool-specific deny -> session-allow -> read-only -> ask."""
        from vector_os_nano.vcli.permissions import PermissionContext

        # 1. Deny fires first
        ctx1 = PermissionContext(deny_tools={"x"}, session_allow={"x"})
        tool_deny = _make_tool("x", check_permissions_result="allow", is_read_only_result=True)
        assert ctx1.check(tool_deny, {}).behavior == "deny"

        # 2. Tool-specific deny before session-allow
        ctx2 = PermissionContext(session_allow={"x"})
        tool_tool_deny = _make_tool(
            "x", check_permissions_result="deny", is_read_only_result=True
        )
        assert ctx2.check(tool_tool_deny, {}).behavior == "deny"

        # 3. Session-allow before read-only (both would allow anyway, but allow fires)
        ctx3 = PermissionContext(session_allow={"x"})
        tool_session = _make_tool("x", check_permissions_result="ask", is_read_only_result=True)
        assert ctx3.check(tool_session, {}).behavior == "allow"

        # 4. Read-only does NOT override the tool's own ask (R3)
        ctx4 = PermissionContext()
        tool_ro = _make_tool("x", check_permissions_result="ask", is_read_only_result=True)
        assert ctx4.check(tool_ro, {}).behavior == "ask"

        # 5. Default ask when nothing matches
        ctx5 = PermissionContext()
        tool_ask = _make_bare_tool("x")
        assert ctx5.check(tool_ask, {}).behavior == "ask"

    def test_no_permission_mode(self) -> None:
        """no_permission=True allows everything, including tools in deny set."""
        from vector_os_nano.vcli.permissions import PermissionContext

        ctx = PermissionContext(deny_tools={"rm_rf"}, no_permission=True)
        tool = _make_bare_tool("rm_rf")
        result = ctx.check(tool, {})
        assert result.behavior == "allow"

    def test_add_deny(self) -> None:
        """add_deny() adds tool to deny set and subsequent check denies it."""
        from vector_os_nano.vcli.permissions import PermissionContext

        ctx = PermissionContext()
        ctx.add_deny("write_file")
        tool = _make_bare_tool("write_file")
        result = ctx.check(tool, {})
        assert result.behavior == "deny"

    def test_permission_result_reason_in_deny(self) -> None:
        """Deny result includes a human-readable reason."""
        from vector_os_nano.vcli.permissions import PermissionContext

        ctx = PermissionContext(deny_tools={"bad"})
        tool = _make_bare_tool("bad")
        result = ctx.check(tool, {})
        assert result.behavior == "deny"
        assert "bad" in result.reason

    def test_check_permissions_called_only_once(self) -> None:
        """check_permissions() on the tool is invoked exactly once per PermissionContext.check() call.

        Regression test: the old implementation called check_permissions twice — once
        in step 2 (deny guard) and again in step 5 (ask propagation).  The fix caches
        the result from step 2 and reuses it in step 5.
        """
        from unittest.mock import MagicMock

        from vector_os_nano.vcli.permissions import PermissionContext
        from vector_os_nano.vcli.tools.base import PermissionResult

        ctx = PermissionContext()

        mock_tool = MagicMock()
        mock_tool.name = "instrumented_tool"
        # check_permissions returns "ask" — triggers propagation path in step 5
        mock_tool.check_permissions.return_value = PermissionResult(behavior="ask")
        # Not read-only — so read-only auto-allow does not short-circuit
        mock_tool.is_read_only.return_value = False

        result = ctx.check(mock_tool, {})

        assert result.behavior == "ask"
        # The critical assertion: check_permissions must have been called exactly once
        mock_tool.check_permissions.assert_called_once()

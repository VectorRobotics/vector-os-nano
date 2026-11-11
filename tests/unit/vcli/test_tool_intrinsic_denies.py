# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Intrinsic tool denies survive --no-permission (moved from
tests/vcli/test_level68_review_fixes.py when its legacy-producer half was deleted).

- the bash deny-list is an unconditional stop; a USER deny may still be overridden;
- file_write / file_edit hard-deny protected paths (write-path parity with read).
"""
from __future__ import annotations

import threading
from pathlib import Path

from vector_os_nano.vcli.permissions import PermissionContext
from vector_os_nano.vcli.tools.base import ToolContext
from vector_os_nano.vcli.tools.bash_tool import BashTool
from vector_os_nano.vcli.tools.file_tools import FileEditTool, FileWriteTool


def _ctx(tmp: Path) -> ToolContext:
    return ToolContext(agent=None, cwd=tmp, session=None,
                       permissions=PermissionContext(), abort=threading.Event())


def test_bash_deny_list_survives_no_permission() -> None:
    ctx = PermissionContext(no_permission=True)
    assert ctx.check(BashTool(), {"command": "rm -rf /"}).behavior == "deny"
    # a safe command under no_permission is still allowed
    assert ctx.check(BashTool(), {"command": "ls"}).behavior == "allow"


def test_user_deny_tools_still_overridable_by_no_permission() -> None:
    # The narrower change: a USER deny preference may still be overridden by
    # --no-permission (only intrinsic tool denies are unconditional).
    ctx = PermissionContext(deny_tools={"bash"}, no_permission=True)
    assert ctx.check(BashTool(), {"command": "ls"}).behavior == "allow"


def test_file_write_denies_protected_path_via_permissions() -> None:
    res = FileWriteTool().check_permissions({"file_path": "~/.ssh/authorized_keys", "content": "x"}, None)
    assert res.behavior == "deny"
    # survives --no-permission too (intrinsic deny)
    ctx = PermissionContext(no_permission=True)
    assert ctx.check(FileWriteTool(), {"file_path": "/etc/shadow", "content": "x"}).behavior == "deny"


def test_file_edit_denies_protected_path() -> None:
    res = FileEditTool().check_permissions(
        {"file_path": "~/.claude/settings.json", "old_string": "a", "new_string": "b"}, None)
    assert res.behavior == "deny"


def test_file_write_execute_blocks_protected_path(tmp_path: Path) -> None:
    r = FileWriteTool().execute({"file_path": "~/.ssh/authorized_keys", "content": "x"}, _ctx(tmp_path))
    assert r.is_error
    assert "protected" in r.content.lower()

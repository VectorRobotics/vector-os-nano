# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E1 launch: '启动 go2 仿真 带机械臂 无窗口' -> a 'ready' line naming go2_piper."""
from __future__ import annotations

import re

import pytest

from tests.e2e_repl import LAUNCH, LAUNCH_STEPS, READY
from tests.harness.pty_cli import run_repl_session


@pytest.mark.sim
@pytest.mark.cli_main
@pytest.mark.xfail(strict=True, reason="acceptance path = subprocess bridge until WP03")
def test_e1_nl_launch_go2_piper_reports_ready() -> None:
    r = run_repl_session([LAUNCH], tool_script={"steps": LAUNCH_STEPS}, answers=["y"], timeout_sec=150)
    text = r.text
    assert not r.timed_out, text[-3000:]
    assert re.search(READY, text) and "go2_piper" in text, text[-3000:]

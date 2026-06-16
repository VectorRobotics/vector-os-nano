# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for IntentRouter — keyword-based tool category routing."""
from vector_os_nano.vcli.intent_router import IntentRouter


class TestIntentRouter:
    def setup_method(self):
        self.router = IntentRouter()

    def test_code_intent_chinese(self):
        result = self.router.route("改一下代码")
        assert result is not None
        assert "code" in result

    def test_code_intent_english(self):
        result = self.router.route("fix the bug in navigate.py")
        assert result is not None
        assert "code" in result

    def test_robot_intent_chinese(self):
        result = self.router.route("去厨房")
        assert result is not None
        assert "robot" in result

    def test_robot_intent_english(self):
        result = self.router.route("navigate to kitchen")
        assert result is not None
        assert "robot" in result

    def test_diag_intent(self):
        result = self.router.route("FAR 为什么不工作")
        assert result is not None
        assert "diag" in result

    def test_sim_intent(self):
        result = self.router.route("启动仿真")
        assert result is not None
        assert "system" in result

    def test_ambiguous_returns_none(self):
        result = self.router.route("你好")
        assert result is None

    def test_mixed_intent(self):
        result = self.router.route("改完代码然后去厨房")
        assert result is not None
        assert "code" in result
        assert "robot" in result

    def test_explore_chinese(self):
        result = self.router.route("探索一下房子")
        assert result is not None
        assert "robot" in result

    def test_result_is_sorted(self):
        result = self.router.route("去厨房看看")
        if result is not None:
            assert result == sorted(result)


class TestSwitchIntentRouting:
    """Campaign #11 R11 — an embodiment-switch NL command must reach the
    answer/tool_use path (where switch_embodiment lives) in BOTH zh and en, and
    must keep raw bash out of the candidate set, without swallowing genuine
    navigate/explore commands (which legitimately go through VGG)."""

    def setup_method(self):
        self.router = IntentRouter()

    def test_switch_intent_bypasses_vgg(self):
        # These previously routed to VGG (Chinese '到' tripped the motor check),
        # where switch_embodiment (a @tool, not a @skill) is unreachable. They
        # must take the tool_use/answer path instead.
        for msg in ("切到go2", "切换到go2", "切到g1", "切换到 g1",
                    "换成go2", "换到go2", "switch to go2", "switch to g1",
                    "change to go2", "变成狗"):
            assert self.router.should_use_vgg(msg) is False, msg

    def test_switch_route_excludes_bash_includes_sim(self):
        # Pin the invariant: a keyword-matched switch route never offers bash
        # (category 'code') and always offers switch_embodiment (category 'sim').
        for msg in ("切到go2", "switch to go2", "切换到go2", "换成go2"):
            result = self.router.route(msg)
            assert result == ["robot", "sim", "system"], (msg, result)
            assert "code" not in result          # bash lives in 'code'

    def test_navigate_not_falsely_bypassed(self):
        # The switch bypass must NOT swallow legitimate motor commands — these
        # must still go through VGG (should_use_vgg True).
        for msg in ("去厨房", "navigate to the kitchen", "走到客厅",
                    "探索一下房子", "走过去"):
            assert self.router.should_use_vgg(msg) is True, msg

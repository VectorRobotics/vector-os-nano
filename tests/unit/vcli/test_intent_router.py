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

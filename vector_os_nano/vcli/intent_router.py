"""Intent-based tool routing for Vector CLI.

Classifies user messages by keyword matching to select relevant tool
categories. Reduces token cost by sending only related tools to the LLM.

Zero-cost: pure keyword match, no LLM call. Falls back to all tools
when intent is ambiguous.
"""
from __future__ import annotations


# (keywords, categories) — checked in order, all matches accumulated
_RULES: list[tuple[frozenset[str], tuple[str, ...]]] = [
    # Code editing
    (frozenset({
        "改", "修改", "编辑", "代码", "文件", "函数", "变量", "类",
        "edit", "fix", "code", "file", "function", "class", "import",
        "read", "write", "bug", "refactor", "重构", "写",
    }), ("code", "system")),

    # Robot control
    (frozenset({
        "去", "走", "站", "坐", "趴", "探索", "导航", "看", "抓", "放",
        "navigate", "walk", "stand", "sit", "explore", "pick", "place",
        "look", "patrol", "stop", "停", "home", "回家", "扫描",
        "wave", "挥手", "turn", "转",
    }), ("robot", "diag")),

    # Diagnostics
    (frozenset({
        "topic", "node", "ros2", "ros", "log", "日志", "状态", "诊断",
        "far", "tare", "terrain", "debug", "为什么", "检查", "查",
        "hz", "频率", "进程", "bridge",
    }), ("diag", "system")),

    # Simulation
    (frozenset({
        "仿真", "sim", "simulation", "reset", "重置", "启动", "模拟",
    }), ("system", "robot")),
]


class IntentRouter:
    """Classify user intent to select relevant tool categories.

    Returns a list of category names, or None when intent is ambiguous
    (meaning all tools should be sent).
    """

    def route(self, user_message: str) -> list[str] | None:
        """Classify user message into tool categories.

        Returns:
            Sorted list of category names, or None for all categories.
        """
        msg = user_message.lower()

        matched: set[str] = set()
        for keywords, categories in _RULES:
            if any(kw in msg for kw in keywords):
                matched.update(categories)

        if not matched:
            return None  # ambiguous → send all tools

        return sorted(matched)

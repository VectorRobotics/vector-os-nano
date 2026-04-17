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



# ---------------------------------------------------------------------------
# Complexity detection keyword sets
# ---------------------------------------------------------------------------

# Sequential: implies ordering / multi-step flow
_SEQUENTIAL_KEYWORDS: frozenset[str] = frozenset({
    "然后", "再", "接着", "之后",
    "and then", "then",
})

# Conditional: implies branching logic
_CONDITIONAL_KEYWORDS: frozenset[str] = frozenset({
    "如果", "假如",
    "if", "whether",
})

# Scope: implies iteration over multiple targets
_SCOPE_KEYWORDS: frozenset[str] = frozenset({
    "所有", "每个", "检查所有",
    "all rooms", "every", "each",
})

# Simultaneous conjunction joining actions
_SIMULTANEOUS_KEYWORDS: frozenset[str] = frozenset({
    "同时",
})

# Perception + judgment patterns (requires multi-step reasoning)
_PERCEPTION_JUDGMENT_PHRASES: tuple[str, ...] = (
    "看看有没有", "看看是否",
    "check if", "see if",
)

# Action verb groups — synonyms within a group count as ONE action.
# If 2+ distinct GROUPS are present, task is multi-action → complex.
_ACTION_VERB_GROUPS: list[tuple[str, frozenset[str]]] = [
    ("navigate", frozenset({"去", "到", "导航", "走到", "去到", "go", "navigate"})),
    ("move", frozenset({"走", "前进", "walk"})),
    ("look", frozenset({"看", "观察", "look", "scan"})),
    ("find", frozenset({"找", "检查", "check", "find"})),
    ("explore", frozenset({"探索", "explore"})),
    ("patrol", frozenset({"巡逻", "patrol"})),
    ("pick", frozenset({"拿", "抓", "pick"})),
    ("place", frozenset({"放", "place"})),
    ("stop", frozenset({"停止", "结束", "stop"})),
    ("start", frozenset({"开始", "start"})),
    ("stance", frozenset({"站", "坐", "stand", "sit"})),
]

# Flat set of all action verbs (for should_use_vgg single-verb check)
_ACTION_VERBS: frozenset[str] = frozenset(
    verb for _, group in _ACTION_VERB_GROUPS for verb in group
)


def _has_multiple_actions(msg: str) -> bool:
    """Return True if message contains 2+ distinct action verb groups."""
    import re
    matched_groups: set[str] = set()
    msg_lower = msg.lower()
    for group_name, verbs in _ACTION_VERB_GROUPS:
        for verb in verbs:
            if verb in msg_lower:
                if verb.isascii():
                    if re.search(r'\b' + re.escape(verb) + r'\b', msg_lower):
                        matched_groups.add(group_name)
                        break
                else:
                    matched_groups.add(group_name)
                    break
        if len(matched_groups) >= 2:
            return True
    return False


# Motor action patterns — these should go through VGG for async execution
_MOTOR_PATTERNS: tuple[str, ...] = (
    "去", "到", "走到", "去到", "导航",
    "go to", "goto", "navigate to",
    "巡逻", "patrol",
    "探索", "explore",
)


class IntentRouter:
    """Classify user intent to select relevant tool categories.

    Returns a list of category names, or None when intent is ambiguous
    (meaning all tools should be sent).
    """

    def is_complex(self, user_message: str) -> bool:
        """Detect whether a user message describes a multi-step / complex task.

        Returns True when the message contains sequential, conditional, scope, or
        perception+judgment keywords — indicating that a single tool call is unlikely
        to satisfy the request and VGG decomposition should be attempted.

        Rules (checked in order):
        1. Empty or very short messages (< 5 chars) → False
        2. Perception+judgment phrases (看看有没有, check if, see if, …) → True
        3. Sequential keywords (然后, and then, …) → True
        4. Conditional keywords (如果, if, whether, …) → True
        5. Scope keywords (所有, every, each, …) → True
        6. Simultaneous conjunction (同时) → True
        7. Otherwise → False

        Note: "if" in English triggers True (conditional). Ambiguous short greetings
        like "hello" or "hi" are short enough to return False without reaching rule 4.
        """
        if not user_message or len(user_message) < 5:
            return False

        msg_lower = user_message.lower()

        # Rule 2: perception + judgment (substring match — order matters)
        if any(phrase in msg_lower for phrase in _PERCEPTION_JUDGMENT_PHRASES):
            return True

        # Rule 3: sequential keywords
        if any(kw in msg_lower for kw in _SEQUENTIAL_KEYWORDS):
            return True

        # Rule 4: conditional keywords
        if any(kw in msg_lower for kw in _CONDITIONAL_KEYWORDS):
            return True

        # Rule 5: scope keywords
        if any(kw in msg_lower for kw in _SCOPE_KEYWORDS):
            return True

        # Rule 6: simultaneous conjunction
        if any(kw in msg_lower for kw in _SIMULTANEOUS_KEYWORDS):
            return True

        # Rule 7: multiple action verbs (2+ distinct verbs → multi-step)
        if _has_multiple_actions(msg_lower):
            return True

        return False

    def should_use_vgg(self, user_message: str, skill_registry: Any = None) -> bool:
        """Return True if this message should go through VGG pipeline.

        VGG is the unified task/information flow framework. ALL actionable
        commands go through it — simple commands produce 1-step GoalTrees,
        complex commands get LLM decomposition.

        Returns True for:
        - Complex tasks (is_complex: multi-step, conditional, scope)
        - Motor actions (navigate, patrol, explore, walk, etc.)
        - Any message that matches a registered skill

        Returns False only for:
        - Empty/trivial input
        - Pure conversation (greetings, questions, no action verb)
        """
        if not user_message or len(user_message) < 2:
            return False

        # System tool keywords → tool_use path, not VGG
        # These are CLI/infra commands, not robot actions.
        msg_lower = user_message.lower()
        _SYSTEM_BYPASS = ("可视化", "foxglove", "visualization", "viz ")
        if any(kw in msg_lower for kw in _SYSTEM_BYPASS):
            return False

        # Complex tasks → VGG
        if self.is_complex(user_message):
            return True

        # Skill match → VGG (1-step GoalTree, no LLM needed)
        if skill_registry is not None:
            try:
                match = skill_registry.match(user_message)
                if match is not None:
                    return True
            except Exception:
                pass

        # Motor pattern keywords → VGG
        if any(pat in msg_lower for pat in _MOTOR_PATTERNS):
            return True

        # Any action verb (word-boundary aware for English, substring for Chinese)
        for v in _ACTION_VERBS:
            if v in msg_lower:
                if v.isascii():
                    # English: require word boundary (avoid "go" in "go2sim")
                    import re
                    if re.search(r'\b' + re.escape(v) + r'\b', msg_lower):
                        return True
                else:
                    # Chinese: substring match is correct (去 in 去厨房)
                    return True

        return False

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

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Intent-based tool routing for Vector CLI.

Classifies user messages by keyword matching to select relevant tool
categories. Reduces token cost by sending only related tools to the LLM.

Zero-cost: pure keyword match, no LLM call. Falls back to all tools
when intent is ambiguous.
"""
from __future__ import annotations


# (keywords, categories) — checked in order, all matches accumulated
_RULES: list[tuple[frozenset[str], tuple[str, ...]]] = [
    # Code editing — "general" carries web_fetch; "system" (robot infra, e.g.
    # skill_reload) is included for the robot world and gated off in the dev
    # world via disable_category("system").
    (frozenset({
        "改", "修改", "编辑", "代码", "文件", "函数", "变量", "类",
        "edit", "fix", "code", "file", "function", "class", "import",
        "read", "write", "bug", "refactor", "重构", "写",
    }), ("code", "general", "system")),

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

    # Simulation — 'sim' (start/stop_simulation) stays enabled even in the dev
    # world, so it must lead here or "start arm sim" routes only to disabled
    # categories and the LLM gets zero tools.
    # Headless phrases are included so they still resolve to the sim tool and the
    # LLM can pass gui=false.
    (frozenset({
        "仿真", "sim", "simulation", "reset", "重置", "启动", "模拟",
        "habitat", "sysnav",
        "headless", "无窗口", "不要窗口", "no window",
    }), ("sim", "system", "robot")),
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

# Meta / first-person-request markers — input that is ABOUT the agent's own
# behaviour or expresses a desire AT the agent, not a robot-actionable command.
# e.g. "我希望你去打开终端" (I want YOU to open a terminal) is a meta request, not a
# skill: it only trips the action-verb hint via "去"/"打开" as substrings. The meta
# guard (see should_use_vgg) fires ONLY when, after stripping the marker, the
# residual carries NO genuine motor/navigation signal — so a politely-phrased REAL
# motor command ("请你巡逻", "I want you to go to the kitchen") still plans, while an
# otherwise-unmappable meta request answers instead of failing a VGG decompose.
# Substring match (Chinese) / prefix-or-substring (English).
_META_REQUEST_MARKERS: tuple[str, ...] = (
    # Chinese first-person desire / directive AT the agent
    "我希望你", "我想让你", "我想要你", "我要你", "希望你能", "我希望", "我想让",
    "你能不能", "你可不可以", "你可以", "你应该", "你需要", "麻烦你", "请你",
    # English first-person desire / polite directive
    "i want you", "i'd like you", "i would like you", "i need you",
    "i wish you", "please ", "could you please", "can you please",
)

# Strong, unambiguous motor/navigation signals — a real robot command regardless of
# polite phrasing. If any survives in the marker residual, the message PLANS.
_STRONG_MOTOR_PATTERNS: tuple[str, ...] = (
    "导航", "巡逻", "探索", "走到", "去到",
    "navigate", "patrol", "explore", "go to", "goto",
)

# Weak directional particles: "去"/"到" mean navigation ONLY when followed by a
# destination, not when followed by another (non-robot) action verb. "去厨房" is
# navigation; "去打开终端" / "去运行测试" is "go DO X" — a meta request, not a skill.
_DIRECTIONAL_PARTICLES: tuple[str, ...] = ("去", "到")

# Non-robot action verbs that, immediately after a directional particle, mark a
# "go do X" meta pattern (X is an app/system action, not a robot skill).
_GO_DO_VERBS: tuple[str, ...] = (
    "打开", "关闭", "开启", "运行", "启动", "执行", "安装", "下载",
    "open", "close", "run", "launch", "install",
)


def _residual_has_motor_signal(msg_lower: str) -> bool:
    """Return True if, after removing meta-request markers, a genuine motor /
    navigation command remains.

    This separates a politely-phrased REAL motor command ("请你巡逻", "I want you
    to go to the kitchen") — which must still PLAN — from a meta request whose only
    "actionability" is incidental ("我希望你去打开终端": "去"/"打开" are substrings of a
    "go do X" ask, not a robot skill) — which must ANSWER.
    """
    residual = msg_lower
    for marker in _META_REQUEST_MARKERS:
        residual = residual.replace(marker, " ")

    # Strong, unambiguous motor verb anywhere in the residual → real command.
    if any(pat in residual for pat in _STRONG_MOTOR_PATTERNS):
        return True

    # Directional particle → navigation, UNLESS immediately followed by a go-do
    # verb (then it is "go DO X", not "go TO <place>").
    for particle in _DIRECTIONAL_PARTICLES:
        start = 0
        while True:
            idx = residual.find(particle, start)
            if idx < 0:
                break
            rest = residual[idx + len(particle):].lstrip()
            if not any(rest.startswith(v) for v in _GO_DO_VERBS):
                return True
            start = idx + len(particle)

    return False


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
        _SYSTEM_BYPASS = (
            "可视化", "foxglove", "visualization", "viz ",
            # Simulation lifecycle — must route to start_simulation /
            # stop_simulation tools, not to gripper_close_skill (which
            # greedily matches "close" / "关闭").
            "仿真", "sim ", " sim", "simulation",
            "go2sim", "go2 sim", "armsim", "arm sim",
            # G1 humanoid sim lifecycle/queries → start_simulation / answer
            # (not a motor skill); "g1" in this robot CLI always means the sim.
            "g1sim", "g1 sim", "启动g1", "start g1", "g1仿真", "g1 仿真",
            # Habitat world + SysNav perception lifecycle — start_simulation /
            # sysnav_perception tools, not motor skills / VGG decompose.
            "habitat", "sysnav", "语义感知",
            # Headless modifier — pass gui=false to start_simulation; not a VGG task.
            "headless", "无窗口", "不要窗口", "no window",
        )
        if any(kw in msg_lower for kw in _SYSTEM_BYPASS):
            return False

        # Complex tasks → VGG
        if self.is_complex(user_message):
            return True

        # Conversational questions / greetings → tool_use (answer directly), so a
        # question like "为什么一开始那么卡" is NOT mis-routed into a plan just because
        # an action verb appears incidentally as a substring (here "开始" inside the
        # adverb "一开始"). Genuinely multi-step asks were already caught by
        # is_complex above, so this only short-circuits simple, non-actionable Q&A.
        _stripped = msg_lower.strip()
        _ZH_Q = (
            "为什么", "为何", "怎么", "怎样", "如何", "是什么", "什么是",
            "是不是", "能不能", "可不可以", "有没有",
        )
        _EN_Q = (
            "why ", "how ", "what ", "what's", "who ", "when ", "where ",
            "which ", "can you", "could you", "do you", "does ", "is it",
            "are you",
        )
        if (
            _stripped.endswith("?")
            or _stripped.endswith("？")
            or _stripped.endswith(("吗", "呢", "吧"))
            or any(q in _stripped for q in _ZH_Q)
            or any(_stripped.startswith(q) for q in _EN_Q)
        ):
            return False

        # Skill match → VGG (1-step GoalTree, no LLM needed). Checked BEFORE the
        # meta-request guard so a genuine command that happens to be phrased as a
        # request ("我希望你去厨房" with a real navigate skill) still plans.
        if skill_registry is not None:
            try:
                match = skill_registry.match(user_message)
                if match is not None:
                    return True
            except Exception:
                pass

        # Meta / first-person-request guard → tool_use (answer directly). Reaching
        # here means the message did NOT match a registered skill (or there is no
        # registry). If it carries a meta/first-person-request marker ("我希望你…",
        # "你能…", "I want you to…", "please …") AND the marker residual has NO genuine
        # motor/navigation signal, it is a request about the agent's own actions — not
        # a robot-actionable command — that only trips the action-verb hint below via
        # an incidental substring (e.g. "去"/"打开" in "我希望你去打开终端"). Routing it to
        # VGG produces a failing decompose (no skill maps), so answer it instead.
        #
        # CRITICAL (regression guard): the residual check runs BEFORE the motor /
        # action-verb checks but is itself gated on having NO real motor signal, so a
        # politely-phrased REAL motor command ("请你巡逻", "I want you to go to the
        # kitchen", "请你导航到厨房") still falls through to the motor-pattern check and
        # PLANS. The marker alone is NOT dispositive — a polite prefix never suppresses
        # a genuine command. (SkillRegistry.match is prefix-only, so "请你巡逻" does NOT
        # skill-match; the residual check is what keeps it on the planning path.)
        if (
            any(marker in msg_lower for marker in _META_REQUEST_MARKERS)
            and not _residual_has_motor_signal(msg_lower)
        ):
            return False

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

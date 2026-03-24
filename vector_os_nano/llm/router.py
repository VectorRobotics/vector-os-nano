"""ModelRouter — select the appropriate LLM model for each pipeline stage.

Reads model assignments from the 'llm.models' config section and falls back
to 'llm.model' (the global default) when a stage-specific entry is absent.

Config format (in default.yaml under 'llm'):
    models:
        classify:     "anthropic/claude-haiku-4-5"
        plan_simple:  "anthropic/claude-haiku-4-5"
        plan_complex: "anthropic/claude-sonnet-4-6"
        chat:         "anthropic/claude-haiku-4-5"
        summarize:    "anthropic/claude-haiku-4-5"

Design notes:
- ModelSelection is a frozen dataclass so it is safe to pass around, log, cache.
- estimate_complexity is a pure static method: no I/O, no side effects.
- Complexity scoring uses >= 2 indicators so a single weak signal stays "simple".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_DEFAULT_FALLBACK = "anthropic/claude-haiku-4-5"

# ---------------------------------------------------------------------------
# Complexity heuristic constants
# ---------------------------------------------------------------------------

SPATIAL_WORDS: frozenset[str] = frozenset({
    # English
    "left", "right", "front", "behind", "between", "next to", "near", "far",
    "above", "below", "on top", "beside",
    # Chinese
    "左", "右", "前", "后", "旁边", "中间", "上面", "下面",
})

MULTI_ACTION_PATTERNS: tuple[str, ...] = (
    # English
    r"\bthen\b",
    r"\band then\b",
    r"\bafter that\b",
    r"\bfirst\b.*\bthen\b",
    r"\bnext\b",
    r"\bfinally\b",
    # Chinese
    r"然后",
    r"接着",
    r"先.*再",
    r"之后",
)

# Pre-compiled for speed
_MULTI_ACTION_RE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in MULTI_ACTION_PATTERNS
)

_INSTRUCTION_LENGTH_THRESHOLD = 50
_WORLD_OBJECTS_THRESHOLD = 4
_COMPLEXITY_SCORE_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSelection:
    """Which model to use for a given LLM call."""

    model: str    # e.g., "anthropic/claude-haiku-4-5"
    reason: str   # For logging: "classify", "simple_task", "complex_task", etc.


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------


class ModelRouter:
    """Select the appropriate LLM model for each pipeline stage.

    Reads model assignments from config. Falls back to the default model
    when a stage-specific config entry is missing.

    Args:
        config: Full config dict. Reads config['llm']['models'] for stage
                mappings and config['llm']['model'] as the fallback default.
    """

    def __init__(self, config: dict) -> None:
        llm_section = config.get("llm", {})
        self._default_model: str = llm_section.get("model", _DEFAULT_FALLBACK)
        self._models: dict[str, str] = llm_section.get("models", {})

    # ------------------------------------------------------------------
    # Per-stage selectors
    # ------------------------------------------------------------------

    def for_classify(self) -> ModelSelection:
        """Model for intent classification (single-word response)."""
        model = self._models.get("classify", self._default_model)
        return ModelSelection(model=model, reason="classify")

    def for_plan(self, instruction: str, world_state: dict) -> ModelSelection:
        """Model for task planning. Auto-selects simple vs complex.

        Calls estimate_complexity() to pick between plan_simple and
        plan_complex stage models.

        Args:
            instruction: Natural-language planning instruction.
            world_state:  Current world state dict (may contain 'objects' list).

        Returns:
            ModelSelection with reason "simple_task" or "complex_task".
        """
        complexity = self.estimate_complexity(instruction, world_state)
        if complexity == "complex":
            model = self._models.get("plan_complex", self._default_model)
            return ModelSelection(model=model, reason="complex_task")
        model = self._models.get("plan_simple", self._default_model)
        return ModelSelection(model=model, reason="simple_task")

    def for_chat(self) -> ModelSelection:
        """Model for conversational chat."""
        model = self._models.get("chat", self._default_model)
        return ModelSelection(model=model, reason="chat")

    def for_query(self) -> ModelSelection:
        """Model for visual queries (vision-capable model preferred)."""
        model = self._models.get("query", self._default_model)
        return ModelSelection(model=model, reason="query")

    def for_summarize(self) -> ModelSelection:
        """Model for execution summarization."""
        model = self._models.get("summarize", self._default_model)
        return ModelSelection(model=model, reason="summarize")

    # ------------------------------------------------------------------
    # Complexity heuristic
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_complexity(instruction: str, world_state: dict) -> str:
        """Heuristic complexity estimation: returns 'simple' or 'complex'.

        Returns 'complex' when the cumulative score reaches
        _COMPLEXITY_SCORE_THRESHOLD (2). The heuristic is deliberately
        conservative — when in doubt it returns 'simple' so the cheaper
        model is used.

        Scoring rules (each contributes 1 point):
        1. A spatial-reasoning word is present in the instruction.
        2. A multi-action pattern matches in the instruction.
        3. Two or more object labels from world_state appear in the instruction.
        4. world_state contains four or more visible objects.
        5. Instruction length exceeds _INSTRUCTION_LENGTH_THRESHOLD characters.

        Args:
            instruction: The user instruction string.
            world_state:  Current world state. Expected key: 'objects' — a list
                          of dicts, each with at least a 'label' field.

        Returns:
            'complex' or 'simple'.
        """
        score = 0

        lower_instruction = instruction.lower()

        # Rule 1 — spatial words
        if _has_spatial_word(lower_instruction):
            score += 1

        # Rule 2 — multi-action patterns
        if _has_multi_action_pattern(instruction):
            score += 1

        if score >= _COMPLEXITY_SCORE_THRESHOLD:
            return "complex"

        # Rule 3 — multiple object labels mentioned
        if _count_objects_mentioned(instruction, world_state) >= 2:
            score += 1

        if score >= _COMPLEXITY_SCORE_THRESHOLD:
            return "complex"

        # Rule 4 — large world state
        objects = world_state.get("objects", [])
        if isinstance(objects, list) and len(objects) >= _WORLD_OBJECTS_THRESHOLD:
            score += 1

        if score >= _COMPLEXITY_SCORE_THRESHOLD:
            return "complex"

        # Rule 5 — long instruction
        if len(instruction) > _INSTRUCTION_LENGTH_THRESHOLD:
            score += 1

        return "complex" if score >= _COMPLEXITY_SCORE_THRESHOLD else "simple"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _has_spatial_word(lower_instruction: str) -> bool:
    """Return True if any spatial word appears in the (already lowercased) instruction."""
    for word in SPATIAL_WORDS:
        if word in lower_instruction:
            return True
    return False


def _has_multi_action_pattern(instruction: str) -> bool:
    """Return True if any multi-action regex matches the instruction."""
    for pattern in _MULTI_ACTION_RE:
        if pattern.search(instruction):
            return True
    return False


def _count_objects_mentioned(instruction: str, world_state: dict) -> int:
    """Count how many world-state object labels appear in the instruction.

    Matching is case-insensitive substring search.

    Args:
        instruction: Raw instruction string.
        world_state:  World state dict with optional 'objects' list.

    Returns:
        Number of distinct labels found.
    """
    objects = world_state.get("objects", [])
    if not isinstance(objects, list):
        return 0

    lower_instruction = instruction.lower()
    count = 0
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        label = obj.get("label", "")
        if label and label.lower() in lower_instruction:
            count += 1
    return count

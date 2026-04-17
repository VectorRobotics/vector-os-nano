"""Dynamic system prompt that refreshes robot context on each LLM turn.

VectorEngine accesses system_prompt as a list. This subclass overrides
__iter__ to refresh the robot context block before the engine serializes
it for the API call — so the LLM always sees current robot state.
"""
from __future__ import annotations

from typing import Any


class DynamicSystemPrompt(list):
    """System prompt list that auto-refreshes robot context."""

    def __init__(self, static_blocks: list[dict], provider: Any = None) -> None:
        super().__init__(static_blocks)
        self._provider = provider
        self._context_idx: int | None = None

        # Find existing robot context block (if build_system_prompt already added one)
        for i, block in enumerate(self):
            text = block.get("text", "") if isinstance(block, dict) else ""
            if "[Robot State]" in text:
                self._context_idx = i
                break

    def __iter__(self):
        if self._provider is not None:
            try:
                block = self._provider.get_context_block()
                if block:
                    if self._context_idx is not None and self._context_idx < len(self):
                        self[self._context_idx] = block
                    else:
                        self.append(block)
                        self._context_idx = len(self) - 1
            except Exception:
                pass
        return super().__iter__()

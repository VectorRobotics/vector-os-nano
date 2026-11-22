# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Scripted LLMs: the packaged test seam that drives the REAL ``vector-cli``.

Switched on ONLY by the test-seam variables: ``VECTOR_FAKE_LLM_TOOLS=<json-path>``
(``runtime.make_backend`` replaces the network LLM with a ``FakeToolScriptBackend``
for an open-loop ``{"turns"}`` script or a ``ReactiveFakeBackend`` for a ``{"steps"}``
script keyed on what the model observes) and ``VECTOR_FAKE_LLM_LOG`` (each reactive
call's observation, one JSON line). Only the network LLM is replaced: every tool,
GoalVerifier, actor-causation grade and verdict still runs, so the verdict stays
honest by construction.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

from vector_os_nano.interfaces.llm import LLMResponse
from vector_os_nano.interfaces.llm import TokenUsage


class FakeToolScriptBackend:
    """A canned ``LLMBackend`` that REPLAYS a SCRIPT of native tool_use turns.

    Drives ``engine.run_turn_native`` deterministically: each ``.call()`` returns
    the NEXT scripted ``LLMResponse`` (a native tool_use turn), advancing a turn
    cursor. A native ReAct turn looks like::

        turn1: tool_calls=[walk(...)]                   stop_reason='tool_use'
        turn2: tool_calls=[verify('at_position(11,3)')] stop_reason='tool_use'  (after the tool_result)
        turn3: tool_calls=[finish()] (or text)          stop_reason='end_turn'

    It returns the ordered native turns the model "would" issue. The REAL native runner still dispatches every skill via
    SkillWrapperTool, evaluates ``verify`` via the REAL GoalVerifier, grades
    actor-causation, and hands the assembled trace to the EXISTING verdict gate —
    so the verdict stays honest by construction (a verify-only script with no walk
    still grades UNCAUSED -> RAN -> verified False).

    After the script is exhausted, every further ``.call()`` returns a terminal
    ``end_turn`` with no tool calls (the loop stops). This makes the runner robust
    to an off-by-one without ever silently re-issuing a stale tool call.
    """

    def __init__(self, turns: list[LLMResponse]) -> None:
        # Defensive copy so a caller mutating the list after construction cannot
        # perturb the replay sequence.
        self._turns: list[LLMResponse] = list(turns)
        self._cursor: int = 0

    # -- serialization seam ----------------------------------------------------

    @classmethod
    def from_json_file(cls, path: str | Path) -> "FakeToolScriptBackend":
        """Build a tool-script backend from a JSON file of scripted turns.

        The file's top-level object is ``{"turns": [<turn>, ...]}`` where each
        ``<turn>`` is ``{"text"?: str, "stop_reason"?: str,
        "tool_calls": [{"id"?, "name", "input"}, ...]}``. Fails LOUD on a missing /
        unparseable / wrong-shaped file — a silent empty script would let a broken
        harness masquerade as a real native run.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "steps" in data:
            # A reactive (closed-loop) script rides the SAME env seam, so the CLI
            # under test stays bare `vector-cli` (the seam names only this class).
            return ReactiveFakeBackend(data["steps"])  # type: ignore[return-value]
        if not isinstance(data, dict) or "turns" not in data:
            raise ValueError(
                "VECTOR_FAKE_LLM_TOOLS payload must be a JSON object with a 'turns' "
                f"(open-loop) or 'steps' (reactive) list, got {type(data).__name__}"
            )
        return cls.from_tool_script(_turns_from_dicts(data["turns"]))

    @classmethod
    def from_tool_script(cls, turns: list[LLMResponse]) -> "FakeToolScriptBackend":
        """Build directly from an ordered list of ``LLMResponse`` turns."""
        if not isinstance(turns, list):
            raise ValueError(f"tool-script turns must be a list, got {type(turns).__name__}")
        return cls(turns)

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: list[dict[str, Any]],
        max_tokens: int,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Return the NEXT scripted native turn; terminal end_turn when exhausted."""
        if self._cursor >= len(self._turns):
            return LLMResponse(
                text="", tool_calls=[], stop_reason="end_turn",
                usage=TokenUsage(input_tokens=0, output_tokens=0),
            )
        turn = self._turns[self._cursor]
        self._cursor += 1
        if on_text is not None and turn.text:
            on_text(turn.text)
        return turn


def _observation(messages: list[dict[str, Any]]) -> tuple[str, bool]:
    """The last thing the model 'sees': (text, is_error) of the final message.

    A tool_result message yields its results' content (is_error if any result is an
    error); a plain user message yields the user's text.
    """
    if not messages:
        return "", False
    content = messages[-1].get("content", "")
    if isinstance(content, str):
        return content, False
    parts: list[str] = []
    is_error = False
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_result":
            is_error = is_error or bool(block.get("is_error"))
            inner = block.get("content", "")
            if isinstance(inner, list):
                inner = " ".join(str(b.get("text", "")) for b in inner if isinstance(b, dict))
            parts.append(str(inner))
        elif block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(parts), is_error


class ReactiveFakeBackend:
    """A CLOSED-LOOP scripted ``LLMBackend``: each step fires on what the model just saw.

    Payload (``VECTOR_FAKE_LLM_TOOLS`` JSON)::

        {"steps": [{"when"?: str, "is_error"?: bool, "text"?: str,
                    "stop_reason"?: str, "tool_calls": [{"name", "input"}, ...]}, ...]}

    On every ``.call()`` the observation is the LAST message (its tool_result content,
    or the user's text). The first not-yet-used step fires iff ``when`` is a substring
    of the observation, ``is_error`` (if given) equals the observation's error flag,
    and every tool it calls is in the offered ``tools`` (a real model cannot call a
    tool it was not offered). Each step fires once; tool-call ids are unique
    per step (``call_<step>_<j>``). No match -> ``end_turn`` with no tools and a
    ``[reactive-fake] no step matched`` text, so a broken script fails visibly.

    ``calls`` records ``(messages, offered tool names)`` per call for assertions; a
    PTY child also appends each call's observation to ``$VECTOR_FAKE_LLM_LOG`` (one
    JSON line per call — the call counter a parent test reads). A step with
    ``"raise": "<msg>"`` raises ``RuntimeError(<msg>)`` instead of answering (an
    injected LLM outage); ``"delay": <s>`` answers *s* seconds late (a slow model, so
    a test can land input while the LLM call is in flight).
    """

    def __init__(self, steps: list[dict[str, Any]]) -> None:
        if not isinstance(steps, list):
            raise ValueError(f"'steps' must be a list, got {type(steps).__name__}")
        self._steps = list(steps)
        self._turns = _turns_from_dicts(self._steps)
        self._used: set[int] = set()
        self.calls: list[tuple[list[dict[str, Any]], list[str]]] = []
        self._log = os.environ.get("VECTOR_FAKE_LLM_LOG", "")

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: list[dict[str, Any]],
        max_tokens: int,
        on_text: Callable[[str], None] | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        offered = [str(t.get("name", "")) for t in (tools or []) if isinstance(t, dict)]
        self.calls.append((list(messages), offered))
        obs, obs_error = _observation(messages)
        if self._log:
            with open(self._log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"obs": obs}, ensure_ascii=False) + "\n")
        for i, step in enumerate(self._steps):
            if i in self._used or str(step.get("when", "")) not in obs:
                continue
            if "is_error" in step and bool(step["is_error"]) != obs_error:
                continue
            if any(tc.name not in offered for tc in self._turns[i].tool_calls):
                continue
            self._used.add(i)
            time.sleep(float(step.get("delay", 0)))
            if "raise" in step:
                raise RuntimeError(str(step["raise"]))
            turn = self._turns[i]
            if on_text is not None and turn.text:
                on_text(turn.text)
            return turn
        return LLMResponse(
            text=f"[reactive-fake] no step matched: {obs[:120]!r}",
            tool_calls=[], stop_reason="end_turn",
            usage=TokenUsage(input_tokens=0, output_tokens=0),
        )


# ---------------------------------------------------------------------------
# JSON <-> LLMResponse turn helpers (single source for the env-var seam)
# ---------------------------------------------------------------------------


def _turns_from_dicts(raw_turns: Any) -> list["LLMResponse"]:
    """Deserialize a list of turn dicts into ``LLMResponse`` native turns."""
    from vector_os_nano.interfaces.llm import LLMToolCall

    if not isinstance(raw_turns, list):
        raise ValueError(f"'turns' must be a list, got {type(raw_turns).__name__}")
    turns: list[LLMResponse] = []
    for i, t in enumerate(raw_turns):
        if not isinstance(t, dict):
            raise ValueError(f"turn {i} must be an object, got {type(t).__name__}")
        raw_calls = t.get("tool_calls", []) or []
        calls: list[LLMToolCall] = []
        for j, c in enumerate(raw_calls):
            if not isinstance(c, dict) or "name" not in c:
                raise ValueError(f"turn {i} tool_call {j} must name a tool")
            calls.append(
                LLMToolCall(
                    id=str(c.get("id", f"call_{i}_{j}")),
                    name=str(c["name"]),
                    input=dict(c.get("input", {}) or {}),
                )
            )
        stop = str(t.get("stop_reason", "tool_use" if calls else "end_turn"))
        turns.append(
            LLMResponse(
                text=str(t.get("text", "")),
                tool_calls=calls,
                stop_reason=stop,
                usage=TokenUsage(input_tokens=0, output_tokens=0),
            )
        )
    return turns


def tool_turn(*tool_calls: Any, text: str = "", end: bool = False) -> "LLMResponse":
    """Build one scripted native ``LLMResponse`` turn for a tool-script.

    Each positional arg is ``(name, input_dict)``. ``end=True`` (or no tool calls)
    marks the turn's stop_reason ``end_turn``; otherwise ``tool_use``.
    """
    from vector_os_nano.interfaces.llm import LLMToolCall

    calls = [
        LLMToolCall(id=f"tc_{i}", name=str(name), input=dict(inp or {}))
        for i, (name, inp) in enumerate(tool_calls)
    ]
    stop = "end_turn" if (end or not calls) else "tool_use"
    return LLMResponse(text=text, tool_calls=calls, stop_reason=stop, usage=TokenUsage())

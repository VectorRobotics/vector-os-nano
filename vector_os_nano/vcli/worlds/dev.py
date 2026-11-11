# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Default "dev" world — a general coding/automation agent (no robot).

Ships with the kernel. Provides:
- the general dev persona (from prompt.py),
- deterministic, side-effect-free verify predicates over the project tree,
- a software-task decompose vocabulary for the GoalDecomposer.

Verify predicates are strictly observational (``file_exists``, ``grep_count``,
``path_contains``). Running a command (e.g. a test suite) is an ACTION — the
permission-gated ``bash`` tool — never a verify predicate.
"""

from __future__ import annotations

import re
from typing import Any

from vector_os_nano.vcli.cognitive.oracle_registry import oracle
from vector_os_nano.vcli.prompt import DEV_ROLE_PROMPT, DEV_TOOL_INSTRUCTIONS
from vector_os_nano.vcli.tools.file_tools import in_project, project_root, real_path, resolved_permission
from vector_os_nano.vcli.worlds.base import DecomposeVocab

# A predicate runs inside the GoalVerifier 5s sandbox: one bounded file per call.
_MAX_BYTES_PER_FILE = 1_000_000


def _read_file(path: str) -> str:
    """The text of the ONE regular file at *path*, if the shared read rule
    (file_tools.read_permission) auto-allows it. A model-authored predicate runs with
    no prompt, so it must never become a content oracle on what file_read would ask
    or refuse, and it never walks a directory (a walk that skips or stops part-way
    lets a negated count verify True).

    Anything else RAISES (GoalVerifier turns that into False): a refused path, a
    directory, a missing path, a non-regular file (FIFO, socket) or one over the size
    cap. Returning False/0 would make 'not path_contains(...)' or
    'grep_count(...) == 0' verify True on a file that was never examined."""
    root = project_root(None)
    p = real_path(path, root)  # resolved ONCE: the decision and the read use this path
    perm = resolved_permission(p, root, path)
    if perm.behavior != "allow":
        raise PermissionError(f"verify may not read {path}: {perm.reason}")
    if not p.is_file():
        if not p.exists():
            raise FileNotFoundError(f"verify found no file {path}: state absence with not file_exists(...)")
        raise OSError(f"verify reads one regular file per call; {path} is not one")
    if p.stat().st_size > _MAX_BYTES_PER_FILE:
        raise OSError(f"verify will not read {path}: larger than {_MAX_BYTES_PER_FILE} bytes")
    return p.read_text(encoding="utf-8", errors="ignore")


@oracle(scope="dev")
def file_exists(path: str) -> bool:
    """True if *path* (relative to cwd) exists. Existence is not content, so a
    protected path inside the project may be asked about; anything read_permission
    would not auto-allow for its location ($HOME, an ancestor of it or a pseudo-fs
    as cwd, outside cwd) raises."""
    p = real_path(path)
    if not in_project(p, project_root(None)):
        raise PermissionError(f"verify may not look outside the project: {path}")
    return p.exists()


@oracle(scope="dev")
def path_contains(path: str, substr: str) -> bool:
    """True if the regular file at *path* contains *substr*."""
    return substr in _read_file(path)


@oracle(scope="dev")
def grep_count(pattern: str, path: str) -> int:
    """Number of regex *pattern* matches in the regular file at *path*."""
    return len(re.compile(pattern).findall(_read_file(path)))


def dev_verify_namespace() -> dict[str, Any]:
    """Return the dev world's read-only verify predicates."""
    return {
        "file_exists": file_exists,
        "grep_count": grep_count,
        "path_contains": path_contains,
    }


# --- Decompose vocabulary --------------------------------------------------

_ONE_FILE = "One file per call: to state something about several files, 'and' the calls together."

_DEV_VERIFY_SIGNATURES: dict[str, str] = {
    "file_exists": (
        "file_exists(path: str) -> bool  # path (relative to cwd) exists; "
        "state absence as not file_exists(path)"
    ),
    "grep_count": (
        "grep_count(pattern: str, path: str) -> int  # regex match count in ONE regular file; "
        "a directory, a missing, secret or non-regular path, a file over "
        f"{_MAX_BYTES_PER_FILE} bytes or an invalid regex verifies False. {_ONE_FILE}"
    ),
    "path_contains": (
        "path_contains(path: str, substr: str) -> bool  # the ONE regular file at path contains "
        f"substr; a directory, a missing, secret or non-regular path verifies False. {_ONE_FILE}"
    ),
}

_DEV_EXAMPLE = """\
Task: "add a greet() function to greet.py and make sure it is defined"
Response:
{
  "goal": "add a greet() function to greet.py and make sure it is defined",
  "sub_goals": [
    {
      "name": "create_greet_file",
      "description": "create greet.py",
      "verify": "file_exists('greet.py')",
      "strategy": "",
      "timeout_sec": 30,
      "depends_on": [],
      "strategy_params": {},
      "fail_action": ""
    },
    {
      "name": "define_greet",
      "description": "define a greet function in greet.py",
      "verify": "grep_count('def greet', 'greet.py') > 0",
      "strategy": "",
      "timeout_sec": 30,
      "depends_on": ["create_greet_file"],
      "strategy_params": {},
      "fail_action": ""
    }
  ],
  "context_snapshot": ""
}"""

# Phase B: the dev world can now *act* via a single tool-backed strategy. A
# "tool_call" sub-goal dispatches one kernel tool (file_write/file_edit/bash/...)
# through the permission gate + the dev allowlist below. Leave strategy empty for
# steps that only need verification; use tool_call when a step must change the tree.
_DEV_TOOL_EXAMPLE = """\
Task: "create config.txt containing the word ready"
Response:
{
  "goal": "create config.txt containing the word ready",
  "sub_goals": [
    {
      "name": "write_config",
      "description": "write config.txt with the content 'ready'",
      "verify": "path_contains('config.txt', 'ready')",
      "strategy": "tool_call",
      "timeout_sec": 30,
      "depends_on": [],
      "strategy_params": {"tool": "file_write", "args": {"file_path": "config.txt", "content": "ready\\n"}},
      "fail_action": ""
    }
  ],
  "context_snapshot": ""
}"""

# Tools a dev sub-goal may invoke autonomously (the kernel's domain-general
# "code" toolset). Anything off this list is denied by ToolDispatcher before any
# permission check — robot/diag/system tools are never reachable from the dev world.
DEV_TOOL_ALLOWLIST: frozenset[str] = frozenset({
    "file_read",
    "file_write",
    "file_edit",
    "bash",
    "glob",
    "grep",
})

_DEV_STRATEGY_PARAMS_HELP = """\
  - tool_call: {"tool": "<tool_name>", "args": {<tool-specific arguments>}}
      Dispatches one kernel tool through the permission gate. Allowed tools:
        file_write  args: {"file_path": str, "content": str}
        file_edit   args: {"file_path": str, "old_string": str, "new_string": str}
        bash        args: {"command": str}
      Use tool_call ONLY for steps that must change the project; otherwise leave
      strategy empty and rely on the verify predicate alone."""

DEV_VOCAB = DecomposeVocab(
    planner_intro=(
        "You are a software task planner. Decompose the user's task into verifiable "
        "sub-goals, each with a deterministic verify predicate over the project. "
        "For steps that must modify the project, set strategy to \"tool_call\"."
    ),
    verify_functions=frozenset(_DEV_VERIFY_SIGNATURES.keys()),
    verify_fn_signatures=_DEV_VERIFY_SIGNATURES,
    strategy_descriptions={
        "tool_call": "Invoke one kernel tool (file_write/file_edit/bash/...) to act on the project",
        "chat": "Ask the chat LLM for free-form text (no project side effect; verify must still hold)",
    },
    strategies=frozenset({"tool_call", "chat"}),
    strategy_params_help=_DEV_STRATEGY_PARAMS_HELP,
    examples=_DEV_EXAMPLE + "\n\n" + _DEV_TOOL_EXAMPLE,
    fallback_verify="True",
)


class DevWorld:
    """The default, robot-free world: a general coding/automation agent."""

    name = "dev"

    def is_robot(self) -> bool:
        return False

    def persona_blocks(self, agent: Any = None) -> tuple[str, str]:
        return DEV_ROLE_PROMPT, DEV_TOOL_INSTRUCTIONS

    def register_tools(self, registry: Any, agent: Any) -> None:
        # General code/web tools are registered by the CLI for every world.
        return None

    def build_verify_namespace(self, agent: Any) -> dict[str, Any]:
        return dev_verify_namespace()

    def register_capabilities(self, registry: Any, agent: Any, backend: Any) -> None:
        # The chat LLM as one routable capability (Phase C seam). No-op without a
        # backend (e.g. before /login), keeping the dev path inert until then.
        if backend is None:
            return
        from vector_os_nano.vcli.cognitive.capabilities import LLMChatCapability
        registry.register(LLMChatCapability(backend))

    def decompose_vocab(self) -> DecomposeVocab:
        return DEV_VOCAB

    def derive_vocab_from_registry(self) -> bool:
        # Dev world ships an explicit DEV_VOCAB; no registry derivation.
        return False

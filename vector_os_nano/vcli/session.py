# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Session persistence for vcli — JSONL format.

Each session is stored as a file of newline-delimited JSON objects:
  <session_id>.jsonl  under $VECTOR_HOME/sessions (or caller-supplied directory).

Entry types
-----------
user        {"type":"user","content":"...","ts":"..."}
assistant   {"type":"assistant","text":"...","tool_use":[...]|null,"ts":"..."}
tool_result {"type":"tool_result","results":[...],"ts":"..."}
meta        {"type":"meta","token_usage":{...},"metadata":{...},"ts":"..."}

Crash-safety: every save() writes + fsyncs.  Corrupt tail lines are silently
skipped on load so a partial write never loses the whole session.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vector_os_nano.schemas import paths
from vector_os_nano.interfaces.llm import TokenUsage

# ---------------------------------------------------------------------------
# Value objects (frozen)
# ---------------------------------------------------------------------------




@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    path: Path


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """A mutable session object backed by a JSONL file."""

    def __init__(
        self,
        session_id: str,
        created_at: str,
        updated_at: str,
        path: Path,
        entries: list[dict[str, Any]] | None = None,
        token_usage: TokenUsage | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.session_id: str = session_id
        self.created_at: str = created_at
        self.updated_at: str = updated_at
        self._path: Path = path
        self._entries: list[dict[str, Any]] = entries if entries is not None else []
        self.token_usage: TokenUsage = token_usage if token_usage is not None else TokenUsage()
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}
        self.read_files: set[str] = set()  # Tracks files read this session (for write-guard)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _non_meta_entries(self) -> list[dict[str, Any]]:
        return [e for e in self._entries if e.get("type") != "meta"]

    # ------------------------------------------------------------------
    # Append methods
    # ------------------------------------------------------------------

    def append_user(self, content: str) -> None:
        self._entries.append({"type": "user", "content": content, "ts": self._now()})
        self.updated_at = self._now()

    def append_assistant(
        self,
        text: str,
        tool_use_blocks: list[dict[str, Any]] | None = None,
    ) -> None:
        self._entries.append(
            {
                "type": "assistant",
                "text": text,
                "tool_use": tool_use_blocks,
                "ts": self._now(),
            }
        )
        self.updated_at = self._now()

    def append_tool_results(self, results: list[dict[str, Any]]) -> None:
        self._entries.append({"type": "tool_result", "results": results, "ts": self._now()})
        self.updated_at = self._now()

    def last_assistant_text(self, since: int = 0) -> str:
        """The model's last non-empty words among the entries after the first *since*."""
        return next((e["text"].strip() for e in reversed(self._entries[since:])
                     if e.get("type") == "assistant" and (e.get("text") or "").strip()), "")

    def add_usage(self, usage: TokenUsage) -> None:
        self.token_usage = self.token_usage.add(usage)
        self.updated_at = self._now()

    def compact(self, keep_recent: int = 12) -> tuple[int, int]:
        """Compress old entries into a summary, keep last *keep_recent*.

        Extracts user messages and assistant texts from old entries into a
        single summary entry at the top.  Recent entries stay verbatim so
        the LLM has full context for the current conversation.

        Returns:
            (before_count, after_count) of non-meta entries.
        """
        non_meta = self._non_meta_entries()
        before = len(non_meta)
        if before <= keep_recent:
            return before, before

        # Snap the keep boundary so the recent window never BEGINS on a
        # tool_result whose producing assistant (tool_use) would be summarized
        # away -> that orphaned tool message makes the chat APIs 400 ("role
        # 'tool' must follow 'tool_calls'") and bricks the REPL (R2-8). Move the
        # cut back to include the producing turn.
        cut = before - keep_recent
        while cut > 0 and non_meta[cut].get("type") == "tool_result":
            cut -= 1
        old_entries = non_meta[:cut]
        recent_entries = non_meta[cut:]

        # Build summary from old entries
        summary_lines: list[str] = []
        for entry in old_entries:
            etype = entry.get("type", "")
            if etype == "user":
                content = entry.get("content", "")
                if content and len(content) < 200:
                    summary_lines.append(f"User: {content}")
            elif etype == "assistant":
                text = entry.get("text", "")
                if text:
                    # Truncate long assistant responses to first sentence
                    first_line = text.split("\n")[0][:100]
                    summary_lines.append(f"V: {first_line}")

        if summary_lines:
            summary_text = (
                "[Earlier conversation summary]\n"
                + "\n".join(summary_lines[-10:])  # keep last 10 exchanges
            )
            summary_entry = {
                "type": "user",
                "content": summary_text,
                "ts": self._now(),
            }
            self._entries = [summary_entry] + recent_entries
        else:
            self._entries = recent_entries

        # Re-add meta entries
        meta = [e for e in self._entries if e.get("type") == "meta"]
        self._entries = [e for e in self._entries if e.get("type") != "meta"] + meta
        self.updated_at = self._now()

        return before, len(self._non_meta_entries())

    # ------------------------------------------------------------------
    # Reconstruct Anthropic API message list
    # ------------------------------------------------------------------

    def to_messages(self, tool_names: Any = None) -> list[dict[str, Any]]:
        """Convert entries to Anthropic API role/content format.

        Drops orphaned tool_results — a tool_result with no open preceding
        assistant tool_use — so a compaction or interruption boundary can never
        produce an API-invalid sequence (a ``tool_result`` / ``role:"tool"`` with
        no preceding ``tool_use`` / ``tool_calls``, which 400s and bricks the
        REPL; R2-8). Both backends consume this list, so the guard protects
        Anthropic and the OpenAI-compatible path alike.

        The mirror case — a tool_use whose result never arrived (Ctrl-C or an
        exception mid-dispatch) — gets a synthetic ``is_error`` "interrupted"
        tool_result right where the next entry (or the end) closes its window, so
        the session never bricks on a dangling tool_use either.

        *tool_names* (the caller's tool list): a tool_use naming a tool outside it
        (another producer's, e.g. native's verify/finish) is sent as TEXT with its
        result instead (so the model never sees a call to a tool it was not given;
        whether a provider would reject one is unverified), so the turn's memory
        survives; its tool_result is then an
        orphan and dropped, and an assistant entry that follows merges into that
        text-only assistant message, so roles still alternate.
        """
        messages: list[dict[str, Any]] = []
        open_tool_use_ids: list[str] = []
        results = {r.get("tool_use_id"): r.get("content") for e in self._entries
                   if e.get("type") == "tool_result" for r in e.get("results") or ()}

        def _foreign(b: Any) -> bool:
            return tool_names is not None and isinstance(b, dict) and b.get("name") not in tool_names

        def _interrupted(ids: list[str]) -> list[dict[str, Any]]:
            return [{"type": "tool_result", "tool_use_id": tid,
                     "content": "interrupted: this tool call did not complete",
                     "is_error": True} for tid in ids]

        def _close_dangling() -> None:
            if open_tool_use_ids:
                messages.append({"role": "user", "content": _interrupted(open_tool_use_ids)})
                open_tool_use_ids.clear()

        for entry in self._entries:
            etype = entry.get("type")
            if etype == "user":
                _close_dangling()  # plain user text closes the tool window
                messages.append({"role": "user", "content": entry["content"]})
            elif etype == "assistant":
                _close_dangling()
                content: list[dict[str, Any]] = []
                # Only include text block if non-empty (Anthropic rejects empty text)
                if entry.get("text"):
                    content.append({"type": "text", "text": entry["text"]})
                tool_use = entry.get("tool_use") or []
                content.extend(
                    {"type": "text", "text": f"[{b.get('name')}({json.dumps(b.get('input') or {}, ensure_ascii=False)})"
                                             f" -> {results.get(b.get('id'), '(no result)')}]"}
                    for b in tool_use if _foreign(b)
                )
                tool_use = [b for b in tool_use if not _foreign(b)]
                content.extend(tool_use)
                if content and messages and messages[-1]["role"] == "assistant":
                    # The previous assistant went text-only (its foreign tool_use became
                    # text, its result dropped): merge, so roles still alternate.
                    messages[-1]["content"].extend(content)
                elif content:
                    messages.append({"role": "assistant", "content": content})
                open_tool_use_ids.extend(
                    b["id"] for b in tool_use if isinstance(b, dict) and "id" in b
                )
            elif etype == "tool_result":
                tool_result_blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": r["tool_use_id"],
                        "content": r["content"],
                        **({"is_error": r["is_error"]} if "is_error" in r else {}),
                    }
                    for r in entry["results"]
                    if r.get("tool_use_id") in open_tool_use_ids
                ]
                if tool_result_blocks:
                    # Every tool_use of the turn is answered in THIS message.
                    answered = {b["tool_use_id"] for b in tool_result_blocks}
                    tool_result_blocks += _interrupted(
                        [t for t in open_tool_use_ids if t not in answered]
                    )
                    open_tool_use_ids.clear()
                    messages.append({"role": "user", "content": tool_result_blocks})
                # else: fully orphaned -> skip (no preceding tool_use)
            # type == "meta" is skipped (internal tracking)
        _close_dangling()
        return messages

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write all entries + meta to the JSONL file (atomic fsync).

        A session is written on its first entry: an empty one leaves no file.
        """
        if not self._entries and not self._path.exists():
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)

        # Build the full set of lines: all content entries + a trailing meta
        meta_entry: dict[str, Any] = {
            "type": "meta",
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "token_usage": self.token_usage.to_dict(),
            "metadata": self.metadata,
            "ts": self._now(),
        }
        all_entries = self._entries + [meta_entry]
        lines = "\n".join(json.dumps(e, ensure_ascii=False) for e in all_entries) + "\n"

        tmp_path = self._path.with_suffix(".jsonl.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                f.write(lines)
                f.flush()
                os.fsync(f.fileno())
            tmp_path.replace(self._path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


# ---------------------------------------------------------------------------
# Module-level factory functions
# ---------------------------------------------------------------------------


def _resolve_dir(directory: Path | None) -> Path:
    return directory if directory is not None else paths.sessions()


def create_session(
    metadata: dict[str, Any] | None = None,
    directory: Path | None = None,
) -> Session:
    """Create a new session (its file is written on its first entry) and return it."""
    session_dir = _resolve_dir(directory)
    session_dir.mkdir(parents=True, exist_ok=True)

    sid = str(uuid.uuid4())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = session_dir / f"{sid}.jsonl"

    session = Session(
        session_id=sid,
        created_at=now,
        updated_at=now,
        path=path,
        metadata=dict(metadata) if metadata else {},
    )
    return session


def load_session(session_id: str, directory: Path | None = None) -> Session:
    """Load an existing session from its JSONL file.

    Corrupt lines (invalid JSON) are skipped so a partial write on crash
    does not destroy the entire session history.
    """
    session_dir = _resolve_dir(directory)
    path = session_dir / f"{session_id}.jsonl"

    entries: list[dict[str, Any]] = []
    meta: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # skip corrupt line
        if obj.get("type") == "meta":
            meta = obj
        else:
            entries.append(obj)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return Session(
        session_id=session_id,
        created_at=meta.get("created_at", now),
        updated_at=meta.get("updated_at", now),
        path=path,
        entries=entries,
        token_usage=TokenUsage.from_dict(meta.get("token_usage", {})),
        metadata=meta.get("metadata", {}),
    )


def _newest_first(session_dir: Path) -> list[Path]:
    paths = []
    for p in session_dir.glob("*.jsonl") if session_dir.exists() else ():
        try:
            paths.append((p.stat().st_mtime, p))
        except OSError:
            continue
    return [p for _, p in sorted(paths, reverse=True)]


def list_sessions(directory: Path | None = None) -> list[SessionSummary]:
    """A SessionSummary of every session that holds a message, newest-modified first."""
    session_dir = _resolve_dir(directory)
    summaries: list[SessionSummary] = []
    for p in _newest_first(session_dir):
        try:
            session = load_session(p.stem, directory=session_dir)
        except Exception:  # noqa: BLE001 — an unreadable file is not a session
            continue
        if session._entries:
            summaries.append(SessionSummary(
                session_id=session.session_id, created_at=session.created_at,
                updated_at=session.updated_at, message_count=len(session._non_meta_entries()), path=p,
            ))
    return summaries


def get_latest_session(directory: Path | None = None) -> Session | None:
    """The most recently modified conversation to resume: never an empty session and
    never a one-shot ``-p`` turn (``metadata['face'] == 'print'``). None if there is none.

    Files are read newest first and only until one qualifies."""
    session_dir = _resolve_dir(directory)
    for p in _newest_first(session_dir):
        try:
            session = load_session(p.stem, directory=session_dir)
        except Exception:  # noqa: BLE001
            continue
        if session._entries and session.metadata.get("face") != "print":
            return session
    return None

"""Session persistence for vcli — JSONL format.

Each session is stored as a file of newline-delimited JSON objects:
  <session_id>.jsonl  under DEFAULT_SESSION_DIR (or caller-supplied directory).

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_SESSION_DIR: Path = Path.home() / ".vector" / "sessions"

# ---------------------------------------------------------------------------
# Value objects (frozen)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TokenUsage":
        return cls(
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            cache_read_tokens=d.get("cache_read_tokens", 0),
            cache_creation_tokens=d.get("cache_creation_tokens", 0),
        )


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

        old_entries = non_meta[:-keep_recent]
        recent_entries = non_meta[-keep_recent:]

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

    def to_messages(self) -> list[dict[str, Any]]:
        """Convert entries to Anthropic API role/content format."""
        messages: list[dict[str, Any]] = []
        for entry in self._entries:
            etype = entry.get("type")
            if etype == "user":
                messages.append({"role": "user", "content": entry["content"]})
            elif etype == "assistant":
                content: list[dict[str, Any]] = []
                # Only include text block if non-empty (Anthropic rejects empty text)
                if entry.get("text"):
                    content.append({"type": "text", "text": entry["text"]})
                if entry.get("tool_use"):
                    content.extend(entry["tool_use"])
                if content:
                    messages.append({"role": "assistant", "content": content})
            elif etype == "tool_result":
                tool_result_blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": r["tool_use_id"],
                        "content": r["content"],
                        **({"is_error": r["is_error"]} if "is_error" in r else {}),
                    }
                    for r in entry["results"]
                ]
                messages.append({"role": "user", "content": tool_result_blocks})
            # type == "meta" is skipped (internal tracking)
        return messages

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write all entries + meta to the JSONL file (atomic fsync)."""
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
    return directory if directory is not None else DEFAULT_SESSION_DIR


def create_session(
    metadata: dict[str, Any] | None = None,
    directory: Path | None = None,
) -> Session:
    """Create a new session, write initial file to disk, and return it."""
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
    session.save()
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


def list_sessions(directory: Path | None = None) -> list[SessionSummary]:
    """Return SessionSummary list for all sessions, newest-modified first."""
    session_dir = _resolve_dir(directory)
    if not session_dir.exists():
        return []

    summaries: list[SessionSummary] = []
    for p in session_dir.glob("*.jsonl"):
        try:
            session = load_session(p.stem, directory=session_dir)
        except Exception:
            continue
        non_meta = session._non_meta_entries()
        summaries.append(
            SessionSummary(
                session_id=session.session_id,
                created_at=session.created_at,
                updated_at=session.updated_at,
                message_count=len(non_meta),
                path=p,
            )
        )

    summaries.sort(key=lambda s: s.path.stat().st_mtime, reverse=True)
    return summaries


def get_latest_session(directory: Path | None = None) -> Session | None:
    """Return the most recently modified session, or None if none exist."""
    summaries = list_sessions(directory=directory)
    if not summaries:
        return None
    latest = summaries[0]
    return load_session(latest.session_id, directory=_resolve_dir(directory))

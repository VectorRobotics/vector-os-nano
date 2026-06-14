# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""On-disk run registry for background (daemonized) runs — Phase E W2.1.

One JSON file per run under ``~/.vector-os-nano/runs/`` (or an injected root),
written ATOMICALLY (tmp + ``os.replace``) so a crash can never leave a torn
file. Entries are frozen dataclasses (additive-only evolution). Listing sweeps
entries whose PID is no longer alive — a crashed run cleans itself up on the
next ``status`` call.

Pure stdlib, no kernel imports — the registry is an operational leaf both the
daemon and the CLI ops consume.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path.home() / ".vector-os-nano" / "runs"


@dataclass(frozen=True)
class RunEntry:
    """Immutable record of one background run."""

    run_id: str
    pid: int
    started_at: float
    world: str
    scenario: str
    log_dir: str
    argv: tuple[str, ...]


def _pid_alive(pid: int) -> bool:
    """True when *pid* exists (signal 0 probe; EPERM still means alive)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class RunRegistry:
    """Name -> RunEntry persistence with atomic writes and dead-PID sweep."""

    def __init__(self, root: "Path | str | None" = None) -> None:
        self._root = Path(root) if root is not None else _DEFAULT_ROOT
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        # run_id is caller-generated (uuid/short id) — keep it filename-safe.
        safe = "".join(c for c in run_id if c.isalnum() or c in "-_") or "run"
        return self._root / f"{safe}.json"

    def register(self, entry: RunEntry) -> None:
        """Persist *entry* atomically (tmp + ``os.replace``)."""
        payload = dataclasses.asdict(entry)
        payload["argv"] = list(entry.argv)
        path = self._path(entry.run_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, path)

    def remove(self, run_id: str) -> None:
        """Delete the entry for *run_id*. Idempotent."""
        try:
            self._path(run_id).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("RunRegistry: remove(%s) failed: %s", run_id, exc)

    def get(self, run_id: str) -> "RunEntry | None":
        """Return the entry for *run_id*, or None (missing/corrupt)."""
        try:
            data = json.loads(self._path(run_id).read_text())
            return RunEntry(
                run_id=str(data["run_id"]),
                pid=int(data["pid"]),
                started_at=float(data["started_at"]),
                world=str(data.get("world", "")),
                scenario=str(data.get("scenario", "")),
                log_dir=str(data.get("log_dir", "")),
                argv=tuple(data.get("argv", ())),
            )
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001 — corrupt entry is not fatal
            logger.warning("RunRegistry: unreadable entry %s: %s", run_id, exc)
            return None

    def list_runs(self, sweep_dead: bool = True) -> tuple[RunEntry, ...]:
        """Return all live entries (sorted by start time).

        With *sweep_dead* (default), an entry whose PID no longer exists is
        removed from disk and omitted — crashed runs self-clean on inspection.
        """
        runs: list[RunEntry] = []
        for path in sorted(self._root.glob("*.json")):
            entry = self.get(path.stem)
            if entry is None:
                continue
            if sweep_dead and not _pid_alive(entry.pid):
                logger.info("RunRegistry: sweeping dead run %s (pid %d)",
                            entry.run_id, entry.pid)
                self.remove(entry.run_id)
                continue
            runs.append(entry)
        return tuple(sorted(runs, key=lambda r: r.started_at))

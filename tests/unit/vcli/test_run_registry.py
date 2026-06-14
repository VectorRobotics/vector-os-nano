# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""W2.1 — on-disk run registry (frozen entries, atomic writes, dead-PID sweep)."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys

import pytest

from vector_os_nano.vcli.run_registry import RunEntry, RunRegistry


def _entry(run_id: str = "r1", pid: int | None = None) -> RunEntry:
    return RunEntry(
        run_id=run_id,
        pid=pid if pid is not None else os.getpid(),
        started_at=1234567890.0,
        world="dev",
        scenario="",
        log_dir="/tmp/x",
        argv=("vector-cli", "--daemon"),
    )


class TestRunEntry:
    def test_frozen(self) -> None:
        e = _entry()
        with pytest.raises(dataclasses.FrozenInstanceError):
            e.pid = 1  # type: ignore[misc]


class TestRegistryPersistence:
    def test_register_and_get_roundtrip(self, tmp_path) -> None:
        reg = RunRegistry(root=tmp_path)
        e = _entry("abc")
        reg.register(e)
        got = reg.get("abc")
        assert got == e

    def test_atomic_file_is_valid_json(self, tmp_path) -> None:
        reg = RunRegistry(root=tmp_path)
        reg.register(_entry("abc"))
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["run_id"] == "abc"
        assert not list(tmp_path.glob("*.tmp"))  # no tmp residue

    def test_remove(self, tmp_path) -> None:
        reg = RunRegistry(root=tmp_path)
        reg.register(_entry("abc"))
        reg.remove("abc")
        assert reg.get("abc") is None
        assert not list(tmp_path.glob("*.json"))

    def test_remove_missing_is_idempotent(self, tmp_path) -> None:
        RunRegistry(root=tmp_path).remove("nope")  # no raise

    def test_corrupt_file_skipped_not_fatal(self, tmp_path) -> None:
        (tmp_path / "bad.json").write_text("{not json")
        reg = RunRegistry(root=tmp_path)
        assert reg.list_runs() == ()


class TestLiveness:
    def test_live_run_listed(self, tmp_path) -> None:
        reg = RunRegistry(root=tmp_path)
        reg.register(_entry("me", pid=os.getpid()))
        runs = reg.list_runs()
        assert [r.run_id for r in runs] == ["me"]

    def test_dead_pid_swept_on_list(self, tmp_path) -> None:
        # A short-lived real process: spawn + wait => the PID is dead.
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        reg = RunRegistry(root=tmp_path)
        reg.register(_entry("dead", pid=proc.pid))
        assert reg.list_runs() == ()           # swept
        assert reg.get("dead") is None         # registry file removed too

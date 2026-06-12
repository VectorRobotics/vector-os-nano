# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""ros_bootstrap — NL start without terminal rituals (owner finding (b)).

vector-cli launched from an UNSOURCED shell must re-exec itself once with the
ROS + SysNav overlays sourced (when they exist on disk), so the in-process
SysNav feed/consumer and the GPU node pair all inherit a working env. The
re-exec must never fire under pytest (conftest guard), never loop, and never
block a box without ROS.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


from vector_os_nano.vcli import ros_bootstrap as rb


def _fake_overlays(tmp_path: Path) -> tuple[Path, Path]:
    """A fake /opt/ros/<distro> and a fake SysNav workspace with setup.bash."""
    ros = tmp_path / "opt_ros" / "jazzy"
    ros.mkdir(parents=True)
    (ros / "setup.bash").write_text("export AMENT_PREFIX_PATH=%s\n" % ros)
    ws = tmp_path / "SysNav"
    (ws / "install").mkdir(parents=True)
    (ws / "install" / "setup.bash").write_text(
        "export AMENT_PREFIX_PATH=%s/install:${AMENT_PREFIX_PATH:-}\n" % ws
    )
    return ros, ws


class TestOverlayScripts:
    def test_finds_ros_and_sysnav(self, tmp_path):
        ros, ws = _fake_overlays(tmp_path)
        scripts = rb.overlay_scripts(
            environ={"VECTOR_SYSNAV_WS": str(ws)}, ros_root=ros.parent
        )
        assert scripts == [ros / "setup.bash", ws / "install" / "setup.bash"]

    def test_no_ros_on_disk_means_no_scripts(self, tmp_path):
        # SysNav alone is useless without ROS underneath — empty list.
        _, ws = _fake_overlays(tmp_path)
        scripts = rb.overlay_scripts(
            environ={"VECTOR_SYSNAV_WS": str(ws)},
            ros_root=tmp_path / "nonexistent",
        )
        assert scripts == []

    def test_ros_without_sysnav_workspace(self, tmp_path):
        ros, _ = _fake_overlays(tmp_path)
        scripts = rb.overlay_scripts(
            environ={"VECTOR_SYSNAV_WS": str(tmp_path / "missing")},
            ros_root=ros.parent,
        )
        assert scripts == [ros / "setup.bash"]

    def test_picks_newest_distro(self, tmp_path):
        ros, _ = _fake_overlays(tmp_path)
        older = ros.parent / "humble"
        older.mkdir()
        (older / "setup.bash").write_text("")
        scripts = rb.overlay_scripts(environ={}, ros_root=ros.parent)
        assert scripts[0] == ros / "setup.bash"  # jazzy > humble lexically


class TestNeedsBootstrap:
    def test_unsourced_env_needs_bootstrap(self, tmp_path):
        ros, ws = _fake_overlays(tmp_path)
        scripts = [ros / "setup.bash", ws / "install" / "setup.bash"]
        assert rb.needs_bootstrap(environ={}, scripts=scripts) is True

    def test_fully_sourced_env_does_not(self, tmp_path):
        ros, ws = _fake_overlays(tmp_path)
        scripts = [ros / "setup.bash", ws / "install" / "setup.bash"]
        env = {
            "AMENT_PREFIX_PATH": f"{ws}/install/tare_planner:{ros}",
        }
        assert rb.needs_bootstrap(environ=env, scripts=scripts) is False

    def test_partially_sourced_env_needs_bootstrap(self, tmp_path):
        # ROS sourced but the SysNav overlay missing — re-exec adds it.
        ros, ws = _fake_overlays(tmp_path)
        scripts = [ros / "setup.bash", ws / "install" / "setup.bash"]
        env = {"AMENT_PREFIX_PATH": str(ros)}
        assert rb.needs_bootstrap(environ=env, scripts=scripts) is True

    def test_no_scripts_no_bootstrap(self):
        assert rb.needs_bootstrap(environ={}, scripts=[]) is False


class TestComposeSourcedEnv:
    def test_captures_exports(self, tmp_path):
        script = tmp_path / "setup.bash"
        script.write_text("export VECTOR_TEST_MARK=sourced-ok\n")
        env = rb.compose_sourced_env([script], base_env={"PATH": os.environ["PATH"]})
        assert env is not None
        assert env["VECTOR_TEST_MARK"] == "sourced-ok"
        assert "PATH" in env  # base env flows through bash

    def test_failing_script_returns_none(self, tmp_path):
        script = tmp_path / "setup.bash"
        script.write_text("exit 3\n")
        env = rb.compose_sourced_env([script], base_env={"PATH": os.environ["PATH"]})
        assert env is None

    def test_multiline_values_survive_nul_parsing(self, tmp_path):
        script = tmp_path / "setup.bash"
        script.write_text('export VECTOR_TEST_ML="a\nb"\n')
        env = rb.compose_sourced_env([script], base_env={"PATH": os.environ["PATH"]})
        assert env is not None
        assert env["VECTOR_TEST_ML"] == "a\nb"


class TestMaybeReexec:
    def _arm(self, monkeypatch, tmp_path):
        ros, ws = _fake_overlays(tmp_path)
        calls: list[tuple] = []
        monkeypatch.setattr(os, "execve", lambda *a: calls.append(a))
        monkeypatch.setattr(
            rb, "compose_sourced_env",
            lambda scripts, base_env: {"AMENT_PREFIX_PATH": "composed", **base_env},
        )
        return ros, ws, calls

    def test_reexecs_with_composed_env_and_guard(self, monkeypatch, tmp_path):
        ros, ws, calls = self._arm(monkeypatch, tmp_path)
        environ = {"VECTOR_SYSNAV_WS": str(ws), "PATH": "/usr/bin"}
        rb.maybe_reexec_with_ros(environ=environ, ros_root=ros.parent)
        assert len(calls) == 1
        path, argv, env = calls[0]
        assert path == sys.executable
        assert argv[:3] == [sys.executable, "-m", "vector_os_nano.vcli.cli"]
        # loop guard, truthy; carries the sourced overlay list for the banner
        assert str(ros / "setup.bash") in env["VECTOR_ROS_BOOTSTRAPPED"]
        assert env["AMENT_PREFIX_PATH"] == "composed"

    def test_already_bootstrapped_never_loops(self, monkeypatch, tmp_path):
        ros, ws, calls = self._arm(monkeypatch, tmp_path)
        environ = {
            "VECTOR_SYSNAV_WS": str(ws),
            "VECTOR_ROS_BOOTSTRAPPED": "1",
        }
        rb.maybe_reexec_with_ros(environ=environ, ros_root=ros.parent)
        assert calls == []

    def test_opt_out_env(self, monkeypatch, tmp_path):
        ros, ws, calls = self._arm(monkeypatch, tmp_path)
        environ = {"VECTOR_SYSNAV_WS": str(ws), "VECTOR_NO_ROS_BOOTSTRAP": "1"}
        rb.maybe_reexec_with_ros(environ=environ, ros_root=ros.parent)
        assert calls == []

    def test_sourced_shell_is_a_noop(self, monkeypatch, tmp_path):
        ros, ws, calls = self._arm(monkeypatch, tmp_path)
        environ = {
            "VECTOR_SYSNAV_WS": str(ws),
            "AMENT_PREFIX_PATH": f"{ws}/install/x:{ros}",
        }
        rb.maybe_reexec_with_ros(environ=environ, ros_root=ros.parent)
        assert calls == []

    def test_compose_failure_continues_without_reexec(self, monkeypatch, tmp_path):
        ros, ws, _ = self._arm(monkeypatch, tmp_path)
        calls: list[tuple] = []
        monkeypatch.setattr(os, "execve", lambda *a: calls.append(a))
        monkeypatch.setattr(rb, "compose_sourced_env", lambda *a, **k: None)
        environ = {"VECTOR_SYSNAV_WS": str(ws)}
        rb.maybe_reexec_with_ros(environ=environ, ros_root=ros.parent)
        assert calls == []  # fail-open: the tool still fails loud later

    def test_no_ros_box_is_a_noop(self, monkeypatch, tmp_path):
        calls: list[tuple] = []
        monkeypatch.setattr(os, "execve", lambda *a: calls.append(a))
        rb.maybe_reexec_with_ros(
            environ={}, ros_root=tmp_path / "nonexistent"
        )
        assert calls == []


class TestPytestGuard:
    def test_conftest_disables_bootstrap(self):
        # The repo conftest must protect every pytest run from execve.
        assert os.environ.get("VECTOR_NO_ROS_BOOTSTRAP") == "1"

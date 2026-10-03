# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Guards for the suite's own hygiene: patterns that once OOM'd or poisoned this box.

Each rule names the ledger key it pins. A guard failure lists every offending line.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_TESTS = _REPO / "tests"
_SELF = Path(__file__).resolve()


def _test_sources() -> list[tuple[Path, list[str]]]:
    return [
        (p, p.read_text(encoding="utf-8").splitlines())
        for p in sorted(_TESTS.rglob("*.py"))
        if p.resolve() != _SELF
    ]


def _offenders(pattern: str, *, unless: str | None = None, skip: tuple[str, ...] = ()) -> list[str]:
    rx, ok = re.compile(pattern), re.compile(unless) if unless else None
    return [
        f"{p.relative_to(_REPO)}:{i}: {line.strip()}"
        for p, lines in _test_sources() if p.name not in skip
        for i, line in enumerate(lines, 1)
        if rx.search(line) and not (ok and ok.search(line))
    ]


def test_no_builtins_open_patch() -> None:
    """tests:isaac-mock-open-oom — yaml read a MagicMock stream forever (12 GB)."""
    assert not _offenders(r"""patch\(\s*["']builtins\.open["']""")


def test_no_recording_sleep_patch_without_a_fake_clock() -> None:
    """tests:isaac-mock-open-oom sibling — a MagicMock time.sleep under a real wall
    clock turns `while time.monotonic() < deadline` into a busy loop that records a
    call per spin. Use tests.harness.clock.fake_clock (or patch the clock alongside).
    """
    assert not _offenders(
        r"""patch\(\s*["'][\w.]*time\.sleep["']\s*(,\s*return_value\s*=\s*None\s*)?\)""",
        unless=r"""patch\(\s*["']time\.(time|monotonic)["']""",
        skip=("clock.py",),
    )


def test_no_machine_wide_kill_commands() -> None:
    """tests:pty-harness-timeouts-and-kill — `rosm nuke` / pkill took down other
    projects' stacks. Kill only your own process group (tests/harness/sim_cleanup.py)."""
    assert not _offenders(
        r"""\[\s*["'](rosm|pkill|killall)["']|["'](pkill|killall) -|["']rosm nuke""",
        skip=("conftest.py",),
    )


def test_no_import_time_env_writes() -> None:
    """tests:env-leak-sim-with-arm — a module-level os.environ write at collection
    made scene choice depend on test order. Use monkeypatch in a fixture."""
    assert not _offenders(r"""^os\.environ(\[.*\]\s*=|\.setdefault\(|\.update\()""",
                          skip=("conftest.py",))


def _tracked_assets() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*.xml", "*.yaml", "*.yml", "*.urdf"],
        cwd=_REPO, capture_output=True, text=True, check=False,
    ).stdout.split()
    return out or ["<no git checkout>"]


@pytest.mark.parametrize("asset", _tracked_assets())
def test_no_absolute_home_paths_in_committed_assets(asset: str) -> None:
    """docs-vs-code:go2piper-abs-mesh-paths — 98 /home/yusen mesh paths made the
    go2+Piper scene uncompilable on any other machine."""
    if asset.startswith("<"):
        pytest.skip("not a git checkout")
    text = (_REPO / asset).read_text(encoding="utf-8", errors="replace")
    hits = [ln.strip()[:120] for ln in text.splitlines() if "/home/" in ln]
    assert not hits, f"{asset}: {len(hits)} absolute /home/ path(s), e.g. {hits[0]}"


def test_ros_python_ahead_of_the_venv_is_detected() -> None:
    """tests:ros-pythonpath-segfault — /opt/ros's pinocchio ahead of the venv's
    segfaults collection; tests/conftest.py exits early with a clear message."""
    from tests.conftest import _ros_shadows_venv

    site = "/v/.venv/lib/python3.12/site-packages"
    assert _ros_shadows_venv(["", "/opt/ros/jazzy/lib/python3.12/site-packages", site], site)
    assert not _ros_shadows_venv(["", site, "/opt/ros/jazzy/lib/python3.12/site-packages"], site)
    assert not _ros_shadows_venv(["", site], site)


def test_stub_restore_drops_stubs_and_modules_bound_to_them() -> None:
    """tests:fake-modules-leak — a fake rclpy (no __spec__) outlived its file and
    broke later collection (importlib.util.find_spec ValueError)."""
    from tests.harness.stubs import stubbed_modules

    fake = types.ModuleType("rclpy_fake_probe_msgs")
    name = "rclpy_fake_probe_msgs"
    assert name not in sys.modules
    with stubbed_modules({name: fake}):
        assert sys.modules[name] is fake
        sys.modules["vector_os_nano._probe_bound_to_stub"] = types.ModuleType("x")
    assert name not in sys.modules
    assert "vector_os_nano._probe_bound_to_stub" not in sys.modules


def test_harness_stub_file_does_not_poison_later_files() -> None:
    """The order that used to ERROR at collection: level9 (fake rclpy) then a file
    calling find_spec('rclpy')."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/harness/test_level9_proxy_e2e.py", "tests/unit/vcli/test_ros2_proxy_logging.py"],
        cwd=_REPO, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]


def test_billed_vlm_benchmarks_are_live_llm_gated() -> None:
    """tests:missed2 — the harness VLM benchmarks called OpenRouter whenever a key
    was in the env/config; billed calls need the explicit VECTOR_LIVE_LLM=1 opt-in."""
    for f in ("test_level0_vlm_api.py", "test_level1_camera_vlm.py",
              "test_level4_patrol.py", "test_level5_vlm_accuracy.py"):
        src = (_TESTS / "harness" / f).read_text(encoding="utf-8")
        assert "pytestmark = pytest.mark.live_llm" in src, f


def test_sim_preflight_skips_the_tier_with_one_reason(tmp_path) -> None:
    """tests:sim-prereq-error-not-skip — on a box without EGL the sim tier used to
    ERROR test by test; now ONE preflight skips it with ONE aggregated reason."""
    probe = tmp_path / "test_probe_sim.py"
    probe.write_text(
        "import pytest\n\n"
        "@pytest.mark.sim\ndef test_a():\n    raise AssertionError('must not run')\n\n"
        "@pytest.mark.sim\ndef test_b():\n    raise AssertionError('must not run')\n"
    )
    env = {**os.environ, "MUJOCO_GL": "egl",
           "__EGL_VENDOR_LIBRARY_FILENAMES": "/nonexistent/egl_vendor.json"}
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-p", "no:cacheprovider",
         "-c", str(_REPO / "pyproject.toml"), "-q", "-rs", "--rootdir", str(tmp_path), str(probe)],
        cwd=_REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "2 skipped" in out, out[-3000:]
    reasons = [ln for ln in r.stdout.splitlines() if ln.startswith("SKIPPED")]
    assert len(reasons) == 1 and "sim preflight: no offscreen GL" in reasons[0], out[-3000:]

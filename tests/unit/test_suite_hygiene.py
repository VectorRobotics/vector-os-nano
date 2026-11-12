# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Guards for the suite's own hygiene: patterns that once OOM'd or poisoned this box.

Each rule names the ledger key it pins. A guard failure lists every offending line.
"""
from __future__ import annotations

import json
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
    """Matching lines in tests/**.py; *skip* holds repo-relative paths (exact files)."""
    rx, ok = re.compile(pattern), re.compile(unless) if unless else None
    return [
        f"{p.relative_to(_REPO)}:{i}: {line.strip()}"
        for p, lines in _test_sources() if p.relative_to(_REPO).as_posix() not in skip
        for i, line in enumerate(lines, 1)
        if rx.search(line) and not (ok and ok.search(line))
    ]


def test_no_builtins_open_patch() -> None:
    """tests:isaac-mock-open-oom — yaml read a MagicMock stream forever (12 GB)."""
    assert not _offenders(r"""patch\(\s*["']builtins\.open["']""")


# Every way to patch time.sleep: patch("...time.sleep", <anything>) and patch.object(time, "sleep").
_SLEEP_PATCH = r"""patch\(\s*["'][\w.]*\btime\.sleep["']|patch\.object\(\s*[\w.]*\btime\s*,\s*["']sleep["']"""


def test_no_recording_sleep_patch_without_a_fake_clock() -> None:
    """tests:isaac-mock-open-oom sibling — a MagicMock time.sleep under a real wall
    clock turns `while time.monotonic() < deadline` into a busy loop that records a
    call per spin. Use tests.harness.clock.fake_clock (or patch the clock alongside).
    """
    assert not _offenders(
        _SLEEP_PATCH,
        unless=r"""patch\(\s*["'][\w.]*\btime\.(time|monotonic)["']""",
        skip=("tests/harness/clock.py",),
    )


@pytest.mark.parametrize("line", [
    'with patch("time.sleep"):',
    'with patch("time.sleep", new=lambda *_: None):',
    'with patch("pkg.mod.time.sleep", side_effect=calls.append):',
    'with patch.object(time, "sleep"):',
])
def test_sleep_guard_catches_every_patch_form(line: str) -> None:
    """The guard once matched only patch("time.sleep"[, return_value=None])."""
    assert re.search(_SLEEP_PATCH, line)


def _tracked_text() -> list[tuple[str, list[str]]]:
    """Every tracked text file, except the docs (docs/, README.md, CLAUDE.md): those
    are excluded only until the release's final docs pass rewrites them."""
    out = subprocess.run(["git", "ls-files", "-z"], cwd=_REPO, capture_output=True, check=False).stdout
    files = []
    for rel in out.decode().split("\0"):
        p = _REPO / rel
        if not rel or rel.startswith("docs/") or rel in ("README.md", "CLAUDE.md") or p == _SELF or not p.is_file():
            continue
        data = p.read_bytes()
        if b"\0" not in data:
            files.append((rel, data.decode("utf-8", "replace").splitlines()))
    assert files, "not a git checkout: the tree-wide guards have nothing to scan"
    return files


def _tracked_offenders(pattern: str, skip: tuple[str, ...] = ()) -> list[str]:
    rx = re.compile(pattern)
    return [f"{rel}:{i}: {line.strip()[:120]}" for rel, lines in _tracked_text() if rel not in skip
            for i, line in enumerate(lines, 1) if rx.search(line)]


# A kill by name or a machine-wide teardown, as a Python argv / string or a shell command.
_KILL = r"""\[\s*["'](rosm|pkill|killall|fuser)["']|(^|[;&|"'])\s*(sudo\s+)?(pkill|killall|rosm\s+nuke|fuser\s+-k)\b"""


def test_no_machine_wide_kill_commands() -> None:
    """tests:pty-harness-timeouts-and-kill, periphery:tools-kill-by-name-and-rosm —
    `rosm nuke` / pkill took down other projects' stacks. Kill only your own process
    group (tests/harness/sim_cleanup.py). Every tracked file, not only tests."""
    assert not _tracked_offenders(_KILL, skip=("tests/conftest.py",))  # the shims that fail those calls


@pytest.mark.parametrize("line, hit", [
    ('subprocess.run(["pkill", "-9", "-f", "verify_fetch_flow"])', True),
    ("subprocess.run(\"pkill -9 -f '[m]ujoco' 2>/dev/null || true\", shell=True)", True),
    ('subprocess.run(["rosm", "nuke", "--yes"])', True),
    ("  killall -9 ros2", True),
    ("trap 'fuser -k 8765/tcp' EXIT", True),
    ("ok || pkill -f tare", True),
    ("# own group under `set -m`). Never pkill by name or wipe /dev/shm", False),
    ("Replaces the old ``rosm nuke --yes`` / ``pkill -f`` teardowns", False),
    ('"tools": "periphery:tools-kill-by-name-and-rosm (pkill -9 -f mujoco, rosm nuke)"', False),
])
def test_kill_guard_catches_invocations_not_prose(line: str, hit: bool) -> None:
    assert bool(re.search(_KILL, line)) is hit


def test_no_import_time_env_writes() -> None:
    """tests:env-leak-sim-with-arm — a module-level os.environ write at collection
    made scene choice depend on test order. Use monkeypatch in a fixture."""
    assert not _offenders(r"""^os\.environ(\[.*\]\s*=|\.setdefault\(|\.update\()""",
                          skip=("tests/conftest.py",))


_MACHINE_PATH = r"""/(home|Users)/[A-Za-z0-9_][\w.-]*/|~/Desktop\b|["']Desktop["']"""


def test_no_machine_paths_in_tracked_files() -> None:
    """tests:hardcoded-machine-paths, docs-vs-code:go2piper-abs-mesh-paths,
    tests-gate-e2e:hygiene-guard-blind-to-this-machine-and-dotfiles — harness tests read
    ~/Desktop/vector_navigation_stack, 98 /home/yusen mesh paths broke the go2+Piper
    scene, a tracked .mcp.json named another user's checkout. No tracked file (code,
    scripts, configs, MJCF, tests, dotfiles) names any account's home: an external
    checkout comes from its env var (VECTOR_NAV_STACK via scripts/nav_env.sh)."""
    assert not _tracked_offenders(_MACHINE_PATH)


@pytest.mark.parametrize("line, hit", [
    ('"command": "/home/yusen/Desktop/vector_os_nano/.venv/bin/python"', True),
    ('<mesh file="/home/alice/robots/go2/base_0.obj"/>', True),
    ("ROOT=/Users/bob/src/nav", True),
    ('Path.home() / "Desktop" / "vector_ws"', True),
    ("cd ~/Desktop/vector_navigation_stack", True),
    ('GlobTool().execute({"pattern": "../home/.ssh/*"})', False),
    ("os.path.expanduser('~/.vector/config.yaml')", False),
])
def test_machine_path_guard_matches_any_account(line: str, hit: bool) -> None:
    assert bool(re.search(_MACHINE_PATH, line)) is hit


def _tracked_but_ignored(repo: Path) -> list[str]:
    return subprocess.run(["git", "ls-files", "-i", "-c", "--exclude-standard"], cwd=repo,
                          capture_output=True, text=True, check=True).stdout.split()


def test_no_tracked_file_is_ignored() -> None:
    """docs-packaging:gitignore-assets-hides-meshes — an unanchored `assets/` matched
    the 232 tracked package meshes; hatchling honours .gitignore, so the wheel shipped
    none of them and a new mesh was silently left untracked."""
    assert not _tracked_but_ignored(_REPO)


def test_tracked_but_ignored_detects_a_swallowed_mesh(tmp_path) -> None:
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run(git + ["init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "pkg" / "assets").mkdir(parents=True)
    (tmp_path / "pkg" / "assets" / "base.obj").write_text("v 0 0 0\n")
    (tmp_path / ".gitignore").write_text("assets/\n")
    subprocess.run(git + ["add", "-f", "pkg/assets/base.obj", ".gitignore"], cwd=tmp_path, check=True)
    assert _tracked_but_ignored(tmp_path) == ["pkg/assets/base.obj"]
    (tmp_path / ".gitignore").write_text("/assets/\n")
    assert _tracked_but_ignored(tmp_path) == []


def test_no_vacuous_tests() -> None:
    """tests:vacuous-tests — a test whose body is only a docstring/pass, or whose
    assert ends in `or True`, passes whatever the product does."""
    import ast

    bad = []
    for p, lines in _test_sources():
        if not p.name.startswith("test_"):
            continue
        for node in ast.walk(ast.parse("\n".join(lines))):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                body = [b for b in node.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant))]
                if all(isinstance(b, ast.Pass) for b in body):
                    bad.append(f"{p.relative_to(_REPO)}:{node.lineno} {node.name}: empty body")
            if (isinstance(node, ast.Assert) and isinstance(node.test, ast.BoolOp)
                    and isinstance(node.test.op, ast.Or)
                    and any(isinstance(v, ast.Constant) and v.value is True for v in node.test.values)):
                bad.append(f"{p.relative_to(_REPO)}:{node.lineno}: assert ... or True")
    assert not bad, bad


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


def test_harness_stub_file_does_not_poison_later_files(tmp_path) -> None:
    """The order that used to ERROR at collection: a file that imports under a fake
    rclpy (the old harness level9) then a file calling find_spec('rclpy')."""
    (tmp_path / "test_a_stubbed.py").write_text(
        "import types\nfrom tests.harness.stubs import stubbed_modules\n\n"
        "with stubbed_modules({'rclpy': types.ModuleType('rclpy')}):\n    import json  # noqa: F401\n\n"
        "def test_a():\n    assert True\n"
    )
    (tmp_path / "test_b_find_spec.py").write_text(
        "import importlib.util\n\nSPEC = importlib.util.find_spec('rclpy')  # ValueError on a leaked stub\n\n"
        "def test_b():\n    assert True\n"
    )
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-p", "no:cacheprovider",
         "-c", str(_REPO / "pyproject.toml"), "-q", "--rootdir", str(tmp_path),
         str(tmp_path / "test_a_stubbed.py"), str(tmp_path / "test_b_find_spec.py")],
        cwd=_REPO, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0 and "2 passed" in r.stdout, r.stdout[-3000:] + r.stderr[-2000:]


def test_billed_vlm_benchmarks_are_live_llm_gated() -> None:
    """tests:missed2 — the harness VLM benchmarks called OpenRouter whenever a key
    was in the env/config; billed calls need the explicit VECTOR_LIVE_LLM=1 opt-in."""
    for f in ("test_level0_vlm_api.py", "test_level1_camera_vlm.py",
              "test_level5_vlm_accuracy.py"):
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
    # ONE aggregated reason that names the GL cause (it also names unprovisioned weights, if any)
    assert len(reasons) == 1 and "sim preflight: " in reasons[0] and "no offscreen GL" in reasons[0], out[-3000:]


def test_gate_counts_a_file_that_fails_to_collect_as_red(tmp_path) -> None:
    """scripts/gate.sh per-file mode listed files from collected items only, so a
    file that failed to import vanished from the run and the gate stayed green."""
    broken = tmp_path / "test_broken_import.py"
    broken.write_text("import nonexistent_module_xyz\n")
    r = subprocess.run(
        ["bash", str(_REPO / "scripts" / "gate.sh"), "hermetic"],
        env={**os.environ, "GATE_PER_FILE": "1", "GATE_PATHS": str(broken),
             "GATE_OUT": str(tmp_path / "out")},
        cwd=_REPO, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode != 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "RED rc=2 collection" in r.stdout, r.stdout[-2000:]


@pytest.mark.parametrize("tier", ["sim", "e2e"])
def test_gate_counts_a_tier_skipped_by_the_sim_preflight_as_red(tmp_path, tier) -> None:
    """C0F1V-1 — with EdgeTAM missing from the HF cache the preflight skipped all 573
    sim tests and the gate printed green; a tier that ran nothing proved nothing."""
    probe = tmp_path / "test_probe_sim.py"
    probe.write_text("import pytest\n\n@pytest.mark.sim\ndef test_a():\n    pass\n\n"
                     "@pytest.mark.sim\ndef test_b():\n    pass\n")
    r = subprocess.run(
        ["bash", str(_REPO / "scripts" / "gate.sh"), tier, "-p", "tests.conftest"],
        env={**os.environ, "GATE_PER_FILE": "0", "GATE_PATHS": str(probe),
             "GATE_OUT": str(tmp_path / "out"), "MUJOCO_GL": "egl",
             "__EGL_VENDOR_LIBRARY_FILENAMES": "/nonexistent/egl_vendor.json"},
        cwd=_REPO, capture_output=True, text=True, timeout=600,
    )
    assert r.returncode != 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert f"[gate] RED: the {tier} tier skipped on the sim preflight" in r.stdout, r.stdout[-2000:]
    assert "[gate] 2 tests, 2 skipped" in r.stdout, r.stdout[-2000:]


def test_own_process_tests_run_in_a_child_a_self_signal_ends_only_it(tmp_path) -> None:
    """tests-inventory:signal-own-pid — native_safety SIGINTed the pytest process from a
    timer; one firing late would abort the whole file. @own_process runs it in a child."""
    probe = tmp_path / "test_probe_own_process.py"
    probe.write_text(
        "import os, signal, pytest\n\n"
        "@pytest.mark.own_process\ndef test_in_child():\n"
        "    assert os.environ.get('VECTOR_TEST_OWN_PROCESS_CHILD') == '1'\n\n"
        "@pytest.mark.own_process\ndef test_child_fails():\n    assert False, 'red in the child'\n\n"
        "@pytest.mark.own_process\ndef test_kills_itself():\n    os.kill(os.getpid(), signal.SIGTERM)\n\n"
        "@pytest.mark.own_process\ndef test_child_skips():\n    pytest.skip('no weights here')\n\n"
        "def test_parent_still_runs():\n    assert 'VECTOR_TEST_OWN_PROCESS_CHILD' not in os.environ\n"
    )
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-p", "no:cacheprovider",
         "-c", str(_REPO / "pyproject.toml"), "-q", "--rootdir", str(tmp_path), str(probe)],
        cwd=_REPO, capture_output=True, text=True, timeout=300,
    )
    out = r.stdout + r.stderr
    assert "2 failed, 2 passed, 1 skipped" in r.stdout, out[-3000:]
    assert "red in the child" in r.stdout and "rc=-15" in r.stdout, out[-3000:]


@pytest.mark.parametrize("state", [".vector", "elsewhere"])
@pytest.mark.parametrize("tier", ["hermetic", "sim", "e2e"])
def test_gate_gives_every_tier_a_throwaway_home_with_the_real_model_caches(tmp_path, tier, state) -> None:
    """tests-gate-e2e:sim-tier-uses-real-home — only the hermetic tier swapped HOME, so
    the sim and e2e tiers wrote the owner's ~/.vector (strategy stats, traces, sessions).
    The real state directory is $VECTOR_HOME wherever it is: the default ~/.vector, or
    a VECTOR_HOME the owner exported."""
    real, out = tmp_path / "real_home", tmp_path / "env.json"
    store = real / state
    (store / "assets" / "g1_gait").mkdir(parents=True)
    (store / "assets" / "g1_gait" / "motion.pt").write_text("provisioned")
    probe = tmp_path / "test_probe_env.py"
    probe.write_text(
        "import json, os, pathlib, pytest\n"
        "from vector_os_nano.schemas import paths\n\n"
        + ("@pytest.mark.sim\n" if tier == "sim" else "")
        + "def test_env():\n"
        "    keys = ('HOME', 'VECTOR_HOME', 'HF_HOME', 'HF_HUB_OFFLINE')\n"
        "    seen = {k: os.environ.get(k) for k in keys}\n"
        "    pt = paths.assets('g1_gait') / 'motion.pt'\n"
        "    seen['assets'] = pt.read_text() if pt.is_file() else None\n"
        "    pathlib.Path(os.environ['PROBE_OUT']).write_text(json.dumps(seen))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("GATE_HOME_SET", "HF_HOME", "VECTOR_HOME")}
    env.update({"VECTOR_HOME": str(store)} if state == "elsewhere" else {})
    r = subprocess.run(
        ["bash", str(_REPO / "scripts" / "gate.sh"), tier],
        env={**env, "HOME": str(real), "GATE_PER_FILE": "0", "GATE_PATHS": str(probe),
             "GATE_OUT": str(tmp_path / "out"), "PROBE_OUT": str(out)},
        cwd=_REPO, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    seen = json.loads(out.read_text())
    assert seen["HOME"] != str(real) and not Path(seen["HOME"]).exists()  # throwaway, removed after
    assert not seen["VECTOR_HOME"].startswith(str(real)) and not Path(seen["VECTOR_HOME"]).exists()
    assert seen["HF_HOME"] == str(real / ".cache" / "huggingface") and seen["HF_HUB_OFFLINE"] == "1"
    assert seen["assets"] == "provisioned"  # fetched robot assets are found, like the caches
    # the gate wrote nothing into the real home or the real state directory
    assert sorted(str(p.relative_to(real)) for p in real.rglob("*")) == [
        state, f"{state}/assets", f"{state}/assets/g1_gait", f"{state}/assets/g1_gait/motion.pt"]


def test_a_bare_pytest_never_writes_the_real_state_directory(tmp_path) -> None:
    """tests-inventory:tests-leave-no-home-guard: outside the gate, tests/conftest.py
    points HOME and VECTOR_HOME at throwaway directories before the package is imported,
    with the real HF cache and the provisioned assets linked; a test writing a session
    or starting a sim (which moves old scene memory in, C1a-V-D2) leaves the real home
    and the real state directory alone."""
    real, home = tmp_path / "real_state", tmp_path / "real_home"
    (real / "assets" / "g1_gait").mkdir(parents=True)
    (real / "assets" / "g1_gait" / "motion.pt").write_text("provisioned")
    (home / ".vector_os_nano" / "worlds" / "apartment").mkdir(parents=True)
    (home / ".vector_os_nano" / "worlds" / "apartment" / "scene_graph.yaml").write_text("rooms: precious\n")
    probe = tmp_path / "test_probe_state.py"
    probe.write_text(
        "import json, os, pathlib\n"
        "from vector_os_nano.schemas import paths\n"
        "from vector_os_nano.vcli.session import create_session\n\n"
        "def test_state():\n"
        "    s = create_session(); s.append_user('hi'); s.save()\n"
        "    seen = {'home': str(paths.home()), 'session': str(s._path), 'HOME': os.environ['HOME'],\n"
        "            'HF_HOME': os.environ['HF_HOME'], 'migrated': paths.migrate_old_memory(),\n"
        "            'assets': (paths.assets('g1_gait') / 'motion.pt').read_text()}\n"
        "    pathlib.Path(os.environ['PROBE_OUT']).write_text(json.dumps(seen))\n"
    )
    out = tmp_path / "seen.json"
    env = {k: v for k, v in os.environ.items() if k not in ("GATE_HOME_SET", "HF_HOME")}
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "tests.conftest", "-p", "no:cacheprovider",
         "-c", str(_REPO / "pyproject.toml"), "-q", "--rootdir", str(tmp_path), str(probe)],
        env={**env, "HOME": str(home), "VECTOR_HOME": str(real), "PROBE_OUT": str(out)},
        cwd=_REPO, capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    seen = json.loads(out.read_text())
    assert not seen["home"].startswith(str(real)) and seen["session"].startswith(seen["home"])
    assert seen["assets"] == "provisioned" and not Path(seen["home"]).exists()  # removed at exit
    assert seen["home"] == str(Path(seen["HOME"]) / ".vector") and not Path(seen["HOME"]).exists()
    assert seen["HF_HOME"] == str(home / ".cache" / "huggingface") and seen["migrated"] == ""
    assert sorted(str(p.relative_to(real)) for p in real.rglob("*")) == [
        "assets", "assets/g1_gait", "assets/g1_gait/motion.pt"]
    assert sorted(str(p.relative_to(home)) for p in home.rglob("*")) == [
        ".vector_os_nano", ".vector_os_nano/worlds", ".vector_os_nano/worlds/apartment",
        ".vector_os_nano/worlds/apartment/scene_graph.yaml"]


def test_a_pty_child_home_finds_the_provisioned_robot_assets(tmp_path, monkeypatch) -> None:
    import shutil

    from tests.harness import pty_cli

    (tmp_path / ".vector" / "assets").mkdir(parents=True)
    monkeypatch.setenv("VECTOR_HOME", str(tmp_path / ".vector"))
    env, home = pty_cli._child_env(live=False, home_prefix="c0_probe_")
    try:
        assert env["VECTOR_HOME"] == str(Path(home) / ".vector")  # its own state directory
        assert (Path(home) / ".vector" / "assets").resolve() == (tmp_path / ".vector" / "assets")
    finally:
        shutil.rmtree(home)
    assert (tmp_path / ".vector" / "assets").is_dir()  # removing the child home left it alone


def test_every_known_red_row_names_an_existing_test_file() -> None:
    """A merge resurrected 88 known_reds rows WP04 had deleted (with the files they
    pinned), and with them a strict xfail whose test WP04 had fixed. A row for a file
    that no longer exists is dead weight a merge can carry back unseen."""
    rows = [
        line.split("|", 1)[0].strip()
        for line in (_TESTS / "known_reds.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not [r for r in rows if not (_REPO / r.split("::", 1)[0]).is_file()]


def test_notice_cites_existing_files_and_every_bundled_license() -> None:
    """docs-packaging:notice-incorrect-incomplete — NOTICE cited a deleted bridge
    directory and said licences sat beside assets that had none."""
    text = (_REPO / "NOTICE").read_text(encoding="utf-8")
    cited = re.findall(r"vector_os_nano/[\w/.*-]+\w", text)
    assert cited and not [c for c in cited if not list(_REPO.glob(c))]
    licenses = subprocess.run(["git", "ls-files", "*/LICENSE"], cwd=_REPO, capture_output=True,
                              text=True, check=True).stdout.split()
    assert licenses and not [p for p in licenses if p.startswith("vector_os_nano/") and p.rsplit("/", 1)[0] not in text]


def test_the_dev_extra_declares_the_gates_pytest_plugin() -> None:
    """tests-inventory:dev-extra-missing-pytest-timeout — gate.sh passes --timeout, so
    on a fresh `pip install -e .[dev]` without pytest-timeout every file ran rc=4."""
    import tomllib

    dev = tomllib.loads((_REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]["dev"]
    names = {re.split(r"[<>=!~\[ ]", d, maxsplit=1)[0].lower() for d in dev}
    assert "--timeout=" in (_REPO / "scripts" / "gate.sh").read_text()
    assert "pytest-timeout" in names, dev


def test_the_ci_install_declares_what_the_hermetic_tier_imports() -> None:
    """R8a-V2: CI installs .[sim,go2,ik,mcp,dev]; front_object (the sim go2 grasp
    perception) needs cv2 and the MCP image resources need PIL, so a fresh CI venv
    without them ran 2 of 226 files red. The extras the workflow names must carry both."""
    import tomllib

    extras = tomllib.loads((_REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    ci = (_REPO / ".github" / "workflows" / "ci.yml").read_text()
    named = re.search(r'-e "\.\[([^\]]+)\]"', ci).group(1).split(",")
    names = {re.split(r"[<>=!~\[ ]", d, maxsplit=1)[0].lower() for x in named for d in extras[x]}
    assert {"opencv-python", "pillow"} <= names, (named, sorted(names))
    assert "opencv-python" in {d.split(">")[0] for d in extras["sim"]}   # front_object is a sim path
    assert "pillow" in {d.split(">")[0] for d in extras["mcp"]}          # mcp/resources encodes PNG

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Repo-wide pytest config — the R2a CI acceptance gate + suite hygiene.

R2a gate: any test that claims a product CAPABILITY (``@pytest.mark.capability``)
MUST also be verified THROUGH the real product (``@pytest.mark.cli_main``); an
offender is turned into a hard failure, so a capability claim cannot regress to a
bypass script.

Hygiene (WP01) — every test runs as if it were alone on a shared box:

- os.environ and the stub-prone sys.modules entries (rclpy*, *_msgs, tf2_ros,
  yaml, mujoco, vector_os_nano.*) are snapshotted and restored around every test;
- the real repo-root ``.env`` never reaches the process (load_dotenv is a no-op,
  LLM keys are dropped) unless ``VECTOR_LIVE_LLM=1``;
- ``live_llm`` tests skip unless ``VECTOR_LIVE_LLM=1`` (one opt-in convention);
- ``sim`` is auto-added to tests in files that build a MuJoCo sim, and the sim
  tier is gated by ONE session preflight (tests/harness/sim_preflight.py) that
  skips it with one loud reason instead of erroring test by test;
- ``rosm`` / ``pkill`` / ``killall`` on PATH are shims that FAIL the calling test
  (machine-wide kills take down other projects' stacks);
- a test that leaves a child process behind is failed and its process groups
  are killed (pytest is a child subreaper, so orphans are caught too);
- ROS defaults to an isolated domain (never 0, where other stacks live);
- tests/known_reds.txt: the committed, commented list of known reds.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import sysconfig
import tempfile
import time
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.harness.sim_cleanup import cmdline, descendants, reap, sim_cleanup  # noqa: E402,F401

_LIVE = os.environ.get("VECTOR_LIVE_LLM", "").strip().lower() in ("1", "true")


def _ros_shadows_venv(path: list[str], site: str) -> bool:
    """True iff an /opt/ros python dir precedes this interpreter's site-packages.

    ROS Jazzy ships its own NumPy-1 pinocchio; importing it ahead of the venv's
    cmeel pinocchio SEGFAULTS the whole process at collection.
    """
    ros = next((i for i, p in enumerate(path) if p.startswith("/opt/ros/")), None)
    own = next((i for i, p in enumerate(path) if p == site), None)
    return ros is not None and own is not None and ros < own


if _ros_shadows_venv(sys.path, sysconfig.get_paths()["purelib"]):
    pytest.exit(
        "ROS python paths shadow the venv (pinocchio would segfault). Run the suite "
        "through scripts/gate.sh, or put the venv site-packages first on PYTHONPATH.",
        returncode=4,
    )

# Never share a DDS domain with the other stacks on this box (they run on 0).
os.environ.setdefault("ROS_DOMAIN_ID", "87")

_LLM_KEYS = re.compile(r"^(DEEPSEEK_|OPENROUTER_|ANTHROPIC_|OPENAI_|VECTOR_LLM_|VECTOR_PROVIDER$)")
if not _LIVE:
    for _k in [k for k in os.environ if _LLM_KEYS.match(k)]:
        del os.environ[_k]

# ---------------------------------------------------------------------------
# Kill-command shims: a test that runs rosm / pkill / killall FAILS.
# ---------------------------------------------------------------------------

_SHIM_DIR = Path(tempfile.mkdtemp(prefix="vos_test_shims_"))
_SHIM_LOG = _SHIM_DIR / "calls.log"
for _name in ("rosm", "pkill", "killall"):
    _p = _SHIM_DIR / _name
    _p.write_text(f'#!/bin/sh\necho "{_name} $*" >> "{_SHIM_LOG}"\nexit 1\n')
    _p.chmod(0o755)
os.environ["PATH"] = f"{_SHIM_DIR}{os.pathsep}{os.environ.get('PATH', '')}"

# Orphaned grandchildren re-parent to pytest, so the leak check below sees them.
try:
    import ctypes

    ctypes.CDLL(None).prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
except Exception:  # noqa: BLE001 — non-Linux: direct children are still checked
    pass


# ---------------------------------------------------------------------------
# Per-test isolation: env + sys.modules snapshot/restore, .env, shims, leaks.
# ---------------------------------------------------------------------------

_WATCHED_MODULES = re.compile(r"^(rclpy|\w+_msgs|tf2_ros|yaml|mujoco|vector_os_nano)(\.|$)")


def _is_stub(mod: object) -> bool:
    return not isinstance(mod, types.ModuleType) or getattr(mod, "__spec__", None) is None


def _restore_modules(before: dict[str, object]) -> None:
    """Undo stub installs: put replaced entries back, drop stubs added by the test.

    A real module first imported by the test is kept (a legit lazy import) unless
    the test also tampered with a watched entry, in which case it may be bound to a
    stub and is dropped too.
    """
    now = {k: v for k, v in sys.modules.items() if _WATCHED_MODULES.match(k)}
    added = {k: v for k, v in now.items() if k not in before}
    tampered = any(before[k] is not v for k, v in now.items() if k in before) or any(
        _is_stub(v) for v in added.values()
    )
    for k, v in added.items():
        if tampered or _is_stub(v):
            del sys.modules[k]
    for k, v in before.items():
        if sys.modules.get(k) is not v:
            sys.modules[k] = v


@pytest.fixture(autouse=True)
def _hygiene(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    env = dict(os.environ)
    mods = {k: v for k, v in sys.modules.items() if _WATCHED_MODULES.match(k)}
    kids = descendants()
    if not _LIVE:
        try:
            import dotenv

            monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
        except ImportError:
            pass
    _SHIM_LOG.unlink(missing_ok=True)
    yield
    _restore_modules(mods)
    os.environ.clear()
    os.environ.update(env)
    problems: list[str] = []
    if _SHIM_LOG.exists():
        problems.append(
            "invoked a machine-wide kill command (" + _SHIM_LOG.read_text().strip()
            + "); kill only the process groups the test spawned "
            "(tests/harness/sim_cleanup.py)"
        )
    leaked: set[int] = set()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        leaked = descendants() - kids
        if not leaked:
            break
        time.sleep(0.1)
    if leaked:
        cmds = [cmdline(p) for p in sorted(leaked)]
        reap(leaked)
        problems.append(f"left {len(leaked)} child process(es) running: {cmds}")
    if problems:
        pytest.fail(f"{request.node.nodeid}: " + "; ".join(problems), pytrace=False)


# ---------------------------------------------------------------------------
# The sim tier: auto 'sim' marker + ONE session preflight.
# ---------------------------------------------------------------------------

# A file that constructs a MuJoCo model / sim (in-process or via a PTY child) is
# sim-tier. ponytail: per-file source regex, so a file's fake-only tests ride along
# into the sim tier; split such files if hermetic coverage matters.
_SIM_SOURCE = re.compile(
    r"MuJoCo(Go2|Arm|Piper|G1)\(|mujoco\.MjModel|MjModel\.from_xml|build_room_scene"
    r"|sim_go2=True|\bsim=True|\"--sim(-go2)?\"|run_repl_session|_init_agent\(|^import mujoco",
    re.M,
)
_sim_file_cache: dict[Path, bool] = {}


def _is_sim_file(path: Path) -> bool:
    if path not in _sim_file_cache:
        try:
            _sim_file_cache[path] = bool(_SIM_SOURCE.search(path.read_text(encoding="utf-8")))
        except OSError:
            _sim_file_cache[path] = False
    return _sim_file_cache[path]


@pytest.fixture(scope="session")
def sim_preflight() -> str:
    """'' when the sim tier can run here, else the ONE reason it cannot.

    Runs tests/harness/sim_preflight.py in a child (an EGL or MuJoCo crash must not
    take the session down): vendored convex_mpc imports, the go2 and go2_piper
    scenes compile from a temp copy of the tracked tree, an EGL Renderer opens, and
    the grounding-dino weights are in the HF cache.
    """
    try:
        r = subprocess.run(
            [sys.executable, "-m", "tests.harness.sim_preflight"],
            cwd=_REPO_ROOT, capture_output=True, text=True, timeout=300,
        )
        reason = r.stdout.strip() if r.returncode == 0 else (
            f"preflight crashed (rc={r.returncode}): {(r.stderr or r.stdout).strip()[-300:]}"
        )
    except subprocess.TimeoutExpired:
        reason = "preflight timed out after 300 s"
    if reason:
        sys.stderr.write(f"\n[sim_preflight] sim tier SKIPPED: {reason}\n")
    return reason


@pytest.fixture(autouse=True)
def _sim_gate(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("sim") is not None:
        reason = request.getfixturevalue("sim_preflight")
        if reason:
            pytest.skip(f"sim preflight: {reason}")


# ---------------------------------------------------------------------------
# Collection: known reds, live gating, auto sim, the R2a capability gate.
# ---------------------------------------------------------------------------

_KNOWN_REDS = Path(__file__).with_name("known_reds.txt")


def _known_reds() -> list[tuple[str, str, str]]:
    """``(nodeid_prefix, mode, reason)`` rows of tests/known_reds.txt."""
    rows = []
    for line in _KNOWN_REDS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        nodeid, mode, reason = (part.strip() for part in line.split("|", 2))
        if mode not in ("deselect", "xfail"):
            raise pytest.UsageError(f"known_reds.txt: bad mode {mode!r} in {line!r}")
        rows.append((nodeid, mode, reason))
    return rows


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    reds = _known_reds() if _KNOWN_REDS.exists() else []
    deselected: list[pytest.Item] = []
    kept: list[pytest.Item] = []
    for item in items:
        row = next((r for r in reds if item.nodeid.startswith(r[0])), None)
        if row is not None and row[1] == "deselect":
            deselected.append(item)
            continue
        if row is not None:
            item.add_marker(pytest.mark.xfail(reason=row[2], strict=False))
        if _is_sim_file(Path(str(item.path))) and item.get_closest_marker("sim") is None:
            item.add_marker(pytest.mark.sim)
        if item.get_closest_marker("live_llm") is not None and not _LIVE:
            item.add_marker(pytest.mark.skip(reason="live_llm: set VECTOR_LIVE_LLM=1 (billed opt-in)"))
        kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept

    # R2a: a `capability` test MUST also carry `cli_main` (fail-closed).
    offenders = sorted(
        item.nodeid for item in items
        if "capability" in item.keywords and "cli_main" not in item.keywords
    )
    if not offenders:
        return
    msg = (
        "CI GATE (R2a): the following @pytest.mark.capability tests are NOT "
        "verified through the real cli.main entrypoint (missing "
        "@pytest.mark.cli_main). A capability MUST be proven through the product, "
        "never a bypass script:\n  - " + "\n  - ".join(offenders)
    )

    def _gate_violation(*_a: object, **_k: object) -> None:
        pytest.fail(msg, pytrace=False)

    for item in items:
        if item.nodeid in offenders and isinstance(item, pytest.Function):
            item.obj = _gate_violation

"""Shared fixtures for the Go2 locomotion harness.

All harness tests require mujoco. convex_mpc is intentionally excluded.
Fixtures are scope="function" so each test gets a clean simulation state.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any

import pytest

# Ensure repo root is on sys.path so vector_os_nano is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Harness config path
_HARNESS_YAML = _REPO_ROOT / ".sdd-locomotion" / "harness.yaml"

# Skip entire harness if mujoco is unavailable
pytest.importorskip("mujoco", reason="mujoco not installed — skipping locomotion harness")


def load_harness_config() -> dict[str, Any]:
    """Load harness.yaml; return empty dict if file is missing."""
    try:
        import yaml  # type: ignore[import]
        with open(_HARNESS_YAML) as fh:
            return yaml.safe_load(fh) or {}
    except ImportError:
        # PyYAML not available — return defaults
        return {}
    except FileNotFoundError:
        return {}


@pytest.fixture(scope="session")
def harness_config() -> dict[str, Any]:
    """Session-scoped harness configuration loaded from harness.yaml."""
    return load_harness_config()


def _import_mujoco_go2():
    """Import MuJoCoGo2 by file path to avoid vector_os_nano/__init__.py cascade.

    The package __init__.py imports httpx and other optional dependencies that
    may not be installed in the harness environment. Loading the sim module
    directly avoids that cascade — MuJoCoGo2 only needs mujoco + numpy.
    """
    import importlib.util

    _MODULE_PATH = (
        _REPO_ROOT
        / "vector_os_nano" / "hardware" / "sim" / "mujoco_go2.py"
    )
    # core/types.py is required by mujoco_go2 (_update_odometry, _update_lidar)
    # Load it first so the relative import inside mujoco_go2.py works.
    _types_path = _REPO_ROOT / "vector_os_nano" / "core" / "types.py"
    types_spec = importlib.util.spec_from_file_location(
        "vector_os_nano.core.types", str(_types_path)
    )
    types_mod = importlib.util.module_from_spec(types_spec)  # type: ignore[arg-type]
    import sys as _sys
    _sys.modules.setdefault("vector_os_nano.core.types", types_mod)
    types_spec.loader.exec_module(types_mod)  # type: ignore[union-attr]

    spec = importlib.util.spec_from_file_location(
        "vector_os_nano.hardware.sim.mujoco_go2", str(_MODULE_PATH)
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    _sys.modules["vector_os_nano.hardware.sim.mujoco_go2"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.MuJoCoGo2


@pytest.fixture
def go2_flat():
    """Connect a headless MuJoCoGo2 on flat ground (room=False).

    Yields the connected instance; disconnects after test completes.
    """
    MuJoCoGo2 = _import_mujoco_go2()
    robot = MuJoCoGo2(gui=False, room=False)
    robot.connect()
    yield robot
    robot.disconnect()


@pytest.fixture
def go2_standing(go2_flat):
    """Connect and stand the robot before the test.

    Builds on go2_flat; calls stand(duration=2.0) so tests start from
    a stable standing posture.
    """
    go2_flat.stand(duration=2.0)
    return go2_flat

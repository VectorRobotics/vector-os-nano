# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Playground (INC6, PART B) — the REAL headless MuJoCoArm sim-oracle guard.

The fakes in ``test_playground_chain.py`` / ``test_playground_predicates.py``
only prove the predicates' *logic*; they cannot prove the fake oracle SHAPE
matches the real arm. This test instantiates the REAL ``MuJoCoArm(gui=False)``
headless on the bundled ``so101_mujoco.xml`` and asserts the sim-oracle contract
the playground predicates depend on:

- ``get_object_positions()`` returns the six known tabletop objects,
- ``arm_at_home()`` (the playground predicate) is True after the arm is driven to
  the home pose,
- the ground truth's objects are the tabletop world manifest's pickables.

POLLUTION SAFETY (critical):
This file is collected by the canonical ``tests/vcli tests/unit/vcli`` gate, so
it MUST NOT destabilise it. The known failure mode is MUJOCO_GL cross-test
pollution. We guard against it two ways:

1. The test only uses PHYSICS (mj_forward / mj_step) — never rendering — so it
   needs no GL backend. We pin ``MUJOCO_GL=disable`` (valid on macOS/Linux) for
   the duration and RESTORE the prior value in a finally block, so the env is
   left exactly as found for every other test in the run.
2. It is marked ``@pytest.mark.integration`` so it can be deselected
   (``-m "not integration"``) in environments where mujoco is unavailable, and
   ``pytest.importorskip`` skips cleanly when the wheel is missing.

Run it in isolation with:
    .venv-nano/bin/python -m pytest tests/vcli/test_playground_real_arm.py -q
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator

import pytest

# Skip the whole module cleanly when the mujoco wheel is absent.
pytest.importorskip("mujoco", reason="mujoco not installed")

from vector_os_nano.playground import PlaygroundWorld
from vector_os_nano.playground.catalog import TABLETOP
from vector_os_nano.skills.home import _DEFAULT_HOME_JOINTS


@contextmanager
def _mujoco_gl_disabled() -> Iterator[None]:
    """Pin MUJOCO_GL=disable for physics-only use, restoring the prior value.

    ``disable`` is a backend-neutral value accepted on every platform (the GL
    context is simply never created). The real arm test runs FK / mj_step only,
    so no render backend is needed. Restoring the previous value (or deleting the
    key when it was unset) keeps this test from polluting any other test's env.
    """
    sentinel = object()
    prev: Any = os.environ.get("MUJOCO_GL", sentinel)
    os.environ["MUJOCO_GL"] = "disable"
    try:
        yield
    finally:
        if prev is sentinel:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = prev


@pytest.fixture
def real_arm() -> Iterator[Any]:
    """A connected, headless real MuJoCoArm on the tabletop scene (pollution-safe)."""
    from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm

    with _mujoco_gl_disabled():
        arm = MuJoCoArm(gui=False)
        arm.connect()
        try:
            yield arm
        finally:
            arm.disconnect()


@pytest.mark.integration
class TestRealArmOracleContract:
    """The real headless MuJoCoArm satisfies the playground sim-oracle contract."""

    def test_get_object_positions_returns_six_known_objects(self, real_arm: Any) -> None:
        positions = real_arm.get_object_positions()
        # Every declared tabletop object is present in the real scene; this is the
        # contract the fake arms in the unit tests must mirror.
        assert set(TABLETOP.object_names) <= set(positions)
        assert len(TABLETOP.object_names) == 6
        # Each position is a 3-vector of floats.
        for name in TABLETOP.object_names:
            xyz = positions[name]
            assert len(xyz) == 3
            assert all(isinstance(float(c), float) for c in xyz)

    def test_arm_at_home_true_after_homing(self, real_arm: Any) -> None:
        # Drive the real arm to the home pose, then the playground predicate
        # (bound to an agent holding the real arm) must read True off the oracle.
        real_arm.move_joints(list(_DEFAULT_HOME_JOINTS), duration=3.0)

        from vector_os_nano.vcli.worlds.sim_oracles import rig_namespace

        assert rig_namespace(real_arm.runtime, "so101")["arm_at_home"]() is True

    def test_ground_truth_objects_are_the_world_manifest_pickables(self, real_arm: Any) -> None:
        import mujoco

        from vector_os_nano.sim.ground_truth import GroundTruth
        from vector_os_nano.vcli.worlds.world_manifest import load_world

        m = real_arm.runtime.model
        gt = GroundTruth(real_arm.runtime, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link"))
        names = set(gt.object_positions())
        # Exactly the scenario's known objects == the tabletop manifest's pickables.
        assert names == set(TABLETOP.object_names) == set(load_world("tabletop").pickables)

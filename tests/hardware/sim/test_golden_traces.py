# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Golden traces: the MuJoCo path moves onto the physics contract with ZERO behaviour change.

Captured in lockstep (VECTOR_SIM_LOCKSTEP=1, headless) on release/1.0 before S1 touched
anything, and replayed here bit for bit:

1. go2: connect, stand, walk(0.4, 0, 0) for 3 sim-s, zero command, wait until the brake
   has settled; (sim_time, q, qd) every 0.1 s.
2. go2_piper: the scripted pick of the green bottle on the pick table (ik_top_down ->
   move_joints -> close -> lift; since S3a through the generic sim/drivers Arm and
   Gripper, which replay MuJoCoPiper's bit for bit); the state at every sampled instant,
   the weld events,
   the final object pose and the holding_object / lifted verdicts.
3. Ground truth at about 50 instants of (1) and (2): every GroundTruth value the oracles
   read (base pose, object positions and speeds, contacts, clearance, welds, arm joints,
   segmentation pixels of the manifest camera).

States compare with ``np.array_equal`` and oracle values exactly. Regenerate (only for
a documented, justified change) with ``python -m tests.hardware.sim.test_golden_traces
<dir>`` under scripts/gate.sh's environment; the ``release_*`` arrays are kept.

Two ground-truth values changed by design when GroundTruth became the one generic class
(S1 step 4); the release/1.0 values are kept as ``release_gt_clearance`` /
``release_gt_contacts`` and :func:`test_ground_truth_moved_only_as_documented` bounds the
change:

* clearance is today's AABB rule over CANONICAL shape AABBs of the published state;
  the old one read the stepped data's geom poses, one physics step behind it. Every
  value moved by at most the object's own motion in that step (max 0.11 mm, carried
  bottle; 0.24 um resting); no 3 cm lift or resting verdict flipped.
* contacts are the world's contract sentence (a gap up to 1 mm is a contact, from a
  probe of the current state); the old one read the stepped data's contact list. One
  value changed: the green bottle touches the pick table after the approach move
  (t = 8.62 s), at a measured gap of 0.92 mm.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

pytestmark = pytest.mark.sim

GOLDEN = Path(__file__).with_name("golden")
_SEAT = (10.46, 3.0)  # the dog 42 cm behind the green bottle, head-on (test_piper_grasp)
_GREEN = "pickable_bottle_green"
_SAMPLE_S = 0.1


# -- ground truth, read through today's API ------------------------------------------




def _gt_row(gt: Any, objects: list[str], camera: tuple[str, int, int], arm: tuple[str, ...]) -> dict:
    """Every value the oracles read, at this instant."""
    pos = gt.object_positions()
    contacts = [sorted(gt.contacts(n)) for n in objects]
    return {
        "t": gt.sim_time,
        "base": np.array(gt.base_pose() or (np.nan,) * 4),
        "pos": np.array([pos[n] for n in objects]),
        "speed": np.array([gt.object_speed(n) for n in objects]),
        "clearance": np.array([gt.clearance(n) for n in objects]),
        "welded": np.array([gt.attached_to(n) is not None for n in objects]),
        "pixels": np.array([gt.visible_pixels(n, *camera) for n in objects]),
        "arm": np.array(gt.joint_positions(arm) if arm else []),
        "contacts": contacts,
    }


# -- the scenarios ---------------------------------------------------------------------


def _free_pose(rt: Any, body: str) -> np.ndarray:
    """x y z qw qx qy qz of a free body, from the published state."""
    sc = rt.world.scene
    return rt.snapshot().q[sc.free_q(sc.body(body))].copy()


def _camera(embodiment: str) -> tuple[str, int, int]:
    from vector_os_nano.embodiments import registry
    from vector_os_nano.vcli.worlds.sim_oracles import _camera as manifest_camera

    return manifest_camera(registry.load(embodiment))


class _Recorder:
    def __init__(self, rt: Any, embodiment: str, arm: tuple[str, ...] = ()) -> None:
        from vector_os_nano.sim.ground_truth import GroundTruth

        self.rt, self.gt, self.arm = rt, GroundTruth(rt), arm
        gt = self.gt
        self.objects = sorted(gt.bid)
        self.camera = _camera(embodiment)
        self.t: list[float] = []
        self.q: list[np.ndarray] = []
        self.qd: list[np.ndarray] = []
        self.rows: list[dict] = []

    def state(self) -> None:
        s = self.rt.snapshot()
        self.t.append(s.sim_time)
        self.q.append(s.q.copy())
        self.qd.append(s.qd.copy())

    def truth(self) -> None:
        self.rows.append(_gt_row(self.gt, self.objects, self.camera, self.arm))

    def sample(self, n: int, *, truth_every: int = 1) -> None:
        for i in range(n):
            assert self.rt.wait_sim(_SAMPLE_S)
            self.state()
            if i % truth_every == 0:
                self.truth()

    def arrays(self) -> dict[str, np.ndarray]:
        out = {"t": np.array(self.t), "q": np.array(self.q), "qd": np.array(self.qd),
               "objects": np.array(self.objects)}
        for k in ("t", "base", "pos", "speed", "clearance", "welded", "pixels", "arm"):
            out[f"gt_{k}"] = np.array([r[k] for r in self.rows])
        nb = int(self.rt.world.scene.model.nbody)
        touch = np.zeros((len(self.rows), len(self.objects), nb), dtype=bool)
        for i, r in enumerate(self.rows):
            for k, ids in enumerate(r["contacts"]):
                touch[i, k, ids] = True
        out["gt_contacts"] = touch
        return out


def run_go2_walk() -> dict[str, np.ndarray]:
    """Scenario 1 (+ its half of scenario 3)."""
    from tests.hardware.sim.dog import stand_up

    dog = stand_up("go2")  # lockstep: the runtime starts no thread
    rt, base = dog.rt, dog.base
    try:
        assert base.gait_backend == "convex_mpc"
        rec = _Recorder(rt, "go2")
        rec.state()
        rec.truth()
        base.set_velocity(0.4, 0.0, 0.0)
        rec.sample(30, truth_every=2)
        base.set_velocity(0.0, 0.0, 0.0)
        n = 0
        while not base.wait_settled(timeout=0.0):
            rec.sample(1, truth_every=2 if n % 2 else 1)
            n += 1
            assert n < 60, "the brake never settled"
        rec.sample(5, truth_every=2)
        return rec.arrays()
    finally:
        dog.close()


def run_go2_piper_pick() -> dict[str, np.ndarray]:
    """Scenario 2 (+ its half of scenario 3)."""
    from tests.hardware.sim.dog import stand_up
    from vector_os_nano.sim.drivers.arm import Arm
    from vector_os_nano.sim.drivers.gripper import Gripper
    from vector_os_nano.sim.session import respawn
    from vector_os_nano.vcli.worlds.sim_oracles import build_namespace
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    dog = stand_up("go2_piper")
    cfg, rt, base = dog.cfg, dog.rt, dog.base
    try:
        arm = Arm(cfg.arm, rt, wait_mount=base.wait_settled)
        grip = Gripper(cfg.arm.gripper, rt)
        respawn(rt, base, cfg, xy=_SEAT)
        assert rt.wait_sim(1.5)
        rec = _Recorder(rt, "go2_piper", tuple(cfg.arm.joints))
        gt = rec.gt
        rec.sample(5)
        g = gt.object_positions()[_GREEN]
        q_above = arm.ik_top_down((g[0], g[1], g[2] + 0.02))
        assert q_above is not None and arm.move_joints(q_above, duration=4.0)
        rec.state()
        rec.truth()
        q_at = arm.ik_top_down(tuple(g))
        assert q_at is not None and arm.move_joints(q_at, duration=2.0)
        rec.state()
        rec.truth()
        assert grip.close()
        rec.state()
        rec.truth()
        lift = arm.get_joint_positions()
        lift[1] = max(0.0, lift[1] - 0.5)
        assert arm.move_joints(lift, duration=2.0)
        rec.state()
        rec.truth()
        rec.sample(20)
        ns = build_namespace(gt, load_world(cfg.world), cfg)
        out = rec.arrays()
        events = gt.attach_events()
        out["weld_t"] = np.array([t for t, _n, _p in events])
        out["weld_body"] = np.array([n for _t, n, _p in events])
        out["weld_on"] = np.array([p == cfg.arm.gripper.weld_body for _t, _n, p in events])
        out["final_pose"] = _free_pose(rt, _GREEN)
        out["verdicts"] = np.array([ns["holding_object"](_GREEN), ns["lifted"](_GREEN),
                                    ns["holding_object"]()])
        return out
    finally:
        dog.close()


SCENARIOS = {"go2_walk": run_go2_walk, "go2_piper_pick": run_go2_piper_pick}


# -- the replay ------------------------------------------------------------------------


def _first_mismatch(name: str, want: np.ndarray, got: np.ndarray) -> str:
    if want.shape != got.shape:
        return f"{name}: shape {got.shape} != golden {want.shape}"
    bad = np.argwhere(~((want == got) | (want != want) & (got != got)))
    i = tuple(bad[0])
    return f"{name}: {len(bad)} values differ; first at {i}: {got[i]!r} != golden {want[i]!r}"


def _replayed(golden: np.lib.npyio.NpzFile) -> list[str]:
    return sorted(k for k in golden.files if not k.startswith("release_"))


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_golden_trace_replays_bit_for_bit(scenario: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTOR_SIM_LOCKSTEP", "1")
    golden = np.load(GOLDEN / f"{scenario}.npz")
    got = SCENARIOS[scenario]()
    assert sorted(got) == _replayed(golden)
    bad = [_first_mismatch(k, golden[k], got[k]) for k in _replayed(golden)
           if not np.array_equal(golden[k], got[k], equal_nan=golden[k].dtype.kind == "f")]
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_ground_truth_moved_only_as_documented(scenario: str) -> None:
    """The step-4 change of clearance and contacts against release/1.0 (module docstring)."""
    g = np.load(GOLDEN / f"{scenario}.npz")
    new, old = g["gt_clearance"], g["release_gt_clearance"]
    assert np.array_equal(np.isinf(new), np.isinf(old))
    fin = np.isfinite(old)
    step = 1.5 * g["gt_speed"] * 1e-3  # one 1 ms physics step of the object's own motion
    assert (np.abs(new - old)[fin] <= step[fin] + 1e-9).all()
    assert ((new >= 0.03) == (old >= 0.03)).all() and ((np.abs(new) < 5e-3) == (np.abs(old) < 5e-3)).all()
    moved = np.argwhere(g["gt_contacts"] != g["release_gt_contacts"])
    assert all(g["gt_contacts"][tuple(i)] for i in moved)  # only contacts gained, none lost
    assert all(g["gt_clearance"][i, k] < 1e-3 for i, k, _b in moved)  # each within the 1 mm sentence
    assert len(moved) == {"go2_walk": 0, "go2_piper_pick": 1}[scenario]


def write(out: Path) -> None:
    os.environ["VECTOR_SIM_LOCKSTEP"] = "1"
    out.mkdir(parents=True, exist_ok=True)
    for name, run in SCENARIOS.items():
        kept = np.load(GOLDEN / f"{name}.npz")
        np.savez_compressed(out / f"{name}.npz", **run(), **{k: kept[k] for k in kept.files if k.startswith("release_")})
        print(f"wrote {out / name}.npz")


if __name__ == "__main__":
    write(Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN)

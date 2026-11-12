# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""FakeGT — a hermetic GroundTruth stand-in over a test's stub devices.

The REAL oracles (``vcli/worlds/sim_oracles.build_namespace``) run on it, so a
spine / executor / producer test exercises the production predicates without
MuJoCo. Truth comes from the stubs the test controls (the real GroundTruth is
tested on the compiled scene in tests/unit/vcli/test_sim_oracles.py):

- base pose      <- ``base.get_position()`` / ``get_heading()``
- object poses   <- ``arm.get_object_positions()`` (or ``objects=`` given here)
- attachments    <- ``gripper.weld_is_active()``; else the object nearest the EE
                    (``arm.fk``) within 8 cm while ``gripper.is_holding()``; the
                    parent is the manifest gripper's weld body
- lifted         <- z above the lowest z the object has been seen at
- contacts       <- the fingers while welded, else *support* while resting
- arm joints     <- ``arm.get_joint_positions()``

``bind_fake_sim(agent, embodiment)`` registers it as the agent's live sim session,
so ``RobotWorld`` / ``PlaygroundWorld`` / the engine bind the oracles as in a sim.
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace
from typing import Any

_EE_RADIUS = 0.08


class FakeGT:
    """*embodiment*: the manifest whose gripper names the weld body and the fingers."""

    def __init__(self, agent: Any = None, *, objects: dict[str, list[float]] | None = None,
                 finger_contact: bool = True, support: str = "table", embodiment: str = "go2_piper") -> None:
        from vector_os_nano.embodiments import registry

        self.agent = agent
        self._objects = objects
        arm = registry.load(embodiment).arm
        grip = arm.gripper if arm is not None else None
        self._parent = grip.weld_body if grip is not None else ""
        self._ids: dict[str, int] = {f: 9001 + i for i, f in enumerate(grip.finger_bodies if grip else ())}
        self.robot_bodies: frozenset[int] = frozenset(self._ids.values())
        self.finger_contact = finger_contact
        self.support = support
        self.events: list[tuple[float, str, str | None]] = []
        self.sim_time = 0.0
        self._rest: dict[str, float] = {}
        self._last_welds: dict[str, bool] = {}
        self.object_positions()  # record the rest heights

    # -- devices ---------------------------------------------------------
    def _dev(self, name: str) -> Any:
        return getattr(self.agent, name, None) if self.agent is not None else None

    @property
    def has_base(self) -> bool:
        return self._dev("_base") is not None

    def base_pose(self):
        base = self._dev("_base")
        if base is None:
            return None
        p = base.get_position()
        return float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0, float(base.get_heading())

    def object_positions(self) -> dict[str, list[float]]:
        arm = self._dev("_arm")
        if self._objects is not None:
            objs = self._objects
        elif arm is not None and hasattr(arm, "get_object_positions"):
            objs = arm.get_object_positions()
        else:
            objs = {}
        out = {n: [float(c) for c in p] for n, p in objs.items()}
        for n, p in out.items():
            self._rest[n] = min(self._rest.get(n, math.inf), p[2])
        return out

    @property
    def bid(self) -> dict[str, int]:
        return {n: i + 1 for i, n in enumerate(sorted(self.object_positions()))}

    def body_id(self, name: str) -> int:
        return self._ids.setdefault(name, 1000 + len(self._ids))

    def _welds(self) -> dict[str, bool]:
        gripper, arm = self._dev("_gripper"), self._dev("_arm")
        objs = self.object_positions()
        welds = {n: False for n in objs}
        reader = getattr(gripper, "weld_is_active", None)
        if callable(reader):
            welds.update({str(k): bool(v) for k, v in reader().items()})
        elif gripper is not None and callable(getattr(gripper, "is_holding", None)) and gripper.is_holding():
            try:
                ee = arm.fk(arm.get_joint_positions())[0]
                near = min(objs, key=lambda n: math.dist(objs[n], ee))
                if math.dist(objs[near], ee) <= _EE_RADIUS:
                    welds[near] = True
            except Exception:  # noqa: BLE001 — no fk: nothing welded
                pass
        for n, on in welds.items():
            if self._last_welds.get(n, False) != on:
                self.events.append((self.sim_time, n, self._parent if on else None))
        self._last_welds = welds
        return welds

    def attachments(self) -> dict[str, str]:
        return {n: self._parent for n, on in self._welds().items() if on}

    def attached_to(self, name: str) -> str | None:
        return self.attachments().get(name)

    def attach_events(self, since: float = -math.inf) -> list[tuple[float, str, str | None]]:
        self._welds()
        return [e for e in self.events if e[0] > since]

    def contacts(self, name: str) -> set[int]:
        if self._welds().get(name):
            return set(self.robot_bodies) if self.finger_contact else set()
        return {self.body_id(self.support)} if self.clearance(name) < 0.005 else set()

    def clearance(self, name: str) -> float:
        z = self.object_positions()[name][2]
        return z - self._rest.get(name, z)

    def object_speed(self, name: str) -> float:
        return 0.0

    def joint_positions(self, joints: tuple[str, ...]) -> list[float]:
        return [float(v) for v in self._dev("_arm").get_joint_positions()]

    def visible_pixels(self, name: str, camera: str, width: int, height: int) -> int:
        return 0


def fake_namespace(agent: Any = None, embodiment: str = "go2_piper", **kwargs: Any) -> dict[str, Any]:
    """The real oracle namespace over a FakeGT of *agent*'s stub devices."""
    from vector_os_nano.embodiments import registry
    from vector_os_nano.vcli.worlds.sim_oracles import build_namespace
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    cfg = registry.load(embodiment)
    gt = kwargs.pop("gt", None) or FakeGT(agent, embodiment=embodiment)
    return build_namespace(gt, load_world(cfg.world), cfg, **kwargs)


def bind_fake_sim(agent: Any, embodiment: str = "so101", **gt_kwargs: Any) -> SimpleNamespace:
    """Register *agent* as the agent of a live (fake) sim session; returns it.

    ``unbind_fake_sim(session)`` (or the session's ``close()``) removes it.
    """
    from vector_os_nano.embodiments import registry
    from vector_os_nano.sim import session as sim_session
    from vector_os_nano.vcli.worlds.world_manifest import load_world

    gt = FakeGT(agent, embodiment=embodiment, **gt_kwargs)
    turn = [(-math.inf, -math.inf)]
    world_id = registry.load(embodiment).world
    s = SimpleNamespace(agent=agent, gt=gt, world=load_world(world_id) if world_id else None)
    s.oracles = lambda: fake_namespace(agent, embodiment, gt=gt, turn_start=lambda: turn[0],
                                       belief=lambda: getattr(agent, "_spatial_memory", None))
    s.mark_turn = lambda phase, goal="": turn.__setitem__(0, (gt.sim_time, time.time())) if phase == "start" else None
    s.close = s.stop = lambda: unbind_fake_sim(s)
    sim_session._LIVE.append(s)
    return s


def bind_real_sim(agent: Any, embodiment: str, runtime: Any) -> Any:
    """Register a SimSession with the REAL GroundTruth over *runtime* (the rig's own:
    no driver exposes one) for a test rig built without SimSession.start;
    ``unbind_fake_sim`` removes it."""
    from vector_os_nano.embodiments import registry
    from vector_os_nano.sim import session as sim_session

    s = sim_session.SimSession(sim_session.SimSpec(embodiment, gui=False), registry.load(embodiment))
    s.agent = agent
    s.runtime = runtime
    s._bind_ground_truth()
    sim_session._LIVE.append(s)
    return s


def unbind_fake_sim(s: SimpleNamespace) -> None:
    from vector_os_nano.sim import session as sim_session

    if s in sim_session._LIVE:
        sim_session._LIVE.remove(s)

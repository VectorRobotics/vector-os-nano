# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""A legged body as the session builds it (sim/session.legged_body), for the sim tests.

On flat ground the body stands at the world origin: the convex MPC brakes measurably worse
far from it (speed 0.2 sim-s after a stop from 0.4 m/s, 21 stop times each: mean 0.048 m/s
at (0, 0), 0.071 at (5, 5), 0.092 at (10, 3.5); /data/vos-analysis/scratch/vos-S4a-flake),
and the flat-ground bounds were measured at the origin.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FLAT = Path(__file__).resolve().parents[2] / "conformance" / "scenes" / "flat.xml"


@dataclass
class Dog:
    cfg: Any
    rt: Any
    base: Any
    lidar: Any

    def close(self) -> None:
        self.base.disconnect()
        self.rt.close()


def stand_up(embodiment: str = "go2", *, room: bool = True, start: bool = True) -> Dog:
    """The body standing in its world's room (or on flat ground), its runtime started
    (threaded unless VECTOR_SIM_LOCKSTEP=1) and its stand posture taken."""
    from vector_os_nano.embodiments import registry
    from vector_os_nano.sim.session import legged_body

    cfg = registry.load(embodiment)
    if room:
        from vector_os_nano.hardware.sim.scene_builder import room_scene
        from vector_os_nano.schemas.world import load_world

        scene = room_scene(cfg, load_world(cfg.world))
    else:  # at the world origin, as the flat-ground tests were calibrated (module docstring)
        from vector_os_nano.sim.scene.compose import compose

        cfg = dataclasses.replace(cfg, spawn=dataclasses.replace(cfg.spawn, xy=(0.0, 0.0)))
        scene = compose(FLAT, cfg, spawn=(0.0, 0.0, 0.0, cfg.spawn.heading), timestep=cfg.base.policy.timestep)
    rt, base, lidar = legged_body(cfg, scene)
    dog = Dog(cfg, rt, base, lidar)
    if start:
        rt.start()
        assert base.set_posture("stand")
    return dog

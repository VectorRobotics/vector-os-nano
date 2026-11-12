# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""BaseProtocol (interfaces/devices.py) is the charter's list, and the ONE generic base
implements it member by member: same kind (property / method), same parameters, same
defaults. No base is built here (the running one: tests/conformance/test_legged.py)."""
from __future__ import annotations

import inspect

import pytest

from vector_os_nano.interfaces import BaseProtocol
from vector_os_nano.sim.drivers.base import LeggedBase

CHARTER = {"connect", "disconnect", "stop", "walk", "set_velocity", "get_position", "get_heading",
           "get_velocity", "get_odometry", "get_lidar_scan", "get_sim_time", "wait_sim", "cmd_motion",
           "set_posture", "limits", "holonomic"}


def _drift(cls: type, name: str) -> str:
    """'' when *cls* implements the protocol member *name* as declared, else what differs."""
    want, got = inspect.getattr_static(BaseProtocol, name), inspect.getattr_static(cls, name, None)
    if got is None:
        return f"{cls.__name__} has no {name}"
    if isinstance(got, property) != isinstance(want, property):
        return f"{name}: property on one side only"
    if not isinstance(want, property) and inspect.signature(got) != inspect.signature(want):
        return f"{name}{inspect.signature(got)} != {name}{inspect.signature(want)}"
    return ""


def test_the_protocol_is_the_charter_list():
    assert set(BaseProtocol.__protocol_attrs__) == CHARTER


@pytest.mark.parametrize("name", sorted(CHARTER))
def test_the_generic_base_implements_each_member(name):
    assert _drift(LeggedBase, name) == ""


def test_a_drifted_base_is_caught():
    """Adversarial: a base whose walk lost its duration default, whose holonomic is a
    method, or that lacks a member, fails the member check."""
    class Drifted(LeggedBase):
        def walk(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0, duration: float = 2.0) -> bool:
            return False

        def holonomic(self) -> bool:  # type: ignore[override]
            return True

    assert "walk" in _drift(Drifted, "walk") and "property" in _drift(Drifted, "holonomic")
    assert "has no" in _drift(object, "set_posture")

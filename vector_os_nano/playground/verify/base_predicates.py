# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Re-export shim: go2 base sim-oracle predicates moved to the kernel.

The deterministic base verify predicates are now SINGLE-SOURCED in
``vector_os_nano.vcli.worlds.go2_sim_oracle`` so BOTH the playground go2 world and
the plain RobotWorld can consume them without the kernel importing the playground
(ADR-008 / kernel rule 2: the dependency edge is one-way, playground -> kernel).
This module stays only so existing imports keep resolving; it adds no logic.
"""

from __future__ import annotations

from vector_os_nano.vcli.worlds.go2_sim_oracle import (
    _AT_POSITION_TOL_M,
    _FACING_TOL_RAD,
    _angle_delta,
    _base_heading,
    _base_position,
    _get_base,
    _is_box,
    make_at_position,
    make_facing,
    make_geodesic_dist,
    make_rooms_producer,
    make_visited,
)

__all__ = [
    "_AT_POSITION_TOL_M",
    "_FACING_TOL_RAD",
    "_angle_delta",
    "_base_heading",
    "_base_position",
    "_get_base",
    "_is_box",
    "make_at_position",
    "make_facing",
    "make_geodesic_dist",
    "make_rooms_producer",
    "make_visited",
]

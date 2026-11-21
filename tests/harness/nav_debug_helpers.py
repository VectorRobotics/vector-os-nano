# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Shared helper for the go2_vnav_bridge source checks (L30, L41)."""
from __future__ import annotations

from importlib.resources import files

def read_bridge_source() -> str:
    folder = files('vector_os_nano.sim.nav.ros2')
    return '\n'.join(folder.joinpath(name+'.py').read_text()
                     for name in ('parameters', 'terrain', 'motion', 'bridge'))

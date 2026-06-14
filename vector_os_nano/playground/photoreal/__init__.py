# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Self-built lightweight co-sim: MuJoCo physics + Blender Cycles/OptiX photoreal
render (campaign #10, DQ-11). The repo venv never imports ``bpy`` — ``server.py``
runs under a standalone Blender (GPL, subprocess-isolated) and this package's
client talks to it over a localhost socket (the proven #9 bridge pattern)."""

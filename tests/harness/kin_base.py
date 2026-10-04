# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""A kinematic stand-in for an in-process base (no MuJoCo): the follower's test body."""
from __future__ import annotations

import math


class KinBase:
    """A unicycle (or holonomic) body integrated in sim time at 100 Hz.

    ``blocked(x, y)`` (optional) is an obstacle the planner does not know about:
    a step into it does not move the body (it pushes against it).
    """

    supports_holonomic = True

    def __init__(self, x=0.0, y=0.0, yaw=0.0, blocked=None, z_at=None):
        self.x, self.y, self.yaw, self.t = x, y, yaw, 0.0
        self.cmd = (0.0, 0.0, 0.0)
        self.blocked = blocked
        self.z_at = z_at or (lambda t: 0.3)
        self.trace: list[tuple[float, float]] = []
        self.stops = 0
        self.waits = 0

    def get_position(self):
        return [self.x, self.y, self.z_at(self.t)]

    def get_heading(self):
        return self.yaw

    def get_sim_time(self):
        return self.t

    def set_velocity(self, vx, vy, vyaw):
        self.cmd = (vx, vy, vyaw)

    def stop(self):
        self.cmd = (0.0, 0.0, 0.0)
        self.stops += 1

    def wait_sim(self, seconds, cancel=None):
        self.waits += 1
        dt = 0.01
        for _ in range(max(1, round(seconds / dt))):
            vx, vy, vyaw = self.cmd
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            nx, ny = self.x + (vx * c - vy * s) * dt, self.y + (vx * s + vy * c) * dt
            if self.blocked is None or not self.blocked(nx, ny, self.t):
                self.x, self.y = nx, ny
            self.yaw = math.atan2(math.sin(self.yaw + vyaw * dt), math.cos(self.yaw + vyaw * dt))
            self.t += dt
            self.trace.append((self.x, self.y))
        return True

    def walk(self, vx, vy, vyaw, duration):
        self.set_velocity(vx, vy, vyaw)
        self.wait_sim(duration)
        self.stop()
        return True

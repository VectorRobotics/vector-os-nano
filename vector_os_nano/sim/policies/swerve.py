# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Joint-torque swerve controller; base.joints pairs steering then rolling wheel.

Wheel rows give steering-axis (x, y), rolling radius and the wheel's lateral
offset from that axis, in metres. Steering is bounded; a half-turn with reversed
wheel direction avoids crossing its stops. No root force or pose writes.
"""
from pathlib import Path

import numpy as np

from vector_os_nano.schemas.robot import BaseSpec
from vector_os_nano.interfaces.policy import PolicyObs


class Policy:
    balances = True  # Own zero-command braking instead of holding rolling-wheel angles.

    def __init__(self, base: BaseSpec, params: dict, tau_limit: np.ndarray, folder: Path):
        self.tick_hz = base.policy.tick_hz
        self._p = params
        self._wheels = np.asarray(params['wheels'], dtype=float)
        self._limit = np.asarray(tau_limit, dtype=float)
        self._range = np.asarray(params['steering_range'], dtype=float)
        n = 2 * len(self._wheels)
        if len(base.joints) != n or self._limit.shape != (n,):
            raise ValueError('swerve needs one steering/rolling joint pair per wheel')
        if self._range.shape != (2,) or not np.isfinite(self._range).all() or np.diff(self._range)[0] < np.pi:
            raise ValueError('swerve steering_range must span at least pi radians')
        if not np.isfinite(self._limit).all() or np.any(self._limit <= 0):
            raise ValueError('swerve needs finite positive joint torque limits')
        for key in ('steering_kp','steering_kd','steering_speed','wheel_kp','wheel_ki','wheel_speed','accel','yaw_accel'):
            if not np.isfinite(params[key]) or params[key] <= 0:
                raise ValueError(f'swerve {key} must be finite and positive')
        self._command = np.zeros(3)
        self._steering = np.zeros(n//2)
        self._integral = np.zeros(n//2)
        self._time = None
        self._requested_motion = False
        self.braking = False

    def reset(self, obs: PolicyObs) -> None:
        self._command[:] = 0
        self._steering = obs.joint_pos[::2].copy()
        self._integral[:] = 0
        self._time = obs.time
        self._requested_motion = False
        self.braking = False

    def torques(self, obs: PolicyObs) -> np.ndarray:
        if (obs.joint_pos.shape != self._limit.shape or obs.joint_vel.shape != self._limit.shape
                or np.asarray(obs.command).shape != (3,)):
            raise ValueError('swerve observation shape does not match the joint layout')
        if not np.isfinite(np.r_[obs.time, obs.joint_pos, obs.joint_vel, obs.command,
                                  obs.base_lin_vel, obs.base_ang_vel]).all():
            raise ValueError('swerve observation must be finite')
        moving = bool(np.linalg.norm(obs.command) >= 1e-5)
        if self._requested_motion and not moving:
            self._integral[:] = 0  # A stop must not preserve the drive bias from pushing an obstruction.
        self._requested_motion = moving
        dt = 1/self.tick_hz if self._time is None else max(0., min(obs.time-self._time, 1/self.tick_hz))
        self._time = obs.time
        delta = np.asarray(obs.command) - self._command
        delta[:2] *= min(1., self._p['accel']*dt/max(np.linalg.norm(delta[:2]), 1e-12))
        delta[2] = np.clip(delta[2], -self._p['yaw_accel']*dt, self._p['yaw_accel']*dt)
        self._command += delta
        tau = np.zeros_like(self._limit)
        vx, vy, yaw = self._command
        for i, (x, y, radius, offset) in enumerate(self._wheels):
            steer, roll = 2*i, 2*i+1
            angle = obs.joint_pos[steer]
            forward = np.array([np.cos(angle), np.sin(angle)])
            contact = np.array([x,y]) + offset*np.array([-forward[1],forward[0]])
            velocity = np.array([vx-yaw*contact[1], vy+yaw*contact[0]])
            if np.linalg.norm(velocity) > 1e-5:
                heading = np.arctan2(velocity[1],velocity[0])
                turns = np.arange(np.ceil((self._range[0]-heading)/np.pi),
                                  np.floor((self._range[1]-heading)/np.pi)+1)
                choices = heading + np.pi*turns
                target = choices[np.argmin(np.abs(choices-angle))]
                self._steering[i] += np.clip(target-self._steering[i],
                                             -self._p['steering_speed']*dt, self._p['steering_speed']*dt)
            tau[steer] = self._p['steering_kp']*(self._steering[i]-angle) - self._p['steering_kd']*obs.joint_vel[steer]
            speed = (velocity@forward + offset*obs.joint_vel[steer])/radius
            speed = np.clip(speed, -self._p['wheel_speed'], self._p['wheel_speed'])
            error = speed-obs.joint_vel[roll]
            candidate = float(np.clip(self._integral[i] + self._p['wheel_ki']*error*dt,
                                      -self._limit[roll], self._limit[roll]))
            effort = self._p['wheel_kp']*error + candidate
            if abs(effort) <= self._limit[roll] or error*self._integral[i] < 0:
                self._integral[i] = candidate
            tau[roll] = self._p['wheel_kp']*error + self._integral[i]
        self.braking = bool(np.linalg.norm(obs.command) < 1e-5 and (
            np.linalg.norm(self._command) > .01 or np.linalg.norm(obs.base_lin_vel[:2]) > .03
            or abs(obs.base_ang_vel[2]) > .05))
        return np.clip(tau, -self._limit, self._limit)

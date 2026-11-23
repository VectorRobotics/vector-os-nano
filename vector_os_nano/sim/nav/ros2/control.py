# SPDX-License-Identifier: Apache-2.0
"""Execute the installed FAR/TARE → localPlanner → pathFollower stack.

ROS callbacks only cache commands. The skill's control-owning thread applies them,
so stop/cancel and the base's limits remain authoritative. No Python path follower.
"""
from __future__ import annotations

import math
import threading
import time

from geometry_msgs.msg import PointStamped, TwistStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Bool

from vector_os_nano.sim.nav.follow import NavResult, arrival_radius


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class NavStack(Node):
    def __init__(self, base, *, context):
        super().__init__('vector_nav_control', context=context)
        self.base = base
        self.mode = None
        self.closed = threading.Event()
        self._command = None
        self._path = None
        self._route = False
        self._started = 0.
        self._finished = False
        self.goal_pub = self.create_publisher(PointStamped, '/goal_point', 5)
        self.waypoint_pub = self.create_publisher(PointStamped, '/way_point', 5)
        self.start_pub = self.create_publisher(Bool, '/start_exploration', 5)
        self.applied_pub = self.create_publisher(TwistStamped, '/vector/navigation/applied_cmd', 5)
        self.create_subscription(PointStamped, '/far/way_point', lambda m: self._waypoint(m, 'far'), 5)
        self.create_subscription(PointStamped, '/tare/way_point', lambda m: self._waypoint(m, 'tare'), 5)
        self.create_subscription(TwistStamped, '/navigation_cmd_vel', self._velocity, 5)
        self.create_subscription(Path, '/path', self._local_path, 5)
        self.create_subscription(Bool, '/exploration_finish', self._finish, 5)

    def _waypoint(self, msg, mode):
        if (self.mode == mode and msg.header.frame_id == 'map'
                and seconds(msg.header.stamp) > self._started
                and all(math.isfinite(v) for v in (msg.point.x, msg.point.y, msg.point.z))):
            self.waypoint_pub.publish(msg)
            self._route = (seconds(msg.header.stamp), time.monotonic())

    def _velocity(self, msg):
        v = msg.twist
        values = (v.linear.x, v.linear.y, v.angular.z)
        if msg.header.frame_id != 'base_link' or not all(math.isfinite(x) for x in values):
            self._command = None
            return
        self._command = (values, seconds(msg.header.stamp), time.monotonic())

    def _local_path(self, msg):
        if (not self.mode or not self._route or msg.header.frame_id != 'base_link'
                or seconds(msg.header.stamp) <= self._started):
            return
        xy = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if not xy or not all(math.isfinite(v) for p in xy for v in p):
            self._path = None
            return
        x, y = self.base.get_position()[:2]
        yaw = self.base.get_heading()
        c, s = math.cos(yaw), math.sin(yaw)
        world_path = [(x+c*a-s*b, y+s*a+c*b) for a,b in xy]
        self._path = (world_path, seconds(msg.header.stamp), time.monotonic())

    def _finish(self, msg):
        if self.mode == 'tare' and msg.data:
            self._finished = True

    def _apply(self, values, command=None):
        if (command or self.base.set_velocity)(*values) is False:
            return False
        msg = TwistStamped()
        t = round(self.base.get_sim_time()*1e9)
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(t, 10**9)
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.angular.z = map(float, values)
        if not self.closed.is_set():
            self.applied_pub.publish(msg)
        return True

    def navigate(self, goal, *, nav, fallen, cancel, tol=None, speed=None, timeout=None, on_progress=None, command=None):
        return self._run(goal, nav=nav, fallen=fallen, cancel=cancel, tol=tol,
                         speed=speed, timeout=timeout, on_progress=on_progress, command=command)

    def explore(self, *, nav, fallen, cancel, timeout=120., on_progress=None, command=None):
        return self._run(None, nav=nav, fallen=fallen, cancel=cancel, timeout=timeout,
                         on_progress=on_progress, command=command)

    def _run(self, goal, *, nav, fallen, cancel, tol=None, speed=None, timeout=None, on_progress=None, command=None):
        base = self.base
        if goal is not None and (len(goal) != 2 or not all(math.isfinite(v) for v in goal)):
            raise ValueError('navigation goal must contain two finite coordinates')
        v = min(float(speed if speed is not None else nav.speed), nav.speed)
        if not math.isfinite(v) or v <= 0:
            raise ValueError('navigation speed must be finite and positive')
        tolerance = min(float(tol if tol is not None else nav.goal_tol or .1), nav.goal_tol or .1, arrival_radius())
        origin = tuple(base.get_position()[:2])
        budget = float(timeout if timeout is not None else 4*math.dist(origin, goal)/v+30)
        if not all(math.isfinite(x) and x > 0 for x in (v, tolerance, budget)):
            raise ValueError('navigation speed, tolerance and deadline must be finite and positive')
        limits = (min(v, base.limits[0]), min(v, base.limits[1]), min(nav.vyaw_max, base.limits[2]))
        start = self._started = base.get_sim_time()
        wall_start = time.monotonic()
        self._command = self._path = None
        self._route = self._finished = False
        self.mode = 'far' if goal is not None else 'tare'
        base._navigation = dict(planner=self.mode, controller='CMU pathFollower', map_source='live_lidar',
                                state='planning', goal=list(goal) if goal is not None else None, path=[])
        last, moved, sent, last_progress = origin, 0., False, -math.inf
        last_send = -math.inf
        last_motion, progress_at = origin, start
        reason, detail = 'timeout', ''
        try:
            while True:
                now, wall = base.get_sim_time(), time.monotonic()
                pos = tuple(base.get_position()[:2])
                moved += math.dist(last, pos); last = pos
                dist = math.dist(pos, goal) if goal is not None else 0.
                if cancel.is_set() or self.closed.is_set():
                    reason = 'aborted'; break
                if fallen():
                    reason = 'fell'; break
                if (goal is not None and dist <= tolerance) or (goal is None and self._finished):
                    reason = 'arrived'; break
                if now-start >= budget:
                    break
                pub = self.goal_pub if goal is not None else self.start_pub
                # FAR discards goals before its graph is initialized; retry until it answers.
                if not self._route and wall-last_send >= 1. and pub.get_subscription_count():
                    if goal is None:
                        pub.publish(Bool(data=True))
                    else:
                        msg = PointStamped()
                        ns = round(start*1e9)
                        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(ns, 10**9)
                        msg.header.frame_id = 'map'
                        msg.point.x, msg.point.y = map(float, goal)
                        msg.point.z = float(base.get_position()[2])
                        pub.publish(msg)
                    sent = True
                    last_send = wall
                cmd, path = self._command, self._path
                fresh = (sent and self._route and cmd is not None and path is not None
                         and all(0 <= now-item[1] <= .75 and wall-item[2] <= 2. for item in (cmd, path))
                         and 0 <= now-self._route[0] <= .75 and wall-self._route[1] <= 2.
                         and cmd[1] > start)
                values = tuple(max(-lim, min(lim, val)) for val,lim in zip(cmd[0], limits)) if fresh else (0.,0.,0.)
                if not self._apply(values, command):
                    reason = 'aborted'; break
                if fresh:
                    base._navigation = {**base._navigation, 'state': 'following', 'path': path[0]}
                else:
                    base._navigation = {**base._navigation, 'state': 'waiting_for_stack'}
                    last_fresh = min(cmd[2] if cmd else wall_start, path[2] if path else wall_start,
                                     self._route[1] if self._route else wall_start)
                    if wall-last_fresh > 20.:
                        detail = 'NAV STACK command/path stream unavailable'; break
                if math.dist(last_motion, pos) >= .05:
                    last_motion, progress_at = pos, now
                if now-progress_at > 20.:
                    reason, detail = 'stalled', 'NAV STACK made no position progress'; break
                if now-last_progress >= 2.:
                    base._navigation = {**base._navigation, 'remaining_m': dist if goal is not None else None,
                                        'elapsed_s': now-start}
                    if on_progress:
                        on_progress(dist, now-start)
                    last_progress = now
                if not base.wait_sim(.02, cancel):
                    reason = 'aborted' if cancel.is_set() else 'timeout'; break
        finally:
            self.mode = None
            self._apply((0.,0.,0.), command)
        pos = base.get_position()
        result = NavResult(reached=reason=='arrived', reason=reason, goal=list(goal) if goal is not None else None,
                           target=list(goal) if goal is not None else None, snapped=False, pos=pos,
                           dist=math.dist(pos[:2], goal) if goal is not None else 0., moved_m=moved,
                           net_m=math.dist(origin, pos[:2]), plan_m=None, elapsed_s=base.get_sim_time()-start,
                           recoveries=0, planner='far' if goal is not None else 'tare', detail=detail)
        base._navigation = {**base._navigation, 'state': reason, 'result': dict(result),
                            'remaining_m': result['dist'], 'elapsed_s': result['elapsed_s']}
        return result

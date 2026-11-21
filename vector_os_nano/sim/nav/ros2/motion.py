# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

"""Path tracking, obstacle checks and recovery for the external navigation bridge."""
import math
import time
import numpy as np
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped
from vector_os_nano.core.run_dir import run_path
from vector_os_nano.sim.nav.follow import sim_tick_dt
from .parameters import _PF_DT, _SENSOR_X, _SENSOR_Y, _nav


class NavigationMotion:
    def _exploration_finish_cb(self, msg) -> None:
        """TARE says exploration is complete — replay terrain, then stop.

        Sequence: terrain replay (seeds FAR V-Graph) → stop robot → notify explore skill.
        """
        if msg.data and not self._exploration_finished:
            self._exploration_finished = True
            self.get_logger().warn("TARE exploration complete — replaying terrain for FAR")
            # Trigger terrain replay so FAR gets complete V-Graph data
            try:
                with open(run_path("vector_terrain_replay"), "w") as f:
                    f.write("1")
            except OSError:
                pass
            # Stop robot after short delay for terrain replay to start
            self._go2.set_velocity(0.0, 0.0, 0.0)
            self._current_path = []
            # Signal explore skill
            try:
                with open(run_path("vector_explore_finished"), "w") as f:
                    f.write("1")
            except OSError:
                pass

    def _path_cb(self, msg: Path) -> None:
        """Store path for Python follower + log."""
        if not self._nav_enabled or self._exploration_finished:
            return

        odom = self._go2.get_odometry()
        heading = self._go2.get_heading()
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        sx = odom.x + cos_h * _SENSOR_X - sin_h * _SENSOR_Y
        sy = odom.y + sin_h * _SENSOR_X + cos_h * _SENSOR_Y

        new_path = []
        for p in msg.poses:
            lx, ly = p.pose.position.x, p.pose.position.y
            mx = sx + lx * cos_h - ly * sin_h
            my = sy + lx * sin_h + ly * cos_h
            new_path.append((mx, my))

        self._current_path = new_path
        self._path_time = time.time()
        # Don't reset _pf_point_id to 0 — the progressive lookahead in
        # _follow_path naturally finds the closest point. Resetting on every
        # path update (5Hz) prevents forward progress at low speeds.

        self._diag_path_count += 1
        now_mono = time.monotonic()
        if new_path and now_mono - getattr(self, '_last_path_log', 0) > 3.0:
            self._last_path_log = now_mono
            pe = new_path[-1]
            self.get_logger().info(
                f"Path: {len(new_path)} pts, "
                f"end=({pe[0]:.1f},{pe[1]:.1f}) "
                f"robot=({odom.x:.1f},{odom.y:.1f})"
            )

    def _check_front_obstacle(self) -> float:
        """Check for obstacles in front cone. Returns min distance (meters).

        Uses cached pointcloud (10 Hz). Checks 60-degree cone in front,
        within 1.0m range, above ground level.
        """
        if not self._cached_points:
            return float('inf')

        odom = self._go2.get_odometry()
        heading = self._go2.get_heading()
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        ground_z = odom.z - 0.28  # approximate ground level

        min_dist = float('inf')
        for x, y, z, _ in self._cached_points:
            # Skip ground points
            if z - ground_z < 0.10:
                continue
            # Skip high points (above obstacle relevance for Go2 at 0.45m standing)
            if z - ground_z > 0.8:
                continue
            # Transform to robot frame
            dx = x - odom.x
            dy = y - odom.y
            # Project onto heading direction
            forward = dx * cos_h + dy * sin_h
            lateral = -dx * sin_h + dy * cos_h

            # Front cone: forward > 0, |lateral| < forward * tan(30deg)
            if forward > 0.1 and forward < 1.0 and abs(lateral) < forward * 0.577:
                dist = math.sqrt(forward * forward + lateral * lateral)
                if dist < min_dist:
                    min_dist = dist

        return min_dist

    def _scan_surroundings(self) -> tuple[float, float, float, float]:
        """Scan cached pointcloud for nearby obstacles in 4 zones.

        Returns (front_dist, left_dist, right_dist, back_dist) in meters.
        Each is the min distance to an obstacle in that zone, or inf.
        Uses body-frame projection: front/back = ±30deg, left/right = 30-120deg.
        """
        if not self._cached_points:
            return (float('inf'), float('inf'), float('inf'), float('inf'))

        odom = self._go2.get_odometry()
        heading = self._go2.get_heading()
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        ground_z = odom.z - 0.28

        front_min = float('inf')
        left_min = float('inf')
        right_min = float('inf')
        back_min = float('inf')

        for x, y, z, _ in self._cached_points:
            dz = z - ground_z
            if dz < 0.10 or dz > 0.8:
                continue
            dx = x - odom.x
            dy = y - odom.y
            fwd = dx * cos_h + dy * sin_h
            lat = -dx * sin_h + dy * cos_h
            d = math.sqrt(fwd * fwd + lat * lat)
            if d > 1.5 or d < 0.05:
                continue

            angle = math.atan2(abs(lat), fwd)
            if fwd > 0 and angle < 0.52:  # front ±30deg
                front_min = min(front_min, d)
            elif fwd < 0 and angle > 2.62:  # back ±30deg (pi-0.52)
                back_min = min(back_min, d)
            elif lat > 0 and angle < 2.1:  # left 30-120deg
                left_min = min(left_min, d)
            elif lat < 0 and angle < 2.1:  # right 30-120deg
                right_min = min(right_min, d)

        return (front_min, left_min, right_min, back_min)

    def _sim_tick_dt(self) -> float:
        """Sim-time elapsed since the previous follower tick (seconds).

        The physics daemon often runs below wall speed (compute-bound: ~0.65x
        measured with GUI + RViz), while this follower ticks on a WALL-clock
        20 Hz timer. Velocity ramps integrated per wall tick then slew ~1/0.65x
        faster in the gait's own (sim) time than they were tuned for and
        destabilize the MPC gait ("飘/瘸腿"). Every ramp/accumulator in
        _follow_path therefore integrates against THIS dt. Falls back to the
        nominal tick when no sim clock is readable (real hardware, error), so
        the pre-fix behavior is the degraded mode, never an exception.
        """
        now_sim: float | None = None
        get_t = getattr(self._go2, "get_sim_time", None)
        if get_t is not None:
            try:
                now_sim = float(get_t())
            except Exception:
                now_sim = None
        last = self._pf_last_sim_t
        self._pf_last_sim_t = now_sim
        return sim_tick_dt(now_sim, last, _PF_DT)

    def _follow_path(self) -> None:
        """Omnidirectional quadruped path follower (20 Hz wall timer).

        Go2 can walk forward, backward, and sideways. This follower:
        1. ALWAYS prefers forward motion (vx > 0)
        2. Uses lateral velocity (vy) to track path when direction error is large
        3. Reverses ONLY when target is directly behind AND very close (dead end)
        4. Applies reactive wall avoidance overlay from cached pointcloud

        All ramps/accumulators integrate against sim-dt (see _sim_tick_dt):
        per-tick constants below are calibrated for a nominal _PF_DT tick and
        scaled by `tick` = actual sim-ticks' worth of time elapsed.
        """
        # Sample the sim clock BEFORE any early return so it stays fresh
        # across teleop/skill-gated ticks (no stale-dt jump on resume).
        dt_sim = self._sim_tick_dt()
        tick = dt_sim / _PF_DT   # nominal-ticks' worth of sim time elapsed
        decay = 0.9 ** tick      # per-tick 0.9 decel-to-stop decay, dt-corrected

        if time.time() < self._teleop_until:
            return
        # A walk / navigate skill holds the base's control: it ignores this thread's
        # set_velocity until the skill ends.

        # --- Wall escape mode: reactive, direction-aware ---
        # Re-scans surroundings EVERY tick to adapt to changing obstacles.
        # Lidar is mounted -20° tilt on head (0.3m forward) — rear coverage
        # is sparser, so back_d uses a conservative clearance threshold.
        now = time.time()
        if self._wall_escape_elapsed < self._wall_escape_total:
            self._wall_escape_elapsed += dt_sim  # sim-time maneuver progress
            front_d, left_d, right_d, back_d = self._scan_surroundings()
            # Lidar rear coverage is sparse due to -20° tilt — treat back_d
            # as less reliable. Use conservative threshold (0.40m) vs front (0.30m).
            back_clear = back_d > 0.40
            left_clear = left_d > 0.25
            right_clear = right_d > 0.25

            if self._wall_escape_elapsed < self._wall_escape_phase2_at:
                # Phase 1: try to reverse — but only if back is clear
                if back_clear:
                    tgt_vx, tgt_vy, tgt_yaw = -0.35, 0.0, 0.0
                else:
                    # Back blocked — turn in place toward most open side
                    if right_d > left_d:
                        tgt_vx, tgt_vy, tgt_yaw = 0.0, 0.0, -0.4
                    else:
                        tgt_vx, tgt_vy, tgt_yaw = 0.0, 0.0, 0.4
            else:
                # Phase 2: strafe toward open side + slight reverse
                if right_clear and right_d > left_d:
                    tgt_vy, tgt_yaw = 0.15, -0.35
                elif left_clear:
                    tgt_vy, tgt_yaw = -0.15, 0.35
                elif right_clear:
                    tgt_vy, tgt_yaw = 0.15, -0.35
                else:
                    tgt_vy = 0.0
                    tgt_yaw = 0.4 if left_d > right_d else -0.4
                tgt_vx = -0.10 if back_clear else 0.0

            # Moderate ramp — responsive but not violent (per-tick, dt-scaled)
            _A = 0.04 * tick
            if self._pf_speed < tgt_vx: self._pf_speed = min(tgt_vx, self._pf_speed + _A)
            else: self._pf_speed = max(tgt_vx, self._pf_speed - _A)
            if self._pf_lat < tgt_vy: self._pf_lat = min(tgt_vy, self._pf_lat + 0.03 * tick)
            else: self._pf_lat = max(tgt_vy, self._pf_lat - 0.03 * tick)
            if self._pf_yawrate < tgt_yaw: self._pf_yawrate = min(tgt_yaw, self._pf_yawrate + 0.06 * tick)
            else: self._pf_yawrate = max(tgt_yaw, self._pf_yawrate - 0.06 * tick)
            self._go2.set_velocity(self._pf_speed, self._pf_lat, self._pf_yawrate)
            self._last_cmd_time = now
            return

        # --- Wall contact detection (accumulate time when stuck against wall) ---
        front_d_check = self._check_front_obstacle()
        cur_speed = self._pf_speed if hasattr(self, '_pf_speed') else 0.0
        if front_d_check < 0.30 and cur_speed < 0.15:
            self._wall_contact_time += dt_sim  # sim-seconds pinned (was wall tick)
        else:
            self._wall_contact_time = 0.0

        if self._wall_contact_time > 0.4:
            front_d, left_d, right_d, back_d = self._scan_surroundings()
            self.get_logger().warn(
                f"Wall escape triggered: F={front_d:.2f} L={left_d:.2f} "
                f"R={right_d:.2f} B={back_d:.2f}"
            )
            all_tight = front_d < 0.35 and left_d < 0.30 and right_d < 0.30
            self._wall_escape_elapsed = 0.0
            if all_tight and back_d > 0.40:
                # Boxed in with only rear open — sustained reverse
                self._wall_escape_phase2_at = 3.0
                self._wall_escape_total = 3.0
            elif back_d > 0.40:
                # Front blocked, back clear — reverse then strafe
                self._wall_escape_phase2_at = 0.8
                self._wall_escape_total = 2.5
            else:
                # Front AND back blocked — turn in place to find opening
                self._wall_escape_phase2_at = 0.0  # skip reverse, straight to strafe/turn
                self._wall_escape_total = 2.0
            self._wall_contact_time = 0.0
            self._current_path = []
            self._stuck_count = 0

        if not hasattr(self, '_pf_speed'):
            self._pf_speed = 0.0       # current forward speed
            self._pf_lat = 0.0         # current lateral speed
            self._pf_yawrate = 0.0     # current yaw rate
            self._pf_point_id = 0      # path progress index
            self._pf_turning = False   # True = turn-in-place mode

        # --- Two-mode quadruped path follower ---
        # Mode 1 (TRACK): heading error < 60° → cos/sin omni-walk, full speed
        # Mode 2 (TURN):  heading error > 60° → stop, turn in place, then go
        # Hysteresis: TRACK→TURN at 60°, TURN→TRACK at 30° (prevents oscillation)
        _MAX_SPEED = 0.6               # m/s forward cruise (balance: speed vs stability)
        _MAX_LAT = 0.20                # m/s max lateral speed (indoor dodging)
        _MAX_YAW_RATE = 1.0            # rad/s max yaw rate
        _YAW_GAIN_TRACK = 4.0          # P-gain for yaw in tracking mode (gentle)
        _YAW_GAIN_TURN = 6.0           # P-gain for yaw in turn mode (snappy)
        _TRACK_THRE = 1.05             # rad (60°) — enter turn mode above this
        _TRACK_RESUME = 0.52           # rad (30°) — resume tracking below this (hysteresis)
        _LOOK_AHEAD = 0.8              # m lookahead
        _STOP_DIS = 0.2                # m — stop within this
        _SLOW_DWN_DIS = 1.0            # m — start decelerating
        _ACCEL = 0.03                  # m/s per nominal tick (sim-dt-scaled below)
        _PATH_TIMEOUT = 8.0            # seconds before path considered stale

        # Stop immediately if exploration is done
        if self._exploration_finished:
            self._pf_speed *= decay
            self._pf_lat *= decay
            self._pf_yawrate *= decay
            if abs(self._pf_speed) < 0.01: self._pf_speed = 0.0
            if abs(self._pf_lat) < 0.01: self._pf_lat = 0.0
            if abs(self._pf_yawrate) < 0.01: self._pf_yawrate = 0.0
            self._go2.set_velocity(self._pf_speed, self._pf_lat, self._pf_yawrate)
            return

        has_path = (self._current_path
                    and time.time() - self._path_time < _PATH_TIMEOUT)

        if not has_path:
            if not self._nav_enabled:
                # Decel to stop (smoothed)
                self._pf_speed *= decay
                self._pf_lat *= decay
                self._pf_yawrate *= decay
                if abs(self._pf_speed) < 0.01: self._pf_speed = 0.0
                if abs(self._pf_lat) < 0.01: self._pf_lat = 0.0
                if abs(self._pf_yawrate) < 0.01: self._pf_yawrate = 0.0
            else:
                front_dist = self._check_front_obstacle()
                if front_dist < 0.30:
                    tgt_vx, tgt_yaw = -0.20, 0.4  # back away + turn harder
                elif front_dist < 0.60:
                    tgt_vx, tgt_yaw = 0.0, 0.3    # stop, turn to find new path
                else:
                    tgt_vx, tgt_yaw = 0.05, 0.0   # gentle creep (was 0.15 — too aggressive)
                _A = 0.03 * tick
                if self._pf_speed < tgt_vx: self._pf_speed = min(tgt_vx, self._pf_speed + _A)
                else: self._pf_speed = max(tgt_vx, self._pf_speed - _A)
                if self._pf_yawrate < tgt_yaw: self._pf_yawrate = min(tgt_yaw, self._pf_yawrate + 0.03 * tick)
                else: self._pf_yawrate = max(tgt_yaw, self._pf_yawrate - 0.03 * tick)
                self._pf_lat *= decay  # decay lateral
            self._go2.set_velocity(self._pf_speed, self._pf_lat, self._pf_yawrate)
            self._last_cmd_time = time.time()
            self._pf_point_id = 0
            return

        odom = self._go2.get_odometry()
        rx, ry = odom.x, odom.y
        heading = self._go2.get_heading()
        path = self._current_path
        path_size = len(path)

        # --- Endpoint distance ---
        ex, ey = path[-1]
        end_dis = math.sqrt((ex - rx)**2 + (ey - ry)**2)

        # --- Progressive lookahead ---
        while self._pf_point_id < path_size - 1:
            px, py = path[self._pf_point_id]
            d = math.sqrt((px - rx)**2 + (py - ry)**2)
            if d < _LOOK_AHEAD:
                self._pf_point_id += 1
            else:
                break
        self._pf_point_id = min(self._pf_point_id, path_size - 1)

        # --- Direction to lookahead point ---
        tx, ty = path[self._pf_point_id]
        dx, dy = tx - rx, ty - ry
        path_dir = math.atan2(dy, dx)

        dir_diff = path_dir - heading
        while dir_diff > math.pi: dir_diff -= 2 * math.pi
        while dir_diff < -math.pi: dir_diff += 2 * math.pi

        abs_err = abs(dir_diff)

        # --- Space-aware speed: slow in tight spaces, fast in open ---
        front_d, left_d, right_d, _back_d = self._scan_surroundings()
        front_gap = front_d - 0.20   # lidar-to-front clearance (lidar on head, ~20cm from nose)
        left_gap = left_d - 0.19     # body side extent
        right_gap = right_d - 0.19

        # Speed based on FRONT gap only — side obstacles don't slow forward motion
        if front_gap > 0.5:
            space_speed = _MAX_SPEED
        elif front_gap > 0.1:
            space_speed = _MAX_SPEED * (front_gap / 0.5)
        else:
            space_speed = 0.10
        # Only crawl when squeezed on BOTH sides AND front tight
        min_side = min(left_gap, right_gap)
        if min_side < 0.05 and front_gap < 0.2:
            space_speed = min(space_speed, 0.10)

        # --- Path curvature: slow down BEFORE turns, not during ---
        # Look 3-5 points ahead on path. If direction changes significantly,
        # reduce speed now so the robot can make the turn without overshooting.
        curve_speed = _MAX_SPEED
        if path_size > 3:
            look_pts = min(self._pf_point_id + 6, path_size - 1)
            fx, fy = path[look_pts]
            future_dir = math.atan2(fy - ry, fx - rx)
            curve = future_dir - path_dir
            while curve > math.pi: curve -= 2 * math.pi
            while curve < -math.pi: curve += 2 * math.pi
            abs_curve = abs(curve)
            if abs_curve > 0.3:  # > 17° turn ahead
                # Scale: 0.3 rad → full speed, 1.5 rad → 30% speed
                curve_speed = _MAX_SPEED * max(0.3, 1.0 - abs_curve / 1.5)

        target_speed = min(space_speed, curve_speed)
        if end_dis < _SLOW_DWN_DIS:
            target_speed = min(target_speed, _MAX_SPEED * (end_dis / _SLOW_DWN_DIS))
        if end_dis < _STOP_DIS or path_size <= 1:
            target_speed = 0.0

        # --- Narrow passage detection ---
        # When both sides are tight but front is clear, robot is in a doorway
        # or corridor. Reduce yaw aggressiveness to prevent spinning.
        _in_narrow = left_gap < 0.20 and right_gap < 0.20 and front_gap > 0.15
        if _in_narrow:
            _eff_yaw_gain_turn = 3.0    # half normal — gentler turning in tight space
            _eff_track_thre = 1.30      # 75° — stay in TRACK mode longer near doors
            _eff_track_resume = 0.40    # 23° — need tighter alignment before resuming
        else:
            _eff_yaw_gain_turn = _YAW_GAIN_TURN   # 6.0
            _eff_track_thre = _TRACK_THRE          # 60°
            _eff_track_resume = _TRACK_RESUME      # 30°

        # --- Two-mode controller ---
        # Mode transition with hysteresis (prevents oscillation at boundary)
        if self._pf_turning:
            if abs_err < _eff_track_resume:
                self._pf_turning = False  # heading aligned → resume tracking
        else:
            if abs_err > _eff_track_thre:
                self._pf_turning = True   # heading way off → stop and turn

        if end_dis <= _STOP_DIS:
            # At goal — stop
            vx = 0.0
            vy = 0.0
            vyaw = _YAW_GAIN_TRACK * dir_diff

        elif self._pf_turning:
            # MODE 2: TURN — heading error large
            if abs_err > 2.1:  # >120° — target is behind
                vx = -0.15     # reverse toward target
                vy = 0.0
                vyaw = _eff_yaw_gain_turn * dir_diff * 0.5
            elif _in_narrow:
                # In doorway/corridor: keep moving forward while turning
                # to avoid spinning in place
                vx = 0.15
                vy = 0.0
                vyaw = _eff_yaw_gain_turn * dir_diff
            else:
                vx = 0.05      # minimal forward creep for gait engagement
                vy = 0.0
                vyaw = _eff_yaw_gain_turn * dir_diff

        else:
            # MODE 1: TRACKING — heading error small
            err_scale = max(0.4, math.cos(abs_err))
            track_speed = target_speed * err_scale
            vx = track_speed * math.cos(dir_diff)
            vy = -track_speed * math.sin(dir_diff)
            vy = max(-_MAX_LAT, min(_MAX_LAT, vy))
            vyaw = _YAW_GAIN_TRACK * dir_diff

        # --- Cylinder body safety (gaps already computed above) ---
        _COMFORT = 0.15      # desired gap between body surface and wall
        _DANGER = 0.05       # gap below this = imminent contact

        # Front: brake based on gap, not raw distance
        if front_gap < _COMFORT and vx > 0:
            if front_gap <= _DANGER:
                vx = 0.0  # body surface ~0.05m from wall — full stop
            else:
                # Scale: full speed at _COMFORT gap, zero at _DANGER gap
                vx *= (front_gap - _DANGER) / (_COMFORT - _DANGER)
            # Slow down when BOTH sides are tight (corridor/gap), but not
            # for single-side obstacles (table legs, door frames)
            if left_gap < _DANGER and right_gap < _DANGER:
                vx *= 0.3  # crawl when squeezed on BOTH sides

        # Sides: proportional push away from wall (never slam to max)
        # Scale push by (1 - forward_speed_ratio) so fast-moving robot
        # gets gentler lateral push (prevents tipping at speed).
        _speed_ratio = min(1.0, abs(self._pf_speed) / _MAX_SPEED)
        _push_scale = 0.15 * (1.0 - 0.5 * _speed_ratio)  # 0.15 at stop, 0.075 at max speed

        if left_gap < _COMFORT:
            push_strength = (_COMFORT - max(0, left_gap)) / _COMFORT
            vy = max(-_MAX_LAT, vy - push_strength * _push_scale)
            if vy > 0:
                vy = 0.0
        if right_gap < _COMFORT:
            push_strength = (_COMFORT - max(0, right_gap)) / _COMFORT
            vy = min(_MAX_LAT, vy + push_strength * _push_scale)
            if vy < 0:
                vy = 0.0

        # --- Smooth acceleration (prevents MPC gait destabilization) ---
        # Per-nominal-tick rates, integrated against sim time via `tick` so
        # the profile the gait sees matches the tuned design at any sim/wall
        # ratio (the un-scaled ramp at 0.65x sim speed was the "飘/瘸腿" bug).
        _ACCEL_FWD = _ACCEL * tick       # m/s this tick forward
        _ACCEL_LAT = 0.01 * tick    # m/s per tick lateral (0→0.15 takes 0.75s sim)
        _ACCEL_YAW_TRACK = 0.04 * tick   # rad/s per tick in tracking (gentle)
        _ACCEL_YAW_TURN = 0.08 * tick    # rad/s per tick in turn mode (faster)

        # Forward/reverse
        if self._pf_speed < vx:
            self._pf_speed = min(vx, self._pf_speed + _ACCEL_FWD)
        elif self._pf_speed > vx:
            self._pf_speed = max(vx, self._pf_speed - _ACCEL_FWD * 2)  # decel 2x faster

        # Lateral
        target_lat = float(np.clip(vy, -_MAX_LAT, _MAX_LAT))
        if self._pf_lat < target_lat:
            self._pf_lat = min(target_lat, self._pf_lat + _ACCEL_LAT)
        elif self._pf_lat > target_lat:
            self._pf_lat = max(target_lat, self._pf_lat - _ACCEL_LAT)

        # Yaw — scale max yaw rate inversely with forward speed (prevent centripetal tip)
        # Fast forward (0.8) → max yaw 0.5 rad/s. Stopped → max yaw 1.2 rad/s.
        _speed_frac = min(1.0, abs(self._pf_speed) / _MAX_SPEED)
        _dynamic_yaw_max = _MAX_YAW_RATE * (1.0 - 0.6 * _speed_frac)

        accel_yaw = _ACCEL_YAW_TURN if self._pf_turning else _ACCEL_YAW_TRACK
        target_yaw = float(np.clip(vyaw, -_dynamic_yaw_max, _dynamic_yaw_max))
        if self._pf_yawrate < target_yaw:
            self._pf_yawrate = min(target_yaw, self._pf_yawrate + accel_yaw)
        elif self._pf_yawrate > target_yaw:
            self._pf_yawrate = max(target_yaw, self._pf_yawrate - accel_yaw)

        # Clamp final speeds
        self._pf_speed = float(np.clip(self._pf_speed, -0.3, _MAX_SPEED))
        self._pf_lat = float(np.clip(self._pf_lat, -_MAX_LAT, _MAX_LAT))

        self._go2.set_velocity(self._pf_speed, self._pf_lat, self._pf_yawrate)
        self._last_cmd_time = time.time()

        if not hasattr(self, '_follow_count'):
            self._follow_count = 0
        self._follow_count += 1
        if self._follow_count % 20 == 0:
            mode = "TURN" if self._pf_turning else "TRACK"
            self.get_logger().info(
                f"PyPF[{mode}]: vx={self._pf_speed:.2f} vy={self._pf_lat:.2f} "
                f"yr={self._pf_yawrate:.2f} endD={end_dis:.1f} "
                f"err={math.degrees(dir_diff):.0f}° "
                f"spd={space_speed:.1f} gap=F{front_gap:.2f}/L{left_gap:.2f}/R{right_gap:.2f}"
            )

    def _log_diagnostics(self) -> None:
        """Log bridge data flow stats every 10s for TARE starvation debugging."""
        self.get_logger().info(
            f"Diag: odom={self._diag_odom_count} "
            f"scan={self._diag_scan_count} "
            f"path={self._diag_path_count} "
            f"nav={'ON' if self._nav_enabled else 'OFF'}"
        )

    def _safety_check(self) -> None:
        # 5s timeout — must be long enough for TARE to initialize and
        # start publishing waypoints after exploration begins.
        # Also skip if exploration background thread is actively running.
        if time.time() - self._last_cmd_time > 5.0:
            try:
                from vector_os_nano.skills.go2.explore import is_exploring
                if is_exploring():
                    return  # don't zero velocity during exploration startup
            except ImportError:
                pass
            self._go2.set_velocity(0.0, 0.0, 0.0)

    def _stuck_detector(self) -> None:
        """Detect when robot is stuck and take recovery action.

        Runs at 0.5 Hz. Two-stage recovery:
        1. After 4s no progress: send /reset_waypoint to TARE
        2. After 8s still stuck: back up 0.3m to escape tight spot
        """
        if not self._nav_enabled:
            self._stuck_count = 0
            self._stuck_pos = None
            self._stuck_wall_clock = 0.0
            return

        odom = self._go2.get_odometry()
        cur = (odom.x, odom.y)

        if self._stuck_pos is not None:
            dx = cur[0] - self._stuck_pos[0]
            dy = cur[1] - self._stuck_pos[1]
            moved = math.sqrt(dx * dx + dy * dy)

            if moved < 0.1:
                if self._stuck_count == 0:
                    # First stuck tick — record wall-clock start
                    self._stuck_wall_clock = time.time()
                self._stuck_count += 1

                # Wall-clock ceiling: abort navigation if stalled too long
                _stall_timeout: float = _nav("stall_timeout", 30.0)
                if self._stuck_wall_clock > 0.0 and (time.time() - self._stuck_wall_clock) > _stall_timeout:
                    self.get_logger().error(
                        "[NAV] Stalled for %.0fs, aborting navigation", _stall_timeout
                    )
                    self._go2.set_velocity(0.0, 0.0, 0.0)
                    try:
                        with open(run_path("vector_nav_stalled"), "w") as fh:
                            fh.write("1")
                    except OSError as exc:
                        self.get_logger().warn(f"[NAV] Could not write stall flag: {exc}")
                    # Reset stuck state so we don't fire again immediately
                    self._stuck_count = 0
                    self._stuck_pos = None
                    self._stuck_wall_clock = 0.0
                    return
            else:
                self._stuck_count = 0
                self._stuck_wall_clock = 0.0

            if self._stuck_count == 1:  # 2s — request TARE replan early
                msg = PointStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "map"
                msg.point.x = odom.x
                msg.point.y = odom.y
                msg.point.z = odom.z
                self._reset_waypoint_pub.publish(msg)
                self._stuck_history.append((odom.x, odom.y))
                self.get_logger().warn(
                    f"Stuck 4s at ({odom.x:.1f},{odom.y:.1f}) — "
                    f"sent /reset_waypoint to TARE"
                )
            elif self._stuck_count == 2:  # 4s — direction-aware escape
                now = time.time()
                front_d, left_d, right_d, back_d = self._scan_surroundings()
                _escape_dur = _nav("wall_escape_duration", 2.5)
                self.get_logger().warn(
                    f"Stuck 8s: F={front_d:.2f} L={left_d:.2f} "
                    f"R={right_d:.2f} B={back_d:.2f}"
                )
                all_tight = front_d < 0.35 and left_d < 0.30 and right_d < 0.30
                # Conservative rear threshold — lidar rear coverage sparse at -20° tilt
                back_clear = back_d > 0.40
                self._wall_escape_elapsed = 0.0
                if all_tight and back_clear:
                    # Boxed in — sustained reverse
                    self._wall_escape_phase2_at = 4.0
                    self._wall_escape_total = 4.0
                elif back_clear:
                    # Front blocked, back clear — reverse then strafe
                    self._wall_escape_phase2_at = 0.8
                    self._wall_escape_total = _escape_dur
                else:
                    # Front AND back blocked — turn in place
                    self._wall_escape_phase2_at = 0.0
                    self._wall_escape_total = _escape_dur
                self._current_path = []
                self._stuck_count = 0
                self._stuck_wall_clock = 0.0
                self._stuck_history.append((odom.x, odom.y))

                # Stuck loop: 3+ escapes at same spot → longer escape
                if len(self._stuck_history) >= 3:
                    recent = self._stuck_history[-3:]
                    cx = sum(p[0] for p in recent) / 3
                    cy = sum(p[1] for p in recent) / 3
                    if all(
                        math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) < 0.5
                        for p in recent
                    ):
                        self.get_logger().warn("Stuck loop — extended escape 4s")
                        self._wall_escape_elapsed = 0.0
                        self._wall_escape_phase2_at = 2.0
                        self._wall_escape_total = 4.0
                        self._stuck_history.clear()

        self._stuck_pos = cur

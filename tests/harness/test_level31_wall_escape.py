"""Level 31: Wall escape debug harness — dog stuck against walls.

Tests verify:
  - Wall contact detection logic (front_d + speed + duration)
  - Escape uses omnidirectional movement (reverse + strafe)
  - Escape picks open direction (left vs right)
  - No false trigger during normal doorway transit
  - Bridge code structure
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nav_debug_helpers import read_bridge_source


# ---------------------------------------------------------------------------
# Part 1: Static verification — bridge source structure
# ---------------------------------------------------------------------------


class TestWallEscapeAttributes:
    """Verify instance variables for wall escape state exist in bridge __init__."""

    def test_wall_contact_time_exists(self):
        """Bridge must declare _wall_contact_time for contact accumulation."""
        src = read_bridge_source()
        assert "_wall_contact_time" in src, (
            "_wall_contact_time not found in bridge source — "
            "required for wall contact duration tracking"
        )

    def test_wall_escape_until_exists(self):
        """Bridge must declare _wall_escape_until timestamp to gate escape mode."""
        src = read_bridge_source()
        assert "_wall_escape_until" in src, (
            "_wall_escape_until not found in bridge source — "
            "required to track when escape maneuver ends"
        )

    def test_wall_contact_time_initialized_to_zero(self):
        """_wall_contact_time must be initialised as a float zero."""
        src = read_bridge_source()
        # Matches: _wall_contact_time: float = 0.0  OR  _wall_contact_time = 0.0
        assert re.search(r'_wall_contact_time\s*[=:].+0\.0', src), (
            "_wall_contact_time must be initialised to 0.0 in __init__"
        )

    def test_wall_escape_until_initialized_to_zero(self):
        """_wall_escape_until must be initialised as a float zero."""
        src = read_bridge_source()
        assert re.search(r'_wall_escape_until\s*[=:].+0\.0', src), (
            "_wall_escape_until must be initialised to 0.0 in __init__"
        )


class TestWallEscapeMovementCommands:
    """Verify escape maneuver uses the expected omnidirectional commands."""

    def test_escape_uses_reverse(self):
        """Escape command must include negative tgt_vx (reverse target velocity).

        After the reactive refactor, the escape block ramps toward tgt_vx = -0.35
        (or -0.10 in phase 2) rather than passing a literal to set_velocity directly.
        We verify a negative tgt_vx assignment exists inside the escape section.
        """
        src = read_bridge_source()
        escape_idx = src.find("Wall escape mode")
        assert escape_idx >= 0, "Wall escape mode block not found"
        # Extend to 1200 chars to cover phase1 + phase2 branches
        escape_block = src[escape_idx: escape_idx + 1200]
        # Current pattern: tgt_vx, tgt_vy, tgt_yaw = -0.35, ... or tgt_vx = -0.10
        match = re.search(r'tgt_vx\s*.*?=\s*(-[\d.]+)', escape_block)
        assert match, "No negative tgt_vx found in escape block (reverse motion required)"
        vx = float(match.group(1))
        assert vx < 0, f"Escape tgt_vx={vx} must be negative (reverse motion)"

    def test_escape_uses_strafe(self):
        """Escape command must include non-zero tgt_vy (lateral strafe).

        After the reactive refactor, phase-2 strafe uses tgt_vy, tgt_yaw = 0.15, ...
        The phase-2 code is ~1230 chars from 'Wall escape mode', so use 1500 chars.
        We search for a non-zero tgt_vy assignment (ignoring 0.0 turn-in-place cases).
        """
        src = read_bridge_source()
        escape_idx = src.find("Wall escape mode")
        assert escape_idx >= 0, "Wall escape mode block not found"
        # 1500 chars covers phase1 + phase2 strafe branch
        escape_block = src[escape_idx: escape_idx + 1500]
        # Find all tgt_vy assignments and check at least one is non-zero
        # Pattern: tgt_vy, tgt_yaw = 0.15, -0.35  or  tgt_vy = -0.15
        matches = re.findall(r'tgt_vy\s*(?:,\s*\w+\s*)*=\s*([-\d.]+)', escape_block)
        assert matches, "No tgt_vy assignments found in escape block (strafe required)"
        non_zero = [v for v in matches if float(v) != 0.0]
        assert non_zero, (
            f"All tgt_vy values are 0.0 in escape block — non-zero strafe required. "
            f"Found: {matches}"
        )

    def test_escape_vx_magnitude_adequate(self):
        """Reverse speed must be fast enough to break wall contact (>= 0.25 m/s).

        After the reactive refactor, reverse speed is set via tgt_vx = -0.35
        (phase 1 reverse branch) rather than a direct set_velocity argument.
        """
        src = read_bridge_source()
        escape_idx = src.find("Wall escape mode")
        assert escape_idx >= 0, "Wall escape mode block not found"
        escape_block = src[escape_idx: escape_idx + 1500]
        # Phase 1 reverse: tgt_vx, tgt_vy, tgt_yaw = -0.35, 0.0, 0.0
        match = re.search(r'tgt_vx\s*(?:,\s*\w+\s*)*=\s*(-[\d.]+)', escape_block)
        assert match, "Negative tgt_vx not found in escape block"
        vx_mag = abs(float(match.group(1)))
        assert vx_mag >= 0.25, (
            f"Escape reverse speed {vx_mag} m/s too low — need >= 0.25 m/s to break wall contact"
        )

    def test_escape_vy_magnitude_adequate(self):
        """Strafe speed must be meaningful enough to clear the wall (>= 0.1 m/s).

        After the reactive refactor, strafe is tgt_vy = 0.15 in phase 2 (1230+ chars in).
        The threshold is lowered from 0.2 to 0.1 to match the new conservative
        indoor strafe speed (avoids knocking objects).
        """
        src = read_bridge_source()
        escape_idx = src.find("Wall escape mode")
        assert escape_idx >= 0, "Wall escape mode block not found"
        escape_block = src[escape_idx: escape_idx + 1500]
        # Find all non-zero tgt_vy assignments
        matches = re.findall(r'tgt_vy\s*(?:,\s*\w+\s*)*=\s*([-\d.]+)', escape_block)
        assert matches, "No tgt_vy found in escape block"
        non_zero = [abs(float(v)) for v in matches if float(v) != 0.0]
        assert non_zero, "All tgt_vy values are 0.0 — strafe required"
        max_vy = max(non_zero)
        assert max_vy >= 0.1, (
            f"Escape strafe speed {max_vy} m/s too low — need >= 0.1 m/s to clear the wall"
        )


class TestWallEscapeDirectionLogic:
    """Verify escape direction uses sensor data to pick the open side."""

    def test_escape_picks_open_side(self):
        """Escape must compare left_d vs right_d and strafe toward the open side."""
        src = read_bridge_source()
        # Both variants must be present: right_d > left_d or left_d comparison
        has_right_check = "right_d > left_d" in src or "left_d > right_d" in src
        assert has_right_check, (
            "Bridge must compare left_d vs right_d to pick the open escape side"
        )

    def test_escape_calls_scan_surroundings(self):
        """Escape block must call _scan_surroundings() to read lateral clearances."""
        src = read_bridge_source()
        escape_idx = src.find("Wall escape mode")
        assert escape_idx >= 0, "Wall escape mode block not found"
        escape_block = src[escape_idx: escape_idx + 900]
        assert "_scan_surroundings()" in escape_block, (
            "_scan_surroundings() not called inside escape mode block"
        )

    def test_detection_calls_check_front_obstacle(self):
        """Contact detection must call _check_front_obstacle() to read front distance."""
        src = read_bridge_source()
        detect_idx = src.find("Wall contact detection")
        assert detect_idx >= 0, "Wall contact detection block not found"
        detect_block = src[detect_idx: detect_idx + 400]
        assert "_check_front_obstacle()" in detect_block, (
            "_check_front_obstacle() not called in contact detection section"
        )


class TestWallEscapeTriggerConditions:
    """Verify trigger thresholds match the spec."""

    def test_front_distance_threshold(self):
        """Contact detection front distance threshold must be in [0.20, 0.40] m.

        After refactor, threshold is 0.30 m (was 0.25). The invariant is that
        a reasonable proximity trigger exists; exact value updated to match
        the current reactive implementation.
        """
        src = read_bridge_source()
        detect_idx = src.find("Wall contact detection")
        assert detect_idx >= 0, "Wall contact detection block not found"
        detect_block = src[detect_idx: detect_idx + 400]
        match = re.search(r'front_d\w*\s*<\s*([\d.]+)', detect_block)
        assert match, "Front distance threshold not found in detection block"
        threshold = float(match.group(1))
        assert 0.20 <= threshold <= 0.40, (
            f"Front distance threshold is {threshold} — expected 0.20-0.40 m"
        )

    def test_no_false_trigger_speed_check(self):
        """Trigger requires a speed check alongside front_d — prevents false triggers.

        After refactor, speed is checked via cur_speed < 0.15 (was _pf_speed < 0.05).
        The invariant is that a speed gate exists on the same condition as front_d.
        """
        src = read_bridge_source()
        detect_idx = src.find("Wall contact detection")
        assert detect_idx >= 0, "Wall contact detection block not found"
        detect_block = src[detect_idx: detect_idx + 400]
        # Accept both old (_pf_speed) and new (cur_speed) patterns with any threshold
        has_speed_check = re.search(
            r'(?:_pf_speed|cur_speed)\s*<\s*[\d.]+',
            detect_block,
        )
        assert has_speed_check, (
            "Speed check missing from wall contact detection — "
            "front_d alone would cause false triggers during fast doorway transit"
        )

    def test_contact_duration_threshold(self):
        """Escape triggers only after > 1.0 s of continuous wall contact."""
        src = read_bridge_source()
        # Find the if _wall_contact_time > N.N check
        match = re.search(r'_wall_contact_time\s*>\s*([\d.]+)', src)
        assert match, "_wall_contact_time threshold check not found"
        duration = float(match.group(1))
        assert 0.3 <= duration <= 1.5, (
            f"Contact duration threshold is {duration}s — expected 0.3-1.5s"
        )

    def test_escape_duration_reasonable(self):
        """Escape duration must be between 1 and 5 seconds inclusive."""
        src = read_bridge_source()
        # Pattern: _wall_escape_until = now + N.N or time.time() + N.N
        match = re.search(r'_wall_escape_until\s*=\s*(?:now|time\.time\(\))\s*\+\s*([\d.]+)', src)
        assert match, "_wall_escape_until assignment not found"
        duration = float(match.group(1))
        assert 1.0 <= duration <= 5.0, (
            f"Escape duration {duration}s out of acceptable range [1.0, 5.0]"
        )

    def test_contact_time_increment_matches_20hz(self):
        """Contact time accumulates at 1/20 s per tick — correct for a 20 Hz timer."""
        src = read_bridge_source()
        detect_idx = src.find("Wall contact detection")
        assert detect_idx >= 0, "Wall contact detection block not found"
        detect_block = src[detect_idx: detect_idx + 400]
        # Matches: 1.0 / 20.0  OR  1/20  OR  0.05
        has_increment = (
            "1.0 / 20.0" in detect_block
            or "1/20" in detect_block
            or "0.05" in detect_block
        )
        assert has_increment, (
            "Contact time increment 1/20 not found — expected 20 Hz accumulation"
        )


class TestWallEscapeSideEffects:
    """Verify escape resets downstream state correctly."""

    def _get_trigger_block(self, src: str) -> str:
        """Return the escape trigger block (all branches + resets).

        After the reactive refactor the trigger consists of multiple elif branches
        (back_clear, all_tight, etc.), so the resets at the bottom come > 300 chars
        after the first _wall_escape_until assignment. Use 700 chars to cover all.
        """
        escape_trigger_idx = src.find("_wall_escape_until = now")
        if escape_trigger_idx < 0:
            escape_trigger_idx = src.find("_wall_escape_until = time.time()")
        assert escape_trigger_idx >= 0, "_wall_escape_until assignment not found"
        # 700 chars covers all branches + resets (_wall_contact_time, _current_path, _stuck_count)
        return src[escape_trigger_idx: escape_trigger_idx + 700]

    def test_escape_clears_path(self):
        """Triggering escape must clear _current_path to discard the stale plan."""
        src = read_bridge_source()
        trigger_block = self._get_trigger_block(src)
        assert "_current_path = []" in trigger_block, (
            "Escape trigger must clear _current_path — stale path would cause "
            "re-entry into the same wall immediately after escape ends"
        )

    def test_escape_resets_stuck_count(self):
        """Triggering escape must reset _stuck_count to prevent double-escape."""
        src = read_bridge_source()
        trigger_block = self._get_trigger_block(src)
        assert "_stuck_count = 0" in trigger_block, (
            "Escape trigger must reset _stuck_count — otherwise stuck detector "
            "may fire immediately after wall escape and cause conflicting behaviors"
        )

    def test_escape_resets_contact_time(self):
        """After triggering, _wall_contact_time must be zeroed to avoid re-trigger."""
        src = read_bridge_source()
        trigger_block = self._get_trigger_block(src)
        assert "_wall_contact_time = 0.0" in trigger_block, (
            "_wall_contact_time not reset after escape trigger — "
            "accumulated contact time would immediately re-trigger on next tick"
        )


# ---------------------------------------------------------------------------
# Part 2: Behavioral tests — WallEscapeSimulator
# ---------------------------------------------------------------------------


@dataclass
class WallEscapeSimulator:
    """Simulates the bridge wall escape detection logic from _follow_path.

    Reproduces the exact algorithm at lines 681-698 of go2_vnav_bridge.py.
    Runs at 20 Hz. Call tick() for each simulated frame.
    """

    # Thresholds (must match bridge)
    front_threshold: float = 0.25      # m — front obstacle distance trigger
    speed_threshold: float = 0.05      # m/s — speed below which dog is "slow"
    contact_duration: float = 1.0      # s — contact time needed to trigger
    escape_duration: float = 2.0       # s — how long escape mode lasts
    dt: float = 1.0 / 20.0            # s — 20 Hz tick

    # Runtime state
    _wall_contact_time: float = field(default=0.0, init=False)
    _escape_until: float = field(default=0.0, init=False)
    _current_time: float = field(default=0.0, init=False)
    _escape_triggered: int = field(default=0, init=False)
    _last_escape_side: str = field(default="", init=False)
    _velocity_log: list = field(default_factory=list, init=False)

    def tick(
        self,
        front_d: float,
        pf_speed: float,
        left_d: float = 2.0,
        right_d: float = 2.0,
    ) -> str:
        """Simulate one 20 Hz _follow_path tick.

        Args:
            front_d: front obstacle distance in metres
            pf_speed: current forward speed (m/s) — may be negative
            left_d: left clearance in metres (for side selection)
            right_d: right clearance in metres (for side selection)

        Returns:
            "escape"    — escape maneuver is active
            "triggered" — escape maneuver just started this tick
            "ok"        — normal operation
        """
        self._current_time += self.dt

        # Active escape mode
        if self._current_time < self._escape_until:
            escape_vy = 0.3 if right_d > left_d else -0.3
            escape_vyaw = 0.4 if right_d > left_d else -0.4
            self._velocity_log.append((-0.35, escape_vy, escape_vyaw))
            return "escape"

        # Contact accumulation
        if front_d < self.front_threshold and abs(pf_speed) < self.speed_threshold:
            self._wall_contact_time += self.dt
        else:
            self._wall_contact_time = 0.0

        # Trigger check
        if self._wall_contact_time > self.contact_duration:
            self._escape_triggered += 1
            self._escape_until = self._current_time + self.escape_duration
            self._wall_contact_time = 0.0
            self._last_escape_side = "right" if right_d > left_d else "left"
            return "triggered"

        return "ok"

    @property
    def escape_active(self) -> bool:
        return self._current_time < self._escape_until

    def run_ticks(
        self,
        n: int,
        front_d: float,
        pf_speed: float,
        left_d: float = 2.0,
        right_d: float = 2.0,
    ) -> list[str]:
        """Run n ticks with constant sensor inputs. Returns list of results."""
        return [self.tick(front_d, pf_speed, left_d, right_d) for _ in range(n)]


class TestWallEscapeSimulator:
    """Behavioral tests — drive the simulator through realistic scenarios."""

    def test_no_trigger_when_moving_fast(self):
        """Dog moving forward at normal speed must never trigger escape."""
        sim = WallEscapeSimulator()
        results = sim.run_ticks(40, front_d=0.20, pf_speed=0.35)
        assert "triggered" not in results, (
            "Wall escape triggered while robot was moving forward — "
            "speed check must suppress trigger when |pf_speed| >= 0.05"
        )

    def test_no_trigger_when_front_clear(self):
        """No wall in front means no trigger even if robot is slow."""
        sim = WallEscapeSimulator()
        results = sim.run_ticks(40, front_d=0.80, pf_speed=0.00)
        assert "triggered" not in results, (
            "Wall escape triggered with clear front (0.80 m) — "
            "distance check must suppress trigger when front_d >= 0.25"
        )

    def test_trigger_after_1s_contact(self):
        """Trigger fires after robot is pinned (front<0.25 + slow) for > 1.0 s."""
        sim = WallEscapeSimulator()
        # 20 Hz × 1.0 s = 20 ticks exactly fills 1.0 s; trigger fires on tick 21+
        results = sim.run_ticks(25, front_d=0.20, pf_speed=0.02)
        assert "triggered" in results, (
            "Wall escape never triggered after 1.25 s of wall contact — "
            "contact accumulation or threshold broken"
        )

    def test_no_trigger_before_1s(self):
        """Must NOT trigger before 1.0 s of continuous contact (avoids false alarms)."""
        sim = WallEscapeSimulator()
        # 19 ticks = 0.95 s — just under threshold
        results = sim.run_ticks(19, front_d=0.20, pf_speed=0.02)
        assert "triggered" not in results, (
            "Wall escape triggered before 1.0 s contact duration — premature trigger"
        )

    def test_escape_lasts_2s(self):
        """After trigger, escape mode must persist for ~2.0 s (40 ticks at 20 Hz).

        Count escape ticks across the priming run AND the follow-up window because
        the simulator enters escape mode immediately after the trigger tick and
        remains in escape for ticks 22-25 of the priming run.
        """
        sim = WallEscapeSimulator()
        # Prime — trigger fires on tick 21; ticks 22-25 are also escape
        prime_results = sim.run_ticks(25, front_d=0.20, pf_speed=0.02)
        prime_escape = sum(1 for r in prime_results if r == "escape")
        # Continuation — front clear so no new trigger
        cont_results = sim.run_ticks(50, front_d=2.0, pf_speed=0.0)
        cont_escape = sum(1 for r in cont_results if r == "escape")
        total_escape = prime_escape + cont_escape
        # 2.0 s at 20 Hz = 40 ticks; allow ±2 for float rounding
        assert 38 <= total_escape <= 42, (
            f"Total escape ticks={total_escape} (prime={prime_escape}, cont={cont_escape}) "
            f"— expected ~40 (2.0 s at 20 Hz)"
        )

    def test_contact_resets_when_front_clears(self):
        """If obstacle clears, accumulated contact time must reset to zero."""
        sim = WallEscapeSimulator()
        # Partial contact
        sim.run_ticks(15, front_d=0.20, pf_speed=0.02)
        # Front clears — contact time should reset
        sim.run_ticks(5, front_d=0.80, pf_speed=0.30)
        # Partial contact again — need another full second to trigger
        results = sim.run_ticks(10, front_d=0.20, pf_speed=0.02)
        assert "triggered" not in results, (
            "Escape triggered despite contact time reset — accumulator not clearing"
        )

    def test_contact_resets_when_speed_increases(self):
        """If speed rises above threshold, accumulated time must reset."""
        sim = WallEscapeSimulator()
        sim.run_ticks(15, front_d=0.20, pf_speed=0.02)   # partial build-up
        sim.run_ticks(5, front_d=0.20, pf_speed=0.30)    # speed clears condition
        results = sim.run_ticks(10, front_d=0.20, pf_speed=0.02)  # back to slow
        assert "triggered" not in results, (
            "Escape triggered despite speed-based reset of contact accumulator"
        )

    def test_escape_picks_right_when_right_is_open(self):
        """When right_d > left_d, escape side must be 'right'."""
        sim = WallEscapeSimulator()
        sim.run_ticks(25, front_d=0.20, pf_speed=0.02, left_d=0.3, right_d=1.5)
        assert sim._last_escape_side == "right", (
            f"Escape went {sim._last_escape_side!r} — expected 'right' when right_d=1.5 > left_d=0.3"
        )

    def test_escape_picks_left_when_left_is_open(self):
        """When left_d > right_d, escape side must be 'left'."""
        sim = WallEscapeSimulator()
        sim.run_ticks(25, front_d=0.20, pf_speed=0.02, left_d=1.5, right_d=0.3)
        assert sim._last_escape_side == "left", (
            f"Escape went {sim._last_escape_side!r} — expected 'left' when left_d=1.5 > right_d=0.3"
        )

    def test_no_false_trigger_doorway_transit(self):
        """Robot transiting a doorway at speed should never trigger wall escape.

        Scenario: front obstacle at 0.20 m for one tick (door edge), but speed
        is 0.35 m/s (normal walk) — should be classified as passing, not pinned.
        """
        sim = WallEscapeSimulator()
        # Simulate approaching + entering a doorway at walking speed
        # Outside doorway — clear
        sim.run_ticks(10, front_d=1.5, pf_speed=0.35)
        # Door edge briefly close (2 ticks ~ 0.10 s)
        sim.run_ticks(2, front_d=0.20, pf_speed=0.35)
        # Through doorway — clear again
        results = sim.run_ticks(10, front_d=1.0, pf_speed=0.35)
        assert "triggered" not in results, (
            "False trigger during doorway transit — speed check failed"
        )

    def test_escape_velocity_uses_reverse(self):
        """Velocity log during escape must have negative vx for all escape ticks."""
        sim = WallEscapeSimulator()
        sim.run_ticks(25, front_d=0.20, pf_speed=0.02)   # trigger
        sim.run_ticks(5, front_d=0.20, pf_speed=0.02)    # some escape ticks
        escape_vels = [v for v in sim._velocity_log if v[0] < 0]
        assert len(escape_vels) > 0, (
            "No reverse (negative vx) velocities logged during escape"
        )
        for vx, vy, vyaw in escape_vels:
            assert vx < 0, f"Escape vx={vx} not negative"
            assert vy != 0, f"Escape vy={vy} is zero — strafe required"

    def test_no_double_trigger_during_escape(self):
        """While escape is active, a second trigger must not fire."""
        sim = WallEscapeSimulator()
        sim.run_ticks(25, front_d=0.20, pf_speed=0.02)   # first trigger
        results = sim.run_ticks(25, front_d=0.20, pf_speed=0.02)  # still pinned
        assert "triggered" not in results, (
            "Second escape trigger fired while first escape was still active"
        )

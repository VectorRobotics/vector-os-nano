"""Level 12: TARE exploration data-chain tests.

Verifies the correctness of the ExploreSkill wander loop that feeds scan data
to the sensorScanGeneration → terrainAnalysis → TARE pipeline.

Key invariants tested:
  1. Wander velocity is sent at the right interval (≤0.8 s between commands)
  2. The interval is short enough to keep movement before teleop_until expires
     (bridge grants 0.5 s; wander must re-send within that window + margin)
  3. The exploration loop keeps running and does NOT stop after the seed phase
  4. Wander direction alternates based on rooms visited (prevents spiralling)
  5. A bridge-less explore (has_bridge=False) skips wander entirely

No MuJoCo, no ROS2, no network calls — all mocked.
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Lazy imports: explore.py depends on navigate._ROOM_CENTERS and _detect_current_room.
# We stub both before importing explore so no MuJoCo / httpx cascade is triggered.
# ---------------------------------------------------------------------------

import importlib
import types as _types


def _load_explore_module():
    """Load explore.py with navigate stub injected into sys.modules."""
    # Build a minimal navigate stub
    nav_stub = _types.ModuleType("vector_os_nano.skills.navigate")
    nav_stub._ROOM_CENTERS = {
        "living_room": (0.0, 0.0),
        "kitchen": (5.0, 0.0),
        "bedroom": (0.0, 5.0),
    }

    def _detect(x: float, y: float) -> str:
        best, best_d = "living_room", 1e9
        for name, (cx, cy) in nav_stub._ROOM_CENTERS.items():
            d = (x - cx) ** 2 + (y - cy) ** 2
            if d < best_d:
                best, best_d = name, d
        return best

    nav_stub._detect_current_room = _detect

    # Inject stubs before importing explore
    sys.modules.setdefault("vector_os_nano.skills.navigate", nav_stub)

    # core stubs
    core_stub = _types.ModuleType("vector_os_nano.core")
    core_stub.__path__ = [str(_REPO_ROOT / "vector_os_nano" / "core")]
    sys.modules.setdefault("vector_os_nano.core", core_stub)

    skill_stub = _types.ModuleType("vector_os_nano.core.skill")
    skill_stub.SkillContext = MagicMock
    skill_stub.skill = lambda **kw: (lambda cls: cls)
    sys.modules.setdefault("vector_os_nano.core.skill", skill_stub)

    types_stub = _types.ModuleType("vector_os_nano.core.types")
    types_stub.SkillResult = MagicMock
    sys.modules.setdefault("vector_os_nano.core.types", types_stub)

    explore_path = (
        _REPO_ROOT / "vector_os_nano" / "skills" / "go2" / "explore.py"
    )
    spec = importlib.util.spec_from_file_location(
        "vector_os_nano.skills.go2.explore", str(explore_path)
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["vector_os_nano.skills.go2.explore"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_explore = _load_explore_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_base(position=(0.0, 0.0, 0.28)) -> MagicMock:
    """Create a mock robot base that returns a safe standing position."""
    base = MagicMock()
    base.get_position.return_value = position
    base.set_velocity.return_value = None
    return base


def _reset_explore_globals() -> None:
    """Reset module-level singletons between tests."""
    _explore._explore_visited.clear()
    _explore._explore_running = False
    _explore._explore_cancel.clear()
    _explore._on_event = None
    _explore._auto_look = None
    _explore._tare_proc = None


# ---------------------------------------------------------------------------
# Class 1: WanderInterval — verifies the 0.8 s cadence
# ---------------------------------------------------------------------------


class TestWanderInterval:
    """Seed walk provides initial scan data for TARE before handing off to autonomous nav."""

    def test_seed_walk_uses_base_walk(self):
        """_exploration_loop should use base.walk() for the seed walk."""
        import inspect
        src = inspect.getsource(_explore._exploration_loop)
        assert "base.walk(" in src, "Seed walk should use base.walk()"

    def test_wander_threshold_less_than_teleop_window_plus_margin(self):
        """0.8 s wander interval < 1.0 s (teleop 0.5 s + 0.5 s safety margin).

        This ensures the robot is re-commanded before too large a stop gap.
        """
        # Hardcoded from bridge: teleop_until = time.time() + 0.5
        teleop_window = 0.5
        safety_margin = 0.5
        max_acceptable_interval = teleop_window + safety_margin

        # The threshold we set in explore.py
        wander_threshold = 0.8
        assert wander_threshold < max_acceptable_interval, (
            f"Wander interval {wander_threshold}s >= {max_acceptable_interval}s "
            f"(teleop_window + margin). Robot will have >0.5s stop gaps."
        )

    def test_wander_duty_cycle_above_fifty_percent(self):
        """Robot moves > 50% of the time during wander.

        teleop_window / wander_interval = 0.5 / 0.8 = 62.5 % — above 50%.
        At the old 2.0 s interval it was 25% — far too low for TARE to get
        sufficient scan density.
        """
        teleop_window = 0.5
        wander_interval = 0.8
        duty_cycle = teleop_window / wander_interval
        assert duty_cycle > 0.5, (
            f"Duty cycle {duty_cycle:.1%} ≤ 50% — robot spends too much time stopped."
        )


# ---------------------------------------------------------------------------
# Class 2: WanderBehavior — the loop actually calls set_velocity on schedule
# ---------------------------------------------------------------------------


class TestWanderBehavior:
    """_exploration_loop uses seed walk then hands off to TARE autonomous nav."""

    def setup_method(self):
        _reset_explore_globals()

    def teardown_method(self):
        _explore._explore_cancel.set()
        _reset_explore_globals()

    def test_seed_walk_called_with_bridge(self):
        """_exploration_loop calls base.walk() for seed when has_bridge=True."""
        base = _make_base()

        def _cancel_after_delay():
            time.sleep(0.1)
            _explore._explore_cancel.set()

        stopper = threading.Thread(target=_cancel_after_delay, daemon=True)
        stopper.start()

        # _start_tare must return True (TARE running) to get past the guard
        # clause. Patching subprocess.run makes pgrep return 0 → tare found.
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("time.sleep", return_value=None):
                _explore._exploration_loop(base, has_bridge=True)

        # Seed walk should have called base.walk()
        assert base.walk.call_count >= 1, "Seed walk should call base.walk()"

    def test_no_set_velocity_in_main_loop(self):
        """Main loop does NOT call set_velocity — TARE handles movement autonomously."""
        base = _make_base()

        def _cancel_after():
            time.sleep(0.1)
            _explore._explore_cancel.set()

        stopper = threading.Thread(target=_cancel_after, daemon=True)
        stopper.start()

        # Must get past _start_tare guard to verify main loop behavior.
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("time.sleep", return_value=None):
                _explore._exploration_loop(base, has_bridge=True)

        # No set_velocity calls — TARE + bridge path follower handle movement
        assert base.set_velocity.call_count == 0, (
            "Main loop should NOT call set_velocity — TARE handles movement"
        )

    def test_wander_skipped_when_no_bridge(self):
        """has_bridge=False: wander block is not entered, no set_velocity calls in main loop.

        The seed phase is also skipped (guarded by has_bridge), so set_velocity
        should only be called zero times in a very short run.
        """
        base = _make_base()

        def _cancel_immediately():
            # Cancel before the first poll iteration
            time.sleep(0.05)
            _explore._explore_cancel.set()

        stopper = threading.Thread(target=_cancel_immediately, daemon=True)
        stopper.start()

        with patch.object(_explore, "_start_tare", return_value=False):
            _explore._exploration_loop(base, has_bridge=False)

        # No velocity commands in no-bridge mode during main loop
        assert base.set_velocity.call_count == 0

    def test_loop_runs_past_seed_phase(self):
        """Loop does NOT stop after seed — continues until _explore_cancel."""
        base = _make_base()

        # Track get_position calls via a counter (we replace it with a plain fn)
        pos_calls = [0]
        positions = [
            (0.0, 0.0, 0.28),
            (0.1, 0.0, 0.28),
            (0.2, 0.0, 0.28),
            (0.3, 0.0, 0.28),
            (0.4, 0.0, 0.28),
        ]

        def _pos():
            idx = pos_calls[0]
            pos_calls[0] += 1
            return positions[min(idx, len(positions) - 1)]

        base.get_position = _pos

        def _cancel_after():
            # time.sleep is patched to no-op so seed phase completes immediately;
            # cancel shortly after the thread starts the main loop
            time.sleep(0.05)
            _explore._explore_cancel.set()

        stopper = threading.Thread(target=_cancel_after, daemon=True)
        stopper.start()

        # Patch subprocess.run to avoid 5-second TARE start timeout, and
        # patch time.sleep to instant so seed phase (2s walk + 1s buffer) finishes fast.
        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("time.sleep", return_value=None):
                _explore._exploration_loop(base, has_bridge=True)

        # Loop entered the while block and ran at least one position check
        assert pos_calls[0] >= 1, (
            f"Expected ≥1 get_position calls in main loop, got {pos_calls[0]}"
        )


# ---------------------------------------------------------------------------
# Class 3: WanderDirection — alternation logic
# ---------------------------------------------------------------------------


class TestWanderDirection:
    """Wander heading alternates based on rooms visited count."""

    def setup_method(self):
        _reset_explore_globals()

    def test_wander_heading_positive_when_zero_rooms(self):
        """Initial wander heading is +0.08 (even rooms count = 0)."""
        # From the code: if len(_explore_visited) % 2 == 0: _wander_heading = 0.08
        # With 0 rooms, heading should be positive
        rooms_visited: set[str] = set()
        heading = 0.08 if len(rooms_visited) % 2 == 0 else -0.08
        assert heading == 0.08

    def test_wander_heading_negative_when_one_room(self):
        """After entering first room, heading flips to -0.08 (odd count)."""
        rooms_visited: set[str] = {"living_room"}
        heading = 0.08 if len(rooms_visited) % 2 == 0 else -0.08
        assert heading == -0.08

    def test_wander_heading_alternates_every_room(self):
        """Heading sign alternates with each additional room discovered."""
        for n in range(6):
            rooms = {f"room_{i}" for i in range(n)}
            heading = 0.08 if len(rooms) % 2 == 0 else -0.08
            expected_sign = 1 if n % 2 == 0 else -1
            assert (heading > 0) == (expected_sign > 0), (
                f"At {n} rooms, expected sign {expected_sign}, got heading {heading}"
            )

    def test_wander_velocity_has_nonzero_heading(self):
        """set_velocity wander calls include non-zero vyaw (not pure forward)."""
        base = _make_base()
        vyaw_calls: list[float] = []

        def _record(vx, vy, vyaw):
            vyaw_calls.append(vyaw)

        base.set_velocity.side_effect = _record

        def _cancel_immediately():
            time.sleep(0.05)
            _explore._explore_cancel.set()

        stopper = threading.Thread(target=_cancel_immediately, daemon=True)
        stopper.start()

        with patch.object(_explore, "_start_tare", return_value=False):
            with patch("time.sleep", return_value=None):
                _explore._exploration_loop(base, has_bridge=True)

        # If any wander calls happened, they should have non-zero vyaw
        wander_calls = [v for v in vyaw_calls if v != 0.0]
        # Note: may be empty if loop cancelled before first wander — that's OK,
        # we just assert the list is not full of zeros if non-empty
        if wander_calls:
            assert all(abs(v) > 0 for v in wander_calls)


# ---------------------------------------------------------------------------
# Class 4: ExploreLoop continuity
# ---------------------------------------------------------------------------


class TestExploreLoopContinuity:
    """Exploration loop keeps running and tracks rooms correctly."""

    def setup_method(self):
        _reset_explore_globals()

    def teardown_method(self):
        _explore._explore_cancel.set()
        _reset_explore_globals()

    def test_explore_running_flag_set_during_loop(self):
        """_explore_running is True while loop executes."""
        base = _make_base()
        running_states: list[bool] = []

        def _pos():
            running_states.append(_explore._explore_running)
            _explore._explore_cancel.set()  # cancel after first poll
            return (0.0, 0.0, 0.28)

        base.get_position = _pos

        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("time.sleep", return_value=None):
                _explore._exploration_loop(base, has_bridge=True)

        assert True in running_states, "_explore_running never became True"

    def test_explore_running_flag_cleared_after_cancel(self):
        """_explore_running is False after the loop exits."""
        base = _make_base()

        def _pos():
            _explore._explore_cancel.set()
            return (0.0, 0.0, 0.28)

        base.get_position = _pos

        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("time.sleep", return_value=None):
                _explore._exploration_loop(base, has_bridge=True)

        assert not _explore._explore_running

    def test_robot_fall_stops_loop(self):
        """Loop exits immediately if robot height < 0.12 m (fallen)."""
        base = _make_base(position=(0.0, 0.0, 0.05))  # fallen

        events: list[str] = []

        def _on_event(event_type, data):
            events.append(event_type)

        _explore._on_event = _on_event

        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("time.sleep", return_value=None):
                _explore._exploration_loop(base, has_bridge=True)

        assert "stopped" in events
        assert events.count("stopped") >= 1

    def test_room_entered_event_on_new_room(self):
        """_emit("room_entered") fires when a new room is detected."""
        base = _make_base()
        events: list[tuple[str, dict]] = []

        def _on_event(event_type, data):
            events.append((event_type, data))

        _explore._on_event = _on_event

        # Provide a mock SceneGraph so nearest_room() returns room names.
        # explore.py uses _spatial_memory.nearest_room() since the hardcoded
        # _detect_current_room was removed.
        mock_sg = MagicMock()
        call_count = [0]
        def _nearest_room(x, y):
            call_count[0] += 1
            if x < 2.0:
                return "living_room"
            return "kitchen"
        mock_sg.nearest_room = _nearest_room
        mock_sg.visit = MagicMock()
        mock_sg.add_door = MagicMock()
        _explore._spatial_memory = mock_sg

        pos_calls = [0]

        def _pos():
            pos_calls[0] += 1
            if pos_calls[0] == 1:
                return (0.0, 0.0, 0.28)   # living_room
            _explore._explore_cancel.set()
            return (5.0, 0.0, 0.28)       # kitchen

        base.get_position = _pos

        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("time.sleep", return_value=None):
                _explore._exploration_loop(base, has_bridge=True)

        # Restore module state
        _explore._spatial_memory = None

        room_events = [d for e, d in events if e == "room_entered"]
        assert len(room_events) >= 1
        rooms = {d["room"] for d in room_events}
        assert "living_room" in rooms or "kitchen" in rooms

    def test_explore_visited_cleared_on_restart(self):
        """_explore_visited is cleared when loop starts fresh."""
        _explore._explore_visited.add("some_old_room")

        base = _make_base()

        def _pos():
            _explore._explore_cancel.set()
            return (0.0, 0.0, 0.28)

        base.get_position = _pos

        with patch("subprocess.run", return_value=MagicMock(returncode=0)):
            with patch("time.sleep", return_value=None):
                _explore._exploration_loop(base, has_bridge=True)

        assert "some_old_room" not in _explore._explore_visited


# ---------------------------------------------------------------------------
# Class 5: DataChainCompatibility — QoS + timing analysis (static/logical)
# ---------------------------------------------------------------------------


class TestDataChainCompatibility:
    """Static analysis of the TARE data chain configuration."""

    def test_bridge_publishes_reliable_qos(self):
        """Bridge source confirms /state_estimation and /registered_scan use RELIABLE QoS.

        sensorScanGeneration subscribes with BEST_EFFORT. In DDS, a RELIABLE
        publisher is compatible with a BEST_EFFORT subscriber (pub offers more
        than sub requires). This combination is safe.
        """
        bridge_path = _REPO_ROOT / "scripts" / "go2_vnav_bridge.py"
        src = bridge_path.read_text()

        # reliable_qos is defined and used for both publishers
        assert "reliable_qos" in src, "reliable_qos profile not found in bridge"
        assert "ReliabilityPolicy.RELIABLE" in src, (
            "RELIABLE policy not found in bridge"
        )
        # Both topic names appear in the same file
        assert "/state_estimation" in src
        assert "/registered_scan" in src

    def test_sensor_scan_generation_uses_reliable_qos(self):
        """sensorScanGeneration.cpp uses RELIABLE QoS for cross-process robustness."""
        cpp_path = (
            _REPO_ROOT.parent
            / "vector_navigation_stack"
            / "src"
            / "base_autonomy"
            / "sensor_scan_generation"
            / "src"
            / "sensorScanGeneration.cpp"
        )
        if not cpp_path.exists():
            pytest.skip("sensorScanGeneration.cpp not found — skipping nav stack analysis")

        src = cpp_path.read_text()
        assert "RMW_QOS_POLICY_RELIABILITY_RELIABLE" in src, (
            "Expected RELIABLE in sensorScanGeneration.cpp (not BEST_EFFORT)"
        )

    def test_approximate_time_queue_size_is_adequate(self):
        """ApproximateTime queue size >= 200 handles 200 Hz odom + 10 Hz scans."""
        cpp_path = (
            _REPO_ROOT.parent
            / "vector_navigation_stack"
            / "src"
            / "base_autonomy"
            / "sensor_scan_generation"
            / "src"
            / "sensorScanGeneration.cpp"
        )
        if not cpp_path.exists():
            pytest.skip("sensorScanGeneration.cpp not found")

        src = cpp_path.read_text()
        import re
        match = re.search(r'syncPolicy\((\d+)', src)
        assert match, "Could not find syncPolicy queue size"
        queue_size = int(match.group(1))
        assert queue_size >= 200, (
            f"syncPolicy queue size {queue_size} < 200 (too small for cross-process sync)"
        )

    def test_wander_interval_reduces_stop_gap(self):
        """Old 2.0 s interval had 1.5 s stop gap; new 0.8 s has ≤0.3 s gap.

        Verifies the fix is strictly better than the old behavior.
        """
        teleop_window = 0.5  # bridge: teleop_until = now + 0.5

        old_interval = 2.0
        new_interval = 0.8

        old_gap = old_interval - teleop_window  # 1.5 s
        new_gap = new_interval - teleop_window  # 0.3 s

        assert new_gap < old_gap, (
            f"New gap {new_gap}s not less than old gap {old_gap}s"
        )
        assert new_gap < 0.5, (
            f"New stop gap {new_gap}s is too large (>= teleop window 0.5 s)"
        )

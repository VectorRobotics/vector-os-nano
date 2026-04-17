"""Level 23: TARE exploration parameter verification tests."""
import os
import yaml
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TARE_CFG = os.path.join(_REPO, "config", "tare_go2_indoor.yaml")


def _load_tare_config():
    with open(_TARE_CFG) as f:
        data = yaml.safe_load(f)
    return data.get("tare_planner_node", {}).get("ros__parameters", {})


class TestTareExtensionDistance:
    """Verify TARE waypoint extension is safe for indoor Go2."""

    def test_config_file_exists(self):
        assert os.path.isfile(_TARE_CFG)

    def test_extend_waypoint_enabled(self):
        cfg = _load_tare_config()
        assert cfg["kExtendWayPoint"] is True

    def test_extend_distance_big_indoor(self):
        """kExtendWayPointDistanceBig must be <= 2.0m for indoor rooms."""
        cfg = _load_tare_config()
        assert cfg["kExtendWayPointDistanceBig"] <= 2.0
        assert cfg["kExtendWayPointDistanceBig"] == 1.5

    def test_extend_distance_small_indoor(self):
        """kExtendWayPointDistanceSmall must be <= 1.0m for indoor."""
        cfg = _load_tare_config()
        assert cfg["kExtendWayPointDistanceSmall"] <= 1.0
        assert cfg["kExtendWayPointDistanceSmall"] == 0.8

    def test_sensor_range(self):
        cfg = _load_tare_config()
        assert cfg["kSensorRange"] == 3.0

    def test_auto_start(self):
        # kAutoStart is False by design: the robot waits for /start_exploration
        # signal rather than exploring immediately on launch. This avoids
        # premature movement before the operator confirms the scene is safe.
        cfg = _load_tare_config()
        assert cfg["kAutoStart"] is False

    def test_collision_margin(self):
        cfg = _load_tare_config()
        assert cfg["kViewPointCollisionMargin"] >= 0.25
        assert cfg["kViewPointCollisionMargin"] <= 0.5

    def test_keypose_min_dist(self):
        cfg = _load_tare_config()
        kp = cfg.get("keypose_graph/kAddNodeMinDist", 0.3)
        assert kp >= 0.2
        assert kp <= 0.5

    def test_terrain_collision_threshold(self):
        cfg = _load_tare_config()
        assert cfg.get("kTerrainCollisionThreshold", 0.3) >= 0.2

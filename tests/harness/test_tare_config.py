# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Validate TARE config values for Go2 indoor exploration."""
import yaml
from importlib.resources import files


def load_tare_config():
    return yaml.safe_load(files('vector_os_nano.sim.nav.ros2').joinpath('tare_go2_indoor.yaml').read_text())


def test_viewpoint_collision_margin_sufficient():
    """kViewPointCollisionMargin must be >= Go2 half-width (0.24m)."""
    cfg = load_tare_config()
    params = cfg["tare_planner_node"]["ros__parameters"]
    margin = params["kViewPointCollisionMargin"]
    assert margin >= 0.24, f"kViewPointCollisionMargin={margin} < 0.24 (Go2 half-width)"


def test_viewpoint_z_margins():
    cfg = load_tare_config()
    params = cfg["tare_planner_node"]["ros__parameters"]
    assert params["kViewPointCollisionMarginZPlus"] >= 0.20
    assert params["kViewPointCollisionMarginZMinus"] >= 0.20


def test_sensor_range_reasonable():
    cfg = load_tare_config()
    params = cfg["tare_planner_node"]["ros__parameters"]
    assert 1.0 <= params["kSensorRange"] <= 10.0


def test_extend_waypoint_distances_safe_for_indoor():
    """Indoor: extension distances must be small enough to avoid pushing waypoints through walls.

    kExtendWayPointDistanceBig <= 2.0m and kExtendWayPointDistanceSmall <= 1.0m
    prevents waypoints from crossing interior walls in the 20x14m MuJoCo house (1.2m doorways).
    """
    cfg = load_tare_config()
    params = cfg["tare_planner_node"]["ros__parameters"]
    big = params["kExtendWayPointDistanceBig"]
    small = params["kExtendWayPointDistanceSmall"]
    assert big <= 2.0, f"kExtendWayPointDistanceBig={big} too large for indoor (max 2.0m)"
    assert small <= 1.0, f"kExtendWayPointDistanceSmall={small} too large for indoor (max 1.0m)"


def test_terrain_collision_enabled():
    cfg = load_tare_config()
    params = cfg["tare_planner_node"]["ros__parameters"]
    assert params["kCheckTerrainCollision"] is True

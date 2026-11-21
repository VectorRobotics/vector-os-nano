# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Path-follower nominal tick. Single source: the _follow_path timer period AND
# the per-tick ramp constants inside it are calibrated against this.
_PF_DT = 1.0 / 20.0

# ---------------------------------------------------------------------------
# Nav config loader (lazy, module-level cache)
# ---------------------------------------------------------------------------

from vector_os_nano.sim.nav.follow import nav_config as _load_nav_config


def _nav(key: str, default: float) -> float:
    """Look up a navigation parameter by key, return default if absent."""
    cfg = _load_nav_config()
    nav_section = cfg.get("navigation", {})
    return float(nav_section.get(key, default))


# Sensor mounting offset — on top of Go2 head, above all leg geoms.
# Must match the body manifest's lidar pos (robot.yaml sensors) and nav stack sensorOffset.
_SENSOR_X: float = 0.3
_SENSOR_Y: float = 0.0
_SENSOR_Z: float = 0.2

# Ceiling filter: points with height above ground exceeding this threshold are
# excluded from /registered_scan. Matches FAR's vehicle_height (indoor.yaml: 1.0).
# Debug harness (test_vgraph_debug.py) confirmed doors are CLEAR at 1.0m.
# Loaded from the shared navigation parameters at startup.
_CEILING_FILTER_HEIGHT: float = _nav("ceiling_filter_height", 1.8)

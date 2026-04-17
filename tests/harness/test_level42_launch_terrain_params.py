"""L42: launch scripts pass terrain_analysis params for Go2.

Scripts that launch terrainAnalysis / terrainAnalysisExt via `ros2 run`
must pass Go2-tuned params (maxRelZ=1.5 etc.) instead of inheriting
C++ defaults (maxRelZ=0.2). Matches the pattern in launch_explore.sh.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parent.parent.parent
_SCRIPTS = ["launch_nav_only.sh", "launch_nav_explore.sh", "test_integration.sh"]


def _content(script: str) -> str:
    return (_REPO / "scripts" / script).read_text()


@pytest.mark.parametrize("script", _SCRIPTS)
class TestTerrainAnalysisParams:
    def test_script_uses_ros_args(self, script):
        """Script uses `terrainAnalysis --ros-args` syntax (not bare `ros2 run`)."""
        assert "terrainAnalysis --ros-args" in _content(script), \
            f"{script}: 'ros2 run terrain_analysis terrainAnalysis' must include '--ros-args'"

    def test_script_passes_maxRelZ(self, script):
        """terrainAnalysis must set maxRelZ >= 1.0 (default 0.2 rejects wall points)."""
        content = _content(script)
        match = re.search(
            r"terrainAnalysis\s+--ros-args(?:\s|\\|\n)+?(?:.*?\s+-p\s+maxRelZ:=([0-9.]+))",
            content, re.DOTALL,
        )
        assert match, f"{script}: terrainAnalysis must pass maxRelZ via --ros-args"
        assert float(match.group(1)) >= 1.0, \
            f"{script}: maxRelZ must be >= 1.0 (got {match.group(1)})"

    def test_script_passes_clearDyObs_true(self, script):
        """terrainAnalysis must set clearDyObs=true for indoor dynamic cleanup."""
        assert "clearDyObs:=true" in _content(script), \
            f"{script}: must pass clearDyObs:=true"

    def test_script_passes_obstacleHeightThre(self, script):
        """terrainAnalysis must set obstacleHeightThre=0.15 for doorsill detection."""
        assert "obstacleHeightThre:=0.15" in _content(script), \
            f"{script}: must pass obstacleHeightThre:=0.15"

    def test_ext_also_has_maxRelZ(self, script):
        """terrainAnalysisExt must also pass maxRelZ >= 1.0."""
        content = _content(script)
        match = re.search(
            r"terrainAnalysisExt\s+--ros-args(?:\s|\\|\n)+?(?:.*?\s+-p\s+maxRelZ:=([0-9.]+))",
            content, re.DOTALL,
        )
        assert match, f"{script}: terrainAnalysisExt must pass maxRelZ via --ros-args"
        assert float(match.group(1)) >= 1.0

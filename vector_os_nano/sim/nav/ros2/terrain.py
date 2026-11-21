# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

class TerrainAccumulator:
    """Accumulates pointcloud data into a 2D voxel grid for terrain persistence."""

    def __init__(self, voxel_size: float = 0.1, z_min: float = -0.5, z_max: float = 2.0):
        self._voxel_size = voxel_size
        self._grid: dict[tuple[int, int], float] = {}  # (ix, iy) → max_z
        self._z_min = z_min
        self._z_max = z_max
        self._count = 0  # total points added

    def add(self, points: list[tuple[float, float, float, float]]) -> None:
        """Add pointcloud (x, y, z, intensity) to grid. Keep max z per voxel."""
        for x, y, z, _ in points:
            if z < self._z_min or z > self._z_max:
                continue
            ix = int(x / self._voxel_size)
            iy = int(y / self._voxel_size)
            key = (ix, iy)
            if key not in self._grid or z > self._grid[key]:
                self._grid[key] = z
        self._count += len(points)

    def save(self, path: str) -> bool:
        """Save grid as numpy npz. Returns True on success."""
        if not self._grid:
            return False
        import numpy as np
        keys = list(self._grid.keys())
        xs = np.array([k[0] for k in keys], dtype=np.int32)
        ys = np.array([k[1] for k in keys], dtype=np.int32)
        zs = np.array([self._grid[k] for k in keys], dtype=np.float32)
        try:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, ix=xs, iy=ys, z=zs, voxel_size=np.float32(self._voxel_size))
            return True
        except Exception:
            return False

    def load(self, path: str) -> bool:
        """Load grid from npz. Returns True on success."""
        try:
            import numpy as np
            data = np.load(path)
            self._voxel_size = float(data['voxel_size'])
            self._grid = {}
            for ix, iy, z in zip(data['ix'], data['iy'], data['z']):
                self._grid[(int(ix), int(iy))] = float(z)
            return True
        except Exception:
            return False

    def to_pointcloud(self) -> list[tuple[float, float, float, float]]:
        """Convert grid back to pointcloud for publishing as /registered_scan."""
        points = []
        half = self._voxel_size / 2
        for (ix, iy), z in self._grid.items():
            x = ix * self._voxel_size + half
            y = iy * self._voxel_size + half
            # intensity = height above ground (same as bridge convention)
            intensity = z
            points.append((x, y, z, intensity))
        return points

    @property
    def size(self) -> int:
        return len(self._grid)

    @property
    def point_count(self) -> int:
        return self._count

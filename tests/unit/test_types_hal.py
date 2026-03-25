"""Tests for Odometry and LaserScan types."""
import math
from vector_os_nano.core.types import Odometry, LaserScan


class TestOdometry:
    def test_defaults(self):
        o = Odometry(timestamp=1.0)
        assert o.x == 0.0 and o.y == 0.0 and o.z == 0.0
        assert o.qw == 1.0
        assert o.vx == 0.0 and o.vyaw == 0.0

    def test_fields(self):
        o = Odometry(timestamp=1.0, x=1.5, y=2.5, z=0.3, qx=0.0, qy=0.0, qz=0.1, qw=0.99, vx=0.3, vy=0.0, vz=0.0, vyaw=0.5)
        assert o.x == 1.5
        assert o.vyaw == 0.5

    def test_frozen(self):
        import pytest
        o = Odometry(timestamp=1.0)
        with pytest.raises(AttributeError):
            o.x = 5.0

    def test_to_dict(self):
        o = Odometry(timestamp=1.0, x=1.0, y=2.0)
        d = o.to_dict()
        assert d["x"] == 1.0
        assert d["timestamp"] == 1.0

    def test_from_dict(self):
        d = {"timestamp": 2.0, "x": 3.0, "y": 4.0, "qw": 1.0}
        o = Odometry.from_dict(d)
        assert o.timestamp == 2.0
        assert o.x == 3.0
        assert o.qw == 1.0

    def test_round_trip(self):
        o1 = Odometry(timestamp=1.5, x=1.0, y=2.0, z=0.3, qz=0.1, qw=0.99, vx=0.5, vyaw=0.2)
        o2 = Odometry.from_dict(o1.to_dict())
        assert o1.x == o2.x and o1.vyaw == o2.vyaw


class TestLaserScan:
    def test_defaults(self):
        s = LaserScan(
            timestamp=1.0,
            angle_min=-math.pi, angle_max=math.pi,
            angle_increment=math.radians(1.0),
            range_min=0.1, range_max=12.0,
            ranges=tuple([1.0] * 360),
        )
        assert len(s.ranges) == 360
        assert s.range_max == 12.0

    def test_frozen(self):
        import pytest
        s = LaserScan(timestamp=1.0, angle_min=0, angle_max=1, angle_increment=0.1, range_min=0.1, range_max=10, ranges=(1.0,))
        with pytest.raises(AttributeError):
            s.ranges = (2.0,)

    def test_to_dict(self):
        s = LaserScan(timestamp=1.0, angle_min=-3.14, angle_max=3.14, angle_increment=0.01, range_min=0.1, range_max=12.0, ranges=(1.0, 2.0, 3.0))
        d = s.to_dict()
        assert d["range_max"] == 12.0
        assert len(d["ranges"]) == 3

    def test_from_dict(self):
        d = {"timestamp": 1.0, "angle_min": -3.14, "angle_max": 3.14, "angle_increment": 0.01, "range_min": 0.1, "range_max": 12.0, "ranges": [1.0, 2.0]}
        s = LaserScan.from_dict(d)
        assert s.range_max == 12.0
        assert len(s.ranges) == 2

    def test_round_trip(self):
        s1 = LaserScan(timestamp=2.0, angle_min=-3.14, angle_max=3.14, angle_increment=0.01, range_min=0.1, range_max=12.0, ranges=(5.0, 6.0, 7.0))
        s2 = LaserScan.from_dict(s1.to_dict())
        assert s1.ranges == s2.ranges

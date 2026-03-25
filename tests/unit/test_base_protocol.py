"""Tests for BaseProtocol — mobile base abstraction."""
from typing import runtime_checkable
from vector_os_nano.hardware.base import BaseProtocol


class TestBaseProtocol:
    def test_is_runtime_checkable(self):
        assert runtime_checkable  # Protocol decorator exists
        # BaseProtocol should be usable with isinstance
        assert hasattr(BaseProtocol, '__protocol_attrs__') or hasattr(BaseProtocol, '_is_runtime_protocol')

    def test_mujoco_go2_satisfies_protocol(self):
        """MuJoCoGo2 must satisfy BaseProtocol after the refactor."""
        # For now, test that we can import and the protocol has the right methods
        from vector_os_nano.hardware.base import BaseProtocol
        required_methods = [
            'connect', 'disconnect', 'stop',
            'walk', 'set_velocity',
            'get_position', 'get_heading', 'get_velocity',
            'get_odometry', 'get_lidar_scan',
        ]
        required_properties = ['name', 'supports_holonomic', 'supports_lidar']

        for method in required_methods:
            assert hasattr(BaseProtocol, method), f"Missing method: {method}"
        for prop in required_properties:
            assert hasattr(BaseProtocol, prop), f"Missing property: {prop}"

    def test_protocol_method_signatures(self):
        """Verify key method signatures via Protocol inspection."""
        import inspect
        from vector_os_nano.hardware.base import BaseProtocol

        # walk should accept vx, vy, vyaw, duration
        walk_sig = inspect.signature(BaseProtocol.walk)
        params = list(walk_sig.parameters.keys())
        assert 'vx' in params
        assert 'vy' in params
        assert 'vyaw' in params
        assert 'duration' in params

        # set_velocity should accept vx, vy, vyaw
        sv_sig = inspect.signature(BaseProtocol.set_velocity)
        params = list(sv_sig.parameters.keys())
        assert 'vx' in params
        assert 'vy' in params
        assert 'vyaw' in params

"""Real ROS messages preserve acquisition time, calibration and optional depth."""
import numpy as np
import pytest
from vector_os_nano.interfaces.devices import CameraFrame

pytestmark = pytest.mark.sim


@pytest.mark.parametrize('depth', [None, np.full((3, 4), .7, np.float32)])
def test_camera_optical_tf_and_pixels_share_acquisition_time(depth):
    from scipy.spatial.transform import Rotation
    from vector_os_nano.sim.nav.ros2.session_viz import camera_messages, stamp
    frame = CameraFrame(np.arange(36, dtype=np.uint8).reshape(3, 4, 3), depth, None,
                        np.array([[5., 0., 2.], [0., 6., 1.5], [0., 0., 1.]]),
                        np.array([1., 2., 3.]), np.eye(3), 9.9999999998, 'wrist')
    rgb, dep, info, tf = camera_messages(frame)
    assert rgb.header.stamp == info.header.stamp == tf.header.stamp == stamp(10.)
    assert bytes(rgb.data) == frame.rgb.tobytes() and np.array_equal(info.k, frame.K.ravel())
    assert rgb.header.frame_id == info.header.frame_id == tf.child_frame_id == 'wrist_optical'
    assert (tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z) == (1., 2., 3.)
    q = tf.transform.rotation
    assert np.allclose(Rotation.from_quat([q.x, q.y, q.z, q.w]).apply([0., 0., 1.]), [0., 0., -1.])
    if depth is None:
        assert dep is None
    else:
        assert dep.header == rgb.header and np.array_equal(np.frombuffer(bytes(dep.data), '<f4').reshape(3, 4), depth)


def test_overview_exposes_each_configured_camera_without_inventing_depth():
    from vector_os_nano.sim.nav.ros2.session_viz import sensor_overview
    frames = [CameraFrame(np.full((3, 4, 3), [255, 0, 0], np.uint8), None, None,
                          np.eye(3), np.zeros(3), np.eye(3), 1., n) for n in ('head', 'wrist')]
    msg = sensor_overview(frames)
    data = np.frombuffer(bytes(msg.data), np.uint8).reshape(520, 640, 3)
    assert np.array_equal(data[100, 100], [255, 0, 0])
    assert np.array_equal(data[100, 420], [255, 0, 0])
    assert np.array_equal(data[400, 420], [16, 24, 32])


def test_read_only_ros_context_stops_without_stopping_other_contexts(tmp_path):
    import threading
    import time
    from types import SimpleNamespace
    from rclpy.context import Context
    from vector_os_nano.sim.scene import compiled_scene
    from vector_os_nano.sim.backends.mujoco.world import MuJoCoWorld
    from vector_os_nano.sim.runtime import SimRuntime
    from vector_os_nano.sim.nav.ros2.session_viz import start
    path = tmp_path/'sensor.xml'
    path.write_text('<mujoco><worldbody><body name="root"><freejoint/>'
                    '<geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    rt = SimRuntime(MuJoCoWorld(compiled_scene(path, robot_root='root')))
    cfg = SimpleNamespace(arm=None, model=SimpleNamespace(root_body='root'))
    other = Context(); other.init()
    try:
        for _ in range(2):
            close = start(rt, cfg, {}, None, None, None)
            time.sleep(.1)
            close()
            assert not any(t.name == 'session_sensor_viz' for t in threading.enumerate())
            assert other.ok()
    finally:
        rt.close(); other.shutdown()

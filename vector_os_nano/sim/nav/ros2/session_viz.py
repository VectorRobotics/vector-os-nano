# SPDX-License-Identifier: Apache-2.0
"""Read-only RViz telemetry of the SAME session the CLI controls.

No command subscriptions, external-object ground truth, or navigation decisions.
All sensor stamps are simulation acquisition times, including the optical TF.
"""
from __future__ import annotations

import json
import os
import threading

import numpy as np
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from builtin_interfaces.msg import Time
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, CameraInfo, JointState, LaserScan, PointCloud2, PointField
from nav_msgs.msg import Odometry, Path as NavPath
from geometry_msgs.msg import TransformStamped, PoseStamped, WrenchStamped, PointStamped, PolygonStamped, Point32, Twist
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

from vector_os_nano.sim.scene import reader


def stamp(seconds):
    ns = round(float(seconds) * 1_000_000_000)
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


def camera_messages(frame):
    """Native pixels and calibration; RGB, depth and optical pose share one stamp."""
    from scipy.spatial.transform import Rotation
    h, w = frame.rgb.shape[:2]
    rgb = Image(height=h, width=w, encoding='rgb8', step=w*3, data=frame.rgb.tobytes())
    rgb.header.stamp = stamp(frame.sim_time)
    rgb.header.frame_id = frame.camera + '_optical'
    info = CameraInfo(height=h, width=w, distortion_model='plumb_bob', d=[0.]*5)
    info.header = rgb.header
    info.k = frame.K.ravel().tolist()
    info.r = np.eye(3).ravel().tolist()
    info.p = np.column_stack((frame.K, np.zeros(3))).ravel().tolist()
    depth = None
    if frame.depth is not None:
        depth = Image(height=h, width=w, encoding='32FC1', step=w*4,
                      data=np.asarray(frame.depth, dtype='<f4').tobytes())
        depth.header = rgb.header
    tf = TransformStamped()
    tf.header.stamp = rgb.header.stamp
    tf.header.frame_id = 'map'
    tf.child_frame_id = rgb.header.frame_id
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, frame.cam_pos)
    q = Rotation.from_matrix(frame.cam_mat @ np.diag([1., -1., -1.])).as_quat()
    tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = map(float, q)
    return rgb, depth, info, tf


class SessionViz(Node):
    def __init__(self, runtime, config, cameras, base, lidar, gripper, *, context, layout=None):
        super().__init__('vector_session_sensors', context=context)
        self.rt, self.cfg = runtime, config
        self.base, self.lidar, self.gripper = base, lidar, gripper
        self.scene = runtime.world.scene
        self.cameras = cameras
        robot_bodies = self.scene.subtree(self.scene.robot_root)
        self.joints = [j for j in reader.joints(self.scene.model)
                       if j.body in robot_bodies and j.type in (reader.JNT_HINGE, reader.JNT_SLIDE)]
        self.tf = TransformBroadcaster(self)
        self.clock = self.create_publisher(Clock, '/clock', 5)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 5)
        self.odom = self.create_publisher(Odometry, '/state_estimation', 5)
        self.base_odom = self.create_publisher(Odometry, '/odom_base_link', 5)
        self.goal = self.create_publisher(PointStamped, '/goal_point', 5)
        self.boundary = self.create_publisher(PolygonStamped, '/navigation_boundary', 5)
        self.layout, self._last_goal = layout, None
        self.scan = self.create_publisher(LaserScan, '/scan', 5)
        self.points = self.create_publisher(PointCloud2, '/registered_scan', 5)
        self.force = self.create_publisher(WrenchStamped, '/vector/fingertip_load', 5)
        self.path = self.create_publisher(NavPath, '/vector/navigation/path', 5)
        self.status = self.create_publisher(String, '/vector/navigation/status', 5)
        self.camera_pubs = {name: (
            self.create_publisher(Image, f'/sensors/{name}/rgb', 2),
            self.create_publisher(Image, f'/sensors/{name}/depth', 2),
            self.create_publisher(CameraInfo, f'/sensors/{name}/camera_info', 2)) for name in cameras}
        self.overview = self.create_publisher(Image, '/sensors/overview', 2)
        self.inventory = self.create_publisher(String, '/vector/sensors',
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.inventory.publish(String(data=json.dumps({
            'cameras': list(cameras), 'lidar': lidar is not None,
            'joint_encoders': [j.name for j in self.joints],
            'fingertip_load': bool(config.arm and config.arm.gripper and config.arm.gripper.force_feedback),
            'odometry': 'ideal simulated root pose', 'imu': 'not configured',
            'calibration': 'simulation estimates; not hardware calibration',
            'navigation': 'FAR/TARE + CMU pathFollower' if base and base._nav_stack is not None
                          else 'VECTOR visibility graph on authored scene',
        })))
        self.create_timer(.05, self.publish_state)
        self.create_timer(.5, self.publish_sensors)
        self.create_timer(.25, self.publish_navigation)
        self._errors = set()

    def publish_state(self):
        snap = self.rt.snapshot()
        now = stamp(snap.sim_time)
        self.clock.publish(Clock(clock=now))
        msg = JointState(name=[j.name for j in self.joints],
                         position=[float(snap.q[j.qadr]) for j in self.joints],
                         velocity=[float(snap.qd[j.dadr]) for j in self.joints])
        msg.header.stamp = now
        self.joint_pub.publish(msg)
        if self.base is None:
            return
        q = snap.q[self.scene.free_q(self.scene.robot_root)]
        v = snap.qd[self.scene.free_qd(self.scene.robot_root)]
        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = now, 'map', self.cfg.model.root_body
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, q[:3])
        tf.transform.rotation.w, tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z = map(float, q[3:7])
        self.tf.sendTransform(tf)
        odom = Odometry()
        odom.header, odom.child_frame_id = tf.header, tf.child_frame_id
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, q[:3])
        odom.pose.pose.orientation = tf.transform.rotation
        # MuJoCo root linear velocity is world-frame; ROS twist is child-frame.
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat([q[4], q[5], q[6], q[3]])
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, rot.inv().apply(v[:3]))
        odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z = map(float, v[3:6])
        self.base_odom.publish(odom)
        if self.lidar is not None:
            import math
            origin, heading = self.lidar.pose(snap)
            odom.child_frame_id = 'sensor'
            odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, origin)
            odom.pose.pose.orientation.x = odom.pose.pose.orientation.y = 0.
            odom.pose.pose.orientation.w, odom.pose.pose.orientation.z = math.cos(heading/2), math.sin(heading/2)
            # Root twist is not the offset sensor's twist; this stream provides pose only.
            odom.twist.twist = Twist()
            odom.twist.covariance = np.diag([1e6]*6).ravel().tolist()
            self.odom.publish(odom)
            tf.child_frame_id = 'sensor'
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, origin)
            tf.transform.rotation = odom.pose.pose.orientation
            self.tf.sendTransform(tf)
        else:
            self.odom.publish(odom)

    def publish_sensors(self):
        frames = []
        for name, camera in self.cameras.items():
            try:
                frame = camera.capture()
                frames.append(frame)
                rgb, depth, info, tf = camera_messages(frame)
                pubs = self.camera_pubs[name]
                pubs[0].publish(rgb)
                if depth is not None:
                    pubs[1].publish(depth)
                pubs[2].publish(info)
                self.tf.sendTransform(tf)
            except Exception as exc:
                if name not in self._errors:
                    self.get_logger().error(f'{name}: {exc}')
                    self._errors.add(name)
        if frames:
            self.overview.publish(sensor_overview(frames))
        if self.lidar is not None:
            scan, points, origin, heading = self.lidar.observation()
            msg = LaserScan(angle_min=scan.angle_min,
                            angle_max=scan.angle_min+(len(scan.ranges)-1)*scan.angle_increment,
                            angle_increment=scan.angle_increment, range_min=scan.range_min,
                            range_max=scan.range_max, ranges=list(scan.ranges))
            msg.header.stamp, msg.header.frame_id = stamp(scan.timestamp), 'scan_frame'
            self.scan.publish(msg)
            import math
            tf = TransformStamped()
            tf.header.stamp, tf.header.frame_id, tf.child_frame_id = msg.header.stamp, 'map', 'scan_frame'
            tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, origin)
            tf.transform.rotation.w, tf.transform.rotation.z = math.cos(heading/2), math.sin(heading/2)
            self.tf.sendTransform(tf)
            cloud = PointCloud2(height=1, width=len(points), point_step=16, row_step=16*len(points), is_dense=True)
            cloud.header.stamp, cloud.header.frame_id = msg.header.stamp, 'map'
            cloud.fields = [PointField(name=n, offset=4*i, datatype=PointField.FLOAT32, count=1)
                            for i, n in enumerate(('x', 'y', 'z', 'intensity'))]
            cloud.data = np.asarray(points, dtype='<f4').tobytes()
            self.points.publish(cloud)
        if self.gripper is not None:
            load = self.gripper.get_finger_load()
            if load is not None:
                msg = WrenchStamped()
                msg.header.stamp, msg.header.frame_id = stamp(load.sim_time), 'map'
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z = map(float, load.force)
                self.force.publish(msg)

    def publish_navigation(self):
        if self.base is None:
            return
        state = self.base.navigation_state()
        state['sim_time'] = self.rt.sim_time
        self.status.publish(String(data=json.dumps(state)))
        msg = NavPath()
        msg.header.stamp, msg.header.frame_id = stamp(state['sim_time']), 'map'
        for x, y in state['path']:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x, p.pose.position.y, p.pose.position.z = float(x), float(y), .04
            p.pose.orientation.w = 1.
            msg.poses.append(p)
        self.path.publish(msg)
        goal = state.get('goal')
        if self.base._nav_stack is None and goal and goal != self._last_goal:
            point = PointStamped()
            point.header = msg.header
            point.point.x, point.point.y = map(float, goal)
            self.goal.publish(point)
            self._last_goal = list(goal)
        if self.layout and self.layout.boxes:
            boxes = list(self.layout.boxes.values())
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            boundary = PolygonStamped()
            boundary.header = msg.header
            boundary.polygon.points = [Point32(x=float(x), y=float(y), z=0.)
                                       for x,y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1),(x0,y0))]
            self.boundary.publish(boundary)


def start(runtime, config, cameras, base, lidar, gripper, *, layout=None):
    """Own an isolated ROS context and executor; no global rclpy shutdown."""
    context = Context()
    context.init()
    control, control_executor, control_thread = None, None, None
    try:
        if os.environ.get('VECTOR_NAV_CONTROL') == '1':
            if base is None or lidar is None:
                raise ValueError('NAV STACK control requires a base and lidar')
            from .control import NavStack
            control = NavStack(base, context=context)
            base._nav_stack = control
            base._navigation.update(planner='far', controller='CMU pathFollower', map_source='live_lidar')
            control_executor = SingleThreadedExecutor(context=context)
            control_executor.add_node(control)
        node = SessionViz(runtime, config, cameras, base, lidar, gripper, context=context, layout=layout)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
    except BaseException:
        if control is not None:
            control.destroy_node()
            base._nav_stack = None
        context.shutdown()
        raise
    if control_executor is not None:
        control_thread = threading.Thread(target=control_executor.spin, daemon=True, name='session_nav_control')
        control_thread.start()
    thread = threading.Thread(target=executor.spin, daemon=True, name='session_sensor_viz')
    thread.start()
    def close():
        if control is not None:
            control.closed.set()
            base.stop()
            control_executor.shutdown()
            control_thread.join()
            control.destroy_node()
            base._nav_stack = None
        executor.shutdown()
        thread.join()
        node.destroy_node()
        context.shutdown()
    return close


def sensor_overview(frames):
    """Display-only RGB/depth mosaic; each tile labels its own acquisition time."""
    from PIL import Image as PILImage, ImageDraw
    canvas = PILImage.new('RGB', (320*len(frames), 520), '#101820')
    draw = ImageDraw.Draw(canvas)
    for i, frame in enumerate(frames):
        canvas.paste(PILImage.fromarray(frame.rgb).resize((320, 240)), (i*320, 20))
        draw.text((i*320+8, 4), f'{frame.camera} RGB | t={frame.sim_time:.3f}s', fill='white')
        if frame.depth is not None:
            # Fixed 0–3 m scale, with invalid returns black; never per-frame normalization.
            d = np.clip(frame.depth/3., 0., 1.)
            rgb = np.stack((d, 1.-abs(2*d-1.), 1.-d), axis=-1)
            rgb[frame.depth <= 0] = 0
            canvas.paste(PILImage.fromarray((rgb*255).astype('uint8')).resize((320, 240)), (i*320, 280))
            draw.text((i*320+8, 264), f'{frame.camera} DEPTH | blue=near red=3m', fill='white')
    msg = Image(height=520, width=canvas.width, encoding='rgb8', step=canvas.width*3, data=canvas.tobytes())
    msg.header.stamp = stamp(max(f.sim_time for f in frames))
    msg.header.frame_id = 'map'
    return msg

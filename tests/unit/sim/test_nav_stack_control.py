"""The external controller cannot fall back, replay stale velocity, or ignore stop."""
import contextlib
import math
import threading
from types import SimpleNamespace

import pytest
from geometry_msgs.msg import PointStamped, PoseStamped, TwistStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool

from tests.harness.kin_base import KinBase
from vector_os_nano.schemas.robot import NavSpec
from vector_os_nano.sim.nav.ros2.control import NavStack
from vector_os_nano.sim.nav.ros2.session_viz import stamp

NAV = NavSpec(body_radius=.42, speed=.15, vyaw_max=.2, goal_tol=.05)


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)

    def get_subscription_count(self):
        return 1


def rig(*, mode='live', stop_at=None, finish_at=None):
    base, cancel = KinBase(), threading.Event()
    node = object.__new__(NavStack)
    node.base, node.mode, node.closed = base, None, threading.Event()
    node._command = node._path = None
    node._route = node._finished = False
    node._started = 0.
    for attr in ('goal_pub', 'waypoint_pub', 'start_pub', 'applied_pub'):
        setattr(node, attr, Publisher())
    wait = base.wait_sim
    def tick(seconds, token=None):
        result = wait(seconds, token)
        now = base.get_sim_time()
        point = PointStamped(); point.header.frame_id = 'map'; point.header.stamp = stamp(now)
        if mode != 'stale_route' or now < .03:
            node._waypoint(point, node.mode)
        path = Path(); path.header.frame_id = 'base_link'; path.header.stamp = stamp(now)
        path.poses = [PoseStamped(), PoseStamped()]; path.poses[1].pose.position.x = 1.
        if mode != 'missing_path': node._local_path(path)
        cmd = TwistStamped(); cmd.header.frame_id = 'base_link'; cmd.header.stamp = stamp(now)
        cmd.twist.linear.x = .5
        if mode == 'stale': cmd.header.stamp = stamp(0.)
        if mode == 'bad_frame': cmd.header.frame_id = 'map'
        if mode == 'nan': cmd.twist.linear.x = math.nan
        node._velocity(cmd)
        if finish_at is not None and now >= finish_at: node._finish(Bool(data=True))
        if stop_at is not None and now >= stop_at: cancel.set()
        return result
    base.wait_sim = tick
    return node, base, cancel


def test_external_commands_drive_and_stop_at_original_goal_without_snapping():
    node, base, cancel = rig()
    out = node.navigate((.3,0.), nav=NAV, fallen=lambda: False, cancel=cancel, timeout=5.)
    assert out and out['planner']=='far' and not out['snapped'] and out['dist']<=.05
    assert base.cmd == (0.,0.,0.) and node.mode is None
    applied = [x.twist.linear.x for x in node.applied_pub.messages]
    assert max(applied)==.15 and applied[0]==applied[-1]==0.
    assert node.waypoint_pub.messages and node.goal_pub.messages


@pytest.mark.parametrize('mode', ['stale','bad_frame','nan','missing_path'])
def test_invalid_or_stale_stream_never_moves(mode):
    node, base, cancel = rig(mode=mode)
    out = node.navigate((1.,0.), nav=NAV, fallen=lambda: False, cancel=cancel, timeout=.2)
    assert not out and out['reason']=='timeout' and out['moved_m']==0.
    assert base.cmd == (0.,0.,0.)


def test_cancel_and_subsequent_goal_require_new_commands():
    node, base, cancel = rig(stop_at=.1)
    out = node.navigate((1.,0.), nav=NAV, fallen=lambda: False, cancel=cancel, timeout=1.)
    assert out['reason']=='aborted' and base.cmd==(0.,0.,0.)
    count = len(node.applied_pub.messages)
    cancel.clear()
    out = node.navigate((2.,0.), nav=NAV, fallen=lambda: False, cancel=cancel, timeout=1.)
    assert node.applied_pub.messages[count].twist.linear.x == 0.
    assert base.cmd == (0.,0.,0.)


def test_stop_racing_with_apply_cannot_rearm_velocity():
    node,base,cancel=rig()
    def guarded(*values):
        base.stop()
        return False
    out=node.navigate((1.,0.),nav=NAV,fallen=lambda:False,cancel=cancel,command=guarded)
    assert out['reason']=='aborted' and base.cmd==(0.,0.,0.)
    assert not node.applied_pub.messages


def test_global_planner_loss_stops_even_when_local_commands_continue():
    node,base,cancel=rig(mode='stale_route')
    out=node.navigate((1.,0.),nav=NAV,fallen=lambda:False,cancel=cancel,timeout=1.2)
    assert not out and 0<out['moved_m']<.12
    assert all(msg.twist.linear.x==0. for msg in node.applied_pub.messages[-10:])


def test_only_selected_planner_can_supply_waypoints_and_tare_drives():
    node, base, cancel = rig(finish_at=.2)
    node.mode='far'; point=PointStamped(); point.header.frame_id='map'; point.header.stamp=stamp(1.)
    node._waypoint(point,'tare')
    assert not node.waypoint_pub.messages
    out=node.explore(nav=NAV,fallen=lambda:False,cancel=cancel,timeout=1.)
    assert out and out['planner']=='tare' and out['moved_m']>0 and node.start_pub.messages
    assert not node.goal_pub.messages and base.cmd==(0.,0.,0.)


def test_frozen_clock_or_exception_releases_velocity():
    node, base, cancel = rig()
    base.wait_sim=lambda *args: False
    out=node.navigate((1.,0.),nav=NAV,fallen=lambda:False,cancel=cancel,timeout=1.)
    assert not out and base.cmd==(0.,0.,0.)
    def interrupted(*args): raise KeyboardInterrupt
    base.wait_sim=interrupted
    with pytest.raises(KeyboardInterrupt):
        node.navigate((1.,0.),nav=NAV,fallen=lambda:False,cancel=cancel,timeout=1.)
    assert base.cmd==(0.,0.,0.) and node.mode is None


@pytest.mark.parametrize('options',[{'speed':0.},{'speed':-1.},{'timeout':math.nan},{'tol':-1.}])
def test_invalid_motion_limits_fail_before_motion(options):
    node,base,cancel=rig()
    with pytest.raises(ValueError):
        node.navigate((1.,0.),nav=NAV,fallen=lambda:False,cancel=cancel,**options)
    assert base.cmd==(0.,0.,0.)


def test_base_dispatch_never_reads_authored_obstacles_for_external_stack():
    from vector_os_nano.sim.drivers.base import LeggedBase
    base=object.__new__(LeggedBase)
    base._cancel=threading.Event();base._stops=0;base._spec=SimpleNamespace(nav=NAV)
    @contextlib.contextmanager
    def control(): yield 0
    base._control=control
    seen=[]
    base._nav_stack=SimpleNamespace(navigate=lambda goal,**kwargs:seen.append((goal,kwargs)) or 'external')
    assert base.navigate_to(2.,3.)=='external'
    assert seen[0][0]==(2.,3.) and not seen[0][1]['cancel'].is_set()


def test_explore_skill_uses_live_stack_and_cancel_does_not_kill_planners(monkeypatch):
    from vector_os_nano.core.skill import SkillContext
    from vector_os_nano.skills.go2 import explore, look
    from vector_os_nano.sim.nav.follow import NavResult
    monkeypatch.setenv('VECTOR_NAV_CONTROL','1')
    monkeypatch.setattr(explore,'_explore_running',False)
    monkeypatch.setattr(look,'_world_vocabulary',lambda ctx:[])
    def forbidden(): raise AssertionError('live control must not kill its planners or fall back')
    monkeypatch.setattr(explore,'stop_tare_only',forbidden)
    monkeypatch.setattr(explore,'_tare_running',forbidden)
    def run(*,cancel):
        assert explore.is_exploring()
        explore.cancel_exploration()
        assert cancel.is_set()
        return NavResult(reached=False,reason='aborted',planner='tare')
    out=explore.ExploreSkill().execute({},SkillContext(base=KinBase(),services={'explore_tare':run}))
    assert not out.success and out.result_data['reason']=='aborted' and not explore.is_exploring()
    explore._explore_cancel.clear()

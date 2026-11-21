from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vector_os_nano.interfaces.policy import PolicyObs
from vector_os_nano.sim import policies


PARAMS = dict(wheels=[[.155,y,.04105,o] for y,o in ((.155,.026),(-.155,-.026))],
              steering_range=[-2.5,2.5], steering_kp=20.,steering_kd=1.,wheel_kp=.4,
              steering_speed=1.,wheel_ki=.4,wheel_speed=10.47,accel=.4,yaw_accel=1.)


def setup(command):
    policy = policies.make('swerve', SimpleNamespace(joints=('s0','w0','s1','w1'),
                            policy=SimpleNamespace(tick_hz=100)), PARAMS, np.array([8.4,2,8.4,2]),Path('.'))
    obs = PolicyObs(0.,np.zeros(3),np.array([1.,0,0,0]),np.zeros(3),np.zeros(3),
                    np.zeros(4),np.zeros(4),np.array(command,dtype=float))
    policy.reset(obs)
    return policy, obs


@pytest.mark.parametrize('speed',[.25,-.25])
def test_forward_and_reverse_keep_wheels_facing_forward(speed):
    policy,obs = setup([speed,0,0])
    for i in range(1,101):
        torque = policy.torques(replace(obs,time=i*.01))
    np.testing.assert_allclose(torque[::2],0,atol=1e-8)
    assert np.all(torque[1::2]*speed > 0)
    assert np.all(np.abs(torque)<=np.array([8.4,2,8.4,2]))


def test_reverse_brakes_without_steering_through_joint_stops():
    policy,obs = setup([-.25,0,0])
    tau = policy.torques(replace(obs,time=.01,joint_pos=np.array([2.49,0,-2.49,0.])))
    assert tau[0] < 0 < tau[2]
    moving = replace(obs,time=.02,command=np.zeros(3),base_lin_vel=np.array([.1,0,0]))
    policy.torques(moving)
    assert policy.braking


def test_zero_command_damps_rolling_wheels_and_invalid_observations_fail():
    policy,obs = setup([0,0,0])
    tau = policy.torques(replace(obs,time=.01,joint_vel=np.array([0,3,0,-3.])))
    assert tau[1] < 0 < tau[3]
    with pytest.raises(ValueError,match='finite'):
        policy.torques(replace(obs,command=np.array([np.nan,0,0])))


def test_bad_wheel_radius_fails_at_manifest_boundary():
    with pytest.raises(ValueError,match='positive radius'):
        policies.owned('swerve',{**PARAMS,'wheels':[[0,0,0,0]]})


def test_steering_reference_is_rate_limited_and_saturated_integral_does_not_wind_up():
    policy,obs = setup([0,.2,0])
    first = policy.torques(replace(obs,time=.01))
    assert np.max(np.abs(first[::2])) <= PARAMS['steering_kp']*PARAMS['steering_speed']*.01
    policy,obs = setup([100,0,0])
    for i in range(1,1001):
        policy.torques(replace(obs,time=i*.01))
    policy.torques(replace(obs,time=10.01,command=np.zeros(3),joint_vel=np.array([0,100,0,100.])))
    assert np.max(np.abs(policy._integral)) <= 2
    policy.reset(obs)
    np.testing.assert_array_equal(policy._integral,0)

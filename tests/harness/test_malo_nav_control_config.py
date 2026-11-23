"""The three installed navigation layers must not stop outside the CLI tolerance."""
from pathlib import Path

import yaml


def test_malo_goal_tolerances_and_control_envelope_agree():
    root=Path(__file__).resolve().parents[2]
    config=root/'examples/household/navigation/malo'
    robot=yaml.safe_load((root/'examples/household/embodiments/malo/robot.yaml').read_text())
    local=yaml.safe_load((config/'local.yaml').read_text())
    far=yaml.safe_load((config/'far.yaml').read_text())['far_planner']['ros__parameters']
    nav=robot['base']['nav']
    planner=local['localPlanner']['ros__parameters']
    follower=local['pathFollower']['ros__parameters']
    assert 0<planner['goalReachedThreshold']<nav['goal_tol']<=.15
    assert 0<follower['stopDisThre']<nav['goal_tol']
    assert 0<far['g_planner/converge_distance']<=nav['goal_tol']
    assert planner['vehicleWidth']>=2*nav['body_radius']
    assert planner['maxSpeed']<=nav['speed'] and follower['maxSpeed']<=nav['speed']
    assert follower['realRobot'] is False
    assert yaml.safe_load((config/'tare.yaml').read_text())['tare_planner_node']['ros__parameters']['kAutoStart'] is False

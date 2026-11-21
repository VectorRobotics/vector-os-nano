# SPDX-License-Identifier: Apache-2.0
"""Bind configured robot devices to the single shared physics runtime."""
from typing import Any, Callable


def joint_groups(cfg: Any, rt: Any, *, abort: Callable[[], bool]) -> dict:
    """Validate all group poses against scalar model limits before exposing commands."""
    from vector_os_nano.sim.drivers.joints import JointGroup
    from vector_os_nano.sim.scene import reader

    model = rt.world.scene.model
    joints = {j.name: j for j in reader.joints(model)}
    actuators = {a.joint: a for a in reader.actuators(model) if a.kind == 'position'}
    for name, spec in cfg.joint_groups.items():
        for i, joint in enumerate(spec.joints):
            j = joints[joint]
            a = actuators.get(j.id)
            if j.type not in (reader.JNT_HINGE, reader.JNT_SLIDE) or a is None:
                raise ValueError(f'{cfg.path}.joint_groups.{name}: {joint} needs a scalar position servo')
            for pose, q in spec.postures.items():
                if ((j.range and not j.range[0] <= q[i] <= j.range[1]) or
                        (a.ctrlrange and not a.ctrlrange[0] <= q[i] * a.gear <= a.ctrlrange[1])):
                    raise ValueError(f'{cfg.path}.joint_groups.{name}.{pose}: {joint} outside model limits')
    return {name: JointGroup(spec, rt, abort=abort) for name, spec in cfg.joint_groups.items()}


def devices(cfg: Any, rt: Any, *, wait_mount: Callable[[], Any] | None = None,
             abort: Callable[[], bool] = lambda: False) -> tuple[Any, Any, Any]:
    """``(arm, gripper, camera)`` of *cfg* on *rt*: the generic drivers, None where
    the manifest has no such block."""
    from vector_os_nano.sim.drivers.arm import Arm
    from vector_os_nano.sim.drivers.camera import Camera
    from vector_os_nano.sim.drivers.gripper import Gripper

    from vector_os_nano.sim.scene.compose import check_fingers

    camera = Camera(cfg.camera, rt) if cfg.camera is not None else None
    if cfg.arm is None:
        return None, None, camera
    arm = Arm(cfg.arm, rt, abort=abort, wait_mount=wait_mount)
    if cfg.arm.gripper is None:
        return arm, None, camera
    check_fingers(rt.world.scene, cfg.arm.gripper.finger_bodies)
    return arm, Gripper(cfg.arm.gripper, rt), camera



# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Build combined Go2 + Piper MJCF via MuJoCo MjSpec attach API.

Regenerate when mount pose (xyz / orientation) needs tuning, or when
upstream Go2 / Piper assets update.

Usage:
    cd vector_os_nano/hardware/sim/mjcf
    python3 go2_piper/build_go2_piper.py
"""
from __future__ import annotations

import os
from pathlib import Path

import mujoco

# Mount pose — Piper base origin relative to Go2 base_link
# x=-0.02: 15cm behind the Livox MID-360 lidar (lidar at x=0.15)
# y=0:    centerline
# z=0.06: top of trunk collision box (trunk half-height = 0.057)
MOUNT_POS = [-0.02, 0.0, 0.06]
MOUNT_QUAT = [1.0, 0.0, 0.0, 0.0]  # identity — Z-up, X-forward (same as Go2 trunk)

# Piper body mass scale. Set to 1.0 to use the real ~2kg arm mass from
# Menagerie — this is required for Piper's own position actuators
# (kp=80, kv=5) to stay numerically stable in stow pose. Scaling down
# causes under-damped oscillation and drifts the arm out of home.
# Real-mass arm is fine for Go2 locomotion (verified by direct Python
# test: full-mass Piper + Go2 walks 0.75m in 3s under sinusoidal gait).
PIPER_MASS_SCALE: float = 1.0


def main() -> None:
    this_dir = Path(__file__).resolve().parent      # .../mjcf/go2_piper
    mjcf_root = this_dir.parent                     # .../mjcf

    go2 = mujoco.MjSpec.from_file(str(mjcf_root / "go2" / "go2.xml"))
    piper = mujoco.MjSpec.from_file(str(mjcf_root / "piper" / "piper.xml"))

    # Dynamically decouple Piper from Go2 locomotion: scale all Piper link
    # masses + inertias by PIPER_MASS_SCALE. With gravcomp=1 already set on
    # every Piper body, the arm holds pose under gravity, and the tiny mass
    # means arm motion does not disturb the dog's trot.
    if PIPER_MASS_SCALE != 1.0:
        for body in piper.bodies:
            if body.mass > 0.0:
                body.mass = float(body.mass) * PIPER_MASS_SCALE
                body.inertia = [float(v) * PIPER_MASS_SCALE for v in body.inertia]

    # Rewrite mesh file paths RELATIVE to this directory (where go2_piper.xml is
    # written). Absolute paths only work on the generating machine; consumers go
    # through scene_builder.build_room_scene, which resolves them at load time.
    go2_assets = Path(os.path.relpath(mjcf_root / "go2" / "assets", this_dir))
    piper_assets = Path(os.path.relpath(mjcf_root / "piper" / "assets", this_dir))
    for mesh in go2.meshes:
        if not os.path.isabs(mesh.file):
            mesh.file = str(go2_assets / mesh.file)
    for mesh in piper.meshes:
        if not os.path.isabs(mesh.file):
            mesh.file = str(piper_assets / mesh.file)
    go2.meshdir = ""
    piper.meshdir = ""

    # Attach Piper at the mount frame inside Go2's base_link
    go2_base = go2.body("base_link")
    mount_frame = go2_base.add_frame(pos=MOUNT_POS, quat=MOUNT_QUAT)
    go2.attach(piper, prefix="piper_", frame=mount_frame)
    go2.compile()

    out = this_dir / "go2_piper.xml"
    out.write_text(go2.to_xml())
    print(f"Wrote: {out}")

    model = mujoco.MjModel.from_xml_path(str(out))
    print(f"Sanity: nbody={model.nbody}, njnt={model.njnt}, nu={model.nu}")


if __name__ == "__main__":
    main()

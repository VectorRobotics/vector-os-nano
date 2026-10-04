# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Sim-tier preflight: can this box run the MuJoCo sim tests at all?

Run as ``python -m tests.harness.sim_preflight`` (tests/conftest.py does, once per
session, in a child so an EGL/MuJoCo crash cannot kill pytest). Prints ``''`` when
the sim tier can run, else ONE line naming every missing prerequisite. Checks:

- the git-TRACKED ``vector_os_nano`` + ``config`` trees, copied to a temp dir, import and build
  the go2 and go2_piper room scenes there (catches absolute mesh paths and files
  that exist locally but were never committed, e.g. an ignored ``assets/``);
- the vendored convex_mpc stack (casadi + the Pinocchio go2 model) initialises;
- an EGL ``mujoco.Renderer`` opens;
- the grounding-dino weights are in the HF cache (perception never downloads).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
GDINO_REPO = "IDEA-Research/grounding-dino-tiny"


def _copy_tracked_tree(dst: Path) -> None:
    try:
        files = subprocess.run(
            ["git", "ls-files", "-z", "vector_os_nano", "config"], cwd=_REPO_ROOT,
            capture_output=True, check=True, timeout=30,
        ).stdout.decode().split("\0")
    except (OSError, subprocess.SubprocessError):
        for top in ("vector_os_nano", "config"):  # the scene build reads config/worlds
            shutil.copytree(_REPO_ROOT / top, dst / top)
        return
    for rel in filter(None, files):
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO_ROOT / rel, dst / rel)


def missing() -> list[str]:
    problems: list[str] = []
    try:
        import mujoco
    except Exception as exc:  # noqa: BLE001 — e.g. an invalid MUJOCO_GL
        return [f"mujoco unusable ({exc})"]
    tmp = Path(tempfile.mkdtemp(prefix="vos_preflight_"))
    try:
        _copy_tracked_tree(tmp)
        sys.path.insert(0, str(tmp))
        try:
            from vector_os_nano.hardware.sim import mujoco_go2

            assert Path(mujoco_go2.__file__).is_relative_to(tmp), mujoco_go2.__file__
            for with_arm in (False, True):
                try:
                    mujoco.MjModel.from_xml_path(str(mujoco_go2._build_room_scene_xml(with_arm=with_arm)))
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{'go2_piper' if with_arm else 'go2'} scene does not compile from a clean checkout ({exc})")
            try:
                import casadi  # noqa: F401
                from vector_os_nano.third_party.convex_mpc.go2_robot_data import PinGo2Model

                PinGo2Model()
            except Exception as exc:  # noqa: BLE001
                problems.append(f"vendored convex_mpc unusable ({type(exc).__name__}: {exc})")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"sim package does not import from a clean checkout ({exc})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light/></worldbody></mujoco>")
        mujoco.Renderer(model, 32, 32).close()
    except Exception as exc:  # noqa: BLE001
        problems.append(f"no offscreen GL (MUJOCO_GL={os.environ.get('MUJOCO_GL')}: {exc})")
    try:
        from huggingface_hub import try_to_load_from_cache

        if not isinstance(try_to_load_from_cache(GDINO_REPO, "config.json"), str):
            problems.append(f"{GDINO_REPO} weights not in the HF cache")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"HF cache unreadable ({exc})")
    return problems


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    print("; ".join(missing()))

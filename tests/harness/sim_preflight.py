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
- the default perception stack's pinned weights are in the HF cache (the gate runs
  with HF_HUB_OFFLINE=1, so a test never downloads).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


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
            from vector_os_nano.embodiments import registry
            from vector_os_nano.hardware.sim import scene_builder
            from vector_os_nano.schemas.world import load_world

            assert Path(scene_builder.__file__).is_relative_to(tmp), scene_builder.__file__
            for body in ("go2", "go2_piper"):
                try:
                    cfg = registry.load(body)
                    scene_builder.room_scene(cfg, load_world(cfg.world))
                except Exception as exc:  # noqa: BLE001
                    problems.append(f"{body} scene does not compile from a clean checkout ({exc})")
            try:
                import casadi  # noqa: F401
                from vector_os_nano.third_party.convex_mpc.go2_robot_data import PinGo2Model

                PinGo2Model()
            except Exception as exc:  # noqa: BLE001
                problems.append(f"vendored convex_mpc unusable ({type(exc).__name__}: {exc})")
            try:
                from vector_os_nano.perception import weights

                problems += [f"{w} weights not in the HF cache ({weights.FETCH_COMMAND})"
                             for w in weights.default_stack() if not weights.cached_dir(w)]
            except Exception as exc:  # noqa: BLE001
                problems.append(f"HF cache unreadable ({type(exc).__name__}: {exc})")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"sim package does not import from a clean checkout ({exc})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    try:
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light/></worldbody></mujoco>")
        mujoco.Renderer(model, 32, 32).close()
    except Exception as exc:  # noqa: BLE001
        problems.append(f"no offscreen GL (MUJOCO_GL={os.environ.get('MUJOCO_GL')}: {exc})")
    return problems


if __name__ == "__main__":
    os.environ.setdefault("MUJOCO_GL", "egl")
    print("; ".join(missing()))

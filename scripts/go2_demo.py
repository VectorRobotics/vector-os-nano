#!/usr/bin/env python3
"""Go2 locomotion visual demo — MuJoCo viewer.

Usage:
    cd ~/Desktop/vector_os_nano
    .venv-nano/bin/python scripts/go2_demo.py              # auto: MPC if available, else sinusoidal
    .venv-nano/bin/python scripts/go2_demo.py --mpc        # force MPC backend
    .venv-nano/bin/python scripts/go2_demo.py --sinusoidal # force sinusoidal backend
"""
import sys
import types
import time
import math
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: stub vector_os_nano packages so we can import mujoco_go2
# without triggering the full package __init__ (which needs httpx etc.)
# ---------------------------------------------------------------------------

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo))

pkg = types.ModuleType("vector_os_nano")
pkg.__path__ = [str(_repo / "vector_os_nano")]
pkg.__package__ = "vector_os_nano"
sys.modules["vector_os_nano"] = pkg

core = types.ModuleType("vector_os_nano.core")
core.__path__ = [str(_repo / "vector_os_nano" / "core")]
core.__package__ = "vector_os_nano.core"
sys.modules["vector_os_nano.core"] = core

hw = types.ModuleType("vector_os_nano.hardware")
hw.__path__ = [str(_repo / "vector_os_nano" / "hardware")]
sys.modules["vector_os_nano.hardware"] = hw

sim = types.ModuleType("vector_os_nano.hardware.sim")
sim.__path__ = [str(_repo / "vector_os_nano" / "hardware" / "sim")]
sys.modules["vector_os_nano.hardware.sim"] = sim


@dataclass(frozen=True)
class Odometry:
    timestamp: float = 0.0
    x: float = 0.0; y: float = 0.0; z: float = 0.0
    qx: float = 0.0; qy: float = 0.0; qz: float = 0.0; qw: float = 1.0
    vx: float = 0.0; vy: float = 0.0; vz: float = 0.0; vyaw: float = 0.0


@dataclass(frozen=True)
class LaserScan:
    timestamp: float = 0.0
    angle_min: float = 0.0; angle_max: float = 0.0
    angle_increment: float = 0.0
    range_min: float = 0.0; range_max: float = 0.0
    ranges: tuple = ()


ct = types.ModuleType("vector_os_nano.core.types")
ct.Odometry = Odometry
ct.LaserScan = LaserScan
sys.modules["vector_os_nano.core.types"] = ct

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

from vector_os_nano.hardware.sim.mujoco_go2 import MuJoCoGo2  # noqa: E402


def _pos_str(go2):
    p = go2.get_position()
    h = math.degrees(go2.get_heading())
    return f"({p[0]:.1f}, {p[1]:.1f})  heading={h:.0f} deg"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Go2 locomotion demo")
    parser.add_argument("--mpc", action="store_true", help="Force MPC backend")
    parser.add_argument("--sinusoidal", action="store_true", help="Force sinusoidal backend")
    parser.add_argument("--flat", action="store_true", help="Use flat ground instead of house")
    args = parser.parse_args()

    backend = "auto"
    if args.mpc:
        backend = "mpc"
    elif args.sinusoidal:
        backend = "sinusoidal"

    print(f"=== Go2 Locomotion Demo (backend={backend}) ===")
    print("Loading scene with MuJoCo viewer...")
    go2 = MuJoCoGo2(gui=True, room=not args.flat, backend=backend)
    go2.connect()

    print("[1/7] Standing up...")
    go2.stand(duration=2.0)
    print(f"      {_pos_str(go2)}")

    print("[2/7] Walking forward 5s...")
    go2.walk(vx=0.3, duration=5.0)
    print(f"      {_pos_str(go2)}")

    print("[3/7] Turning left 3s...")
    go2.walk(vyaw=1.0, duration=3.0)
    print(f"      {_pos_str(go2)}")

    print("[4/7] Walking forward 5s...")
    go2.walk(vx=0.3, duration=5.0)
    print(f"      {_pos_str(go2)}")

    print("[5/7] Turning right 3s...")
    go2.walk(vyaw=-1.0, duration=3.0)
    print(f"      {_pos_str(go2)}")

    print("[6/7] Walking backward 3s...")
    go2.walk(vx=-0.3, duration=3.0)
    print(f"      {_pos_str(go2)}")

    print("[7/7] Sitting down...")
    go2.sit(duration=2.0)
    print(f"      {_pos_str(go2)}")

    print("\nDemo complete. Close the viewer window to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    go2.disconnect()


if __name__ == "__main__":
    main()

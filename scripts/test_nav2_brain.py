#!/usr/bin/env python3
"""Integration test: Vector OS Nano brain controlling Nav2 in Gazebo.

Prerequisites:
    Terminal 1: ros2 launch vector_go2_gazebo full_stack.launch.py mode:=nav
    Terminal 2: python3 scripts/test_nav2_brain.py [--rooms] [--single x y]

Tests:
    Default (--rooms): Navigate to 5 rooms sequentially, verify arrival.
    --single x y: Navigate to a specific coordinate.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import threading

import rclpy
from rclpy.node import Node

sys.path.insert(0, ".")
from vector_os_nano.core.nav_client import NavStackClient

# Room targets (matching NavigateSkill room database)
ROOMS: list[tuple[str, float, float]] = [
    ("hallway",        10.0,  5.0),
    ("kitchen",        17.0,  2.5),
    ("living_room",     3.0,  2.5),
    ("master_bedroom",  3.5, 12.0),
    ("guest_bedroom",  16.0, 12.0),
]


def wait_for_state_estimation(
    nav: NavStackClient, timeout: float = 10.0
) -> bool:
    """Block until state estimation is received or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        odom = nav.get_state_estimation()
        if odom is not None:
            print(f"[OK] State estimation: ({odom.x:.1f}, {odom.y:.1f}, z={odom.z:.2f})")
            return True
        time.sleep(0.5)
    print("[FAIL] No state estimation received")
    return False


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def navigate_and_verify(
    nav: NavStackClient,
    name: str,
    target_x: float,
    target_y: float,
    timeout: float = 60.0,
    arrival_radius: float = 1.0,
) -> bool:
    """Navigate to target and verify arrival within radius."""
    odom = nav.get_state_estimation()
    start_dist = distance(odom.x, odom.y, target_x, target_y) if odom else float("inf")
    print(f"\n{'='*60}")
    print(f"[NAV] {name}: ({target_x:.1f}, {target_y:.1f})  start_dist={start_dist:.1f}m")
    print(f"{'='*60}")

    start = time.time()
    result = nav.navigate_to(target_x, target_y, timeout=timeout)
    elapsed = time.time() - start

    odom = nav.get_state_estimation()
    if odom:
        final_dist = distance(odom.x, odom.y, target_x, target_y)
        arrived = final_dist < arrival_radius
        status = "OK" if arrived else "FAIL"
        print(f"[{status}] {name}: result={result}, final_dist={final_dist:.2f}m, "
              f"elapsed={elapsed:.1f}s, pos=({odom.x:.1f}, {odom.y:.1f})")
        return arrived
    else:
        print(f"[FAIL] {name}: no state estimation after navigation")
        return False


def test_rooms(nav: NavStackClient) -> None:
    """Navigate to multiple rooms sequentially."""
    results: list[tuple[str, bool]] = []

    for name, x, y in ROOMS:
        ok = navigate_and_verify(nav, name, x, y, timeout=60.0)
        results.append((name, ok))
        if not ok:
            print(f"[WARN] Failed to reach {name}, continuing...")
        time.sleep(2.0)  # brief pause between goals

    # Summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    passed = 0
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if ok:
            passed += 1
    print(f"\n{passed}/{len(results)} rooms reached successfully")

    if passed < len(results):
        sys.exit(1)


def test_single(nav: NavStackClient, x: float, y: float) -> None:
    """Navigate to a single coordinate."""
    ok = navigate_and_verify(nav, f"target({x},{y})", x, y, timeout=60.0)
    if not ok:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nav2 brain integration test")
    parser.add_argument("--rooms", action="store_true", default=True,
                        help="Navigate to multiple rooms (default)")
    parser.add_argument("--single", nargs=2, type=float, metavar=("X", "Y"),
                        help="Navigate to a single coordinate")
    parser.add_argument("--mode", default="auto", choices=["auto", "nav2", "cmu"],
                        help="NavStackClient mode (default: auto)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Default navigation timeout in seconds")
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node("vector_nav2_brain_test")

    nav = NavStackClient(node=node, mode=args.mode, timeout=args.timeout)
    print(f"NavStackClient mode={nav.mode}, available={nav.is_available}")

    if not nav.is_available:
        print("[FAIL] NavStackClient not available. Is Nav2 running?")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    # Spin in background thread
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # Wait for state estimation
    if not wait_for_state_estimation(nav, timeout=15.0):
        print("[FAIL] Cannot proceed without state estimation")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    try:
        if args.single:
            test_single(nav, args.single[0], args.single[1])
        else:
            test_rooms(nav)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        nav.cancel()
    finally:
        print("Shutting down...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

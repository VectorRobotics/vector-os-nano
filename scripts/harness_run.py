#!/usr/bin/env python3
"""Go2 Locomotion Harness Runner.

Runs the 5-level locomotion benchmark (L0–L4) and reports a summary table.

Usage:
    python scripts/harness_run.py [--level N] [--verbose] [--no-capture]

Options:
    --level N       Run only level N (0-4). Default: all levels.
    --verbose       Pass -v to pytest for verbose test output.
    --no-capture    Pass -s to pytest (show print/log output).
    --timeout N     Per-test timeout in seconds (default: 60).
    --help          Show this help message.

Exit code:
    0   All selected levels passed.
    1   One or more levels failed or errored.
    2   mujoco not installed (harness cannot run).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HARNESS_DIR = _REPO_ROOT / "tests" / "harness"

_LEVELS: dict[int, dict] = {
    0: {
        "name": "Physics Validity",
        "file": "test_level0_physics.py",
        "marker": "level0",
        "description": "MuJoCo model loads, actuators respond, gravity works",
    },
    1: {
        "name": "Standing Stability",
        "file": "test_level1_standing.py",
        "marker": "level1",
        "description": "PD controller holds upright standing posture",
    },
    2: {
        "name": "Open-Loop Walking",
        "file": "test_level2_walking.py",
        "marker": "level2",
        "description": "Locomotion produces displacement without falling",
    },
    3: {
        "name": "Velocity Tracking",
        "file": "test_level3_velocity.py",
        "marker": "level3",
        "description": "set_velocity() achieves commanded velocities",
    },
    4: {
        "name": "Navigation (P-ctrl)",
        "file": "test_level4_navigation.py",
        "marker": "level4",
        "description": "Proportional controller reaches 2m waypoint",
    },
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class LevelResult(NamedTuple):
    level: int
    name: str
    status: str       # PASS | FAIL | ERROR | SKIP
    passed: int
    failed: int
    errors: int
    duration_s: float
    skipped: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_mujoco() -> bool:
    result = subprocess.run(
        [sys.executable, "-c", "import mujoco"],
        capture_output=True,
    )
    return result.returncode == 0


def _has_pytest_timeout() -> bool:
    """Return True if pytest-timeout plugin is installed."""
    result = subprocess.run(
        [sys.executable, "-c", "import pytest_timeout"],
        capture_output=True,
    )
    return result.returncode == 0


_PYTEST_TIMEOUT_AVAILABLE = _has_pytest_timeout()


def _run_level(level_id: int, verbose: bool, no_capture: bool, timeout: int) -> LevelResult:
    """Run a single harness level and return structured results."""
    info = _LEVELS[level_id]
    test_file = _HARNESS_DIR / info["file"]

    if not test_file.exists():
        return LevelResult(
            level=level_id,
            name=info["name"],
            status="ERROR",
            passed=0, failed=0, errors=1, skipped=0,
            duration_s=0.0,
        )

    cmd = [
        sys.executable, "-m", "pytest",
        str(test_file),
        "--tb=short",
        "-q",  # quiet by default; -v overrides below
    ]
    # Only add --timeout if pytest-timeout is installed
    if _PYTEST_TIMEOUT_AVAILABLE:
        cmd.append(f"--timeout={timeout}")
    if verbose:
        cmd.append("-v")
    if no_capture:
        cmd.append("-s")

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=not (verbose or no_capture),
        text=True,
    )
    elapsed = time.perf_counter() - t0

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    combined = stdout + stderr

    # Parse pytest summary line: "X passed, Y failed, Z error in Ns"
    passed = _parse_count(combined, "passed")
    failed = _parse_count(combined, "failed")
    errors = _parse_count(combined, "error")
    skipped = _parse_count(combined, "skipped")

    if result.returncode == 5:
        # pytest exit code 5 = no tests collected
        status = "SKIP"
    elif failed > 0 or errors > 0:
        status = "FAIL"
    elif result.returncode != 0:
        status = "ERROR"
    else:
        status = "PASS"

    if verbose or no_capture:
        # Output already printed to terminal by pytest
        pass
    else:
        # Print captured output only on failure
        if status in ("FAIL", "ERROR"):
            print(combined)

    return LevelResult(
        level=level_id,
        name=info["name"],
        status=status,
        passed=passed,
        failed=failed,
        errors=errors,
        skipped=skipped,
        duration_s=elapsed,
    )


def _parse_count(text: str, keyword: str) -> int:
    """Extract 'N keyword' from pytest output."""
    import re
    match = re.search(rf"(\d+)\s+{keyword}", text)
    return int(match.group(1)) if match else 0


def _print_summary(results: list[LevelResult]) -> None:
    """Print a formatted summary table."""
    col_widths = [5, 24, 8, 8, 8, 8, 10]
    header = ["Level", "Name", "Status", "Passed", "Failed", "Errors", "Duration"]

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    row_fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_widths) + " |"

    print()
    print("=" * 80)
    print("  Go2 Locomotion Harness — Results")
    print("=" * 80)
    print(sep)
    print(row_fmt.format(*header))
    print(sep)

    all_passed = True
    for r in results:
        status_label = r.status
        if r.status == "PASS":
            status_indicator = "[PASS]"
        elif r.status == "FAIL":
            status_indicator = "[FAIL]"
            all_passed = False
        elif r.status == "ERROR":
            status_indicator = "[ERR ]"
            all_passed = False
        else:
            status_indicator = "[SKIP]"

        duration_str = f"{r.duration_s:.1f}s"
        print(row_fmt.format(
            f"L{r.level}",
            r.name,
            status_indicator,
            str(r.passed),
            str(r.failed),
            str(r.errors),
            duration_str,
        ))

    print(sep)
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total_errors = sum(r.errors for r in results)
    total_time = sum(r.duration_s for r in results)
    print(row_fmt.format(
        "TOTAL",
        "",
        "[PASS]" if all_passed else "[FAIL]",
        str(total_passed),
        str(total_failed),
        str(total_errors),
        f"{total_time:.1f}s",
    ))
    print(sep)
    print()

    if all_passed:
        print("  All levels PASSED. Locomotion harness verified.")
    else:
        failed_levels = [f"L{r.level} ({r.name})" for r in results if r.status in ("FAIL", "ERROR")]
        print(f"  FAILED levels: {', '.join(failed_levels)}")
        print("  Review test output above for details.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Go2 locomotion harness benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--level", type=int, choices=list(_LEVELS.keys()),
        help="Run only this level (0-4). Default: all.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose pytest output.")
    parser.add_argument("--no-capture", "-s", action="store_true", help="Show stdout/stderr.")
    parser.add_argument("--timeout", type=int, default=60, help="Per-test timeout in seconds.")
    args = parser.parse_args()

    # Check mujoco availability first
    if not _check_mujoco():
        print("ERROR: mujoco is not installed. Install it with:")
        print("  pip install mujoco")
        return 2

    levels_to_run = [args.level] if args.level is not None else list(_LEVELS.keys())

    print(f"\nRunning locomotion harness — levels: {levels_to_run}")
    print(f"Test directory: {_HARNESS_DIR}")
    print(f"Per-test timeout: {args.timeout}s\n")

    results: list[LevelResult] = []
    all_ok = True

    for level_id in levels_to_run:
        info = _LEVELS[level_id]
        print(f"  L{level_id}: {info['name']} — {info['description']}")
        r = _run_level(level_id, verbose=args.verbose, no_capture=args.no_capture,
                       timeout=args.timeout)
        results.append(r)

        if r.status in ("FAIL", "ERROR"):
            all_ok = False
            print(f"       => FAILED ({r.failed} failed, {r.errors} errors)")
            # Continue running remaining levels (full report is more useful)
        else:
            print(f"       => {r.status} ({r.passed} passed in {r.duration_s:.1f}s)")

    _print_summary(results)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

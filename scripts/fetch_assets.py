#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Fetch what the package does not ship (interface charter section 5).

  python scripts/fetch_assets.py <robot> [--dest DIR] [--check]
      The robot manifest's ``assets:`` block: the pinned upstream revision is
      fetched with git into a temporary directory, exactly the listed files are
      copied into ~/.vector/assets/<name> (or --dest), every sha256 is verified and
      the upstream licence is written beside them. Verified files are left alone;
      --check only verifies (exit 1 with the command to run). A checksum mismatch
      names the file and deletes nothing.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from vector_os_nano.embodiments import assets
from vector_os_nano.embodiments.config import AssetsSpec, load_embodiment_config


def _git_checkout(spec: AssetsSpec, into: Path) -> Path:
    """Only the files *spec* names, at its pinned revision (a blobless sparse fetch)."""
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(into), *args], check=True)

    paths = sorted({spec.upstream(rel) for rel in spec.sha256} | {spec.license["file"]})
    subprocess.run(["git", "init", "-q", str(into)], check=True)
    git("sparse-checkout", "set", "--no-cone", *paths)
    git("fetch", "-q", "--depth", "1", "--filter=blob:none", spec.git, spec.rev)
    git("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD")
    return into


def fetch_robot(robot: str, dest: Path | None, check: bool) -> int:
    spec = load_embodiment_config(robot).assets
    if spec is None:
        print(f"{robot} ships its model and policy in the package: nothing to fetch")
        return 0
    dest = dest or assets.fetched_dir(spec)
    bad = assets.problems(spec, dest)
    if not bad:
        print(f"{robot}: {len(spec.sha256)} files verified in {dest}")
        return 0
    if check:
        print(f"{robot}: {len(bad)} of {len(spec.sha256)} files in {dest} not verified "
              f"(e.g. {next(iter(bad))} {next(iter(bad.values()))}); fetch them with: "
              f"{assets.fetch_command(robot)}", file=sys.stderr)
        return 1
    altered = [rel for rel, why in bad.items() if why == assets.MISMATCH]
    if altered:
        print(f"{robot}: {', '.join(altered)} in {dest} {assets.MISMATCH}; it is left untouched: "
              "move it aside and run this again", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory(prefix="vos_fetch_") as tmp:
        print(f"{robot}: fetching {spec.git} @ {spec.rev[:12]} ({len(bad)} files) ...")
        src = _git_checkout(spec, Path(tmp) / "src")
        upstream_bad = assets.problems(spec, _staged(spec, src, Path(tmp) / "stage"))
        if upstream_bad:
            rel, why = next(iter(upstream_bad.items()))
            print(f"{robot}: upstream {spec.upstream(rel)} @ {spec.rev[:12]} {why} ({rel}); "
                  f"nothing was written to {dest}", file=sys.stderr)
            return 1
        for rel in bad:
            _install(Path(tmp) / "stage" / rel, dest / rel)
        _install(src / spec.license["file"], dest / "LICENSE")
    left = assets.problems(spec, dest)
    if left:  # a concurrent writer; never claim success
        print(f"{robot}: {dest} still not verified: {left}", file=sys.stderr)
        return 1
    print(f"{robot}: {len(spec.sha256)} files verified in {dest} ({spec.license['spdx']}, "
          f"licence in {dest / 'LICENSE'})")
    return 0


def _staged(spec: AssetsSpec, src: Path, stage: Path) -> Path:
    """Copy the listed files out of the checkout under their asset-folder names."""
    for rel in spec.sha256:
        if (src / spec.upstream(rel)).is_file():
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / spec.upstream(rel), stage / rel)
    return stage


def _install(src: Path, dst: Path) -> None:
    """Atomic per file: a crash never leaves a half-written file under its real name."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robot", help="embodiment id whose manifest has an assets: block")
    ap.add_argument("--dest", type=Path, help="asset folder (default: ~/.vector/assets/<name>)")
    ap.add_argument("--check", action="store_true", help="only verify; exit 1 if not provisioned")
    args = ap.parse_args(argv)
    return fetch_robot(args.robot, args.dest, args.check)


if __name__ == "__main__":
    sys.exit(main())

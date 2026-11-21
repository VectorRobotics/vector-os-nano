# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Regenerable scene binaries keyed by source, asset, manifest and compiler contents."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco as mj

from vector_os_nano.embodiments.assets import model_root
from vector_os_nano.schemas import paths
from vector_os_nano.schemas.robot import EmbodimentConfig
from vector_os_nano.schemas.world import WorldManifest


def cache_path(robot: EmbodimentConfig, world: WorldManifest, spec) -> Path:
    """Hash actual bytes, including included XML and meshes/textures outside model folders."""
    sources = {world.scene, Path(__file__), Path(__file__).with_name('compose.py'),
               Path(__file__).with_name('compiled.py')}

    def models(cfg):
        if cfg.compose:
            c = cfg.compose
            return (cfg.model, models(c.base), models(c.arm), c.body, c.pos, c.quat, c.prefix)
        sources.add(model_root(cfg) / cfg.model.file)
        return cfg.model

    model = models(robot)
    for kind in ('meshes', 'textures', 'hfields', 'skins'):
        for asset in getattr(spec, kind):
            if asset.file:
                sources.add(Path(asset.file))
            if kind == 'textures':
                sources.update(Path(f) for f in asset.cubefiles if f)
    # Only values consumed by compose() belong here; controller tuning and map labels do not change physics.
    cameras = tuple((c.name, c.body, c.pos, c.quat, c.fovy) for c in robot.sensors
                    if c.role == 'camera' and c.pos is not None)
    gripper = robot.arm.gripper if robot.arm else None
    pairs = (gripper.weld_body, tuple(world.pickables)) if gripper and gripper.assist else ()
    timestep = robot.base.policy.timestep if robot.base else None
    identity = (model, cameras, world.spawn, pairs, timestep)
    digest = hashlib.sha256(f'{mj.__version__}\0{identity!r}'.encode())
    seen = set()

    def include(path):
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        digest.update(os.fsencode(path) + b'\0')
        with path.open('rb') as f:
            while chunk := f.read(1024 * 1024):
                digest.update(chunk)
        if path.suffix.lower() in ('.xml', '.urdf'):
            for child in ET.parse(path).iter('include'):
                include(path.parent / child.attrib['file'])

    for path in sorted({p.resolve() for p in sources}):
        include(path)
    return paths.scene_cache() / f'scene_{digest.hexdigest()}.xml'


def atomic_write(path: Path, writer) -> None:
    """Publish a complete XML or MJB; concurrent builders never expose partial bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.stem}-', suffix='.tmp')
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)

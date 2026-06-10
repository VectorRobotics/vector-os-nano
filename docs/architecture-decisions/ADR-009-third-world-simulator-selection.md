# ADR-009: Third-World Simulator Selection — Habitat-Sim as the Photoreal Navigation Backend

- Status: **Proposed — awaiting CEO approval (new external dependency, gate #1)**
- Date: 2026-06-10
- Related: ADR-005 (Isaac, superseded/paused), ADR-008 (playground parallel track),
  [ARCHITECTURE.md](../ARCHITECTURE.md)

## Context

The campaign goal is a third world: photoreal, semantically real indoor navigation —
VLN first (NL instruction → navigate → deterministic geodesic verify), SysNav revival
riding along (it needs equirect panorama + registered pointcloud + odometry; v2.4
stalled exactly because MuJoCo box-room renders are semantically empty). Locomotion
fidelity is NOT required here — Go2 MPC gait stays proven in MuJoCo; this world runs a
kinematic velocity base behind `BaseProtocol`.

Evidence: live web research (2026-06-10) + two spikes on the target machine (Ubuntu
24.04, RTX 5080 Laptop 16GB, driver 580.126.20).

**Spike results (conda py3.9 env, habitat-sim 0.3.3 headless):** install clean; EGL
headless rendering works on this exact laptop GPU; photoreal egocentric RGB frame
verified; `EquirectangularSensor` rendered a 640×1280 color+depth panorama; spherical
unprojection of the equirect depth yielded an 819K-point registered pointcloud; exact
ground-truth odometry. The full SysNav input chain demonstrated in ~60 lines.

## Decision (proposed)

Adopt **habitat-sim 0.3.3/0.3.4 (pinned, conda py3.9 headless variant)** as the
third-world backend, integrated per ADR-008 as a playground-track world:

- **Process split:** habitat runs in its own conda subprocess (the repo venv stays
  py3.12/uv); a thin bridge connects them — the proven go2-sim subprocess pattern.
  v1 bridge is plain socket/JSON (no ROS2 needed for VLN); the M4 SysNav bridge adds
  ROS2 topic publishers mirroring `go2_vnav_bridge` patterns.
- **Verify stays the moat:** `PathFinder.find_path` geodesic distance, `snap_point`,
  navmesh containment — deterministic oracle predicates (`at_position` / `visited` /
  `object_visible` / `geodesic_dist`), never VLM-graded.
- **Base:** `habitat_sim.physics.VelocityControl` + `pathfinder.try_step` (the VLN-CE
  recipe) implements `BaseProtocol.walk/set_velocity/get_position/get_heading`.
- **Scenes:** start license-free (habitat_test_scenes now; HSSD 211 synthetic scenes
  via HF CC BY-NC terms; Replica 18 scans) while the owner signs the HM3D research
  agreement (DQ-1) for the 1000-scan photoreal corpus + HM3D-Semantics.
- **Scenario DTO evolves additively:** `sim_backend: str = "mujoco"` (+ generic
  `scene_ref`) appended with defaults — frozen-dataclass rule preserved; `scene_xml`
  stays for MJCF worlds.

## Decision matrix (5 candidates × 8 dimensions, 2026-06-10)

| Dimension (weight) | Habitat 0.3.x | Isaac Sim | Genesis 1.1.1 | AI2-THOR/ProcTHOR | OmniGibson 3.7 |
|---|---|---|---|---|---|
| Photoreal indoor scenes (MUST) | HM3D 1000 scans + MP3D + HSSD 211 + Replica | imports only (USD conv.) | **none bundled** | stylized, not photoreal | 50 interactive scenes |
| Deterministic geodesic oracle (MUST) | first-class navmesh API ✔ | build yourself | build yourself | Unity navmesh ✔ | Isaac-based, partial |
| Headless on this RTX 5080 (MUST) | **spike-verified ✔** | verified 2026-04 (Docker) | driver 575+ ok (Nyx) | CloudRendering hangs reported | yes (heavy) |
| Equirect pano (SHOULD, SysNav) | **native, spike-verified ✔** | no native | no | no (stitch) | no |
| Pointcloud/odom (SHOULD, SysNav) | depth-unproject, spike ✔ | **RTX lidar (in-tree config)** | raycaster lidar ✔ | back-project | via Isaac cams |
| py3.12 repo fit (SHOULD) | conda py3.9 → subprocess | py3.11 (5.x) → subprocess | **pip py3.12 native ✔** | untested, deps 2021 | py3.10 pin |
| License (MUST) | MIT code; datasets NC-research | EULA | Apache-2.0 | Apache-2.0 | MIT + encrypted dataset |
| Maintenance 2026 | **sunset 2026-05 (final v0.3.4)** | active (NVIDIA) | active, 1.0 is 2 wks old | life-support | active |

## Why Habitat despite the May-2026 sunset

Every MUST is spike-verified on the target machine today; it is the only candidate
with native equirect (SysNav's hard input) AND a benchmark-standard VLN ecosystem
(R2R/VLN-CE episodes are portable JSON). The sunset risk is acceptable because our
usage surface is mature read-mostly code (rendering + navmesh), we pin versions, and
the kernel/world seam means the backend is swappable without kernel changes — the
seam IS the hedge. Runner-up triggers: Isaac (in-tree Docker assets, RTX lidar) if M4
shows depth-unprojected clouds can't drive SysNav clustering, or when interactive
manipulation scenes matter; Genesis re-evaluated when 1.x stabilizes and a scene
ecosystem (InternScenes?) emerges; VLNVerse/InternUtopia tracked as Isaac-based
scene/benchmark donors.

## Campaign uncertainties — answered

1. **habitat py3.12/uv:** no official wheels; conda stable = py3.9 only (spike-
   confirmed working); nightly py3.10/3.11; py3.12 unconfirmed → subprocess bridge.
2. **Pseudo-lidar quality:** one 640×1280 equirect depth pano → 819K registered
   points, denser than a Mid-360 sweep; voxel-clustering input adequacy is plausible
   but pattern differs from spinning lidar → validate against SysNav clustering in M4.
3. **Genesis maturity:** 1.0 released 2026-05-27, 1.1.1 on 2026-06-09; clean py3.12
   pip + lidar sensors, but zero indoor scene ecosystem and 2-week-old stability.
4. **HM3D licensing:** Matterport account + free research agreement (owner action,
   DQ-1); HSSD/Replica/test-scenes are agreement-free interim paths — M2 is unblocked.

## Consequences

- This world is **Linux-first** (EGL headless ≠ macOS); macOS kernel-track devs use
  the conda osx-arm64 windowed build or treat it like go2-nav (already Linux-only).
- A frozen upstream means we vendor patches if needed (none required by the spikes).
- New repo deps: NONE in the py3.12 venv (bridge is stdlib); the conda env is an
  external runtime like the SysNav sibling workspace — documented, not vendored.

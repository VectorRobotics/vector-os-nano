# Agent Status

**Updated:** 2026-05-18 (v2.4 SysNav — infra LANDED, bridge wiring + launch NOT delivered; master merge prep)
**Branch:** `feat/v2.0-vectorengine-unification`

## Current state

**v2.4 SysNav Simulation Integration — PAUSED (infra + tests green, T6 + G5 NOT delivered)**

5 core modules + 215 tests landed (lidar360, pano360, gt_odom, sysnav_bridge,
sysnav_sim_tool, xmat G3 fix). Tasks T6 (vnav_bridge sensor wiring) + G5 (launch file)
were NOT delivered — sensors exist but do not wire into scripts/go2_vnav_bridge.py
or a working launch file. Pipeline does not run end-to-end. CEO decision: infra retained,
integration deferred to v2.5. Local manipulation (Piper pick/place) decoupled/deferred
(skills in-tree, not registered by default). Branch is 34 commits ahead of
origin/master; integrate via PR (NOT fast-forward — see Master merge readiness).

CEO directive 2026-04-25: auto-approve architectural decisions, start
implementation, emphasize broad test coverage. v2.4-perception-overhaul
(YOLOE + SAM3 + own pointcloud / sanity gates) is archived because
SysNav already provides equivalent capabilities. SysNav (CMU sibling
lab, PolyForm-NC) runs as a separate ROS2 workspace; we publish
ground-truth `/registered_scan`, `/camera/image`, `/camera/depth`,
`/state_estimation` from MuJoCo and consume `/object_nodes_list` via
the existing `sysnav_bridge` adapter (Apache 2.0 boundary preserved).

## Cleanup landed pre-spec (this session)

Deleted (~2570 LoC removed):
- `vector_os_nano/perception/vlm_qwen.py`
- `vector_os_nano/perception/go2_perception.py`
- `vector_os_nano/perception/go2_calibration.py`
- `tests/unit/perception/test_{vlm_qwen,go2_perception,go2_calibration}.py`
- `tests/integration/test_sim_tool_perception_wire.py`
- `scripts/verify_perception_pick.py`
- `docs/v2.3_live_repl_checklist.md`

Modified:
- `vector_os_nano/vcli/tools/sim_tool.py` — Qwen wire-up block
  replaced by a comment pointing at `sysnav_bridge`. `agent._perception`
  / `agent._calibration` set to `None` until LiveSysnavBridge
  populates `world_model` directly.

Verified post-cleanup:
- 70/70 existing tests still green
  (`test_pick_top_down.py` 33, `test_mobile_pick.py` 22, `test_sysnav_bridge_mapping.py` 15).
- Targeted import smoke OK across perception/skills/integrations.

## v2.4 — landed cycle summary (PAUSED — infra only, T6 + G5 deferred)

8 tasks planned; 6 delivered, 2 NOT delivered:

| Wave | Task | Status | Tests | Notes |
|------|------|--------|-------|-------|
| W0 | env probe (mj_ray, cube-face benchmarks) | ✅ done | (probe only) | |
| W1 | T1 lidar360 / T2 gt_odom / T3 G3 xmat fix | ✅ done | 56 | xmat REP-103 fix: right camera pose now correct |
| W2 | T4 pano360 / T5 LiveSysnavBridge | ✅ done | 47 | bridge runs, subscribers ready |
| W3 | T6 sensor integration / T7 SysnavSimTool | ⚠️ partial | 21 | T7 tool done; T6 wiring into vnav_bridge NOT done |
| W4 | T8 smoke + docs | ⚠️ partial | (smoke) | smoke_sysnav_sim.py runs; sysnav_simulation.md exists; g5_launch does NOT exist |
| — | **T6 NOT DELIVERED** | ❌ deferred | — | vnav_bridge.py never modified to wire lidar360/pano360/gt_odom |
| — | **G5 NOT DELIVERED** | ❌ deferred | — | sysnav_sim.launch.py never created; no working launch file |

**Test result (2026-05-18, post-decoupling)**: green on the memory-safe
subset run in chunks — `tests/skills` 90, sensors+sysnav_bridge unit +
safe integration 125, `tests/unit/vcli` prompt/sysnav 35 (≈250 tests, 0
failures). 3 tests fixed this session: 1 sysnav-pause contract + 2
pre-existing stale prompt-brand assertions (red since a2ad980, 5 wks).
NOTE: the full repo suite is NOT run as one process — `pytest tests/unit`
whole-dir OOMs a 64 GB host (pre-existing repo characteristic; use CI
shards / chunked runs). Coverage ≥ 90 % on each new module.
Infrastructure is solid; pipeline is incomplete.

## Commit chain (v2.4 cycle, this branch)

```
[smoke + docs]
TBD       T8 smoke_sysnav_sim.py + docs/sysnav_simulation.md
7e56458   T6+T7 sensor integration tests + SysnavSimTool CLI
966ef44   T4+T5 MuJoCoPano360 + LiveSysnavBridge
06cb3e9   T1+T2 MuJoCoLivox360 + GroundTruthOdomPublisher
9317007   T3 G3 xmat REP-103 fix
8071b3f   pivot + cleanup (-2570 LoC v2.3 Qwen perception)
886ec4d   sysnav_bridge adapter (pre-cycle, foundation)
2ae7c3f   relicense MIT → Apache 2.0
```

## Reference

- Spec: `.sdd/spec.md` (v2.4 SysNav Sim)
- Plan: `.sdd/plan.md`
- Tasks: `.sdd/task.md`
- Status: `.sdd/status.json` (phase=tasks, all approved)
- Bringup integration (real-robot side): `docs/sysnav_integration.md`
- Sim integration: `docs/sysnav_simulation.md` (T8 will write)
- Adapter (already landed): `vector_os_nano/integrations/sysnav_bridge/`
- SysNav repo (sibling lab): https://github.com/zwandering/SysNav

## Archive index

- `.sdd/archive-v2.4-perception-overhaul/` — YOLOE+SAM3 SDD (now redundant)
- `.sdd/archive-v2.3/` — Qwen perception cycle (impl deleted this session)
- `.sdd/archive-v2.2/` — loco manipulation infrastructure
- `.sdd/archive-v2.1-pick/` — Piper top-down grasp
- earlier archives unchanged.

## Master merge readiness (2026-05-18)

Branch is 34 commits ahead of origin/master, 0 real-code commits behind
(post-`git fetch` 2026-05-18). NOT fast-forward-mergeable: master already
absorbed the v2.0 line via PR #8 (2026-04-16) + PR #7 sim-to-real-nav, so
master carries merge commits not in this linear branch. BUT
`git log --no-merges HEAD..origin/master` is EMPTY — master has zero code
this branch lacks; the divergence is pure merge topology, no real conflict.
Integration path: open a PR (repo convention, PR #7/#8 precedent) — GitHub
merge expected conflict-free. The 34 = 4 merge-prep commits + the v2.4 work
landed after the 2026-04-16 PR#8 point. Green on the memory-safe subset
(≈250 tests, chunked — see Test result above; full single-process OOMs,
pre-existing). Diff is +730K lines, ~660K = Piper/Menagerie vendored mesh
assets under vector_os_nano/hardware, pre-existing since v2.1 — flagged for
CEO merge decision, not introduced by this cleanup.

Docs have been truth-corrected (2026-05-18):
- status.md: v2.4 PAUSED narrative (infra + tests, T6 + G5 deferred)
- progress.md: v2.4 entry corrected; v2.4 NOT "landed", is "paused"
- docs/sysnav_simulation.md + docs/sysnav_integration.md: header block added (STATUS: PAUSED/DEFERRED)
- docs/pick_top_down_spec.md + docs/pick_top_down_known_issues.md: header block added (STATUS: DEFERRED)
- docs/v2.2_live_repl_checklist.md: DELETED (stale, obsolete)

Living .md count: 21 (before cleanup), 20 (after v2.2 checklist deletion). All production + ADR + `.sdd/` docs verified.

## Next session starter

```
cd ~/Desktop/vector_os_nano
git log --oneline feat/v2.0-vectorengine-unification -10   # recent commits
cat progress.md                       # current status
# chunked (whole tests/unit OOMs a 64 GB host — never run it as one process):
.venv-nano/bin/python -m pytest tests/skills -q
.venv-nano/bin/python -m pytest tests/unit/hardware/sim/sensors tests/unit/integrations/sysnav_bridge tests/unit/vcli/test_sysnav_sim_tool.py tests/unit/vcli/test_prompt.py tests/integration/test_sysnav_bridge_mapping.py tests/integration/test_xmat_rep103_regression.py -q
```

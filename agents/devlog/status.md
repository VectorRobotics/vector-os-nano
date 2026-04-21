# Agent Status

**Updated:** 2026-04-20 (v2.4 perception overhaul SDD artifacts drafted while CEO away)
**Branch:** `feat/v2.0-vectorengine-unification`
**Upstream:** ahead by 5 commits (3 bookkeeping + spec + plan + task)

## Current state

**v2.4 Perception Overhaul — SDD artifacts DRAFTED, awaiting CEO approval**

CEO (Yusen) directed YOLO + SAM 3 grounding + high-fidelity object assets during the 2026-04-20 debrief after v2.3 live-REPL smoke failed. CEO stepped away for a few hours and authorised autonomous drafting; artifacts are committed and pushed for review on return.

v2.3 closed — all 10 v2.3 code commits on remote, SDD artifacts archived
to `.sdd/archive-v2.3/`. Smoke identified root cause: Qwen VLM bbox–thumbnail
coordinate-system bug (not the superficial xmat flip). v2.3.1 hot-fix
spec preserved in `archive-v2.3/NEXT_SESSION.md` — G4 xmat fix carries
forward into v2.4, but Qwen grounding is deleted entirely.

## What landed this session (5 commits)

```
8dda396  docs(sdd): draft v2.4 task breakdown
d3bcac1  docs(sdd): draft v2.4 perception plan
88c82ec  docs(sdd): draft v2.4 perception overhaul spec
e21fd8f  docs(sdd): archive v2.3 cycle to .sdd/archive-v2.3/
43c0b26  docs(sdd): finalize v2.3 artifacts + v2.3.1 next-session spec
```

## v2.4 spec headlines

Full details in `.sdd/spec.md`. Short form:

- **11 MUST goals** including YoloeDetector, Sam3Segmenter, mask→pointcloud
  projection, sanity gates, xmat REP-103 fix, Google Scanned Objects
  scene swap (10 meshes), Qwen removed from grounding.
- **7 SHOULD** latency/coverage/diagnostic script targets.
- **4 MAY** IOU fusion, domain randomisation; FoundationPose deferred v2.5.
- **8 open questions** — CEO review required on: O3 (SAM3 vs SAM2.1
  primary), O4 (object count — 10 vs 5 vs 20), O6 (delete or archive
  QwenVLMDetector).
- **Single new dep**: `ultralytics>=8.3.237` (bundles both YOLOE and SAM 3 via `SAM3SemanticPredictor`).
- SAM 3 auto-falls back to SAM 2.1 when HF-gated weights absent.

## v2.4 plan headlines

Full details in `.sdd/plan.md`:

- 14 technical decisions with rationale.
- 11 module designs (new: `detectors/`, `segmenters/`, `pointcloud_projection`, `sanity_gates`; modified: `go2_perception`, `go2_ros2_proxy`, `go2_room.xml`, `sim_tool`, `detect`, `mobile_pick`; deleted: `vlm_qwen`).
- Pointcloud math: numpy + scipy KDTree (open3d optional).
- HSV colour-fraction filter for "blue bottle" subqueries (resolves query specificity without asking YOLOE for colour).
- Git-LFS for GSO meshes; download-script fallback path documented.
- 10 risks with mitigations (SAM 3 HF access being the top risk, solved by auto-fallback).

## v2.4 task headlines

Full details in `.sdd/task.md`:

- 10 tasks across 6 waves + CEO smoke.
- Serial subagent dispatch per wave (narrow pytest, no MuJoCo import in subagent prompts).
- TDD RED→GREEN→REFACTOR per task with specific test names.
- Subagent prompt template included.
- Estimated wall-clock: 2–3 days.

## Env probe (zero-risk, done autonomously)

See `agents/devlog/v24-env-probe.md`:

- `ultralytics` missing — T0 will add.
- Local `/usr/bin/python3` torch is CPU-only; Yusen's GPU env distinct.
- GSO repo sparse-cloned to `/tmp/gso_probe` — 1030 models confirmed.
- Candidate list of **10 GSO pickable objects** drafted (to be finalised in T7).

## Blocking decisions for CEO

1. Approve spec as drafted, with any scope revisions? (`.sdd/spec.md`)
2. Resolve 3 blocking open questions — O3 / O4 / O6 (listed at end of spec).
3. Confirm path forward: execute v2.4 as planned (2–3 days), OR keep spec but delay implementation.

## Session starter (next time)

```
cd ~/Desktop/vector_os_nano
cat agents/devlog/status.md                       # this file
cat .sdd/spec.md                                   # v2.4 draft
cat .sdd/plan.md                                   # v2.4 plan
cat .sdd/task.md                                   # v2.4 task list
cat agents/devlog/v24-env-probe.md                 # T0 prep
git log --oneline 543dfd4..HEAD                    # this session's commits
```

## Archived cycles (reference)

- `.sdd/archive-v2.3/` — v2.3 Go2 Perception Pipeline (Qwen, 10 commits landed + smoke failed)
- `.sdd/archive-v2.2/` — v2.2 Loco Manipulation Infrastructure
- `.sdd/archive-v2.1-pick/` — v2.1 Piper top-down grasp
- `.sdd/archive-v2.0-vectorengine/` — v2.0 VectorEngine unification
- `.sdd/archive-mujoco-render/` — MuJoCo render milestone
- `.sdd/archive-isaac-sim/` — Isaac Sim attempts
- `.sdd/archive-gazebo/` — Gazebo attempts

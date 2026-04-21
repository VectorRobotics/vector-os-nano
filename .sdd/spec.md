# v2.4 Perception Overhaul — Specification

**Status**: DRAFT — awaiting CEO approval
**Scope**: MuJoCo-sim first; protocol-clean path for real D435 after.
**Branch**: `feat/v2.0-vectorengine-unification` (continuation; v2.3 archived)

---

## 1. Overview

Rebuild the Go2 grounding stack on SOTA perception models — **YOLOE**
(real-time open-vocabulary detection) + **SAM 3** (concept-prompt
segmentation) — driven by mask-aware depth projection, and replace the
toy cylinder pick scene with **Google Scanned Objects** high-fidelity
meshes. Kills the two structural failure modes exposed by the 2026-04-20
v2.3 live-REPL smoke:

1. Qwen-VLM bbox thumbnail/full-resolution coordinate mismatch →
   catastrophic depth projection errors (6 m phantom "blue bottle").
2. Low-fidelity capsule objects provide no realistic RGB → sim-to-real
   gap prevents any grounded detector from generalising.

Target user flow after v2.4:

```
go2sim with_arm=1
抓起蓝色瓶子
  └─ MobilePickSkill._resolve_target miss
      └─ auto-detect via context.perception (YOLOE+SAM3 backend)
          ├─ YOLOE detect(RGB, "blue bottle") → bbox (~30 ms)
          ├─ SAM 3 segment(RGB, bbox) → mask (~80 ms)
          ├─ mask ∩ depth → pointcloud
          ├─ sanity gates (distance, height, density, area)
          ├─ centroid in camera frame
          └─ calibration.camera_to_world → ObjectState → upsert
      └─ resolve retry → hit
  └─ approach_pose / navigate / wait_stable
  └─ PickTopDownSkill → held=True
```

## 2. Background & Motivation

### Why v2.3 failed live-REPL smoke

Code inspection confirmed two stacking bugs:

- **Bbox-thumbnail scale (CRITICAL)**: `QwenVLMDetector.detect` resizes
  the input frame to 160 × 120 (`_VLM_IMAGE_MAX_DIM=160`) before sending
  to OpenRouter, but `_detection_from_raw` uses the **original**
  `image.shape[:2]` for bbox normalisation — and only rescales when
  `max(coord) ≤ 1.0`. Qwen returns pixel coordinates in the 160 × 120
  thumbnail (e.g. `(50, 40, 80, 100)`) — never normalised — so bboxes get
  applied **at half-scale** onto the 320 × 240 depth frame, pointing at
  the upper-left quadrant. Median depth of a distant wall/ceiling
  (~6 m) is projected as the object location → world_model receives
  `(16.8, 1.62, 1.37)` for a bottle physically at `(11, 3, 0.25)`.
- **xmat LEFT/RIGHT flip (G4 from v2.3.1)**: `get_camera_pose`
  computes `right = (-sin h, cos h, 0)` which at heading=0 is
  `(0, 1, 0)` — ROS-LEFT under REP-103, not right. Contributes ~1.4 m
  mirrored lateral error.

### Why patching v2.3 is the wrong answer

Even a thorough `v2.3.1` hot-fix (bbox rescale + sanity gate + xmat
swap) still leaves three architectural gaps that every 2025 SOTA mobile-
manipulation stack addresses:

1. **Qwen-VLM is not a grounding model.** It is trained for chat-style
   description. Its bbox output is a capability hack — documentation
   reports robust grounding only between 480 × 480 and 2560 × 2560, and
   we feed it 160 × 120.
2. **bbox-centre depth is brittle.** Narrow or off-centre objects, or
   objects with occlusion, get depth sampled from foreground/background
   pixels outside the target.
3. **No mask → no pointcloud → no geometric sanity.** We cannot verify
   a detection is a plausible object (size, shape, density).

2025 SOTA (OK-Robot, DynaMem, Grounded-SAM 2/3, SVM, GeFF, LOVMM) all
follow the same pattern: **open-vocab detect → promptable segment →
mask-filtered depth → geometric/semantic consensus → persistent memory
→ close-loop refinement**. v2.4 ports the first three stages and leaves
persistent voxel memory + close-loop servoing for v3.0.

### Why realistic assets matter now

The current `pickable_*` bodies are 2.8 cm cylinders with flat RGBA
colours. Any detector — Qwen, YOLOE, or SAM 3 — is trained on realistic
object appearance. A "blue bottle" stimulus with zero texture detail
and uniform colour is an **out-of-distribution** query. Failures at
the detector stage cannot be separated from failures in the projection
stage. Replacing the scene with Google Scanned Objects meshes (1030
real household objects, MJCF-compatible via `kevinzakka/
mujoco_scanned_objects`) gives the grounding models realistic inputs
and closes the sim-to-real gap for appearance-dependent perception.

## 3. Goals

### MUST (blocking for release)

- **G1** `YoloeDetector` class implementing an `OpenVocabDetector`
  protocol on top of `ultralytics` YOLOE-S or YOLOE-M weights. Returns
  `list[Detection]` with pixel-space bboxes in the native RGB resolution
  (no resize-based bbox corruption).
- **G2** `Sam3Segmenter` class wrapping Meta SAM 3 / SAM 3.1 inference.
  Takes RGB + list[bbox prompt] → `list[np.ndarray mask]` at native
  resolution. Thread-safe single-instance cache.
- **G3** Replace `Go2Perception.track` implementation: `detect()` runs
  YOLOE, `track()` runs SAM 3, then mask-filtered depth → pointcloud →
  centroid. Populates `TrackedObject.mask` (already a field in
  `core.types.TrackedObject`).
- **G4** `pointcloud_projection.py` utility: takes `mask`, `depth_frame`,
  `intrinsics` → `(N,3)` camera-frame pointcloud. RANSAC plane removal
  option for on-table grasping. Statistical outlier removal.
- **G5** Sanity gate module: reject detections where
  `depth_median ∉ [0.15, 5.0]`, `world_z ∉ [-0.2, 1.5]`, mask area
  `∉ [200, 50_000] px`, or pointcloud density < 50 points. Returns
  `TrackedObject(pose=None)` on rejection.
- **G6** Fix `Go2ROS2Proxy.get_camera_pose` xmat convention —
  `right = (sin h, -cos h, 0)` to match ROS REP-103.
- **G7** Replace 3 capsule `pickable_*` bodies in `go2_room.xml` with
  10 Google Scanned Objects meshes (selection: ~4 bottles/cans, 3 mugs,
  3 miscellaneous small objects — all pickable with 35 mm Piper jaws).
  Collision submeshes via V-HACD (already provided by the asset repo).
- **G8** `QwenVLMDetector` removed from grounding path. `QwenVLM`
  (natural-language describe/visual_query) stays — rewire through
  `Go2VLMPerception` only.
- **G9** DetectSkill / MobilePickSkill external behaviour unchanged —
  tool inputs + `SkillResult.result_data` schema same as v2.3 plus
  `mask_area_px` and `pointcloud_points` fields on each object summary.
- **G10** CLI summary: skill output shows `label / world_xyz / mask_area
  / confidence` per object (fixes v2.3.1 G1 UX gap).
- **G11** Live-REPL end-to-end: on `go2sim with_arm=1`, `抓起蓝色瓶子`
  returns a valid world_xyz within 0.1 m of ground truth for at least
  4 / 5 attempts.

### SHOULD

- **S1** YOLOE-S ≤ 50 ms per call on RTX 5080 @ 640 × 480.
- **S2** SAM 3 ≤ 200 ms per call on RTX 5080 @ 640 × 480.
- **S3** End-to-end detect latency ≤ 400 ms wall-clock.
- **S4** Unit coverage ≥ 90 % on `yoloe_detector.py`, `sam3_segmenter.py`,
  `pointcloud_projection.py`, `sanity_gates.py`.
- **S5** `MobilePickSkill` logs a distance warning when `nav_dist > 3 m`
  (carry forward v2.3.1 S1).
- **S6** `DetectSkill` returns `diagnosis="all_3d_invalid"` when every
  tracked object fails sanity gates (carry forward v2.3.1 S2).
- **S7** `debug_perception_live.py` script: one-shot capture →
  YOLOE + SAM 3 + projection trace with per-stage timings and
  visualisations written to `/tmp/perception_debug/`.

### MAY

- **M1** IOU-based temporal fusion: same-label detections within 5 cm
  across frames merge into one `ObjectState` with confidence-weighted
  position update.
- **M2** Domain-randomised materials in `go2_room.xml` (lighting /
  texture variation for training robustness).
- **M3** FoundationPose 6-DoF pose estimate — **deferred to v2.5**.
- **M4** Persistent VoxelMap + CLIP memory (OK-Robot style) — **deferred
  to v3.0**.

## 4. Non-Goals (explicitly out of scope)

- Real RealSense D435 integration — `Go2ROS2Proxy` path only. Real
  Piper hardware integration.
- 6-DoF pose estimation (FoundationPose / GenPose). Piper does top-down
  grasp only; centroid + gripper orientation from task geometry suffices.
- Voxel map / persistent scene memory across sessions.
- Close-loop visual servoing during approach.
- Multi-view fusion across multiple detect calls.
- SAM 3D single-image reconstruction (deferred v3.0+).
- Nav-stack tuning.

## 5. User Scenarios

### US1 — `抓起蓝色瓶子` from empty world_model

Dog at home position, no prior perception. User says "抓起蓝色瓶子".

1. LLM routes to `mobile_pick_skill(label="blue bottle")` (CN→EN alias).
2. `_resolve_target` misses — world_model empty.
3. `run_autodetect_retry` helper fires — calls `DetectSkill(query="blue
   bottle")`.
4. `Go2Perception.detect` → YOLOE returns bbox for the blue bottle
   mesh on the table.
5. `Go2Perception.track` → SAM 3 returns mask for that bbox.
6. Mask ∩ depth → pointcloud → centroid `(x_cam, y_cam, z_cam)`.
7. Sanity gates pass (distance 0.7 m, height 0.25 m, mask area 400 px,
   pointcloud 150 points).
8. `Go2Calibration.camera_to_world` → world `(≈11.0, ≈2.85, ≈0.25)`.
9. `world_model.add_object` → retry `_resolve_target` → hit.
10. Approach + grasp → `held=True`.

### US2 — `检测所有物体` enumerating the table

User asks "检测桌上所有物体". DetectSkill with `query="all objects"`.
YOLOE returns 3–5 bboxes for bottles/mugs/cans on the pick table.
SAM 3 masks each. World model gets 3–5 ObjectState entries with valid
world coordinates. CLI shows per-object summary.

### US3 — no-match case

`抓起香蕉` when no banana on scene. YOLOE returns empty list. DetectSkill
returns `diagnosis="no_detections"`. MobilePick reports
`object_not_found`.

### US4 — sanity-gate rejection

Camera pointed at far wall. YOLOE false-positive detects "bottle"
on a poster 6 m away. SAM 3 masks it. Pointcloud median depth > 5 m →
sanity gate rejects → `TrackedObject.pose=None`. DetectSkill marks
`has_3d=False` and does **not** seed world_model.
`diagnosis="all_3d_invalid"`. MobilePick does **not** navigate to
phantom coords.

## 6. Technical Constraints

- **Hardware**: RTX 5080 16 GB VRAM. Dog sim + ROS2 + MuJoCo + Piper +
  YOLOE + SAM 3 must coexist. Target VRAM budget for perception ≤ 6 GB.
- **No network calls** in the grounding path — YOLOE + SAM 3 weights
  local on disk (~1.5 GB + ~2 GB). First-run download acceptable; CI
  should skip model-dependent tests.
- **Python 3.12**; **torch 2.x CUDA 13**; must not break existing
  `vector-cli` boot or ROS2 bridge. No changes to `core.types` public
  API.
- **PerceptionProtocol** unchanged — `get_point_cloud(mask)` already in
  the protocol; previously `NotImplementedError` in Go2Perception, now
  implemented.
- **MuJoCo 3.2+** — mesh assets must be MJCF 3.x-compatible; texture
  maps via `<texture>` + `<material>` elements (no PBR until upstream
  issue #2674 resolves).
- **Imports must not cascade into MuJoCo during test collection** —
  `feedback_no_parallel_agents.md` still applies.

## 7. Interface Definitions

### 7.1 New `perception/detectors/base.py`

```python
@runtime_checkable
class OpenVocabDetector(Protocol):
    """2D open-vocabulary object detector."""

    def detect(
        self, image: np.ndarray, query: str
    ) -> list[Detection]: ...
```

### 7.2 New `perception/detectors/yoloe_detector.py`

```python
class YoloeDetector:
    """Ultralytics YOLOE open-vocabulary detector."""

    def __init__(
        self,
        model_name: str = "yoloe-s",      # s | m | l
        confidence_threshold: float = 0.25,
        device: str | None = None,          # auto-detect CUDA
    ) -> None: ...

    def detect(
        self, image: np.ndarray, query: str
    ) -> list[Detection]: ...
```

### 7.3 New `perception/segmenters/sam3_segmenter.py`

```python
class Sam3Segmenter:
    """Meta SAM 3 / SAM 3.1 promptable segmenter."""

    def __init__(
        self,
        model_name: str = "sam3.1-base",    # base | large
        device: str | None = None,
    ) -> None: ...

    def segment(
        self,
        image: np.ndarray,
        prompts: list[tuple[float, float, float, float]],   # bboxes
    ) -> list[np.ndarray]: ...   # masks, same H×W as image
```

### 7.4 New `perception/pointcloud_projection.py`

```python
def mask_depth_to_pointcloud(
    mask: np.ndarray,                       # H×W bool
    depth: np.ndarray,                      # H×W float32 metres
    intrinsics: CameraIntrinsics,
    depth_min: float = 0.15,
    depth_max: float = 5.0,
    statistical_outlier_k: int = 20,
    statistical_outlier_std: float = 2.0,
) -> np.ndarray: ...                        # (N,3) float64, camera frame


def pointcloud_centroid(
    pointcloud: np.ndarray,                 # (N,3)
    method: str = "median",                 # median | mean | plane_top
) -> np.ndarray: ...                         # (3,)
```

### 7.5 New `perception/sanity_gates.py`

```python
@dataclass(frozen=True)
class SanityGateConfig:
    depth_range: tuple[float, float] = (0.15, 5.0)
    height_range: tuple[float, float] = (-0.2, 1.5)
    mask_area_range: tuple[int, int] = (200, 50_000)
    min_pointcloud_points: int = 50


def apply_sanity_gates(
    mask_area_px: int,
    pointcloud: np.ndarray,
    centroid_camera: np.ndarray,
    centroid_world: np.ndarray | None,
    config: SanityGateConfig,
) -> tuple[bool, str]: ...    # (ok, reason_if_rejected)
```

### 7.6 Updated `perception/go2_perception.py`

Same `PerceptionProtocol` methods. Internals:

```python
class Go2Perception:
    def __init__(
        self,
        camera: Any,
        detector: OpenVocabDetector,
        segmenter: Sam3Segmenter | None,
        gates: SanityGateConfig | None = None,
        intrinsics: CameraIntrinsics | None = None,
    ) -> None: ...

    def detect(self, query: str) -> list[Detection]: ...
    def track(self, detections: list[Detection]) -> list[TrackedObject]: ...
    def get_point_cloud(
        self, mask: np.ndarray | None = None
    ) -> np.ndarray: ...
```

`segmenter=None` falls back to bbox-centre depth (a degraded mode for
development / low-VRAM smoke).

### 7.7 Scene assets

New directory `vector_os_nano/hardware/sim/mjcf/pickable_assets/`
containing selected Google Scanned Objects: 10 meshes, each with
`<object>.xml` (MJCF fragment) + `model.obj` (visual) +
`model_collision_0..31.obj` (V-HACD convex decomposition) + `texture.png`.
`go2_room.xml` includes the fragment and instantiates 10 `pickable_*`
bodies on the pick table (replacing 3 cylinders).

## 8. Test Contracts

### Unit tests

- **yoloe_detector**: weight-loading mock, bbox output shape/type,
  confidence threshold filtering, device fallback, CN→EN query unchanged
  (not this class's job).
- **sam3_segmenter**: mock SAM 3 inference, bbox prompt → mask shape
  matches image, empty prompt list → empty masks.
- **pointcloud_projection**: synthetic mask + synthetic depth → known
  centroid; plane removal removes a 1m × 1m plane; statistical filter
  removes injected outliers.
- **sanity_gates**: each gate individually (depth too far, height too
  low, mask too small, pointcloud too sparse), composed cases.
- **go2_perception**: detect + track end-to-end with mocked detector +
  segmenter + synthetic frames; pose=None on gate reject; calls
  `get_point_cloud` correctly.
- **go2_calibration** (regression): verify xmat fix → right = -Y at
  heading=0. Update fixture expected values.
- **go2_ros2_proxy camera pose** (regression): two heading cases
  verified.

### Integration tests

- **sim_tool wire-up**: `_start_go2(with_arm=True)` constructs
  `Go2Perception(YoloeDetector, Sam3Segmenter)` when weights present,
  degrades to stubbed fallback on missing weights.
- **DetectSkill + MobilePick**: mocked detector+segmenter returning
  crafted bboxes/masks → world_model populated with correct world
  coords, respect sanity gates.
- **Scene load**: `go2_room.xml` with new GSO meshes loads without
  MuJoCo error, pickable bodies present with `pickable_*` naming.

### E2E dry-run (no GPU model calls)

- `scripts/verify_perception_pick.py --dry-run`: same exit-0 contract
  as v2.3, updated to use mocked detector/segmenter.

### Live smoke (GPU required, Yusen-run)

- `go2sim with_arm=1` loads 10 GSO pickables on table.
- `抓起蓝色瓶子` → world_model populated with valid coords
  (|err| < 0.1 m), pick succeeds ≥ 4 / 5 runs.

## 9. Acceptance Criteria

1. All MUST goals G1–G11 satisfied.
2. `pytest tests/unit tests/integration` passes on a GPU-less dev box
   using mocked detector+segmenter (real weight load behind
   `@pytest.mark.gpu`).
3. Baseline v2.3 test count (194) preserved or replaced; net new ≥ 30
   unit tests.
4. Live-REPL smoke signed off by Yusen (5-run pass criterion).
5. Qwen removed from grounding path (ruff + grep can't find
   `QwenVLMDetector` references in `skills/` or `perception/go2_*.py`).
6. Branch pushable to main once CEO approves.

## 10. Open Questions

| # | Question | Default (if no CEO input) | Alternatives |
|---|---|---|---|
| O1 | SAM 3 vs SAM 3.1 base/large? | `sam3.1-base` (balance) | `sam3-base` (stability), `sam3.1-large` (accuracy, +VRAM) |
| O2 | YOLOE size: S/M/L? | `yoloe-s` (speed) | M/L trade accuracy |
| O3 | SAM 3 segmentation deferred — use **Arch B** (YOLOE + SAM 2.1) instead? Faster (50–150 ms), battle-tested. | Proceed with SAM 3.1 (Yusen direction); keep SAM 2.1 path documented as fallback if VRAM / speed fails. | Full fallback to SAM 2.1 |
| O4 | Number of GSO pickable objects? | 10 (balance variety vs scene load time) | 5 (faster), 20 (more tests) |
| O5 | Keep bbox-centre fallback when SAM unavailable? | Yes — degraded-mode smoke without GPU | No — require mask always |
| O6 | Delete `QwenVLMDetector` now or keep as archived? | Delete (+20 tests removed, ~500 LoC) — code-review agent will flag unused | Archive under `perception/archive/` |
| O7 | FoundationPose 6-DoF — v2.5 or inline in v2.4? | v2.5 (Piper is top-down only) | Inline — +5 days, side-grasp capability |
| O8 | Scene: keep 3 cylinder bottles alongside GSO as regression anchor? | No — replace fully, GSO covers diversity | Yes — hybrid |

Yusen to resolve O3, O4, O6 at minimum before `/sdd plan`.

## 11. Known debt carried forward

- `VECTOR_SHARED_EXECUTOR=0` spin-thread leak (v2.3 LOW, legacy path).
- `coverage` package / `numpy 2.4` C-tracer conflict — workaround via
  `sys.settrace`.
- `_normalise_color_keyword` still in private API (H3 v2.3 review).
- VGG `last_seen('blue bottle')['room']` literal-string goal bug
  (decomposer strategy whitelist needed).
- `_wait_stable` duplicated between `pick_top_down` and `mobile_pick`
  (should extract to `skills/utils/mobile_helpers.py`).

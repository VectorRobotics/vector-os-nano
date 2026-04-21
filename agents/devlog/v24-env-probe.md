# v2.4 Environment Probe Report

**Date**: 2026-04-20 (while CEO away)
**Dispatcher**: Main Claude (acting as architect for v2.4 draft)

Autonomous probe run during CEO absence. Full dependency install
deferred to T0 wave — this is research only.

## Python / CUDA

```
torch   : 2.11.0+cpu    (CUDA=False on /usr/bin/python3)
cv2     : 4.13.0
scipy   : 1.11.4 (warns on numpy 2.4.3 — known, not blocking)
numpy   : 2.4.3
open3d  : in pyproject.toml [perception] extras
opencv-python : in pyproject.toml [perception] extras
ultralytics   : NOT installed — T0 to add
```

**Implications**:
- Default `/usr/bin/python3` is CPU-only torch. Yusen's dev box likely
  has a separate CUDA torch in a conda/venv. T0 probe must re-run in
  the GPU environment.
- For unit tests, CPU torch is sufficient (all detector/segmenter tests
  use mocks).
- For live smoke (W6), Yusen's GPU env must have `torch+cu128` +
  `ultralytics>=8.3.237`.

## Google Scanned Objects

```
repo     : https://github.com/kevinzakka/mujoco_scanned_objects
total    : 1030 directories under /models/
clone    : sparse-checked to /tmp/gso_probe (only model.xml + README)
structure: model.xml + model.obj + 32× model_collision_*.obj + texture.png
```

**V-HACD convex decomposition** is pre-computed for each object (up to
32 submeshes) — drop-in for MuJoCo convex contact.

## Candidate pickable objects for v2.4 scene

Draft shortlist of 10 (to be finalised in T7 after size / simulation
compatibility check). Selection criteria: diameter ≤ 5 cm fits 35 mm
Piper jaws, diverse shape categories, recognisable by YOLOE vocabulary.

| # | Category | GSO ID | Rationale |
|---|---|---|---|
| 1 | Mug | `ACE_Coffee_Mug_Kristen_16_oz_cup` | Standard mug, graspable by handle |
| 2 | Mug | `Cole_Hardware_Mug_Classic_Blue` | Blue colour — anchor for "blue X" query |
| 3 | Can | `Brisk_Iced_Tea_Lemon_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt` | Beverage can, ~6 cm — verify ≤5 cm body |
| 4 | Container | `BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup` | Small ramekin, low grasp |
| 5 | Toy | `Android_Figure_Chrome` | Small humanoid — shape variation |
| 6 | Toy | `Android_Figure_Orange` | Orange colour — alt colour anchor |
| 7 | Bowl | `Cole_Hardware_Bowl_Scirocco_YellowBlue` | Small bowl — wide shape class |
| 8 | Box | `Crunch_Girl_Scouts_Candy_Bars_Peanut_Butter_Creme_78_oz_box` | Rectangular — another shape class |
| 9 | Cup | `Ecoforms_Cup_B4_SAN` | Tall cylinder alternative |
| 10 | Honey dipper | `Cole_Hardware_Mini_Honey_Dipper` | Narrow/slender — worst-case grasp |

Alternates if some exceed 5 cm envelope:
- `Cole_Hardware_Plant_Saucer_Brown_125`
- `Ecoforms_Plant_Bowl_Atlas_Low`
- `Android_Figure_Panda`
- `BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup` (repeat — small size anchor)

## T0 expected work (once CEO approves)

1. Install `ultralytics>=8.3.237` in project venv (`pip install -U 'ultralytics>=8.3.237'`).
2. Attempt YOLOE weight download (`yoloe-11s-seg.pt` — expected auto-download from GitHub releases).
3. Attempt SAM 2 weight download (`sam2.pt` — open access, auto).
4. Document SAM 3 HF access flow for Yusen (`hf auth login` + request on facebook/sam3 page).
5. Measure actual size of each candidate GSO object mesh (may need xvfb render or mesh trimesh inspection).
6. Finalise 10 picks + record their bounding box dimensions in this
   report for scene XML placement.
7. Add `ultralytics` to `[project.optional-dependencies].perception`.

## Notes for CEO review

- Spec assumes SAM 3 as primary with SAM 2.1 fallback — ensure this
  matches intent. If pivoting to SAM 2.1 primary (proven path), delete
  the access-gating complexity.
- Spec targets 10 GSO objects; reduce to 5 if scene load time becomes
  a problem.
- `Cole_Hardware_Mug_Classic_Blue` is the proposed "blue X" anchor for
  regression parity with v2.3 smoke (`抓起蓝色瓶子` → re-aim at blue
  mug since "bottle" may no longer map to the same scene object).

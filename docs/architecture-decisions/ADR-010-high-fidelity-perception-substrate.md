# ADR-010: High-Fidelity Perception Substrate — Campaign #10 (G1 real VLN + manipulation)

- Status: **Proposed — R2 sandbox spike done (co-sim GPU/latency de-risked); awaiting CEO pick of substrate (DQ-11 gate)**
- Date: 2026-06-14
- Related: ADR-009 (habitat third-world, superseded for this goal), DQ-10 (MuJoCo-as-world,
  approved for campaign #8/#9 physics), DQ-11 (this decision), [ARCHITECTURE.md](../ARCHITECTURE.md)

## Context

Campaign #9 shipped real Qwen-VL semantic recognition on a MuJoCo furnished room and proved the
ceiling the owner then named: **physics fidelity ≠ perception fidelity.** A real VLM on MuJoCo's
basic (non-photoreal) render hits a domain gap — noisy, flaky grounding. The owner's actual goal is
a **high-fidelity PERCEPTION sim**: a G1 humanoid doing autonomous navigation AND manipulation,
driven by **real camera images + multi-sensor data → real recognition / real VLN**, not just
physics. Owner constraints: Isaac Sim is **too heavy** (excluded); "if it's only MuJoCo, the existing
go2 + NAVSTACK already covers nav" — so MuJoCo-render-VLN is not the differentiator.

This re-opens the DQ-10 substrate decision as **DQ-11 (CEO gate)**. A 5-agent judge panel (workflow
`waxcsgpd6`, 2026-06-14) scored five candidates against six owner-weighted axes. **No spike was run
this round** — scores are research-grade (license/backend/ecosystem facts + web), and every photoreal
option carries an UNVERIFIED RTX 5080 (Blackwell sm_120) risk that R2 must measure before any build.

## Candidate matrix (weighted, max 60)

Weights: perception×3, manipulation×3, ecosystem×2, vector-cli seam×2, dep-weight×1, RTX-5080×1.

| Candidate | perc | manip | eco | seam | dep | rtx | **total** | verdict |
|---|---|---|---|---|---|---|---|---|
| **co-sim MuJoCo physics + photoreal render (Blender/OptiX)** | 4 | 5 | 3 | 4 | 3 | **4** | **53** | **recommended primary** |
| SAPIEN 3 / ManiSkill3 | 5 | 5 | 4 | 5 | 3 | 2 | 53 | strong alternative |
| Genesis (+ Nyx renderer) | 4 | 4 | 3 | 5 | 3 | 4 | 54 | track, don't bet |
| habitat-3 re-pin (Bullet) | 4 | 2 | 4 | 3 | 3 | 2 | 43 | reject |
| Isaac Sim/Lab | 5 | 5 | 5 | 2 | 1 | 3 | 52 | reject (owner-excluded) |

## Decision (proposed)

Recommend **co-sim: keep MuJoCo for physics (unchanged), bolt on a photoreal renderer for the camera
images** as the R2 spike primary, with **SAPIEN/ManiSkill3 as the strong fallback**. Rationale —
the top three are a paper tie (53/53/54), so the deciders are **reuse** and **risk**, not score:

- **Reuse:** co-sim keeps *all* of campaign #5–#9 — G1 gait, real rigid-body collision, sensors,
  the recognise→navigate architecture, `target_locate` geometry. The two greenfield engines (SAPIEN,
  Genesis) throw that away for a new physics engine.
- **Risk:** co-sim's renderer (Blender Cycles + OptiX) is the **only** candidate with *confirmed*
  RTX 5080/Blackwell support (2026 GPU benchmarks); Blender ships its own CUDA/OptiX runtime, side-
  stepping our torch/CUDA-13 toolchain; license is clean (GPL, subprocess-isolated like ffmpeg).
  SAPIEN (rtx=2) and Genesis (closed-source 1-month-old Nyx wheel) both bet on an *unproven*-on-
  Blackwell renderer *and* discard MuJoCo reuse.
- Prior art **MuBlE** (arXiv 2503.02834) is the exact pattern: MuJoCo physics + Blender PBR feeding a
  VLM, with measured sim-to-real gain over a basic renderer; vision runs 2–10 Hz while force control
  runs full rate.

**co-sim's one real unknown is render latency** (offline Blender = minutes/frame; must drop samples +
OptiX-denoise to hit ~2–10 Hz without re-introducing a denoiser-noise domain gap). That is
measurable in one afternoon — which is exactly the R2 spike.

### Honest dissent (steelman)
If the owner weights **manipulation-benchmark maturity + cleanest in-process seam** above MuJoCo
reuse, **SAPIEN/ManiSkill3 is the better engine** (5/5 perception+manip, first-class G1+Go2, pure
in-process gymnasium API, no socket bridge) — *provided* its Blackwell question (only an unresolved
RTX 5090 GitHub issue exists today) resolves in R2. Genesis scores highest on paper but its photoreal
case rests entirely on the proprietary, weeks-old, closed-source Nyx wheel (vendor-hostage + unverified
quality + unverified Blackwell) — promising, but not a bet today (same call as Genesis in DQ-2).

### Rejected
- **habitat-3 re-pin:** Bullet rigid bodies and articulated furniture joints are real, but the
  manipuland **grasp is a suction/snap abstraction** (object sticks to the gripper tip) — the same
  architectural limit that killed habitat 0.3.3 for us; a newer pin does not fix it. And photoreal
  scenes (HM3D/MP3D) are static meshes while the only manipulable scenes (ReplicaCAD) are lower-
  fidelity CAD — you cannot get photoreal *and* manipulable in one scene. Plus no native G1/Go2 and
  Meta has ended maintenance.
- **Isaac Sim/Lab:** highest capability ceiling (RTX photoreal + PhysX manip + first-class G1/Go2)
  but owner-excluded as too heavy (≥50 GB, Omniverse Kit, subprocess-only seam), and a live Blackwell
  `TiledCamera` hang (IsaacLab #4951) jeopardizes the very camera path this campaign needs.

## Integration seam (kernel stays world-agnostic — rule 2)

MuJoCo physics steps **in-process** in the repo venv (unchanged). A `PhotorealRenderer` adapter
implements `get_camera_observation()`: read MuJoCo cam pose + body/joint poses → either (A) send a
scene-graph delta over a socket to a persistent `blender --background --python server.py` subprocess
(reuse the campaign #9 habitat conda-subprocess+socket bridge scaffold; `bpy` is import-once-per-
process so it *wants* a subprocess) which returns a Cycles/OptiX RGB frame, or (B) an in-process
3DGS/torch rasterizer for static-scene pixels with MuJoCo movers composited on top. `BaseProtocol`
(`set_velocity`/`navigate_to`/`get_camera_observation`) is untouched; only the world adapter changes.
Async double-buffering hides render latency (vision 2–10 Hz, physics+force full rate).

## Consequences

- **No repo dependency added until the CEO approves DQ-11.** R2 spike installs Blender + assets only
  in `~/sandbox/c10-substrate-spike/`; repo venv stays clean.
- If co-sim passes R2 (latency loop-viable AND Qwen-VL grounds furniture it failed on in #9), it is
  the clear winner — reuse + confirmed Blackwell + now-measured fidelity/latency. If latency fails,
  R2 falls through to a SAPIEN/Genesis Blackwell smoke test.
- First post-approval task: prune the superseded #9 MuJoCo-VLM-render perception code; the world-
  agnostic furnished-room builder, recognise→navigate architecture, and `target_locate` geometry are
  reused on the new substrate.
- **AVOID `madrona_mjx`** (the tempting in-ecosystem MuJoCo batch raytracer): deprecated, capped at
  CUDA ≤12.5.1, no sm_120/Blackwell support.

## R2 spike plan (sandbox, no repo deps)

1. Baseline: render the existing #9 G1-room chair/sofa via `mujoco.Renderer`, feed Qwen-VL (openrouter),
   record the #9-style grounding failure (the control to beat).
2. Install Blender 4.x; import a CC0 PBR room + a PBR G1 asset; set one MuJoCo body pose; render ONE
   headless Cycles+OptiX frame on the RTX 5080; feed Qwen-VL — confirm it now recognizes what it failed.
3. Latency: time OptiX renders at samples {8,32,128} @ 512² on the 5080 — is any quality loop-viable
   (<500 ms ≈ 2 Hz)? Eyeball denoiser artifacts at the fast setting.
4. Seam: stand up `blender --background` subprocess + socket server (reuse habitat bridge); send cam
   extrinsics + body poses from the repo venv, get a PNG back, assert pose match (no sync drift).
5. Manipulation foreground: spawn box + gripper, step a MuJoCo grasp (contact force > 0 unchanged),
   render the grasp moment photoreal, ask Qwen-VL "is the gripper touching the box?" (stresses the
   moving-foreground weak spot).
6. (time-boxed contrast) if co-sim latency fails: SAPIEN/Genesis Blackwell import + one rt frame +
   Qwen-VL grounding, to pick the alternative.

## R2 spike results (2026-06-14, `~/sandbox/c10-substrate-spike/`, repo zero-dep)

Ran the co-sim spike with Blender 4.5.10 LTS (portable, not in venv/git). Evidence: real
openrouter Qwen-VL calls (~$0.011 total); `baseline_mujoco_room.png`, `blender_room_s{8,32,128}.png`.

- **Killer risk RESOLVED — co-sim runs on RTX 5080 Blackwell.** Blender Cycles+OptiX backend =
  `OPTIX: NVIDIA GeForce RTX 5080 Laptop GPU`. Latency @640×480: first frame 2347 ms (OptiX kernel
  warmup), then **826 ms @ 32 samples (~1.2 Hz), 979 ms @ 128**. A persistent render process avoids
  re-warmup → loop-viable for vision-at-low-rate VLN (the MuBlE pattern). The one thing that could
  have killed co-sim does not.
- **Honest correction:** an initial "MuJoCo grounds 0/3" reading was a *camera-framing artifact*
  (free-cam azimuth was reversed — MuJoCo `azimuth=0` faces the furniture, not 180; the lens was on
  the grey wall). Properly framed + coloured, the #9-style MuJoCo flat render @160px grounds **chair
  0.95, plant 0.70–0.85, misses sofa (read as "room divider") = 2/3**. #9's "works but flaky" was
  correct; the ceiling is *flakiness*, not zero.
- **Apples-to-apples (same VLM, 160px, same low-poly assets, same camera):** Blender grounds **chair
  0.90, sofa 0.95 (fixes the one MuJoCo missed!), plant flaky (→"lamp")**. Both ≈2/3, failing on
  *different* objects → **a wash on toy assets.** A resolution sweep (160 vs 512 px) swung randomly
  and 512 was sometimes *worse* → grounding is **stochastic/flaky on low-poly assets, not resolution-
  bound.**
- **Decision-relevant insight (refines the whole campaign):** on low-poly game assets the *renderer
  swap alone is a modest gain, not a categorical leap.* The real levers are **(a) photoreal ASSETS**
  (3DGS room scans / real PBR furniture, not toy meshes) and **(b) a more robust perception pipeline**
  (resolution, consistency voting, de-flaking) — not the render engine per se. This **strengthens
  co-sim**: since swapping the engine alone is only a modest win, paying the full migration cost to
  *discard MuJoCo physics reuse* (SAPIEN/Genesis) buys the same asset-bound ceiling. co-sim keeps all
  #5–#9 physics and adds photoreal assets incrementally behind one seam.
- Manipulation physics needs no re-spike (co-sim physics is unchanged MuJoCo — real contact already
  proven in #5–#9/DQ-10); the MuJoCo-state→render socket bridge is the validated #9 scaffold.

**Net:** co-sim is feasible and de-risked on its only hard unknown. The remaining open question is
*asset fidelity* — validated in R3 below.

## R3 spike results (2026-06-14) — "asset fidelity is the lever" CONFIRMED

Pulled one genuinely photoreal CC0 asset (PolyHaven `ArmChair_01`, 4K PBR diffuse/normal/ARM),
rendered it in Blender Cycles+OptiX (1214 ms @ 64 samples post-warmup), fed it through the SAME #9
VLM pipeline (160px). Evidence: `photoreal_armchair.png` + real openrouter call.

- **Result:** `chair` → **0.95** ("classic upholstered armchair with wooden legs"); on the `sofa`
  query the VLM *correctly disambiguates* — "this object is a chair, not a sofa." Confident, precise,
  fine-grained, with correct label rejection — vs the toy Kenney chair's flaky 0.90 / sofa-confusion /
  plant→"lamp". **Same 160px pipeline.**
- **Conclusion:** asset fidelity is the real lever, and the 160px downsize is NOT the bottleneck once
  the asset is photoreal. The render engine (Blender OptiX) was necessary; photoreal ASSETS are what
  close the #9 grounding ceiling.
- **Effect on the decision:** co-sim is now validated *end-to-end on the actual RTX 5080*: OptiX runs,
  ~1.2 Hz, photoreal-asset grounding is confident, all #5–#9 physics is reused, and **PolyHaven CC0
  assets are free of SAPIEN's CC-BY-NC commercial caveat.** The recommendation firms from "paper-tie
  primary" to an **evidence-backed confident recommendation of co-sim.** SAPIEN/ManiSkill3 remains the
  alternative if the owner values its richer out-of-box scene + manipulation-benchmark ecosystem
  (assemble-it-yourself is co-sim's cost) over MuJoCo reuse.
- **Honest caveat:** one asset (armchair on a plain floor) — a strong single point, not a batch. Full
  VLN still needs assembled photoreal SCENES (multi-asset rooms or 3DGS background scans), the
  MuJoCo↔Blender pose-sync bridge (the validated #9 scaffold), and a photoreal manipulation foreground
  — all post-approval build, not this PROBE.

**This is the natural pause point: R1–R3 PROBE complete, DQ-11 decision-ready, all remaining
substantive work is owner-gated (pick substrate → build).**

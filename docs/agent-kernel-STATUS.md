# Verified Agent Kernel — STATUS (resume anchor)

One-page "where are we / what's next". Read this first when resuming; durable design is
[ARCHITECTURE.md](ARCHITECTURE.md); hidden-bug lessons are [tricky-bugs.md](tricky-bugs.md);
full round-by-round history is in `git log` + the loop journal (`~/.vector-nano-loop/`).

- Branch: `feat/playground-vln` (campaigns #2–#10 live here; #2–#8 merged to `master` via DQ-4 @ `3e82996`).
- Last updated: 2026-06-14 (campaign #10 R16 — median-detection added; FK shows top-down arm reaches ~below itself, object@x=11 (table centre) is BEYOND reach from the table-edge-blocked dog = scene-geometry limit (localization solved). R17 = place graspable object in reach).).
- Scope guard: this is **vector-os-nano only** — not the UniLab go2arm-grasp work.

## North star

**Vector OS Nano = natural language controls everything**, via a built-in agent that
decomposes NL → plans → executes → verifies → replans (a grounded CLOSED loop). VGG is that
engine; verify is the moat (deterministic, never LLM-graded). Robots are the end; the dev/
macOS path is a means. Generalize across embodiments (arm, go2, g1) — never one-off patches.

## Current state / what's next

**Campaign #10 (NEXT, owner 2026-06-14) — HIGH-FIDELITY perception sim for real VLN + manipulation.**
The owner's goal: G1 doing autonomous navigation AND manipulation in a sim whose PERCEPTION is
real enough to drive genuine VLN — real camera images + multi-sensor data feeding real image
recognition / VLN, **not just physics**. Constraints/lessons:
- Isaac Sim = too heavy.
- **MuJoCo alone is a physics engine** — campaign #9 proved a VLM on MuJoCo's basic render hits a
  perception-fidelity ceiling (domain gap, noisy/flaky grounding). "If it's only MuJoCo, the
  existing go2 + NAVSTACK already covers nav" — so MuJoCo-VLM-render is NOT the differentiator.
- The real need is a **high-fidelity perception substrate** (photoreal/co-sim/other) good enough
  for real recognition + VLN + manipulation. **This re-opens the DQ-10 substrate decision = CEO
  gate (DQ-11).** Spike in `~/sandbox/` before any repo dependency.
- **R1 PROBE done** (workflow `waxcsgpd6`, 5-agent panel; ADR-010): top three are a paper tie
  (Genesis 54 / SAPIEN 53 / co-sim 53 of 60), so the deciders are **reuse + risk**, not score.
  **Recommended primary = co-sim** (keep MuJoCo physics unchanged → reuses all of #5–#9; bolt on a
  Blender/OptiX photoreal renderer = the *only* candidate with confirmed RTX 5080/Blackwell support;
  clean license). **SAPIEN/ManiSkill3 = strong fallback** (best paper scores + cleanest in-process
  seam, but rtx=2 Blackwell unknown + discards MuJoCo reuse). Genesis = track-don't-bet (closed-source
  1-month Nyx wheel). habitat re-pin REJECTED (suction-grasp is architectural; can't get photoreal +
  manipulable in one scene). Isaac REJECTED (owner-excluded + live Blackwell TiledCamera hang).
  co-sim's one real unknown = render latency.
- **R2 PROBE-spike done** (`~/sandbox/c10-substrate-spike/`, repo zero-dep): **co-sim's killer risk
  RESOLVED** — Blender Cycles+OptiX runs natively on the RTX 5080 Blackwell (`OPTIX: RTX 5080`), ~826 ms
  @32 samples (~1.2 Hz, persistent process avoids warmup) = loop-viable. Honest finding: on low-poly
  Kenney assets a renderer swap is only a *modest* grounding gain (Blender 2/3, MuJoCo 2/3, failing on
  *different* objects; grounding is stochastic/flaky, not resolution-bound). **The real lever is
  photoreal ASSETS + perception-pipeline robustness, not the engine** — which *strengthens* co-sim
  (keep all #5–#9 physics reuse; add photoreal assets incrementally; greenfield migration buys the
  same asset-bound ceiling).
- **R3 PROBE-spike done** — "asset fidelity is the lever" CONFIRMED: a photoreal CC0 armchair
  (PolyHaven 4K PBR) rendered in Blender OptiX grounds **chair 0.95, precise + correctly disambiguates
  "not a sofa"** through the same 160px pipeline that gave flaky 2/3 on toy meshes. **co-sim now
  validated end-to-end on the RTX 5080** (OptiX + ~1.2 Hz + confident photoreal grounding + full
  physics reuse + free CC0 assets) → recommendation firms to a confident **co-sim** (SAPIEN alt if its
  richer out-of-box ecosystem outweighs MuJoCo reuse). **DQ-11 RESOLVED (owner 2026-06-14,
  option "borrow don't depend"): self-built lightweight co-sim — MuJoCo physics (reuse #5–#9) +
  Blender Cycles/OptiX photoreal render behind a `PhotorealRenderer` world adapter (kernel untouched,
  rule 2), MuJoCo-state→frame over the #9 subprocess+socket scaffold. Fits our 24.04 / RTX 5080 /
  non-ROS vector-cli stack. Covers BOTH G1/Go2 photoreal VLN AND manipulation (Piper, #17/#19) — what
  the evaluated MATRiX (zsibot, MuJoCo+UE5) doesn't (quadruped-only, no public image, 22.04/Humble-only).
  MATRiX/UE5 = architecture validation, not a dependency; UE5-plugin = optional real-time upgrade only
  if Blender latency blocks. Heavy external deps remain CEO-gated.**
- **BUILD sequence (campaign.md):** (1) `PhotorealRenderer` seam: MuJoCo state → Blender subprocess →
  RGB frame, pose-aligned, behind `get_camera_observation()`, one frame end-to-end (TDD). (2) photoreal
  scene: PolyHaven CC0 PBR furniture + photoreal room (or 3DGS scan), VLM-grounding step-change vs #9.
  (3) G1/Go2 photoreal VLN (reuse recognise→navigate on photoreal frames). (4) manipulation: Piper
  grasp driven by photoreal perception. vector-cli only + screenshots; real VLM always called (rule 5).
- **BUILD R1 DONE** (`vector_os_nano/playground/photoreal/`): the co-sim wire plumbing, repo zero-dep.
  `camera.py` = pure MuJoCo→Blender camera transform (identity-of-columns; both engines share Z-up
  world + camera -Z-forward/+Y-up — TDD pins the axes so a refactor can't slip in drift). `bridge.py` =
  `BlenderBridge` client (clones the #9 subprocess+socket pattern; `blender_bin()`/`blender_available()`
  via `VECTOR_BLENDER`; PNG over base64). `server.py` = Blender-side Cycles/OptiX render server (GPL,
  subprocess-isolated — repo venv never imports `bpy`). 12 tests green (pure pose + fake-server protocol
  + Blender-gated e2e). End-to-end proven on the box: pose → bridge → **OPTIX RTX 5080** render (640×480
  @64 = 1085 ms) of a photoreal CC0 armchair → real Qwen-VL grounds **chair 0.95 + correctly rejects
  sofa** — same fidelity as R3, now through the production seam. Evidence: `~/sandbox/c10-substrate-spike/
  r1_e2e_armchair.png`. NOT yet done (R2): the `PhotorealRenderer` adapter that reads a live MuJoCo
  `(model,data)` (cam pose + body→asset scene spec) behind `get_camera_observation()`, + the photoreal
  multi-asset room. (Note: live qwen-VL hit a transient upstream 429 mid-round — called for real, failed
  loud, succeeded on retry; never faked.)
- **BUILD R2 DONE** (commit pending): the `PhotorealRenderer` world adapter + scene builder. `scene.py` =
  pure `build_room_scene_spec(placements, asset_map, ...)` → Blender scene dict that SUBSTITUTES photoreal
  CC0 assets for the toy meshes at the room's visible centres (unmapped labels skipped — partial libraries
  are valid). `camera.py` gains `look_at_camera_matrix` (build demo/test poses in MuJoCo convention).
  `renderer.py` = `PhotorealRenderer.observe(model, data)` reads the LIVE `g1_head_cam` pose
  (`cam_xpos`/`cam_xmat`/`cam_fovy`) and renders the photoreal frame from EXACTLY that pose → returns the
  `{rgb, cam_pos, cam_mat, fovy}` dict shape recognise→navigate already consumes. 18 playground tests green
  (pure scene + look-at + fake-bridge renderer asserting matrix_world == MuJoCo pose, no drift end-to-end;
  uses a tiny hand-built MJCF, no Blender/network). E2E on the box: a real MuJoCo first-person `g1_head_cam`
  pose → adapter → OPTIX 640×480@64 (~1.08 s) of the photoreal armchair room → real Qwen-VL **chair 0.95**.
  Evidence `~/sandbox/c10-substrate-spike/r2_room_photoreal.png`. Honest: the single-seat armchair also
  grounds as **sofa 0.9** (genuine VLM ambiguity on one upholstered seat — a distinct sofa asset
  disambiguates; asset-library expansion is later).
- **BUILD R3 DONE** (commit pending): photoreal wired into `mujoco_g1.get_camera_observation` as a HYBRID —
  rgb=Blender(OptiX), depth/pose=MuJoCo, one camera frame (the Blender render runs from the exact pose
  MuJoCo rendered the depth at, and over a socket so it's safe off the control thread, unlike MuJoCo's GL
  Renderer). Env-gated `photoreal` flag (default OFF → furnished-room behaviour byte-identical, rule 2/9);
  `boot_g1_agent` enables it on `g1_room_vlm` when `VECTOR_G1_PHOTOREAL` is set. Lazy `BlenderBridge`
  (injectable for tests), closed in `disconnect`; fails loud if requested with no Blender; assets from
  `VECTOR_PHOTOREAL_ASSETS` (heavy CC0 assets never vendored — unmapped targets skipped). `renderer.py`
  gained `render_from_pose`. 22 playground tests green + full suite 1670 passed, no g1 regression. **E2E
  through the REAL g1 base** (booted room+furnished+photoreal headless): `get_camera_observation` returns
  `{rgb(640×480 Blender), rgb_mujoco, depth, cam_pos, cam_mat, fovy}` and the real Qwen-VL grounds **chair
  0.9** on the photoreal frame. Evidence `~/sandbox/c10-substrate-spike/r3_g1_photoreal_obs.png` (+ the
  MuJoCo frame side-by-side).
- **BUILD R5 — vector-cli photoreal ACCEPTANCE PASSED** (owner-requested): live REPL,
  `vector-cli --scenario g1_room_vlm` + `VECTOR_G1_PHOTOREAL=1` (boot banner shows "[PHOTOREAL co-sim:
  Blender/OptiX]"). NL command `找到椅子并走过去` → VGG planner → `recognize_navigate` →
  **[PASS] verify `visited('chair')`, 1/1 verified (21.1s)** — the real VLM grounded the chair on the
  offscreen photoreal Blender frame, depth-at-bbox located it, `navigate_to` drove there, and the
  DETERMINISTIC verify confirmed arrival at the chair's GT coords (rule 5, no teleport). Live viewer
  screenshot shows the G1 standing AT the chair: `~/sandbox/c10-substrate-spike/r5_vcli_viewer.png`. The
  hybrid co-sim is confirmed in the real CLI: physics+viewer=MuJoCo, perception=photoreal Blender. Clean
  teardown (Blender subprocess closed on quit; exit segfault is the known tolerated GL-teardown 139).
  **builds #1–#3 DONE + CLI-accepted.** Remaining: scene-quality polish (per-asset yaw so the chair faces
  the camera, walls, sofa/plant CC0 assets to disambiguate) — optional; and **build #4 = Piper
  manipulation** (reuse #17/#19 MuJoCoPiper + PickTopDownSkill driven by photoreal perception).
- **BUILD R6 DONE — photoreal co-sim generalized to the Go2/Piper embodiment** (commit pending). Extracted
  the furnished-room co-sim wiring into a shared factory `playground/photoreal/cosim.py`
  (`furnished_room_renderer` + `furnished_room_asset_map`) so G1 and Go2 share ONE asset-mapping +
  bridge-lifecycle (rule 7, world-agnostic); refactored g1's `_ensure_photoreal_renderer` onto it (g1
  tests still green). `MuJoCoGo2` gained the same env-gated `photoreal` flag: `get_camera_frame` returns a
  Blender render from the resolved head cam's (`d435_rgb`/`RECOG_CAM`) LIVE pose; lazy bridge closed on
  disconnect; `go2_runtime` enables it on `VECTOR_G1_PHOTOREAL`/`VECTOR_GO2_PHOTOREAL`. 28 playground tests
  green + suite 1676 passed, no go2 regression. E2E through the real go2 base (headless furnished+photoreal):
  `get_camera_frame` → Blender 640×480 → real Qwen-VL **chair 0.9** (`r6_go2_photoreal.png`). This is the
  structural enabler for **build #4**: the Go2 carries BOTH the head camera and the Piper arm, so the pick
  skills' perception (autodetect/recognise) now grounds on photoreal frames.
- **BUILD R7 DONE — photoreal PICK-scene rendering** (commit pending). The pick scene is the apartment
  (table at x=11 + 3 graspable `pickable_*` cylinders: blue/green bottle, red can), a different world from
  the furnished VLM room. Added: `scene.py` `objects` primitive passthrough; `server.py` renders
  cylinder/box primitives (Blender, coloured — no mesh asset); `cosim.py` `build_pick_scene_spec` +
  generic `scene_renderer`; `MuJoCoGo2._ensure_photoreal_renderer` now branches (furnished→room,
  else→pick scene built from LIVE `pickable_*` body poses + colours/sizes read from the model, no
  hardcoding) via new `_pick_objects`/`_pick_table`. 29 playground tests green + suite 1676 passed (the
  3 deepseek reds + 1 habitat ticketed-drive TIMING flake that PASSES in isolation and is outside the
  changed files — not an R7 regression). E2E through the real go2 pick base (headless, photoreal):
  `_pick_objects` correctly reads the 3 live cylinders, `get_camera_frame` renders them photoreal, and
  the real Qwen-VL grounds **"red can" 0.9** (`r7_go2pick_photoreal.png`). Honest: generic "can"/"bottle"
  → `[]` — the simple primitives ground by COLOUR (the R2 toy-primitive flakiness); photoreal can/bottle
  MESH assets would harden it. This proves the photoreal PERCEPTION substrate for manipulation. NEXT (R8):
  close the loop — `vector-cli` go2_piper "认出红色的罐子→抓起来" with the grasp target PERCEPTION-derived
  (recognise→pick, not registered GT) + verify grasped + screenshot; optional photoreal object meshes.
- **First post-approval task:** prune the superseded MuJoCo-VLM-render perception code from #9 (kept
  for now — tested + interconnected; the world-agnostic builder / recognise→navigate / target_locate
  geometry are reused on the new substrate).

**Campaign #9 outcome (MuJoCo furnished-room VLN, feat/playground-vln) — sound engineering, capped by
the perception substrate:**
- World-agnostic furnished-room builder (one builder, two embodiments — g1 + go2; rule #2/#7).
- Real Qwen-VL furniture recognition on the MuJoCo render (both embodiments) — works, but flaky on
  MuJoCo's non-photoreal images (the ceiling above).
- Reliable arrival via **recognise → depth-at-bbox locate → navigate_to** (g1 headless 3/3) after
  visual-servoing proved flaky (tricky Cases 21–23).
- g1 integrated into vector-cli (`start_simulation(sim_type='g1')`, "启动g1"); live viewer shows the
  furnished room; in-REPL g1 walks via a daemon control thread (Cases 22/24). Suite 1648 green.

## Campaign history (one line each; details in git log + loop journal)

- **#2** full-stack居住世界: house world (ReplicaCAD) + 50Hz cmd_vel + real nav stack + full VLN 5/5.
- **#3** kernel refactor: false-PASS family killed (pre_satisfied / evidence-gate / single-source verify).
- **#4** robustness + LLM param-binding + E2E GUI smoke + N6 gait PROBE.
- **#5** G1 real gait (unitree_rl_gym pretrained policy, MuJoCo, zero new deps).
- **#6** G1 closed-loop waypoint navigate (obstacle-aware vgraph; euclidean geodesic on flat).
- **#7** visibility-graph obstacle planner (pure geometry); batches folded into #8.
- **#8** MuJoCo furnished-physics VLN (DQ-10=A): gait+collision+avoidance+lidar/occupancy/camera+
  explore+colour-recognition→go, one vector-cli command, GUI-verified. Merged master (DQ-4).
- **#9** real VLM semantic recognition + recognise→navigate + double-embodiment + g1 vector-cli
  integration (this campaign — see Current state).

## Run / verify

Tests: see CLAUDE.md "Build / test" (canonical chunked command, expected environmental reds:
3 deepseek `.env` + level71 GL segfault — always `--deselect` it; exit 139 + full summary = green).
Quirk: go2 sim load rewrites `mjcf/go2/scene_room_piper.xml` abs paths — `git checkout` it before
committing. Live validation: real CLI + deepseek headless first; GUI/timing behaviours are
owner-window checks (tmux real vector-cli + screenshot) — never claimed verified headless.

## Pointers

- Rules + read order: [../CLAUDE.md](../CLAUDE.md)
- Design: [ARCHITECTURE.md](ARCHITECTURE.md) · Hidden bugs: [tricky-bugs.md](tricky-bugs.md)
- Subsystems: [cli-tool-system.md](cli-tool-system.md) · [skill-protocol.md](skill-protocol.md) · [sim-dev-guide.md](sim-dev-guide.md)
- ADRs: [ADR-006](architecture-decisions/ADR-006-agent-kernel-world-plugin.md) ·
  [ADR-007](architecture-decisions/ADR-007-closed-loop-controller.md) ·
  [ADR-008](architecture-decisions/ADR-008-playground-parallel-track.md) ·
  [ADR-009](architecture-decisions/ADR-009-third-world-simulator-selection.md)
- Plans: none active. Open CEO gate: campaign #10 substrate selection (decision-queue.md).
- Superseded docs/STATUS detail live in git history (`git log --all -- <path>`). No working-tree archive.

## Autonomous loop

Owner-away iterations are campaign-driven and live OUTSIDE the repo:
`~/.vector-nano-loop/{constitution,campaign,journal,next-prompt,decision-queue}.md`.
Start with `/loop` + the constitution prompt (constitution.md = fixed state machine; campaign.md =
current milestones). Past campaign journals are archived as `journal-cN-DONE.md`.

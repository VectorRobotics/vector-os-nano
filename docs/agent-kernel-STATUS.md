# Verified Agent Kernel — STATUS (resume anchor)

One-page "where are we / what's next". Read this first when resuming; durable design is
[ARCHITECTURE.md](ARCHITECTURE.md); hidden-bug lessons are [tricky-bugs.md](tricky-bugs.md);
full round-by-round history is in `git log` + the loop journal (`~/.vector-nano-loop/`).

- Branch: `feat/playground-vln` (campaigns #2–#10 live here; #2–#8 merged to `master` via DQ-4 @ `3e82996`).
- Last updated: 2026-06-14 (campaign #10 DQ-11 RESOLVED — self-built lightweight co-sim, MuJoCo physics + Blender/OptiX photoreal; BUILD unlocked).
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

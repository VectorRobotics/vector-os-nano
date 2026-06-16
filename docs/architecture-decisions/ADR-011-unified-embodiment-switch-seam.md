# ADR-011 — Unified embodiment-switch seam (G1 ⇄ Go2 in one vector-cli session)

Status: PROPOSED (CEO gate — DQ-14). Date: 2026-06-15. Campaign #11 M0.

## Context

Campaign #11's intermediate goal: in ONE vector-cli sim, freely switch G1/Go2, each
running the full capability stack (locomotion + navigation stack + VLN + SysNav), one
NL command end-to-end, all real-sim accepted. Today the embodiments are **two separate
runtimes** (`g1_runtime.boot_g1_agent` / `go2_runtime.boot_go2_agent`) chosen once at
sim start — there is no in-session switch.

A 4-agent capability audit (workflow `wbikcsq4w`) read the real code and produced the
readiness matrix below (each cell verified against `file:line`, not assumed):

| Capability | Go2 | G1 |
|---|---|---|
| **Locomotion** | READY — sinusoidal trot (1.06 m / 3×1.5 s); convex MPC optional/unverified | READY (conditional) — unitree_rl_gym RL policy; `motion.pt`/scene assets not vendored, setup state UNKNOWN |
| **Nav stack** | PARTIAL — only `--sim-go2` ROS2 FAR subprocess path has `navigate_to` (returns bool); furnished `go2_room_vlm` has none | READY — `NavigateToPointSkill` → in-process `g1_vgraph`, 3-value dict, CLI-verified |
| **VLN** | PARTIAL — perception only (real Qwen-VL on Blender frame); no `recognize_navigate` (no `navigate_to`/depth), only `vlm_seek` | READY — `recognize_navigate` full chain CLI-verified |
| **SysNav** | MISSING — no wiring; no `get_pano`; `sysnav_tool.py:86` gate requires `get_pano` | MISSING — same; `MuJoCoPano360` exists but unattached |

Cross-cutting gaps: `navigate_to`/`get_camera_observation`/`get_pano` are **not in
BaseProtocol** (duck-typed optional extensions); signatures diverge (G1 `navigate_to`
→ dict, Go2 proxy → bool); walk semantics diverge (Go2 blocking `drive_for` vs G1
streaming deadman); SysNav consumer chain works but the **feed side** is missing on both.

## Decision

Add a **`switch_embodiment` seam in the vcli runtime/tool layer** — kernel/BaseProtocol
unchanged (rule 2/7). The key finding: the rebind already exists — `start_simulation`
(`sim_tool.py:216-271`) does the full rebind (app state + `wrap_skills` + `init_vgg`) but
**refuses** when a base is connected (`sim_tool.py:145-156`, "stop_simulation first").

1. **Reuse `app_state` as the active-embodiment holder** (no new manager class — lower
   risk). Extract the rebind block into a single-source `SimStartTool._rebind_agent(app,
   context, agent, world)` (rule 3 — start & switch must not drift into split-brain).
2. **New `SwitchEmbodimentTool`** (`vcli/tools/switch_tool.py`, category `sim`):
   no-op if already on target → **boot target runtime first, then swap** (boot-then-swap so
   a failed switch never leaves zero embodiments) → `_shutdown_agent(old)` (already kills
   Go2 subprocess group + disconnects + joins physics thread + closes GL/viewer) →
   `_rebind_agent(new)`. Fail loud (rule 8) on unknown target / missing assets / boot error.
3. **Skill registration by capability probe**, not embodiment name: `boot_*_agent` builds a
   **fresh `SkillRegistry`**, registering a skill only when the base supports it
   (`callable(base, 'navigate_to')` / `hasattr(base, 'get_camera_observation'/'get_pano')`).
   This makes the NAV/VLN/SysNav asymmetry fall out into rule-3 single-source vocab — the
   planner never gets a skill the current base can't run (`go2_runtime` already omits
   `NavigateToPointSkill` this way).

### M0 verification corrections (why we verify, not assume)
- The audit claimed `registry.unregister` exists (`base.py:257`) — **FALSE**: `SkillRegistry`
  (`core/skill.py:296`) has `register`/`list_skills` but **no `unregister`**. Correct approach:
  **rebuild a fresh registry on boot** (which `boot_*_agent` already does) instead of
  unregistering; the stale robot-category **tool** entries on the ToolRegistry side are what
  must be cleared on swap — exact mechanism to confirm in M1.
- Verified true: refuse-branch `sim_tool.py:145-156`; rebind `:216-271`; `_shutdown_agent:359`
  (killpg + disconnect); `boot_g1_agent`/`boot_go2_agent`; `BaseProtocol` has
  `walk`/`set_velocity`/`disconnect` but NOT `navigate_to`/`get_pano` (duck-typed confirmed).

## Milestone order (de-risked, smallest seam first)

- **M1 (minimal closed loop):** locomotion-only g1⇄go2 runtime switch. Add `switch_embodiment`
  + extract `_rebind_agent`. Real-sim accept: one vector-cli session — `walk 1m` (Go2 trot,
  measured displacement) → `switch to g1` → `walk 1m` (G1 RL policy, measured) → `switch to
  go2`; **switch ≥5× and confirm VRAM / thread / GL-handle counts do not grow**, no viewer
  leak. (Verifies the seam itself: teardown/rebind/registry.)
- **M2:** VLN symmetry — give `MuJoCoGo2` `navigate_to` (+ geodesic), register
  `RecognizeNavigateSkill` via capability probe (lidar fallback exists). Accept: same session,
  "找到椅子并走过去" on G1 (R5) → switch → same on Go2 (real Qwen-VL + real arrival).
- **M3:** Nav-stack unification (most divergent) — runtime `NavProvider` abstraction
  (`navigate_to(x,y,tol)→dict` + `geodesic_distance`); Go2 ROS2 proxy adapts bool→dict; single
  `NavigateToPointSkill`. (Must first real-sim confirm FAR actually routes indoors, not just
  door-chain fallback.)
- **M4:** SysNav cross-embodiment — attach `MuJoCoPano360` to G1/Go2, expose `get_pano`; lift
  `wire_sysnav_feed` out of `habitat_runtime` to an embodiment-agnostic seam; replace the
  `sysnav_tool.py:86` habitat-only gate with a capability probe. (First verify
  `mapping_mecanum_sim.yaml` install/share path — UNKNOWN.)

## Consequences

- **New nodes/topics:** none (pure in-process Python; ROS2 interfaces untouched).
- **Files:** `switch_tool.py` (new); `sim_tool.py` (extract `_rebind_agent` + hint on the
  refuse branch); later `go2_runtime`/`habitat_runtime`/`mujoco_go2`/`pano360` wiring.
- **Model change:** embodiment lifecycle goes from "choose-once-at-start" to "runtime-switchable".
- **kernel/BaseProtocol:** zero change (rule 2/7).

## Risks

- **GL/EGL context + MuJoCo physics-thread leak across reconnect** (historical crash class) —
  must real-sim switch ≥5× and confirm VRAM/handles/threads do not grow (NOT a unit test).
- **G1 gait assets** (`motion.pt`/scene, ~57 MB) setup state UNKNOWN (rule 5, not peeked) —
  switching to G1 may fail boot; run `setup_g1_gait.sh` before M1 accept.
- Go2 convex MPC backend + FAR indoor routing both UNVERIFIED — M2/M3 must confirm in real sim.
- Rebind logic must be single-source (`_rebind_agent`) or start/switch drift → rule-3 split-brain.

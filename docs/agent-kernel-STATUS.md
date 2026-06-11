# Verified Agent Kernel — STATUS (resume anchor)

One-page "where are we / what's next". Read this first when resuming; durable design is
[ARCHITECTURE.md](ARCHITECTURE.md); hidden-bug lessons are [tricky-bugs.md](tricky-bugs.md).

- Branch: `feat/playground-vln` (off `master` @ PR #13 merge; gait-fix branch merged as PR #13).
  Campaign: third world — photoreal/semantic nav world (VLN-first, SysNav revival rides along);
  M0 DONE: ADR-009 (Proposed) recommends pinned habitat-sim 0.3.3 conda-py3.9 subprocess +
  bridge — spike-verified on the Linux box (photoreal RGB, equirect pano+depth, 819K-pt
  unprojected cloud, GT odom). owner APPROVED 2026-06-10; HM3D agreement (DQ-1) still pending — interim scenes are license-free (test-scenes/HSSD/Replica).
  M1 DONE (backend-agnostic, ungated): `Scenario` carries additive `sim_backend`/`scene_ref`;
  a non-MJCF world registers/resolves through `WorldRegistry` with the engine surface intact
  (tests/vcli/test_scenario_backend_seam.py). M2 (the habitat world itself) awaits the gate.
- Last updated: 2026-06-10.
- Scope guard: this is **vector-os-nano only** — not the UniLab go2arm-grasp work.

## Current state (2026-06-11)

- **Go2 explore gait (飘/瘸腿): FIXED, owner-confirmed live.** Root cause was two-clock skew
  (physics ~0.65× real-time vs wall-tick velocity ramps in the nav bridge) — full case in
  [tricky-bugs.md](tricky-bugs.md) Case 1. Fix `d7e158b`: `_follow_path` ramps + wall-escape
  state machine integrate against sim-dt (`hardware/sim/sim_clock.py` + `MuJoCoGo2.get_sim_time()`).
  Escalation to `/clock`+`use_sim_time` (plan B) NOT needed. Env-gated diagnostics
  (`VECTOR_PHYS_LOG`/`VECTOR_MPC_LOG`/`VECTOR_CMDVEL_LOG` in `mujoco_go2.py`) kept — zero cost
  when off; remove when no longer useful.
- **Arm touchstone: hardened end-to-end** (perception, grounding, decompose target binding,
  real-time timeouts, singular/plural intent, long chains incl. foreach grab-everything,
  place/handover, honest+fast failure, grounded verify). Single-skill AND long-chain NL control
  work live on real deepseek, 中/英.
- **Suite + venv reality:** see CLAUDE.md "Build / test" (venv is `.venv` now; 4 documented
  environmental reds; sim tests open real GL windows on a desktop).

## North star (restated 2026-06-05)

**Vector OS Nano = natural language controls everything**, via a built-in agent that
decomposes NL → plans → executes → verifies → replans (a grounded CLOSED loop). VGG is that
engine; verify is the moat (deterministic, never LLM-graded). Robots are the end; the dev/
macOS path is a means. Generalize across embodiments (arm, go2, future) — never one-off patches.

## Shipped (condensed; details in `git log` + tricky-bugs.md)

- **Phases A–C.2:** kernel/world decoupling (robot-free boot); dev world acts + StrategyStats
  (`80916f4`/`f5b9eb4`, hardening `8e961f8`, e2e `bee46f7`); capability seam + cross-capability
  routing (`62fcfc1`/`2a7c942`).
- **Arm + Stages 0–2** (`aebd61e`, `cdbfada`): SO-101 NL control; window-by-default sim
  (`--headless` opt-out, mjpython auto re-exec); Blackboard + `${step.path}` binding +
  `result_data`; registry-derived decompose vocab (no split-brain).
- **Playground v1 + ADR-008 seam prelude;** **Stage 4** control-flow IR (`foreach` expand from a
  producing step via Blackboard, obs-driven replan hook) + **Go2 second embodiment**;
  **Stage 5** unified controller (every turn — chat included — is a verified trace;
  `VECTOR_LEGACY_TURN=1` one-release fallback; stage5 plan deleted as fully shipped).
- **Live-hardening I–VII** (real CLI + deepseek): retry-strategy fix, REPL log quiet, meta-input
  routing, decompose JSON robustness, mjpython locate fix, **P0 segfault** sync-exec gate
  (`118f886`), RobotWorld sim-oracle grounding (`75cbdba`), LLM-side target binding (`4a2edf7`,
  `f53bd04`), sim grasp z_offset (`c953d72`) → headless NL grasp chain works end-to-end.
- **R2 round (real GUI run findings):** R2-2 skill-declared `typical_duration_sec` timeout floor ·
  R2-3 singular/plural intent + unbound→nearest (sim-gated) · R2-5 sim motor auto-allow
  (ALL-semantics safety) · R2-6 ROS2-absent logs at DEBUG · R2-7 detect/foreach `name`+`label`
  contract, pick fail-fast on absent target, detect honest-empty, grounded place verify
  (`not holding_object()`), target-aware `holding_object(target)` + `picked_object` recorded ·
  R2-8 session compaction orphaned-tool-message 400 fix · go2 base verify grounding (`2edeb25`).
- **Phase E:** W1.1 learning-tier evidence gate (`85d59a2`) · W1.2 fail-loud world-registration
  preflight (`7afe2c0`) · W1.3 scene_graph TextLLM adapter (`c3103ad`) · W1.4 playground
  step-primitives wired live → **Wave 1 complete**; W2.4 typed `failure_class` into replan
  (`14ae43c`) · **W2.3 ObjectMemory re-query-freshest** (2026-06-10, implemented inline after
  the harness-flake warning: readers re-query the live SceneGraph TTL-gated, decay cache as
  fallback, byte-identical without a ref; 46 level57 + 49 level60/61 green).
- **2026-06-10 third-world campaign session (feat/playground-vln, 13 commits, autonomous loop):**
  G0 Linux-clone bootstrap (.venv on the verified stack, sim.sh hardened) · M0 ADR-009
  simulator selection (habitat-sim recommended, 2 hardware spikes: photoreal RGB, equirect
  pano+depth, 819K-pt unprojected cloud, GT odom) · M1 backend-aware Scenario + non-MJCF world
  registration proven · FALSE-SUCCESS class killed (#2b `verify_strengthen` named-target verify;
  #3 kernel `step_output()` per-step self-verify) · Phase E Wave 2 complete (W2.1 run-registry/
  daemon/CLI ops + W2.2 RUN_ID watchdog /proc sweep) · Stage 3 referring-expression grounding
  (`Objects (live)` in world_context + exact-name binding guidance) · W3.1 WorldBlueprint ·
  W3.3 first slice (protocol-based provider resolution, `_base` seam) · hygiene (dead examples/
  deleted, doc truth-ups). Suite 1140→1186 green.
- **This branch (gait fix, merged PR #13):** engine sync-exec gated on mjpython only — Linux REPL responsive (`fcc6b20`) ·
  go2 gated diagnostics + `get_sim_time()` (`6a39f6c`) · venv reconcile `.venv-nano`→`.venv`
  (`13a9429`) · **the gait sim-dt fix** (`d7e158b`) · docs: tricky-bugs casebook + STATUS condense
  (this commit).

## OPEN — prioritized backlog

0. **OWNER LIVE-TEST FINDING (2026-06-11, vector-cli --scenario apartment): the persona /
   status surface does not know the habitat world exists.** Banner correctly shows
   `Base: habitat_kinematic` and the SysNav feed is up, but asking "怎么启动" gets "还没跑
   仿真" + Go2/arm sim offers, and "怎么启动habitat kinematic" sends the LLM into bash/tool
   exploration. Root: `robot_status` tool + DynamicSystemPrompt's robot-state block (and the
   persona's sim-start guidance) are MuJoCo-era — they must reflect a connected habitat
   base/world: base name, scenario id, position, live world-model object count, and "the
   world is ALREADY running; no start needed". Fix first next session.

1. **M2 — the habitat third world: OWNER APPROVED DQ-2 (2026-06-10) — IN PROGRESS.**
   Part 1 SHIPPED: `playground/habitat/` server (standalone py3.9, conda subprocess, JSON/socket,
   navmesh `try_step` kinematics, geodesic/snap/semantic-objects oracle ops) + `HabitatBridge`
   (PORT handshake, fail-loud, watchdog-tagged) + `HabitatBase` (full BaseProtocol + narrow
   provider specs; vy honestly unsupported) — REAL subprocess e2e green on the Linux box
   (walk on skokloster navmesh, exact odom, geodesic consistency). Part 2 SHIPPED — M2 COMPLETE: `apartment` preset (embodiment `mobile`, license-free
   apartment_1, rooms honestly empty until HM3D-Semantics/DQ-1), `geodesic_dist` predicate
   (kernel oracle, fail-safe inf — the VLN success criterion), `--scenario apartment` boots
   the kinematic base via `_maybe_init_habitat_agent` (M1 `sim_backend` dispatch), and the
   acceptance e2e passes LIVE: registry world → real conda-subprocess base → real GoalVerifier
   sandbox evaluating at_position (discriminating) / geodesic_dist / facing. M3 muscle layer SHIPPED (2026-06-10): server `navigate_to` (shortest-path waypoint
   following, deterministic + bounded, honest stuck-stop) + always-on 256x256 egocentric RGB
   + `render` op (base64 PNG, lazy PIL in the conda env); `HabitatBase.navigate_to/render_rgb_png`.
   Live e2e: cross-apartment navigation verified by `geodesic_dist < 0.5` through the REAL
   sandbox + a real PNG frame. **M3 COMPLETE (2026-06-11)**: `NavigateToPointSkill` (base-generic, verify_hint =
   the VLN criterion) + mobile skill set (walk/turn/stop/navigate_to — the habitat agent's
   registry is rebuilt base-only per rule 3, no arm skills taught) + LIVE acceptance
   (tests/vcli/test_habitat_nl_slice.py, VECTOR_LIVE_LLM=1): 5 NL instructions 中/英 incl. a
   two-waypoint chain, real LLM + real conda subprocess + real harness, all verified-done,
   ≥4 hard-evidence, every verify a deterministic sandbox predicate (geodesic_dist /
   at_position / True) — never an LLM judge. M0-M3 COMPLETE. **M4 IN PROGRESS (owner directive 2026-06-11)** — Part A SHIPPED:
   habitat equirect color+depth pano op (512x1024, pose-synced) + `HabitatSysnavBridge`
   publishing the SysNav input triplet (/camera/image equirect RGB, /registered_scan
   WORLD-frame cloud unprojected from the same-frame depth — reuses the tested LidarSample
   PointCloud2 layout, /state_estimation GT odom). rclpy works IN the venv; the node
   roundtrip is integration-tested (discovery-then-drain pattern — single-threaded wait-set
   starvation documented in the test). Live: real pano pair + plausible world cloud verified.
   Part B: `tare_planner` BUILT in the sibling workspace (3m34s, msgs importable) and the
   CONSUMPTION path is green with REAL messages — stub /object_nodes_list →
   LiveSysnavBridge → WorldModel (sysnav_<id> canonical ids, VLM-confirmed confidence 1.0 —
   the v2.4 contract works unchanged; tests/integration/test_sysnav_consumption_path.py,
   skips where the workspace isn't sourced). **M4 COMPLETE (2026-06-11, DQ-3 approved)**: the FULL SysNav graph runs on the photoreal
   world — habitat pano → detection_node (YOLOE-26x TensorRT engine, exported on this GPU) →
   semantic_mapping_node (SAM2.1 base+) → /object_nodes_list → LiveSysnavBridge → WorldModel
   produced REAL labelled nodes (sofa/light/picture in apartment_1, conf 0.70 unverified tier).
   SysNav runs on ITS OWN venv (.venv-sysnav, numpy 1.26 — ROS cv_bridge is numpy-1-ABI;
   lesson: uv silently upgraded numpy when adding spacy/rerun — re-pin after every install
   batch). Harness: scripts/verify_habitat_sysnav.py. FOLLOW-UPS: object z-coords look
   frame-shifted (sofa z=-1.57) — audit the fusion pixel/extrinsic convention vs our equirect;
   old MuJoCo-era SysnavSimTool stays unregistered (superseded by the habitat runner).
   **M5 (2026-06-11): demo harness shipped + owner live-testing handover.**
   scripts/demo_third_world.py runs the full visible loop (semantic objects → NL → navigate →
   deterministic verify, artifacts under ~/sandbox/m5_demo/). The demo EXPOSED and we FIXED two
   real gaps: (1) Objects (live) now carries per-label best-confidence COORDS (semantic
   navigation was vacuous without them); (2) navigate-to-OBJECT false success killed — the
   navigate analogue of #2b: NavigateToPointSkill takes a `label` param resolved against the
   LIVE world model, unknown object FAILS LOUDLY with the known-object set (tested), guidance
   forbids inventing x/y. HabitatBridge.request is now thread-safe (demo drives nav + pano
   from two threads). Demo verdict PARTIAL (3/4: one run hit the LLM shortening strategy
   'navigate_to'→'navigate' — fail-loud worked; naming-affordance polish item). CLI handover:
   VECTOR_HABITAT_SYSNAV=1 wires the pano feed + /object_nodes_list consumer into the REPL
   agent (scripts/launch_sysnav_nodes.sh runs the perception pair). Owner testing live. Everything ungated that M2/M3 need is in place: seam (M1), grounding
   (referring expressions, step_output verify), ops (registry/daemon/watchdog), WorldBlueprint.
   On approval: conda-py3.9 habitat subprocess + socket bridge (go2-sim pattern), kinematic
   `BaseProtocol` over navmesh `VelocityControl`/`try_step`, oracle predicates
   (at_position/visited/object_visible/geodesic_dist), `vector-cli --scenario hm3d_house`.
2. **Phase E remainder:** `--daemon` goal-runner design (align with real need after M2) ·
   W3.2 capability factory + model-zoo bridge (strategic — do WITH the C.3 owner framing,
   not before it) · W3.3 remaining getattr-by-string sites (incremental, mechanical).
3. **Stage 3 grounding remainder:** referring-expression resolution SHIPPED 2026-06-10 —
   `world_context` now carries `Objects (live): ...` (world-model labels + property hints,
   sim-oracle body-name fallback, capped, fail-safe) and `_TARGET_BINDING_GUIDANCE` teaches
   resolving the user's reference (any language, attributes) to the EXACT listed name at
   plan/replan time; an unresolved binding fails loudly and replan re-binds from the fresh
   list (kernel stays deterministic — the LLM is the language component; combined with the
   #2b target-aware verify the wrong-object path cannot false-pass). STILL OPEN: the VLM
   `MuJoCoPerception`/`DetectSkill` real-perception path on no-oracle hardware.
   **Phase C.3/C.4** stays blocked behind that
   ([agent-kernel-phase-c-plan.md](agent-kernel-phase-c-plan.md)).
4. **Owner-gated window checks (cannot verify headless):** R2-1 `--sim-go2` macOS in-process
   window (opens, no segfault, walk animates; frozen-when-idle is expected v1) · R2-4 single ^C
   aborts to prompt under mjpython, second ^C exits.
5. **Chores:** pin/vendor `convex_mpc` in `pyproject.toml` (a venv rebuild silently loses the
   numpy2 fixes — tricky-bugs Case 2) · cli.py 32 pre-existing ruff errors · version is still
   0.1.0 in pyproject/version.py while history speaks of v2.x (owner call) ·
   `ros2/nodes/agent_node.py` still calls the removed `agent.execute` (every /execute service
   call errors into the broad except) · hygiene 2026-06-10: `examples/` DELETED (all five
   taught the removed v0.1 `agent.execute` API and crashed on it; git history is the archive),
   `run_turn_unified` dark-launch docstring corrected (it is live), phase-d Stage-4/5 status
   headers truthed up, cli-tool-system diagram now shows the unified controller ·
   canary test for the
   private `mujoco.viewer._MJPYTHON` probe (`viewer_mode`) · remove the TEMP gated diagnostics
   when gait work is truly done. Linux clone (`~/Desktop/vector_os_nano`) reconciled 2026-06-10:
   synced to master, `.venv` rebuilt (mujoco 3.9 / numpy 2.4.6 / pin 4.0 + convex_mpc editable),
   `sim.sh` hardened (no silent system-python fallback). NOTE on Linux the known-red
   `test_sim_tool_lifecycle_dev_to_arm_to_dev` is a process-killing SEGFAULT
   (`mujoco.viewer.launch_passive` GL thread) — deselect it there; an intermittent
   at-exit GL-teardown segfault observed twice now (GLFW 'not initialized' warning, AFTER the
   summary prints — results complete both times; treat exit 139 with a complete tail as green).

## Run / verify

Tests: see CLAUDE.md "Build / test" (canonical command, expected environmental reds).
Quirk: go2 sim load rewrites `mjcf/go2/scene_room_piper.xml` abs paths — `git checkout` it
before committing. Live validation pattern: real CLI + deepseek headless first; GUI/timing/^C
behaviors are owner-window checks — never claim them verified headless.

## Pointers

- Rules + read order: [../CLAUDE.md](../CLAUDE.md)
- Design: [ARCHITECTURE.md](ARCHITECTURE.md) · Hidden bugs: [tricky-bugs.md](tricky-bugs.md)
- Plans: [agent-kernel-phase-e-plan.md](agent-kernel-phase-e-plan.md) (CURRENT, Wave 2) ·
  [agent-kernel-phase-d-plan.md](agent-kernel-phase-d-plan.md) (Stage 3 remainder open) ·
  [agent-kernel-phase-c-plan.md](agent-kernel-phase-c-plan.md) (C.3/C.4 decisions open)
- ADRs: [ADR-006](architecture-decisions/ADR-006-agent-kernel-world-plugin.md) ·
  [ADR-007](architecture-decisions/ADR-007-closed-loop-controller.md) ·
  [ADR-008](architecture-decisions/ADR-008-playground-parallel-track.md)
- Superseded docs live in git history (`git log --all -- <path>`). No working-tree archive.

## Autonomous /loop prompt (the standing mission for owner-away iterations)

Run via `/loop <this prompt>` (no interval => self-paced). Mission-oriented + high-autonomy: each firing
advances the mission as far as it safely can, not a single tiny edit.

> **Mission: advance vector-os-nano toward a generalizable PHYSICAL agent for robots.** Iterate autonomously
> (owner away; auto-approve on; current work branch; ONLY this project, never UniLab). This is
> a mission, not a checklist: make natural language truly control a robot through a grounded CLOSED loop
> (understand -> decompose -> plan -> execute -> verify -> replan -> recover), generalizing across embodiments
> (arm, go2, future) AND tasks. Simulation is a MEANS; the end is a physical robot agent. Push the LLM through
> the whole cognitive layer (language, decomposition, planning, strategy/verify selection, recovery); keep
> grounding/verify/safety DETERMINISTIC — verify is the moat, never LLM-graded. Prefer fixes that remove an
> embodiment asymmetry or generalize a mechanism over one-off patches.
>
> Each iteration, ORIENT then act with judgment — you have wide latitude:
> - ORIENT: read `docs/agent-kernel-STATUS.md` (OPEN backlog + current state), `docs/ARCHITECTURE.md`,
>   `docs/tricky-bugs.md`, and memories `vector-os-nano-live-hardening` / `-language-layer` /
>   `workflow-model-tiering`. Optionally run the real cli + deepseek to feel current state and discover issues.
> - CHOOSE a meaningful objective — a bug class, a capability, an architectural improvement — that moves the
>   mission forward. You MAY pursue a FARTHER goal across several workflows/edits in one iteration; don't
>   artificially stop at one tiny change. Decompose it yourself and advance as far as you safely can.
> - BUILD: reproduce/diagnose first, then implement via focused dynamic **Workflows** (implement -> 2-3
>   adversarial reviewers -> critic), chaining as many as the objective needs. Pin agent models per
>   `workflow-model-tiering`. Write/extend tests for logic that matters; add evals where output quality matters.
> - VERIFY HONESTLY: keep the canonical suite green (`.venv/bin/python -m pytest tests/vcli
>   tests/unit/vcli -q`); validate behavior headless with the real cli + deepseek wherever possible. Some
>   things only reproduce in the owner's mjpython window (GUI render, real-time timing, Ctrl-C under mjpython)
>   — reason carefully, add what headless coverage you can, and CLEARLY hand the visual/timing confirmation to
>   the owner. Never claim a GUI-visual works unverified.
> - COMMIT + RECORD: self-review the real diff; green-then-commit in isolated, logically-scoped commits,
>   updating STATUS (+ ARCHITECTURE if structure/contracts changed; + tricky-bugs.md if a hidden bug was
>   cracked) and the relevant memory in the SAME commit (Doc Governance). Record what you did + what's next so
>   the next iteration resumes cleanly. **Do NOT push.**
>   `git checkout mjcf/go2/scene_room_piper.xml` if a go2 test dirtied it.
>
> DON'T interrupt the owner to ask — make reasonable decisions and proceed. Only stop/surface on a GENUINE
> blocker: the canonical suite goes red and you can't get it green (halt-on-red; salvage + commit partial
> green), something needs the owner's GUI/hardware confirmation, or an action would be destructive /
> irreversible / outward-facing (push, deploy, delete owner data). Otherwise keep advancing the mission,
> iteration after iteration, then schedule the next one.
>
> Current backlog: the **OPEN — prioritized backlog** section above (advance any; not exhaustive — discover more).

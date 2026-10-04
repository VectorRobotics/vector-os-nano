# Plan — sim mobile manipulation, world-class (CURRENT phase plan / GOAL)

> Resume anchor for the 2026-10 refactor campaign. Cold-read THIS doc, then
> `docs/agent-kernel-STATUS.md`, then `git log goal/sim-mobile-manip`.
> Branch `goal/sim-mobile-manip` (local; push only when the owner says so).
> Analysis artefacts (not in git): `/data/vos-analysis/` — 14 subsystem reports
> (`*.md`), `understand.json` (304 claimed bugs, 254 confirmed / 44 plausible /
> 6 refuted after adversarial verification), `design.json` (4 architectures, 3 judges).

## Goal (owner directive, 2026-10-03)

"完全提升和重构整个项目的代码库，完成所有的目标，把它做成全世界最强的 agentic system，
让机器人能够在 sim mobile manipulation，解决所有 bug。" This approves every stage the
docs previously queued as a CEO gate (S4 embodiment registry, S5 policy plugin, S6
capability exposure, S8 legacy-producer retirement, new deps). Unchanged non-negotiables:
bare `vector-cli` + NL is the only acceptance face; the honest-verify spine only gets
stricter; embodiments are config (Rule 11); every capability has a sim e2e (Rule 12).

## Acceptance (all measured through bare `vector-cli` + NL unless stated)

- **A1 fetch** — go2+Piper, MuJoCo house, live LLM: "把绿色瓶子拿过来" / "bring me the red
  can" → (explore if unknown) → localize → navigate → grasp → carry → deliver, GROUNDED
  by independent GT. Paired-seed campaign N≥20 over ≥3 objects × ≥2 spawns: ≥ 80%
  (Wilson 95% lower bound reported); place-on-surface N≥10: ≥ 70%.
- **A2 honest** — 0 false GROUNDED over every trial, checked by a GT grader that shares
  no code with the oracles; the adversarial moat corpus has a mutation-sensitivity check.
- **A3 recover** — injected faults (perception offset, object nudged, nav blocked, grasp
  miss) end GROUNDED via the attempt semantics or as an honest failure with a reason;
  ≥ 60% recovered.
- **A4 control** — "停" (or Ctrl-C) zeroes the base command within 0.1 s and the body is < 0.05 m/s by 1.0 sim-s while a long skill runs; Ctrl-C never
  exits the REPL; embodiment switch go2 ↔ go2_piper ↔ so101 leaks no threads/pgids.
- **A5 gate** — one sanitized, memory-capped regression command: hermetic tier 100%
  green (documented deselects only), scripted bare-REPL e2e 3/3 consecutive passes.
- **A6 bugs** — every confirmed/plausible bug in the ledger is closed by a failing-first
  regression test or an explicit DECISIONS entry.
- **A7 shape** — one sim runtime, one producer, one grasp, one place; no module > 800
  lines (cli.py < 300); dead tracks deleted; docs match code (doc-drift gate).

## Architecture decision (judge panel 2026-10-04: unify-runtime, with grafts)

Root cause of most bugs = two of everything (two go2 runtimes, two producers, four pick
variants). End state:

- **One in-process sim session** behind NL `start_simulation(embodiment=…)`: the
  in-process MuJoCoGo2 + MuJoCoPiper rig (where every measured grasp number came from);
  the subprocess bridge + `Go2ROS2Proxy`/`PiperROS2Proxy` are deleted for sim.
  `SimRuntime` owns MjData behind a lock, applies mutations at step boundaries, hands out
  atomic RGB-D+pose snapshots, has a lockstep mode for deterministic e2e, and a GT view
  that skills/perception may not import (firewall test).
- **External TARE/FAR = optional capability**: in-process rclpy bridge thread over the
  runtime + the CMU stack as an owned subprocess (`VECTOR_NAV_STACK`, read-only,
  `--params-file`, own pgid). In-repo vgraph (Minkowski inflation, goal snapping) +
  coverage explore are the default when no stack is configured.
- **One producer** (native ReAct) behind a TurnRouter: permission gate on every tool,
  per-run CancelToken, single-flight, stop pre-check without an LLM call, a turn worker
  so NL "停" is read mid-turn, no legacy fallback. Legacy VGG producer deleted (S8) once
  its live dependencies (GoalVerifier, CapabilityRegistry, ObjectMemory binding) re-home.
- **Spine, stricter only**: allowlist verify grammar; `@oracle(channel, transition,
  scope)` tag — only tagged callables are evidence (stubs, memory readers, dev predicates
  fail closed); F→T transition rule (already-true at baseline never grounds); attempt
  semantics (FAIL→PASS on the same predicate, CAUSED, verifies); turn-level task contract
  (fetch ⇒ delivered, place ⇒ object_on; False at turn-start GT, True at turn-end GT);
  each actuating skill's `verify_hint` auto-verified; verify-probe refusal. New GT
  oracles: near_object, in_room, holding_object (weld ∧ contacts ∧ lift relative to rest),
  lifted, released, kept_holding, object_on, localized, arm_at_home (manifest).
- **Mobile-manip skills**: ONE `grasp` (target from scene-graph instance / box with
  provenance / vocab; approach from manifest; re-perceive at standoff; live-base re-IK;
  contact-gated weld; ≤2 regrasps; honest failure), ONE `place` (surface height,
  descend to contact, release, retreat), carry (holding invariant), honest `home`.
- **World manifest** (`config/worlds/*.yaml`): rooms, pickables (aliases/colour/category),
  surfaces, user location — replaces hardcoded `_COLOR_TO_SCENE` / weld triples.
- **Perception & memory**: snapshot-only perception (K/shape asserted), localizer
  plausibility gates, GDINO-over-vocab discovery without a VLM, instance identity +
  last_seen in the scene graph, atomic world-keyed persistence, no (0,0) sentinel.
  Detectors are routable capabilities (GDINO default; RynnBrain at :8786 as a second
  backend — the North Star's "route to the right model" demonstrated, not claimed).
- **Embodiment as config (S4/S5)**: generalise MuJoCoGo2 in place (manifest-driven
  constants, `STEP_FNS[policy.ref]`), runtime arm attach; a zero-code `go2_alt`
  manifest must stand and walk. SO-101 untouched behind its manifest.

## Waves (each wave = one workflow; each WP in its own worktree, independently verified, then merged)

| wave | work packages |
|---|---|
| 0 | WP01 test hygiene + sanitized gate + bare-REPL PTY harness + reactive fake LLM · WP02 spine S1 (allowlist grammar, @oracle) · WP03 strangler flip: NL start_simulation → in-process SimSession (+ same-process realloc spike, re-exec fallback) |
| 1 | WP04 dead-track sweep · WP05 native safety (permissions, CancelToken, single-flight, turn worker, no double-exec) · WP06 SimRuntime (lock, snapshots, lockstep mode) |
| 2 | WP07 spine S2 (attempts, causation channels, task contract, auto verify_hint) · WP08 nav + coverage explore · WP09 Piper hardening (contact-gated weld, live-base IK, tracking moves) |
| 3 | WP10 world manifest + GT oracles · WP11 perception + memory + detector capabilities · WP13 TARE/FAR capability · WP14 native persona/context/memory for mobile manip |
| 4 | WP12 grasp/place/carry + paired-seed trial runner with independent GT grader · WP15 retire legacy producer (S8) |
| 5 | WP16 split cli.py, one runtime builder, credentials/sessions · WP17 manifest-driven driver + policy dict (S4/S5) |
| 6 | WP18 docs reconciliation + final live acceptance campaign (A1–A7) |

## Disciplines (every WP, every agent)

- Tests only via `/data/vos-analysis/run_tests.sh` (ROS sourced with venv pinocchio
  first, random ROS domain, 16 GB cgroup, rosm shim). Sims only via
  `/data/vos-analysis/bin/with-sim-slot` (≤2 concurrent, domains 42/43, memory cap).
  Never `rosm nuke`, never `pkill -f` by name, never touch `/dev/shm`.
- This box is shared: other projects' ROS stacks on domain 0, RynnBrain/ollama/z-manip
  containers must stay up. Never write into `~/Z-Robotics-Lab/Z-Navigation-Stack-go2w`.
- Verify through the real launcher (bare `vector-cli` on a PTY), never a hand-built rig.
  Never trust a self-report: the merge step re-runs the WP's acceptance.
- Spine edits are stricter-only; every new predicate has an adversarial negative test.
- Each bug fix lands with a failing-first regression test; update the ledger.

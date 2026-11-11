# Plan — release 1.0 (CURRENT phase plan)

> Resume anchor for the 2026-11 release campaign. Cold-read THIS doc, then `git log release/1.0`.
> Branch `release/1.0` (local, forked from `goal/sim-mobile-manip`; push / tag only when the owner says so).
> Supersedes `plan-sim-mobile-manip.md` (kept until R9 only because wave 1 cites its WP07/WP11/WP14 definitions).
> Working artefacts (not in git): `/data/vos-analysis/release-1.0/` — `read/*.json` (17 subsystem
> reports, blocker/high adversarially verified), `branches/*.json` (the 4 unmerged-branch assessments),
> `ledger.json` (464 open rows; one file per work package under `wp/`), `gate/` (gate runs).

## Goal (owner directive, 2026-11-11)

"阅读并且重构代码库 … release 第一个正式版本，确保是一个闭环的项目。"

**Closed loop** means: a fresh clone installs; bare `vector-cli` + natural language plans → routes →
executes → verifies → recovers for every capability the README advertises; one gate proves it; the docs
say exactly what the code does. Anything that cannot be made true *and verified on this box* is cut from
the 1.0 promise and labelled experimental — never shipped as a silent half.

## Scope decisions (coordinator defaults — the owner may override any row)

| item | in 1.0 | reason |
|---|---|---|
| go2, go2+Piper, SO-101 sims | supported | proven through the bare REPL (E-suite) |
| native tool-use producer | the ONLY producer | legacy VGG producer, IntentRouter, primitives, playground, `vector-eval` are deleted (S8 was approved 2026-10-03) |
| MCP server | rebuilt on the native producer: stdio default, loopback SSE, sim bodies | advertised in README; today it cannot start (mcp SDK 2.x API) and is open on 0.0.0.0 |
| perception weights | fetched on first use with a visible notice; `VECTOR_OFFLINE=1` forbids it | a fresh clone has none and the loaders are offline-pinned |
| install | editable clone is the documented path; the wheel is also made correct | `config/` sits outside the package; `.gitignore` `assets/` drops 232 meshes from a build |
| G1 humanoid | experimental: honest error + the pinned asset-fetch script restored; full closure (skills, causation, e2e) awaits the owner | its model/policy (57 MB, BSD-3, unitree_rl_gym) were never in git |
| real SO-101 hardware | experimental, SDK-only, unadvertised | unreachable from `vector-cli`, no hardware on this box to verify |
| external TARE/FAR stack | experimental opt-in (`VECTOR_NAV_STACK`) with an honest readiness check; the in-process bridge (old WP13) moves to 1.1 | needs a stack a fresh clone does not have |
| one generic embodiment driver (old WP17) | 1.1 | XL; the docs stop claiming zero-code robots |
| version | 1.0.0 | first official release |

## Acceptance (internal gate + bare `vector-cli`; the owner's face is the bare REPL)

- **A1 install** — fresh clone → `uv pip install -e ".[all]"` → `vector-cli` starts; no key ⇒ one clear
  instruction; a built wheel contains the MJCF assets and `config/`, and runs a sim from outside the repo.
- **A2 flows** — scripted bare-REPL e2e green for: dev write (E0), launch + switch (E1, E6), SO-101
  pick / place / home (E7+), go2 rooms + explore (E8, E9), go2+Piper grasp → carry → deliver and
  place-on-surface (new), stop + Ctrl-C (E5), MCP stdio smoke (new), slash commands (new).
- **A3 honest** — 0 false GROUNDED: moat corpus (incl. vacuous compares, verify probes) green; the
  independent `tools/gt_grade.py` audit over the e2e tier reports none.
- **A4 one of everything** — one producer, one runtime builder (REPL / `-p` / MCP), one sim session
  path, one grasp + one place for the mobile arm; grep finds no import of a deleted module.
- **A5 gate** — `scripts/gate.sh all`: hermetic 100 %, sim 100 %, e2e 3 consecutive green runs;
  `tests/known_reds.txt` holds environment rows only; a hermetic CI workflow exists.
- **A6 safe + portable** — no machine paths, no kill-by-name; grep / file / web tools guarded and
  tested; MCP on loopback.
- **A7 docs** — README + getting-started commands each verified; ARCHITECTURE / STATUS / DECISIONS
  current; CHANGELOG; the doc-drift gate green.
- **A8 live smoke** (billed, small) — real model through the bare REPL: fetch, navigate, SO-101 pick;
  numbers recorded in STATUS.

## Target shape

```
vcli/      cli.py (argparse + main) · runtime.py (ONE build_runtime) · repl.py · slash.py · render.py
           engine.py (bind world, tools, permissions) · native_loop.py (THE producer)
           cognitive/ = the honest-verify spine only (trace_store, evidence_classifier,
           actor_causation, goal_verifier, object_goal, coord_goal, oracle_registry, abort, types)
           tools/ (guarded code tools, sim lifecycle, scene-graph query, go_to_room, skill wrapper)
           worlds/ (dev, robot, world_manifest, sim_oracles) · backends/ (anthropic, openai, scripted)
sim/       SimSession · SimRuntime · GroundTruth · nav/ (vgraph, follow, coverage)
hardware/  sim/ (MuJoCoGo2, MuJoCoPiper, MuJoCoArm, MuJoCoG1*) · so101/* (real, experimental)
skills/    so101 pack · go2 locomotion/nav pack · mobile-manip pack (grasp, place, carry)
perception/ detector capability (GDINO | RynnBrain) · EdgeTAM · localizer · grasp point · vocab
core/      Agent · Skill/Registry/Context · SceneGraph · types      mcp/  thin server on the same runtime
config/    worlds/*.yaml · nav.yaml (packaged)                      embodiments/  robot.yaml manifests
```

## Work packages (each: own worktree + branch `rel/<WP>`, independent verifier, then merged by the coordinator)

| WP | wave | scope | status |
|---|---|---|---|
| I1 | 1 | fast-forward `wave4j-int2` (3 gate reds: reset_pose race, OSQP swallows Ctrl-C, `localized` causation) | merged f3f1276 |
| I23 | 1 | merge `wave3-wp11` (perception + memory) and `wave3-wp14` (native persona / context / lifecycle); one `localized` channel (WP11's `observation_count`); post-merge fixes | open |
| I4 | 1 | land `wave2-wp07` (spine S2) on top: tag-driven causation, attempt groups, task contract bound to the manifest (fetch, place), probe hole closed, vacuous-compare corpus | open |
| R1 | 2 | ONE producer: delete the legacy producer + IntentRouter + primitives + playground + `vector-eval` + their tests; `engine.init_vgg` → a plain world binder | open |
| R2 | 2 | skills hygiene: skill packs registered per embodiment, no auto-appended home, dead skills deleted, coordinate-nav honesty | open |
| R3 | 2 | tool security: `rg` argv, resolved-path guard + credential deny-list, `web_fetch` asks + redirect checks; dead diagnostics tools deleted | open |
| R4 | 3 | fetch closure: one place skill for the mobile arm, carry, deliver; E-scenarios fetch + place | open |
| R5 | 3 | split `cli.py`; ONE `build_runtime`; slash-command table; explicit credentials (no borrowed OAuth) | open |
| R6 | 3 | MCP server on `build_runtime` + native producer; SDK 2.x; stdio / loopback; verdict returned | open |
| R7 | 3 | sim + perception hardening: SO-101 stop + `ctrl_motion`, posture persistence, nav-stack honesty, headless GL, weight provisioning, loud perception failures | open |
| R8 | 4 | packaging + hygiene: `config/` in the package, `.gitignore`, pyproject, dead periphery, NOTICE, gate honesty, CI | open |
| R9 | 4 | docs: README, getting-started, ARCHITECTURE, STATUS, DECISIONS, CHANGELOG, version | open |
| G1 | 4 | G1: restored fetch script + honest error (default) or full closure (owner) | open |
| REL | 5 | final acceptance A1–A8, adversarial review of the whole diff, tag `v1.0.0` (local) | open |

## Disciplines (every WP, every agent)

- Tests only through `scripts/gate.sh` (memory-capped, own ROS domain, throwaway HOME). Anything that
  starts a sim or `vector-cli` goes through `/data/vos-analysis/bin/with-sim-slot` (≤ 2 sims on the
  box). Any other python goes through `/data/vos-analysis/bin/capped`. Never `rosm nuke`, never kill
  by name, never touch `/dev/shm`, never write into `~/Z-Robotics-Lab/`. This box is shared.
- Work only inside your worktree, on branch `rel/<WP>`; commit there; never push; never touch the
  main checkout or another WP's branch.
- Every ledger row of the WP gets a disposition: fixed (with the failing-first test that pins it),
  not-a-bug (evidence), or deferred (reason). Nothing is left silent.
- Deletion over addition; no new layers, registries or plugin systems; the shortest diff that closes
  the loop. The spine only gets stricter; every new predicate or guard has an adversarial negative test.
- Docs: WP agents do NOT edit `docs/` or `README.md` (one docs pass, R9, rewrites them from the final
  code); they return doc notes instead. This overrides the per-commit STATUS rule for this campaign.
- Verify through the real launcher (bare `vector-cli` on a PTY), never a hand-built rig. A self-report
  is never accepted: the verifier re-runs the acceptance, the coordinator re-runs the gate after merging.

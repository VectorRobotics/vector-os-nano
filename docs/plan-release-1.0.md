# Plan — release 1.0 (CURRENT phase plan)

> Resume anchor for the 2026-11 release campaign. Cold-read THIS doc, then `git log release/1.0`.
> Branch `release/1.0` (local, forked from `goal/sim-mobile-manip`; push / tag only when the owner says so).
> Supersedes `plan-sim-mobile-manip.md` (kept until R9 only because wave 1 cites its WP07/WP11/WP14 definitions).
> Working artefacts (not in git): `/data/vos-analysis/release-1.0/` — `read/*.json` (17 subsystem
> reports, blocker/high adversarially verified), `branches/*.json` (the 4 unmerged-branch assessments),
> `ledger.json` (464 open rows; one file per work package under `wp/`), `gate/` (gate runs),
> `assets/g1_gait/` (the fetched G1 model + policy).

## What 1.0 is (owner directive, 2026-11-11)

"阅读并且重构代码库 … release 第一个正式版本，确保是一个闭环的项目。" — and — "我的底层理念是 agentic
robot system，第一个版本可能只能仿真了，但我也希望是真正的 agentic + 跨仿真，跨本体。"

**Vector OS Nano 1.0 is an agentic robot runtime whose agent does not depend on the simulator or on
the robot body.** A natural-language goal goes to one model-driven tool-use loop; its tools are
derived from what the current body can do; it executes on a body inside a simulator backend; every
step is graded by oracles over that backend's ground truth; it recovers inside the loop. Sim-only.

Three pillars, each with a proof that runs through bare `vector-cli` + natural language:

- **P1 agentic** — one producer (the native tool-use loop), a turn-level task contract, attempt
  semantics (FAIL → retry → PASS verifies), tools and persona derived from the body's capabilities,
  the agent starts / switches simulator and embodiment by itself, spatial memory.
  *Proof:* scripted e2e including injected-fault recoveries; a small live-model matrix.
- **P2 cross-simulator** — one physics contract; MuJoCo and PyBullet both pass one conformance
  suite; the same NL task ends GROUNDED on both, graded by each backend's own ground truth.
  *Proof:* conformance suite + e2e on both backends + "add a simulator" guide (one class).
- **P3 cross-embodiment** — a robot is a manifest (+ assets, + policy); bodies are composed
  (base ⊕ arm ⊕ sensors); skills bind by capability, never by robot name. Five bodies in four
  morphology classes: `so101` (arm), `piper` (arm), `go2` (quadruped), `go2_piper` (mobile
  manipulator = go2 ⊕ piper), `g1` (humanoid). *Proof:* e2e per body + a robot added by manifest
  only, with zero Python + "add a robot" guide.

**Closed loop** on top: a fresh clone installs; everything the README advertises works and is
proven by one gate; the docs say exactly what the code does. What cannot be made true and verified
on this box is cut from the promise and labelled experimental, never shipped as a silent half.

## Scope decisions (owner-confirmed where marked ✔; otherwise coordinator defaults)

| item | in 1.0 | note |
|---|---|---|
| second simulator | **PyBullet**, in-process ✔ | builds on py3.12 (probed); Isaac Sim / ManiSkill plug in later as out-of-process backends behind the same contract (1.1) |
| G1 humanoid | supported ✔ | assets (57 MB, BSD-3, unitree_rl_gym @ pinned commit) are fetched by a script, never committed |
| native tool-use producer | the ONLY producer | legacy VGG producer, IntentRouter, primitives, playground, `vector-eval` deleted |
| MCP server | rebuilt on the native producer: stdio default, loopback SSE, sim bodies | today it cannot start (mcp SDK 2.x) and is open on 0.0.0.0 |
| perception weights | fetched on first use with a visible notice; `VECTOR_OFFLINE=1` forbids | a fresh clone has none |
| install | editable clone documented; the wheel is also made correct | `config/` outside the package; `.gitignore` `assets/` drops 232 meshes |
| real SO-101 hardware | experimental, SDK-only, unadvertised | no hardware here to verify |
| external TARE/FAR stack | experimental opt-in (`VECTOR_NAV_STACK`), honest readiness | the in-process bridge (old WP13) is 1.1 |
| version | 1.0.0 | |

## Architecture — N simulators × M robots with N + M code

```
 skills · perception · kernel · oracles            never import a simulator
        │  BaseProtocol · ArmProtocol · GripperProtocol · Camera · GroundTruth
 generic drivers, one per morphology class         LeggedBase · Arm · Gripper · Camera · Lidar
 GroundTruth, ONE implementation                   (oracle-only; skills and perception never see it)
        │  control policies, one per controller    servo · convex_mpc · torchscript · sinusoidal
        │  PhysicsWorld  (~25 methods)             ← the ONLY thing a simulator implements
 backends, one per simulator                       sim/backends/mujoco · sim/backends/pybullet
        ▲  SceneDescription (canonical IR)         MJCF / URDF compiled once by MuJoCo: bodies, joints,
        │                                          inertia, geoms, meshes, actuators, cameras
 manifests, one per robot / world                  embodiments/<id>/robot.yaml (compose: base ⊕ arm)
                                                   config/worlds/<world>.yaml
```

- **PhysicsWorld** (`vector_os_nano/sim/physics.py`) — load(scene) · close · time / step · name →
  opaque handle · body pose / twist · joint position / velocity · joint servo (target, kp, kd,
  max force) and joint torque · contacts · attach / detach (the grasp constraint) with a transition
  log · AABB · raycast · render(camera) → RGB-D + segmentation. A new simulator is this one class
  plus a green conformance suite.
- **SimRuntime** — the one stepper thread (or lockstep), step-boundary mutations, `paused()`,
  `wait_sim`, one render thread. Simulator-neutral: it drives any PhysicsWorld.
- **GroundTruth** — written once over PhysicsWorld (object poses, contacts, attachments, clearance,
  visible pixels). The firewall test stays: skills and perception never import it.
- **Generic drivers** read the manifest and talk to PhysicsWorld only: `Arm` (IK / FK on the
  canonical model, settle-checked moves, abort, commanded-motion counter), `Gripper` (contact-gated
  attach), `LeggedBase` (velocity command → policy → torques, posture, stop brake, odometry),
  `Camera`, `Lidar`. Uniform protocols, no per-robot method drift (Rule 11).
- **Policies** (`sim/policies/`) are chosen by `policy.ref` in the manifest.
- **Manifests** hold every per-robot number. `compose:` builds `go2_piper` from `go2` ⊕ `piper`
  at scene-build time (the committed `go2_piper.xml` goes away).
- **Skill packs bind by capability**: locomotion + navigation (base), manipulation (arm + gripper),
  perception (camera), mobile manipulation (base + arm). Per-robot parameters live in the manifest.
- **Session** — `start_simulation(embodiment, backend)`; `backend.supports(config)` returns a
  reason or None, so the backend × embodiment matrix is derived, never hand-written.

Staging (strangler; every stage leaves the gate green):
S1 contract + generic runtime + generic GroundTruth, MuJoCo behind PhysicsWorld, no driver change →
S2 SceneDescription export + PyBullet PhysicsWorld + kinematic / dynamic parity tests →
S3 generic Arm / Gripper / Camera (piper tabletop on both backends; go2_piper's arm moved over) →
S4 generic LeggedBase + policy plugins (go2 on both backends; G1 on MuJoCo) →
S5 session × backend × embodiment: NL selection, composition manifests, capability-bound skill packs,
conformance suite, zero-code robot test.
Fallbacks, decided by evidence and stated in the docs: if the MPC gait cannot be made stable on
PyBullet, go2 / go2_piper stay MuJoCo-only in 1.0 and `supports()` says why; G1 on PyBullet is not
attempted; SO-101 moves to the generic arm driver only if its 5-DoF pick survives, otherwise it keeps
its dedicated driver behind the same protocols.

## Acceptance (internal gate + bare `vector-cli`; the owner's face is the bare REPL)

- **A1 install** — fresh clone → `uv pip install -e ".[all]"` → `vector-cli` starts; no key ⇒ one
  clear instruction; a built wheel contains the MJCF assets and `config/` and runs a sim from
  outside the repo.
- **A2 flows** — scripted bare-REPL e2e green for every body: dev write; launch + switch body and
  backend; SO-101 and Piper pick / place / home; go2 rooms + explore; go2+Piper grasp → carry →
  deliver and place-on-surface; G1 stand / walk / navigate; stop + Ctrl-C; MCP stdio smoke; slash
  commands.
- **A3 honest** — 0 false GROUNDED: moat corpus (vacuous compares, verify probes) green; the
  independent `tools/gt_grade.py` audit over the e2e tier reports none, on both backends.
- **A4 one of everything** — one producer, one runtime builder (REPL / `-p` / MCP), one session
  path, one grasp + one place for the mobile arm, one GroundTruth, one driver per morphology class.
- **A5 cross-sim** — the conformance suite is green on MuJoCo and PyBullet; body poses agree to
  1e-5 at random joint configurations; simulators are imported only under `sim/backends/<name>/`
  (import-firewall test); arm pick-and-place and base navigation end GROUNDED on both backends
  (or the fallback is documented with its evidence).
- **A6 cross-embodiment** — five bodies e2e; a test manifest (different spawn / stance / camera)
  stands and walks with zero Python; go2_piper is composed from go2 + piper; no arm skill is
  offered on an arm-less body and the locomotion pack is offered on every legged one.
- **A7 gate** — `scripts/gate.sh all`: hermetic 100 %, sim 100 %, e2e 3 consecutive green runs;
  `tests/known_reds.txt` holds environment rows only; a hermetic CI workflow exists.
- **A8 safe + portable** — no machine paths, no kill-by-name; grep / file / web tools guarded and
  tested; MCP on loopback.
- **A9 docs** — README + getting-started commands each verified; ARCHITECTURE / STATUS / DECISIONS
  current; add-a-simulator and add-a-robot guides; CHANGELOG; the doc-drift gate green.
- **A10 live matrix** (billed, small) — real model through the bare REPL across body × backend ×
  task (fetch, navigate, pick / place); success and false-GROUNDED counts recorded in STATUS.

## Work packages (each: own worktree + branch `rel/<WP>`, independent verifier, merged by the coordinator)

| WP | wave | scope | status |
|---|---|---|---|
| I1 | 1 | fast-forward `wave4j-int2` (3 gate reds) | merged f3f1276, verified (sim 10/10, E5 + E10 green) |
| I23 | 1 | merge `wave3-wp11` (perception + memory) and `wave3-wp14` (native persona / context / lifecycle) | running |
| I4 | 1 | land `wave2-wp07` (spine S2) + bind the task contract to the manifest + close the probe hole + vacuous-compare corpus | queued behind I23 |
| R3 | 1 | tool security: `rg` argv, resolved-path guard + credential deny-list, `web_fetch` asks + redirect checks | running |
| R1 | 2 | ONE producer: delete the legacy producer + IntentRouter + primitives + playground + `vector-eval` + their tests | open |
| S1 | 2 | physics contract on MuJoCo: `PhysicsWorld`, simulator-neutral `SimRuntime`, ONE `GroundTruth` over it; drivers untouched | open |
| S2 | 2 | `SceneDescription` export from the compiled model + PyBullet `PhysicsWorld` + parity tests | open |
| S3 | 3 | generic Arm / Gripper / Camera drivers; `piper` tabletop body on both backends; go2_piper's arm moved over; SO-101 stop + commanded-motion counter | open |
| R5 | 3 | split `cli.py`; ONE `build_runtime`; slash-command table; explicit credentials (no borrowed OAuth) | open |
| S4 | 4 | generic LeggedBase + policy plugins (convex_mpc, torchscript, sinusoidal); go2 on both backends; G1 on MuJoCo with skills + causation; posture persistence | open |
| R4 | 4 | fetch closure: one place skill for the mobile arm, carry, deliver; e2e fetch + place | open |
| R6 | 4 | MCP server on `build_runtime` + native producer; SDK 2.x; stdio / loopback; verdict returned | open |
| S5 | 5 | session × backend × embodiment: NL selection, derived support matrix, `compose:` manifests, capability-bound skill packs, conformance suite, zero-code robot test, asset fetch (G1, weights) | open |
| R8 | 5 | packaging + hygiene: `config/` in the package, `.gitignore`, pyproject (`[pybullet]` extra), dead periphery, NOTICE, gate honesty, CI | open |
| R9 | 6 | docs: README, getting-started, ARCHITECTURE, STATUS, DECISIONS, add-a-simulator / add-a-robot guides, CHANGELOG, version | open |
| REL | 6 | final acceptance A1–A10, adversarial review of the whole diff, tag `v1.0.0` (local) | open |

## Disciplines (every WP, every agent)

- Tests only through `scripts/gate.sh` (memory-capped, own ROS domain, throwaway HOME). Anything that
  starts a sim or `vector-cli` goes through `/data/vos-analysis/bin/with-sim-slot` (≤ 2 sims on the
  box). Any other python goes through `/data/vos-analysis/bin/capped`. Never `rosm nuke`, never kill
  by name, never touch `/dev/shm`, never write into `~/Z-Robotics-Lab/`. This box is shared.
- Work only inside your worktree, on branch `rel/<WP>`; commit there; never push; never touch the
  main checkout or another WP's branch.
- Every ledger row of the WP gets a disposition: fixed (with the failing-first test that pins it),
  not-a-bug (evidence), or deferred (reason). Nothing is left silent.
- Deletion over addition; no layer without two users; the shortest diff that closes the loop. The
  spine only gets stricter; every new predicate or guard has an adversarial negative test.
- Never resolve a conflict by taking one whole side of a file.
- Docs: WP agents do NOT edit `docs/` or `README.md` (one docs pass, R9, rewrites them from the final
  code); they return doc notes instead. This overrides the per-commit STATUS rule for this campaign.
- Verify through the real launcher (bare `vector-cli` on a PTY), never a hand-built rig. A self-report
  is never accepted: the verifier re-runs the acceptance, the coordinator re-runs the gate after merging.

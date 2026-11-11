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

## Goal — run until done (owner directive, 2026-11-11)

"设置个 goal，一直到做完为止；代码结构也要 concise，删除仓库废代码，重构结构也是目标的一环。"

The campaign runs wave after wave, without waiting for the owner, until ALL of this is true:

1. **Acceptance A1–A11 below is green**, each proven by the gate or by a recorded run.
2. **The code base is concise** (A11): the tree is the charter's target tree; no dead code; one
   implementation per capability; no compatibility shims; fewer source lines than at the start.
3. **`v1.0.0` is tagged locally on `release/1.0`.** Nothing is pushed until the owner says so.

The coordinator stops only for a decision that is genuinely the owner's (a product choice the plan
does not already settle) or for a guardrail (anything destructive outside this repository).
Progress lives in the work-package table below and in `/data/vos-analysis/release-1.0/ledger.json`.

## Scope decisions (owner-confirmed where marked ✔; otherwise coordinator defaults)

| item | in 1.0 | note |
|---|---|---|
| second simulator | **PyBullet**, in-process ✔ | builds on py3.12 (installed in the venv); go2 walks there on the sinusoidal policy, the convex-MPC trot is a time-boxed stretch; Isaac Sim / ManiSkill plug in later as out-of-process backends behind the same contract (1.1) |
| G1 humanoid | supported ✔ | assets (57 MB, BSD-3, unitree_rl_gym @ pinned commit) are fetched by a script, never committed |
| native tool-use producer | the ONLY producer | legacy VGG producer, IntentRouter, primitives, playground, `vector-eval` deleted |
| MCP server | rebuilt on the native producer: stdio default, loopback SSE, sim bodies | today it cannot start (mcp SDK 2.x) and is open on 0.0.0.0 |
| perception weights | resolved through the Hugging Face cache, fetched on first use with one visible notice; `HF_HUB_OFFLINE=1` forbids | a fresh clone has none |
| install | editable clone documented; the wheel is also made correct | `config/` outside the package; `.gitignore` `assets/` drops 232 meshes |
| real SO-101 hardware | experimental, SDK-only, unadvertised | no hardware here to verify |
| external TARE/FAR stack | experimental opt-in (`VECTOR_NAV_STACK`), honest readiness | the in-process bridge (old WP13) is 1.1 |
| version | 1.0.0 | |

## Architecture — N simulators × M robots with N + M code

The binding contract is [plan-release-1.0-interfaces.md](plan-release-1.0-interfaces.md) (the
interface charter, reviewed against the code by four adversarial reviewers). In short:

```
 skills · perception · kernel                      never import a simulator, never see ground truth
        │  BaseProtocol · ArmProtocol · GripperProtocol · CameraProtocol (CameraFrame: no segmentation)
 generic drivers, one per morphology class         LeggedBase · Arm · Gripper · Camera · Lidar
        │  ControlPolicy, one per controller       convex_mpc · torchscript · sinusoidal
 world side, ONE implementation each               SimRuntime · grasp decision · GroundTruth · oracles
        │  PhysicsWorld (14 methods)               ← the ONLY thing a simulator implements
 backends                                          sim/backends/mujoco · sim/backends/pybullet
        ▲  CompiledScene (canonical model)         MJCF / URDF compiled once by MuJoCo; ids, state
        │                                          layout and kinematics are canonical on every backend
 data plug-ins                                     embodiments/<id>/ (robot.yaml + model files, compose:)
                                                   worlds/<id>/ (world.yaml + scene)
```

The owner's four requirements map to the charter's sections: explicit interfaces (§3, the
`interfaces/` package), explicit config-file interfaces (§4, the `schemas/` package: `robot.yaml`,
`world.yaml`, the user config, the environment catalogue), explicit file / model interfaces (§5: a
robot is a folder, a world is a folder, one state directory, policy and weight files, versioned
outputs), everything modular (§6: a layering test with an allowlist that only shrinks).

Measured facts the design rests on (charter review): body-frame kinematic parity MuJoCo ↔ PyBullet
is ~1e-7 m; PyBullet clears torque commands every step (the backend re-issues them); the go2 PD
stance matches to the millimetre; the sinusoidal gait walks on PyBullet (0.37 m/s) while the
convex-MPC trot falls within 2 s there; a fixed constraint holds a carried object under 20 m/s²;
ray-based clearance would have graded a resting object as lifted (so the AABB rule stays).

## Acceptance (internal gate + bare `vector-cli`; the owner's face is the bare REPL)

- **A1 install** — fresh clone → `uv pip install -e ".[all]"` → `vector-cli` starts; no key ⇒ one
  clear instruction; a built wheel contains the MJCF assets and `config/` and runs a sim from
  outside the repo.
- **A2 flows** — scripted bare-REPL e2e green for every body: dev write; launch + switch body and
  backend; SO-101 and Piper pick / place / home; go2 rooms + explore; go2+Piper grasp → carry →
  deliver and place-on-surface; G1 stand / walk / navigate; stop + Ctrl-C; MCP stdio smoke; slash
  commands.
- **A3 honest** — 0 false GROUNDED: moat corpus (vacuous compares, verify probes) green; the
  independent `scripts/gt_grade.py` audit over the e2e tier reports none, on both backends.
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
- **A11 concise structure** — (a) the tree equals the charter's target tree: no `hardware/sim/`, no
  repo-root `config/`, `tools/`, `foxglove/`, no stray scripts; (b) no dead code: every package
  module is reachable from an entry point (test), every test targets live behaviour, no
  commented-out blocks, no compatibility shims or re-exports, no generated file tracked; (c) one
  implementation per capability; (d) no source file over 800 lines (vendored third-party code and
  listed, justified exceptions only); (e) `vector_os_nano/` has fewer Python source lines than the
  53.3k it started with, the second backend included — every work package reports its delta.

## Work packages (each: own worktree + branch `rel/<WP>`, independent verifier, merged by the coordinator)

| WP | wave | scope | status |
|---|---|---|---|
| I1 | 1 | fast-forward `wave4j-int2` (3 gate reds) | merged f3f1276, verified (sim 10/10, E5 + E10 green) |
| I23 | 1 | merge `wave3-wp11` (perception + memory) and `wave3-wp14` (native persona / context / lifecycle) | merged f07070a; verified round 2 (hermetic 231/231, sim 51/51, e2e 11/11); 52 ledger rows closed |
| I4 | 1 | land `wave2-wp07` (spine S2) + bind the task contract to the manifest + close the probe hole + vacuous-compare corpus | merged 0898728 after the redesign of charter 3.5 (the contract compiles only from closed whole-sentence forms; the verdict reports `goal_check` proven / failed / unchecked); verified (a 5400-turn runner corpus: proven never without the end state; hermetic 238/238, sim 51/51, e2e 12/12 twice); post-merge hermetic 238/238, sim 56/56, e2e 12/12; 27 ledger rows closed; 3 small findings → I4e |
| R3 | 1 | tool security: `rg` argv, ONE resolved-path guard (credential names refused by path component anywhere, project-root auto-allow, else ask), `bash` in its own process group, `web_fetch` asks + every hop public + capped, dev verify oracles read one regular file per call (no directory walk) | merged 2eeb6c6 after the surface was shrunk (R3c); verified; post-merge hermetic 234/234, sim 51/51, e2e 10/11 (the E10 carry scenario alternates: a fixed 1.2 m walk leaves 0.1 m of margin; the I4 branch replaces it); 9 ledger rows closed; 5 minor follow-ups → R8a |
| S2 | 1b | the contract, standalone, on both engines (new files only): `interfaces/`, `sim/scene/{reader, canonical}`, `sim/backends/{mujoco, pybullet}/world.py`, `tests/conformance/` (kinematic + inertial parity, every binding sentence, oracle-parity battery, go2 walking on PyBullet) | merged 425161e after 4 verification rounds (conformance 134/134 twice on both engines, 23+ mutants killed, parity 9.3e-7 m, stepped physics bit-identical across the contact fix); post-merge hermetic 235/235, sim 57/57; 2 minor PyBullet contact-geometry gaps (mesh hulls, edges) → S3 |
| R1 | 2 | ONE producer: delete the legacy producer + IntentRouter + primitives + playground + `vector-eval` + their tests | merged acb55f2 through MR1 (see below): `engine.py` 2089 → 248 lines, the package 55.5k → 42.4k lines |
| R2 | 2 | skills hygiene: no auto-appended home (it drops a held object), dead skills deleted (mobile_pick / mobile_place / place_region / terminal_dock / the ROS door-chain), coordinate navigation succeeds only on the planner's result, a named grasp fails loud without its detector, SO-101 defaults no longer registered on other bodies | running |
| S1 | 2 | MuJoCo runs on the contract: golden traces first; `SimRuntime` over `PhysicsWorld`; `fn(model, data)` → `fn(world)`; `sim/grasp.py`; the one `GroundTruth` | merged d9cc7eb; verified round 2 (goldens regenerated from the base and compared, bit-exact; two documented ground-truth changes; navigate and piper tests 20/20; sim and e2e twice); post-merge hermetic 242/242, sim 57/57, e2e 12/12; −176 package lines |
| R8a | 2 | periphery sweep (independent of the rest): `tools/` deleted (`gt_grade` → `scripts/`), dead scripts, `.mcp.json`, the dead ROS runtime and diagnostics tools, machine paths and kill-by-name, `.gitignore` anchoring, the `[dev]` extra, a throwaway HOME for every gate tier, test hygiene + `known_reds`, NOTICE, a hermetic CI workflow + a fresh-clone attempt, the tool-security follow-ups | merged 3c65dac; verified round 2 (hermetic 230/230 also on a fresh clone with the CI extras, e2e 11/11, sim 49/50 + a known flake); post-merge hermetic 231/231 (after the dead-tracks pin learned to ignore stale bytecode), e2e 11/11, sim 55/56 + the navigate segfault flake (green alone; S1 owns it); 37 ledger rows closed; −4.3k lines in the repository, −536 in the package |
| C0 | 2 | provisioning (independent of the rest): G1 assets by manifest (`assets:` block, pinned revision, sha256) + `scripts/fetch_assets.py`; perception weights through the Hugging Face cache only (first-use fetch with one notice, `HF_HUB_OFFLINE=1` forbids), one capability line at sim start, no silent model fallback; dead perception code | verified round 3 (the fetch script run for real: 30 files, sha256 match; G1 starts from the fetched folder; weights only through the Hugging Face cache; a tier skipped by the sim preflight is now red); merged on an integration branch, gate running; this workstation is provisioned for the gate (G1 assets linked into the state directory, a campaign weights cache) |
| I4e | 2 | three small findings on the merged spine: a filler token that changes the end state (放下一点), a bare colour word bound as an object, one unpinned test | merged 2d31e21 (the merged tree is the verified one: hermetic 240/240, e2e 12/12) |
| MR1 | 2 | merge `rel/R1-fix1` into release/1.0 hunk by hunk (native_loop, cli, engine and their tests), prune the ratchet allowlists, delete what the two packages left dead for each other | verified (every hunk of both sides accounted for; a 180-row runner corpus: no gate decision changed; hermetic 172/172, sim 51/51, e2e 12/12 twice), then integrated with the S1 runtime by the coordinator (4 conflicts: three legacy files stay deleted, the legacy-executor tests of one file go) and fast-forwarded: release/1.0 = eacc963; hermetic 174/174 on the merged tree, sim and e2e running |
| S3a | 3 | generic `Arm` / `Gripper` / `Camera` (`sim/drivers/`) over the runtime, kinematics on the canonical model, protocols moved to `interfaces/`; the `piper` body (tabletop) on both backends; go2_piper's arm moved over; no ground truth through any arm, gripper or camera method | running |
| S3b | 3 | SO-101 on the generic drivers: colliding jaws + the contact-gated grasp, model perception instead of the ground-truth one; the documented fallback otherwise | open |
| R5 | 3 | split `cli.py`; ONE `build_runtime`; slash-command table; explicit credentials (no borrowed OAuth) | running |
| S4 | 4 | generic LeggedBase + policies (convex_mpc, torchscript, sinusoidal); go2 on both backends; G1 on MuJoCo with skills + causation; posture persistence | open |
| R4 | 4 | manipulation closure: ONE grasp + ONE place for every arm on the generic drivers (the SO-101 and Piper skill families become one pack), carry, deliver, one object store (`WorldModel` deleted); e2e fetch + place | open |
| R6 | 4 | MCP server on `build_runtime` + native producer; SDK 2.x; stdio / loopback; verdict returned | open |
| S5 | 5 | session × backend × embodiment × world: NL selection with honest errors, the one scene compile, `compose:`, capability-bound skill packs + derived persona, a robot added by manifest only | open |
| C1 | 5 | config + file interfaces: `schemas/` (robot, world, user, env, paths), `$VECTOR_HOME`, robot and world folders, asset fetch script, perception weight provisioning | open |
| R8 | 5 | final packaging: wheel contents, `config/` and the external-navigation files into the package, extras layout, real-hardware extra, the layering ratchet at zero | open |
| R9 | 6 | docs: README, getting-started, ARCHITECTURE, STATUS, DECISIONS, add-a-simulator / add-a-robot guides, CHANGELOG, version | open |
| DC | 6 | dead-code and structure sweep: static reachability, unused functions, duplicate helpers, the 800-line rule, stale tests | open |
| REL | 6 | final acceptance A1–A11, adversarial review of the whole diff, tag `v1.0.0` (local) | open |

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

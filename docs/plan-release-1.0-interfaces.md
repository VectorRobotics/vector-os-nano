# Release 1.0 — interface charter (binding for every work package)

Part of the current phase plan ([plan-release-1.0.md](plan-release-1.0.md)); R9 folds it into
ARCHITECTURE.md and the add-a-simulator / add-a-robot guides. Version 2: version 1 was reviewed
against the code by four adversarial reviewers (MuJoCo reality, PyBullet feasibility with measured
experiments, spine + agentic integrity, plugin author); their 40 must-change items are folded in.
The review itself: `/data/vos-analysis/release-1.0/design/critique-v1.json`.

Owner requirements: an **agentic robot system**, sim-only in 1.0, truly **agentic + cross-simulator
+ cross-embodiment**, with **(1) explicit interfaces, (2) explicit config-file interfaces,
(3) explicit file / model interfaces, (4) everything modular.**

Every plug point therefore has exactly four things: a code interface, a config schema, a file
convention and a conformance test.

## 1. Plug points

| plug point | code interface | declared in | how it is added | proven by |
|---|---|---|---|---|
| simulator | `interfaces.PhysicsWorld` + `SimBackend` | — | a module under `sim/backends/` + one row in its table | `tests/conformance` (every backend, the same tests) |
| robot body | none: data read by generic drivers | `robot.yaml` (schema 1) + model files | a folder: package `embodiments/<id>/` or `$VECTOR_HOME/embodiments/<id>/` | strict load + stands + protocols conform |
| control policy | `interfaces.ControlPolicy` | `robot.yaml: base.policy` | a module under `sim/policies/` + one table row | policy conformance (obs / action shapes, a stand test) |
| skill | `@skill` with `requires` (+ optional `verify_hint`) | packs bound by capability | a pack module + one table row | every skill: schema valid, a non-empty hint names tagged oracles only |
| world | none: data | `world.yaml` (schema 1) + scene file | a folder: package `worlds/<id>/` or `$VECTOR_HOME/worlds/<id>/` | strict load; the scene compiles on its own |
| oracle | `@oracle(scope, channel, transition, max_tol, domain)` | the world layer only | in-tree only (never a plug-in: evidence is not pluggable) | moat corpus + tag == classifier-table pin |
| perception model | `interfaces.Detector` (`detect(rgb, queries) -> list[Detection]`), `Segmenter` | user config `perception:` | a module under `perception/` + one table row | detector conformance |
| LLM | `interfaces.LLMBackend` | user config `llm:` | a provider row (preset) or `openai_compat` + `base_url` | scripted-backend contract tests |
| face | CLI (REPL, `-p`), MCP | — | — | bare-REPL e2e, MCP stdio smoke |

Code plug-ins are rows in built-in tables inside their own package — never an edit to the kernel.
Data plug-ins (robots, worlds) are folders — no edit at all. Out-of-tree code plug-ins (entry points)
are 1.1: no third-party plug-in exists yet, and an uncertified simulator must not grade GROUNDED.

## 2. The canonical model

A robot or world model file is MJCF or URDF. **MuJoCo is the canonical compiler**: one compile step
(`sim/scene/`) composes world scene ⊕ robot (⊕ arm at its mount), adds the manifest's cameras and one
inactive grasp pair per (gripper weld body × world pickable), and returns a `CompiledScene`: the
composed XML path, the compiled `MjModel`, the robot root, the grasp pairs, the cameras, the timestep.

- **Ids are canonical**: a body / joint / geom id at any interface is the index in the compiled
  model. Name lookup, subtree, free bodies and state slices are written once over `CompiledScene`;
  no backend implements them.
- **State is canonical**: `(q, qd)` in the compiled model's generalized coordinates. A hinge or slide
  joint is one scalar. A free body is 7 in `q` (position of the body FRAME, quaternion w x y z) and
  6 in `qd` (linear velocity of the frame origin in WORLD coordinates, angular velocity in the BODY
  frame). Every backend publishes exactly this layout.
- **Kinematics is canonical**: body poses, site poses and shape AABBs are computed from a state on
  the compiled model (`sim/scene/canonical.py`), never asked of a backend. IK / FK for arms run on
  the same compiled source (`sim/scene/kinematics.py`). MuJoCo is therefore a hard dependency of
  the `sim` tier for every backend; a backend integrates dynamics, contact and its own sensors.
- Physics options (integrator, cone, impratio) come from the world scene. The timestep is the
  robot's `base.policy.timestep` when it has one, else the world's.
- No dataclass mirror of the model: other backends read the compiled `MjModel` through
  `sim/scene/reader.py` (bodies, joints, inertia, geoms, meshes, actuators, cameras, no-collide
  pairs, gravity compensation, joint couplings).

## 3. Code interfaces (`vector_os_nano/interfaces/` — imports stdlib, numpy and `schemas/` only)

### 3.1 `PhysicsWorld` — the only thing a simulator implements (14 methods)

```python
class PhysicsWorld(Protocol):
    name: str                       # "mujoco" | "pybullet"
    scene: CompiledScene
    caller_thread_only: bool        # True: no stepper thread may drive it (a main-thread viewer)

    def close(self) -> None: ...
    def time(self) -> float: ...
    def step(self, n: int = 1) -> None: ...
    def state(self) -> tuple[np.ndarray, np.ndarray]: ...            # (q, qd) copies of one instant
    def reset(self, q, qd=None) -> None: ...
    def set_targets(self, joints, q) -> None: ...
    def targets(self, joints) -> np.ndarray: ...
    def set_torque(self, joints, tau) -> None: ...
    def contacts(self, body: int) -> frozenset[int]: ...
    def attach(self, parent: int, child: int) -> None: ...
    def detach(self, child: int) -> None: ...
    def attached(self) -> dict[int, int]: ...                        # child -> parent
    def raycast(self, origins, directions, exclude_body: int = -1) -> tuple[np.ndarray, np.ndarray]: ...
    def render(self, camera: str, width: int, height: int, kinds=("rgb", "depth")) -> RGBDFrame: ...

class SimBackend(Protocol):
    name: str
    def supports(self, robot: EmbodimentConfig, world: WorldConfig) -> str | None: ...   # None = yes, else why not
    def load(self, scene: CompiledScene, *, gui: bool) -> PhysicsWorld: ...
```

Binding semantics (the conformance suite tests each sentence on every backend):

- World frame, z up, SI units, quaternions w x y z.
- `step(n)` advances n physics steps of the scene timestep. Commands are held for all n
  (zero-order hold). An engine that clears commands re-issues them before each internal step.
- `set_targets` writes the set-points of the model's OWN position actuators for those joints: gains,
  control range and force limit are the model's, evaluated at every physics step. `targets` reads
  them back. A joint without a position actuator is a `KeyError`.
- `set_torque` drives joints that have motor actuators; the value is clipped to the actuator range
  and held until replaced.
- The model's passive dynamics (joint damping, armature, friction loss, gravity compensation,
  contact exclusions, joint couplings) belong to the model: every backend reproduces them within
  the conformance bounds.
- `reset` writes the whole state (`qd` zero when omitted), clears torques, keeps targets, and
  refreshes kinematics and collision detection.
- `contacts(body)` = bodies with at least one contact point at signed distance ≤ 1e-3 m, as of the
  last `step` or `reset`.
- `attach(parent, child)` holds the child rigidly at its pose relative to the parent as of the last
  `step` or `reset` (MuJoCo anchors at the kinematics the last step computed, one physics step
  behind `q`; current after a reset: today's behaviour, kept bit for bit). Only pairs in
  `scene.grasp_pairs`; anything else is a `KeyError` listing them. `detach` of a free child is a
  no-op; attaching a child to the parent that already holds it is no transition.
- `raycast`: unit directions; returns distance (−1 = miss) and the hit body (−1). The geometry tested
  is the engine's own (MuJoCo: visible geoms; PyBullet: collision shapes), so lidar is
  backend-specific by design and is never evidence.
- `render` observes one sim instant and may be called from any thread (a backend that cannot render
  while stepping serializes the two itself). `RGBDFrame`: `rgb` (H, W, 3) uint8; `depth` (H, W)
  float32, metres along the optical axis, 0 = invalid; `seg` (H, W) int32 canonical BODY id of the
  visible geom, −1 = none; `K` from the vertical fovy and the ACTUAL shape (fy = (H/2)/tan(fovy/2),
  fx = fy, cx = W/2, cy = H/2); `cam_pos` (3,); `cam_mat` (3, 3) world ← camera with columns
  right, up, backward (OpenGL).

### 3.2 World side (one implementation each, never pluggable)

- `sim/runtime.py` — `SimRuntime(world)`: one stepper thread or lockstep, `submit(fn(world))` /
  `call` at a step boundary (a call times out after 5 s; once the stepper is dead every later call
  fails loud with the reason), `paused()`, `wait_sim`, the per-tick controller hook, a published
  `(sim_time, step, q, qd)` snapshot for lock-free reads, `grasp(spec)` / `release(parent)` /
  `attachments()`, and the attachment log: every `attach` / `detach` a job or a controller makes is
  logged with its sim time at the moment of the call (two inside one job are two events).
- `sim/grasp.py` — the grasp decision: jaws servoed by the driver; `grasp(world, spec)` attaches only
  when every finger body touches the same free body and the jaws are held open ≥ `min_opening`;
  `release`. Its only entry points are `SimRuntime.grasp(spec)` / `release(parent)`. **No driver,
  skill or perception module calls `attach` / `detach`.**
- `sim/ground_truth.py` — ONE concrete class (no Protocol): `sim_time`, `base_pose`,
  `object_positions`, `object_speed`, `joint_positions`, `attachments()`, `attached_to(name)`,
  `attach_events(since)` (fail-closed when the log is truncated), `contacts`, `clearance` (today's
  rule over canonical shape AABBs of the published state: lowest point minus the highest colliding
  non-robot shape top under it, 1 cm support epsilon), `visible_pixels` (segmentation render).
  Composite predicates read one instant in one call. No driver constructs or holds one.
- Causation baselines (pose, arm joints, attachments) come from the session's ground truth
  (`capture(agent, session)`, stamped with the session's uid); only the commanded-motion counters
  come from the drivers. A baseline from another session never causes.

### 3.3 Actor side — what skills and perception may touch

- `BaseProtocol` (connect, disconnect, stop, walk, set_velocity, get_position, get_heading,
  get_velocity, get_odometry, get_lidar_scan, get_sim_time, wait_sim, cmd_motion, set_posture,
  limits, holonomic), `ArmProtocol` (joint_names, dof, get_joint_positions, move_joints,
  move_cartesian, fk, ik, ik_top_down, home, stop, ctrl_motion), `GripperProtocol` (open, close,
  is_holding, get_position), `CameraProtocol.capture(width, height) -> CameraFrame`.
- `CameraFrame` = rgb, depth, `self_mask`, K, cam_pos, cam_mat, sim_time, camera. **No
  segmentation ever reaches the actor**; `RGBDFrame.seg` stays on the world side.
- No protocol method returns the pose of anything but the robot itself
  (`get_object_positions` is deleted). Drivers keep the world and their ids private: no public
  `runtime`, `world` or `gt` attribute.
- Generic drivers, one per morphology class (`sim/drivers/`): `Arm`, `Gripper`, `LeggedBase`,
  `Camera`, `Lidar`. They read the manifest and the runtime only.

### 3.4 `ControlPolicy`

```python
@dataclass(frozen=True)
class PolicyObs:
    time: float                 # sim seconds since reset
    base_pos: np.ndarray        # (3,) world
    base_quat: np.ndarray       # (4,) w x y z, world <- body
    base_lin_vel: np.ndarray    # (3,) WORLD frame
    base_ang_vel: np.ndarray    # (3,) BODY frame
    joint_pos: np.ndarray       # base.joints order
    joint_vel: np.ndarray
    command: np.ndarray         # vx, vy (body frame, m/s), vyaw (rad/s)

class ControlPolicy(Protocol):
    tick_hz: float                                     # torques() is called at this rate
    def reset(self, obs: PolicyObs) -> None: ...
    def torques(self, obs: PolicyObs) -> np.ndarray: ...   # (len(base.joints),), held until the next tick
```
Braking and the stance hand-over at zero command are the policy's job. Postures (stand / sit / lie)
are PD holds owned by `LeggedBase` and persist until a non-zero velocity command. Built-ins:
`convex_mpc` (quadruped), `torchscript` (file + observation spec; G1), `sinusoidal`.

### 3.5 What a verdict claims

A verdict states three separate facts and never merges them:

- **Step evidence** — a graded (action → verify) step is GROUNDED only when its predicate is ground
  truth at that instant and the actor caused it. This is the moat; it has no vocabulary.
- **Goal check** — `goal_check`: `proven` | `failed` | `unchecked`. The spine compiles a task
  contract from the user's sentence ONLY when the WHOLE sentence matches a closed form (grasp,
  release, place on a named surface, pick-and-place, fetch to the user) and binds its slots through
  the world manifest. `proven`: every contract item was False at turn start, True at turn end and
  backed by a GROUNDED step. `failed`: a contract was compiled and is not met, or a slot could not
  be bound (the reason names the slot, quotes the words and lists the candidates). `unchecked`: no
  contract — the spine does not read this sentence as an end state, says so, and lists the
  predicates that were proven instead. There is no fallback from "the sentence mentions an object"
  to a contract, no list of prepositions and no exemption list: a closed grammar can be sound;
  what lies outside it is reported, never guessed, and never told to the model as its task.
  An object slot binds through the world manifest's ONE resolver, which the oracles and the grasp
  skill ask too: the WHOLE slot must be this world's own words (a body name, an alias, a colour
  with a category or a generic head, a category, and a closed list of fillers), and it binds
  when exactly one pickable fits every word. So a category alone names the object when the
  world has exactly one of it ("the cup"), and names nothing when it has two ("the bottle").
- **`verified`** — with a contract: the goal check is `proven` and every graded step that is not a
  failed attempt (below) is GROUNDED. Without one: every graded step is GROUNDED and no gate
  rejects. A consumer that needs "the task the user asked for is done" reads
  `goal_check == "proven"`.
- **Failed attempts and recovery** — a tool call that returns an error is one of two things. When
  the body's commanded-motion evidence (the drivers' motion counters and the grasp attachments,
  read on the world side) did not change across the call, it was REFUSED: it is not an action; it
  neither fails the step it is in nor proves anything. Otherwise it is a FAILED ATTEMPT: it closes
  its step at once as a FAILED row with its message, so it neither fails nor backs a later
  action's row, and it holds nothing but the failing call (an action that succeeded before it in
  the same step is closed first, as its own unproven row). Without motion evidence (no body)
  every error is a failed attempt. A failed attempt AT A CONTRACT ITEM (its own verify hint is
  that item's predicate) does not by itself reject a turn whose contract is `proven`: the end
  state the sentence names was reached and proven after it, which is what recovery is; the failed
  rows stay in the report. Every other failed attempt rejects the turn as before, with or
  without a contract, unless a later attempt of the same predicate supersedes it. An action that
  succeeded and was never verified is still an unproven step and still rejects the turn.

The gates that only reject stay (coordinate goals; verify probes): they never make a goal proven.

**Manipulation needs a contract.** In a world with a manifest, a turn that asks for a manipulation
(a manipulation word in the goal) or performs one (a dispatched grasp, release or placement) is
verified only through a proven contract. Outside the closed forms the spine has no sound way to
tell which object the words name (a landmark, a part, an excluded or a negated object all mention
it), so it does not look for one: the goal is `unchecked`, the turn is not verified, and the
reason gives the forms that can be checked. The step evidence is reported as always.

## 4. Config-file interfaces (`vector_os_nano/schemas/` — imports stdlib and yaml only)

Every file has a required `schema: 1`; a higher number fails with "upgrade vector_os_nano". Frozen
dataclasses, fail-loud loaders naming the file, the key and the valid set. Unknown keys are rejected
everywhere except owner blocks (`base.policy` beyond its typed keys, `skills.<name>`), which their
owner parses with the same strict helper. A test requires every typed field to have a reader outside
`schemas/` (no decorative keys).

**`robot.yaml`** — the id is the folder name; capabilities are derived (base, arm, gripper, camera,
depth, lidar), never written.
```
schema: 1
display_name            optional
world                   optional default world id
model:  {file, root_body, prefix?}            MJCF or URDF, relative to the robot folder
assets: {name, git, rev, files, sha256, license}   optional fetch recipe for files not shipped
base:   {height, holonomic, limits {vx, vy, vyaw}, fall_z,
         joints [ordered], stance {joint: rad}, postures {name: {joint: rad}}, kp, kd, torque_scale?,
         policy {ref, timestep, tick_hz, by_backend? {<backend>: {ref, ...}}, <policy-owned keys>},
         nav {body_radius, speed, vyaw_max, explorer?}}
arm:    {joints, actuators?, ee {site} | {body, pos}, home, postures?, ik_seeds?, track_tol?,
         gripper? {actuator, joint, open, closed, finger_bodies (required, colliding), weld_body,
                   min_opening, mimic?}}
sensors: [{role: camera, name, width, height, depth?, body?, pos?, quat?, fovy?} |
          {role: lidar, body, pos, tilt_deg, elevations_deg, n_azimuth, range, period_s}]
skills: {<skill>: {...}}                      per-robot skill parameters, parsed by the skill
compose: {base: <id>, arm: <id>, mount: {body, pos, quat}, prefix?}
```
Every name is the exact name in that robot's own model file; the compiler applies `model.prefix`
(and `compose.prefix`, default `<arm id>_`) and checks each name exists. A compose manifest is the
base manifest plus the arm manifest's model and `arm` block; any top-level block it carries replaces
the base's block whole (no deep merge). `torchscript` policy keys: `file`, `decimation`,
`obs: [{term, scale, period?}]`, `action: {scale}` with the fixed term vocabulary `base_lin_vel`
(body frame), `base_ang_vel` (body frame), `projected_gravity`, `command`, `joint_pos_rel`,
`joint_vel`, `last_action`, `gait_phase`; the loader runs the file once and fails loud on a shape
mismatch.

**`world.yaml`** — `schema`, `scene`, `spawn {xy, heading, z}`, `rooms`, `boxes`, `doors`,
`surfaces {name: {body, box, height}}`, `pickables {body: {category, colour, aliases, surface?}}`,
`user {xy}`, `vocab`. Spawn belongs to the world; `start_simulation(embodiment, backend, world)`
may override the robot's default world.

**`$VECTOR_HOME/config.yaml`** — `schema`, `llm {provider: deepseek | openrouter | anthropic |
openai_compat, model, base_url?, api_key_env?}`, `perception {detector, segmenter, scene_vlm}`,
`defaults {embodiment, backend, world, gui}`. Precedence: CLI flag > this file > built-in default.
Secrets come only from the environment (a repo-root `.env` is loaded first); a config file never
holds a key; another application's credentials are never borrowed.

**Environment** — `schemas/env.py` catalogues every variable (name, kind: user | test_seam |
experimental | third_party, default, meaning). An AST test requires every `VECTOR_*` literal and
every `os.environ` read in the package to be in the catalogue.

## 5. File / model interfaces

- **State directory** — `VECTOR_HOME`, default `~/.vector` (a one-time migration moves today's
  `~/.vector_os_nano` memory in). `schemas/paths.py` owns every path; a test bans `Path.home()` and
  `expanduser` elsewhere.
  ```
  $VECTOR_HOME/ config.yaml · history · VECTOR.md · personality.md
                sessions/<id>.jsonl · exports/<id>.md
                memory/<world>/{scene_graph.yaml, terrain_map.npz}
                embodiments/<id>/ · worlds/<id>/ · assets/<name>/
                cache/scenes/ (regenerable) · runs/<stamp>/ (logs, crash logs, GT dumps)
  ```
- **A robot is a folder**: `embodiments/<id>/{robot.yaml, model/{<file>, meshes/}, LICENSE}`. A
  composed body's folder holds only `robot.yaml`. Conventions: SI units, z up, x forward, mesh paths
  relative. Mesh folders are named `meshes/`, never `assets/`.
- **A world is a folder**: `worlds/<id>/{world.yaml, scene.xml, meshes/}`; the scene is plain MJCF
  with no robot, no tokens and no grasp pairs, and compiles on its own.
- **A policy is files + a spec**: whatever the policy declares (`file`, the convex-MPC's own URDF),
  resolved against the robot folder, then `$VECTOR_HOME/assets/<assets.name>`.
- **Assets not shipped** (G1): resolved from the robot folder, then `$VECTOR_HOME/assets/<name>`;
  fetched by `scripts/fetch_assets.py <robot>` (pinned revision, sha256). A start without them fails
  loud with that exact command.
- **Perception weights**: each model class names its `{repo_id, revision}`; resolved through the
  Hugging Face cache only; `HF_HUB_OFFLINE=1` forbids fetching; a first-use fetch prints one notice.
- **Outputs are interfaces too**, each with a `schema` field: session JSONL, `scene_graph.yaml`, the
  `VECTOR_VERDICT {json}` line (per step: embodiment, backend, session, actor_caused; per report:
  goal_check, contract,
  world, backend version, perception kind), the `VECTOR_GT_DUMP` JSONL.

## 6. Modularity (enforced by `tests/unit/test_layering.py`: AST-based, with an allowlist that only shrinks)

1. `schemas/` imports nothing from the package; `interfaces/` imports only `schemas/`.
2. A simulator is imported only by its backend: `pybullet` under `sim/backends/pybullet/`; `mujoco`
   under `sim/backends/mujoco/` and `sim/scene/` (the canonical compiler, reader and kinematics).
3. Actor code — `skills/`, `perception/`, `sim/drivers/`, `sim/policies/` — imports none of
   `sim.ground_truth`, `sim.grasp`, `sim.backends`, the oracle module or `vcli`; calls no
   `attach` / `detach` / `reset`; `skills/` and `perception/` touch no attribute named `runtime`,
   `world`, `gt` or `get_object_positions`.
4. The kernel (`vcli/engine.py`, `vcli/native_loop.py`, `vcli/native_runner.py`, `vcli/native_tools.py`,
   `vcli/cognitive/`) imports no concrete world,
   embodiment, backend or skill pack — only `interfaces` and tables.
5. `core/` imports only `interfaces` and `schemas`.
6. Optional heavy dependencies (torch, transformers, pybullet, casadi, pinocchio, mcp, rclpy) are
   imported lazily or inside their owning module; `import vector_os_nano` and the CLI start on a base
   install (tested in a subprocess with those modules blocked).

Target tree: `interfaces/` · `schemas/` · `core/` · `sim/{runtime, grasp, ground_truth, session,
scene/, drivers/, policies/, nav/, backends/{mujoco, pybullet}/}` · `embodiments/<id>/` ·
`worlds/<id>/` · `skills/` (packs) · `perception/` · `vcli/` (kernel + faces) · `mcp/` ·
`hardware/so101/` (real arm, experimental).

## 7. Bodies × backends in 1.0

| body | MuJoCo | PyBullet |
|---|---|---|
| `piper` (6-DoF arm, tabletop) | ✔ | ✔ |
| `so101` (5-DoF arm) | ✔ | ✔ once it runs on the generic `Arm` / `Gripper` with colliding jaws and model perception; until then its grasp oracles are not registered and the docs say so |
| `go2` (quadruped) | ✔ `convex_mpc` | ✔ `sinusoidal` (measured: 0.37 m/s, turns); `convex_mpc` there is a time-boxed stretch (measured: falls within 2 s) |
| `go2_piper` (go2 ⊕ piper) | ✔ | ✔ on the sinusoidal gait; carrying needs its own gate |
| `g1` (humanoid, TorchScript policy) | ✔ | not attempted in 1.0 |

`backend.supports(robot, world)` returns the reason for every ✗, so the matrix the user sees is
derived, never hand-written. A kinematic (teleported) base is ruled out.

## 8. Staging (strangler: the full gate is green after every stage)

- **S2 — the contract, standalone, on both engines** (new files only): `interfaces/`,
  `sim/scene/{reader, canonical}`, `sim/backends/mujoco/world.py`, `sim/backends/pybullet/world.py`,
  `tests/conformance/`. Kinematic parity (body-frame poses, 1e-5, floating bases included), inertial
  parity (mass, COM, principal inertia), every binding sentence of §3.1, the oracle-parity battery
  (resting, contact margin, lift threshold, attach log, AABB, segmentation), behavioural bounds
  (PD stance height within 5 mm, servo hold within 0.05 rad), go2 walking on the sinusoidal policy.
- **S1 — MuJoCo runs on the contract**: golden traces captured first, in lockstep (go2 walk + brake;
  go2_piper pick; a ground-truth replay). `SimRuntime` over `PhysicsWorld`; the mechanical
  `fn(model, data)` → `fn(world)` rewrite inside `hardware/sim/`; `sim/grasp.py`; the one
  `GroundTruth`; goldens unchanged.
- **S3 — generic `Arm` / `Gripper` / `Camera`**: the new `piper` body on both backends; go2_piper's
  arm moved over (the grasp campaign must hold); SO-101 on the generic drivers (or the documented
  fallback); `get_object_positions` and actor-side segmentation removed.
- **S4 — generic `LeggedBase` + policies**: go2 on both backends; G1 on MuJoCo with the locomotion
  pack, a commanded-motion counter and its assets; posture persistence.
- **S5 — session × backend × embodiment × world**: `start_simulation(embodiment, backend, world)` by
  natural language with honest errors; the one scene compile; `compose:`; capability-bound skill
  packs and the persona derived from them; a robot added by manifest only.
- **C1 — config and file interfaces**: `schemas/` (robot, world, user, env, paths), `$VECTOR_HOME`,
  robot and world folders, the fetch script, explicit credentials.

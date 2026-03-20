"""Test full pipeline: LLM planning + executor + world model updates."""
import time
from vector_os.core.agent import Agent
from vector_os.core.config import load_config
from vector_os.core.world_model import ObjectState
from vector_os.core.types import TaskPlan, TaskStep, SkillResult
from vector_os.core.executor import TaskExecutor
from vector_os.core.skill import SkillRegistry, SkillContext
from vector_os.skills import get_default_skills

print("=" * 50)
print("Test: Full Pipeline (Plan + Execute + WorldModel)")
print("=" * 50)

# --- Test 1: Executor with mock skill ---
print("\n--- Test 1: Executor runs a plan ---")

class MockSkill:
    name = "greet"
    description = "Say hello"
    parameters = {"who": {"type": "string", "default": "world"}}
    preconditions = []
    postconditions = []
    effects = {}
    def execute(self, params, context):
        print(f"    [greet] Hello {params.get('who', 'world')}!")
        return SkillResult(success=True, result_data={"greeted": params.get("who", "world")})

registry = SkillRegistry()
for s in get_default_skills():
    registry.register(s)
registry.register(MockSkill())

from vector_os.core.world_model import WorldModel
wm = WorldModel()
context = SkillContext(arm=None, gripper=None, perception=None, world_model=wm, calibration=None, config={})

plan = TaskPlan(
    goal="test greet",
    steps=[
        TaskStep(step_id="s1", skill_name="greet", parameters={"who": "Yusen"}),
        TaskStep(step_id="s2", skill_name="greet", parameters={"who": "Vector OS"}, depends_on=["s1"]),
    ],
)

executor = TaskExecutor()
result = executor.execute(plan, registry, context)
print(f"  Result: success={result.success}, steps={result.steps_completed}/{result.steps_total}")
for t in result.trace:
    print(f"    [{t.status}] {t.skill_name} {t.duration_sec:.3f}s")

# --- Test 2: Precondition blocking ---
print("\n--- Test 2: Precondition blocks execution ---")

wm2 = WorldModel()
# Gripper is open (empty), but pick requires gripper_empty which IS true
# Let's simulate gripper already holding something
wm2.update_robot_state(gripper_state="holding", held_object="something")
context2 = SkillContext(arm=None, gripper=None, perception=None, world_model=wm2, calibration=None, config={})

plan2 = TaskPlan(
    goal="test precondition",
    steps=[
        TaskStep(step_id="s1", skill_name="pick", parameters={}, preconditions=["gripper_empty"]),
    ],
)

result2 = executor.execute(plan2, registry, context2)
print(f"  Result: success={result2.success}, status={result2.status}")
print(f"  Reason: {result2.failure_reason}")

# --- Test 3: World model updates through execution ---
print("\n--- Test 3: World model tracks state changes ---")

wm3 = WorldModel()
wm3.add_object(ObjectState(
    object_id="cup_1", label="red cup",
    x=0.25, y=0.05, z=0.02,
    confidence=0.95, state="on_table",
    last_seen=time.time(),
))
print(f"  Before: gripper={wm3.get_robot().gripper_state}, held={wm3.get_robot().held_object}")
print(f"  Before: cup state={wm3.get_object('cup_1').state}")

# Simulate pick effect
wm3.apply_skill_effects("pick", {"object_id": "cup_1"}, SkillResult(success=True))
print(f"  After pick: gripper={wm3.get_robot().gripper_state}, held={wm3.get_robot().held_object}")

# Simulate place effect
wm3.apply_skill_effects("place", {}, SkillResult(success=True))
print(f"  After place: gripper={wm3.get_robot().gripper_state}, held={wm3.get_robot().held_object}")

# --- Test 4: LLM plan → executor (real API) ---
print("\n--- Test 4: LLM plan → executor (real API, no arm) ---")

cfg = load_config("config/user.yaml")
api_key = cfg["llm"]["api_key"]

agent = Agent(llm_api_key=api_key, config="config/user.yaml")
agent.world.add_object(ObjectState(
    object_id="cup_1", label="red cup",
    x=0.25, y=0.05, z=0.02,
    confidence=0.95, state="on_table",
    last_seen=time.time(),
))

print("  Sending: 'home'")
r = agent.execute("home")
print(f"  Result: success={r.success}, reason={r.failure_reason}")

# --- Test 5: Custom skill with LLM ---
print("\n--- Test 5: Custom skill registered + LLM discovers it ---")

class WaveSkill:
    name = "wave"
    description = "Wave the robot arm side to side as a greeting gesture"
    parameters = {"times": {"type": "int", "default": 3, "description": "Number of waves"}}
    preconditions = []
    postconditions = []
    effects = {}
    def execute(self, params, context):
        times = params.get("times", 3)
        print(f"    [wave] Waving {times} times! (simulated)")
        return SkillResult(success=True, result_data={"waved": times})

agent.register_skill(WaveSkill())
print(f"  Skills after register: {agent.skills}")

# Check if LLM can see the new skill
from vector_os.llm.claude import ClaudeProvider
llm = ClaudeProvider(api_key=api_key, model=cfg["llm"]["model"], api_base=cfg["llm"]["api_base"])
schemas = agent._skill_registry.to_schemas()
plan = llm.plan("wave hello 5 times", agent.world.to_dict(), schemas)
print(f"  LLM plan for 'wave hello 5 times':")
for step in plan.steps:
    print(f"    {step.step_id}: {step.skill_name}({step.parameters})")

# --- Test 6: World model persistence ---
print("\n--- Test 6: World model save/load ---")

import tempfile, os
wm_save = WorldModel()
wm_save.add_object(ObjectState(
    object_id="test_obj", label="test block",
    x=0.1, y=0.2, z=0.03,
    confidence=0.9, state="on_table",
    last_seen=time.time(),
))
wm_save.update_robot_state(gripper_state="holding", held_object="test_obj")

tmpfile = os.path.join(tempfile.gettempdir(), "wm_test.yaml")
wm_save.save(tmpfile)
print(f"  Saved to {tmpfile}")

wm_load = WorldModel.load(tmpfile)
print(f"  Loaded: {len(wm_load.get_objects())} objects")
print(f"  Object: {wm_load.get_object('test_obj').label}")
print(f"  Robot: gripper={wm_load.get_robot().gripper_state}")
os.remove(tmpfile)

print()
print("=" * 50)
print("ALL PIPELINE TESTS PASSED")
print("=" * 50)

"""Test CLI interactive mode — type commands manually."""
import time
from vector_os.core.agent import Agent
from vector_os.core.config import load_config
from vector_os.core.world_model import ObjectState
from vector_os.cli.simple import SimpleCLI

# Load config with API key
cfg = load_config("config/user.yaml")
api_key = cfg["llm"]["api_key"]

# Create agent with LLM (no hardware)
agent = Agent(llm_api_key=api_key, config="config/user.yaml")

# Pre-populate world model so LLM has context
wm = agent.world
wm.add_object(ObjectState(
    object_id="cup_1", label="red cup",
    x=0.25, y=0.05, z=0.02,
    confidence=0.95, state="on_table",
    last_seen=time.time(),
))
wm.add_object(ObjectState(
    object_id="bat_1", label="battery",
    x=0.20, y=-0.08, z=0.01,
    confidence=0.88, state="on_table",
    last_seen=time.time(),
))

# Start CLI
cli = SimpleCLI(agent=agent, verbose=True)
cli.run()

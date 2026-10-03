# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""E-suite: the acceptance scenarios, each one bare ``vector-cli`` session on a PTY.

Every scenario types natural language into the REAL interactive REPL with no flags
(tests/harness/pty_cli.run_repl_session). Only the LLM is replaced, by a reactive
script on the ``VECTOR_FAKE_LLM_TOOLS`` env seam, so the run is deterministic and
network-free while every skill, verify and verdict is the product's own.

A scenario the product cannot pass yet is ``xfail(strict=True)`` naming the work
package that turns it green: the honest baseline. Flipping one means deleting its
xfail in the WP that fixes it; an unexpected pass fails the run (strict).
"""

# NL go2_piper launch shared by the go2 scenarios. The harness gives every child a
# blank VECTOR_NAV_STACK: a launch must not depend on the external CMU stack, and
# the e2e tier never starts it on a shared box.
LAUNCH = "启动 go2 仿真 带机械臂 无窗口"
LAUNCH_STEPS = [
    {"when": "启动 go2 仿真", "tool_calls": [
        {"name": "start_simulation", "input": {"sim_type": "go2", "with_arm": True, "gui": False}}]},
    {"when": "", "text": "go2 仿真已启动", "tool_calls": []},
]

# A readiness line: the word 'ready' on a line with no 'not' ('already' and 'not
# ready' do not count). WP03 owns the real ready-line format; tighten this to it
# when E1/E6 flip green.
READY = r"(?im)^(?!.*\bnot\b).*\bready\b"

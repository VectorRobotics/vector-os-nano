# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Core agent engine — pure Python, no ROS2 imports.

Contains:
- types.py      : shared frozen data types (Pose3D, Detection, SkillResult, ...)
- config.py     : configuration loading
- agent.py      : Agent, the body the skills act on
- world_model.py: the session object store
- skill.py      : @skill, the Skill protocol, SkillContext, SkillRegistry
- scene_graph.py: spatial memory (rooms, objects, viewpoints)
- scene_nodes.py: its node types (RoomNode, ViewpointNode, ObjectNode)
- scene_store.py: its persistence (save / load / autosave / quarantine)
"""

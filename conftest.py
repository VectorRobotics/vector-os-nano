"""Pytest root conftest — adds the project root to sys.path so that
`vector_os_nano` is importable without installing the package.
"""
import sys
import os

# Insert the repo root so `import vector_os_nano` works in all test files
sys.path.insert(0, os.path.dirname(__file__))

# The CLI re-execs itself with ROS overlays sourced when launched unsourced
# (vcli/ros_bootstrap.py). execve would REPLACE the pytest process — any test
# that reaches cli.main() must see the opt-out.
os.environ["VECTOR_NO_ROS_BOOTSTRAP"] = "1"

"""Tests for Isaac Sim Docker configuration files.

Level: Isaac-L3
Validates Docker config without running Docker.
All tests are pure file-parsing — no network, no container runtime needed.
"""
from __future__ import annotations

import os
import stat
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent  # vector_os_nano/
_ISAAC_DOCKER_DIR = _REPO_ROOT / "docker" / "isaac-sim"
_DOCKERFILE = _ISAAC_DOCKER_DIR / "Dockerfile"
_COMPOSE_FILE = _ISAAC_DOCKER_DIR / "docker-compose.yaml"
_CYCLONE_XML = _ISAAC_DOCKER_DIR / "cyclonedds.xml"
_ENTRYPOINT_SH = _ISAAC_DOCKER_DIR / "docker-entrypoint.sh"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
_LAUNCH_ISAAC_SH = _SCRIPTS_DIR / "launch_isaac.sh"
_STOP_ISAAC_SH = _SCRIPTS_DIR / "stop_isaac.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_file(path: Path) -> str:
    assert path.exists(), f"Expected file not found: {path}"
    return path.read_text(encoding="utf-8")


def _parse_yaml(path: Path) -> dict:
    text = _read_file(path)
    return yaml.safe_load(text)


def _parse_xml(path: Path) -> ET.Element:
    text = _read_file(path)
    return ET.fromstring(text)


def _is_executable(path: Path) -> bool:
    mode = os.stat(path).st_mode
    return bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


# ---------------------------------------------------------------------------
# 1. Dockerfile
# ---------------------------------------------------------------------------


class TestDockerfile:
    """Dockerfile structure and content validation."""

    def test_dockerfile_exists(self) -> None:
        assert _DOCKERFILE.exists(), f"Dockerfile not found at {_DOCKERFILE}"

    def test_dockerfile_has_from_directive(self) -> None:
        content = _read_file(_DOCKERFILE)
        assert "FROM" in content, "Dockerfile must have a FROM directive"

    def test_dockerfile_from_is_nvidia_isaac_sim(self) -> None:
        content = _read_file(_DOCKERFILE)
        assert "nvcr.io/nvidia/isaac-sim" in content or "isaac-sim" in content

    def test_dockerfile_installs_ros2(self) -> None:
        content = _read_file(_DOCKERFILE)
        assert "ros-humble" in content or "ros-jazzy" in content, \
            "Dockerfile must install a ROS2 distribution"

    def test_dockerfile_installs_cyclonedds(self) -> None:
        content = _read_file(_DOCKERFILE)
        assert "cyclonedds" in content.lower(), \
            "Dockerfile must install CycloneDDS for reliable DDS transport"

    def test_dockerfile_copies_cyclonedds_config(self) -> None:
        content = _read_file(_DOCKERFILE)
        assert "cyclonedds.xml" in content, \
            "Dockerfile must COPY cyclonedds.xml into the image"

    def test_dockerfile_copies_bridge_scripts(self) -> None:
        content = _read_file(_DOCKERFILE)
        assert "bridge" in content, \
            "Dockerfile must copy bridge scripts"

    def test_dockerfile_sets_ros_domain_id(self) -> None:
        content = _read_file(_DOCKERFILE)
        assert "ROS_DOMAIN_ID" in content

    def test_dockerfile_sets_rmw_implementation(self) -> None:
        content = _read_file(_DOCKERFILE)
        assert "RMW_IMPLEMENTATION" in content

    def test_dockerfile_sets_entrypoint(self) -> None:
        content = _read_file(_DOCKERFILE)
        assert "ENTRYPOINT" in content

    def test_dockerfile_has_python_package_install(self) -> None:
        content = _read_file(_DOCKERFILE)
        assert "pip install" in content or "requirements.txt" in content

    def test_dockerfile_bridge_requirements_file_exists(self) -> None:
        bridge_req = _ISAAC_DOCKER_DIR / "bridge" / "requirements.txt"
        assert bridge_req.exists(), \
            f"bridge/requirements.txt not found at {bridge_req}"

    def test_dockerfile_no_hardcoded_passwords(self) -> None:
        content = _read_file(_DOCKERFILE)
        forbidden = ["password=", "passwd=", "secret=", "token="]
        for f in forbidden:
            assert f.lower() not in content.lower(), \
                f"Hardcoded credential found in Dockerfile: '{f}'"


# ---------------------------------------------------------------------------
# 2. docker-compose.yaml
# ---------------------------------------------------------------------------


class TestDockerCompose:
    """docker-compose.yaml structure and Isaac Sim requirements."""

    def test_compose_file_exists(self) -> None:
        assert _COMPOSE_FILE.exists(), f"docker-compose.yaml not found at {_COMPOSE_FILE}"

    def test_compose_is_valid_yaml(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        assert isinstance(data, dict), "docker-compose.yaml must parse to a dict"

    def test_compose_has_services(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        assert "services" in data, "docker-compose.yaml must define services"

    def test_compose_has_isaac_sim_service(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        services = data.get("services", {})
        assert "isaac-sim" in services, "docker-compose.yaml must have 'isaac-sim' service"

    def test_compose_has_gpu_reservation(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        isaac = data["services"]["isaac-sim"]
        deploy = isaac.get("deploy", {})
        resources = deploy.get("resources", {})
        reservations = resources.get("reservations", {})
        devices = reservations.get("devices", [])
        assert any(d.get("driver") == "nvidia" for d in devices), \
            "Isaac Sim service must reserve NVIDIA GPU"

    def test_compose_uses_network_mode_host(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        isaac = data["services"]["isaac-sim"]
        assert isaac.get("network_mode") == "host", \
            "Isaac Sim must use network_mode: host for DDS discovery"

    def test_compose_has_volumes(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        isaac = data["services"]["isaac-sim"]
        assert "volumes" in isaac, "Isaac Sim service must define volumes"

    def test_compose_has_environment_section(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        isaac = data["services"]["isaac-sim"]
        assert "environment" in isaac

    def test_compose_sets_ros_domain_id(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        isaac = data["services"]["isaac-sim"]
        env = isaac.get("environment", [])
        env_str = str(env)
        assert "ROS_DOMAIN_ID" in env_str

    def test_compose_sets_rmw_cyclonedds(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        isaac = data["services"]["isaac-sim"]
        env = str(isaac.get("environment", []))
        assert "rmw_cyclonedds" in env.lower() or "cyclonedds" in env.lower()

    def test_compose_has_health_check(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        isaac = data["services"]["isaac-sim"]
        assert "healthcheck" in isaac, \
            "Isaac Sim service must define a healthcheck"

    def test_compose_health_check_has_test(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        hc = data["services"]["isaac-sim"]["healthcheck"]
        assert "test" in hc, "healthcheck must have a 'test' command"

    def test_compose_health_check_has_interval(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        hc = data["services"]["isaac-sim"]["healthcheck"]
        assert "interval" in hc

    def test_compose_container_name_is_vector_isaac_sim(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        isaac = data["services"]["isaac-sim"]
        assert isaac.get("container_name") == "vector-isaac-sim", \
            "Container name must match the name checked by IsaacSimProxy"

    def test_compose_volumes_include_scenes(self) -> None:
        data = _parse_yaml(_COMPOSE_FILE)
        volumes = str(data["services"]["isaac-sim"].get("volumes", []))
        assert "scenes" in volumes, "Must mount scenes directory for USD assets"


# ---------------------------------------------------------------------------
# 3. cyclonedds.xml
# ---------------------------------------------------------------------------


class TestCycloneDDSConfig:
    """CycloneDDS XML must configure correct DDS settings."""

    def test_cyclonedds_xml_exists(self) -> None:
        assert _CYCLONE_XML.exists(), f"cyclonedds.xml not found at {_CYCLONE_XML}"

    def test_cyclonedds_xml_is_valid_xml(self) -> None:
        root = _parse_xml(_CYCLONE_XML)
        assert root is not None

    def test_cyclonedds_xml_has_root_element_cyclonedds(self) -> None:
        root = _parse_xml(_CYCLONE_XML)
        assert root.tag == "CycloneDDS", \
            f"Root element must be 'CycloneDDS', got '{root.tag}'"

    def test_cyclonedds_has_domain_element(self) -> None:
        root = _parse_xml(_CYCLONE_XML)
        domain = root.find("Domain")
        assert domain is not None, "CycloneDDS config must have a Domain element"

    def test_cyclonedds_domain_id_is_zero(self) -> None:
        root = _parse_xml(_CYCLONE_XML)
        domain = root.find("Domain")
        assert domain is not None
        assert domain.get("id") == "0", \
            "Domain id must be 0 to match ROS_DOMAIN_ID=0"

    def test_cyclonedds_shared_memory_disabled(self) -> None:
        root = _parse_xml(_CYCLONE_XML)
        # SharedMemory/Enable must be 'false'
        shm_enable = root.find(".//SharedMemory/Enable")
        if shm_enable is not None:
            assert shm_enable.text.strip().lower() == "false", \
                "SharedMemory must be disabled (not supported across container boundaries)"
        else:
            # If no element, SHM is disabled by default — acceptable
            content = _read_file(_CYCLONE_XML)
            assert "SharedMemory" in content and "false" in content.lower(), \
                "SharedMemory must be explicitly disabled"

    def test_cyclonedds_xml_not_empty(self) -> None:
        content = _read_file(_CYCLONE_XML)
        assert len(content.strip()) > 50, "cyclonedds.xml appears too short/empty"

    def test_cyclonedds_xml_has_network_section(self) -> None:
        content = _read_file(_CYCLONE_XML)
        assert "General" in content or "Network" in content, \
            "CycloneDDS config must have network configuration"


# ---------------------------------------------------------------------------
# 4. docker-entrypoint.sh
# ---------------------------------------------------------------------------


class TestDockerEntrypoint:
    """docker-entrypoint.sh must source ROS2 and exec the bridge."""

    def test_entrypoint_exists(self) -> None:
        assert _ENTRYPOINT_SH.exists(), \
            f"docker-entrypoint.sh not found at {_ENTRYPOINT_SH}"

    def test_entrypoint_is_executable(self) -> None:
        assert _is_executable(_ENTRYPOINT_SH), \
            "docker-entrypoint.sh must be executable (chmod +x)"

    def test_entrypoint_has_bash_shebang(self) -> None:
        content = _read_file(_ENTRYPOINT_SH)
        first_line = content.split("\n")[0]
        assert "bash" in first_line or "sh" in first_line, \
            "Entrypoint must have a bash/sh shebang"

    def test_entrypoint_sources_ros2(self) -> None:
        content = _read_file(_ENTRYPOINT_SH)
        assert "source" in content and "ros" in content.lower(), \
            "Entrypoint must source ROS2 setup"

    def test_entrypoint_two_process_architecture(self) -> None:
        content = _read_file(_ENTRYPOINT_SH)
        assert "ISAAC_PID" in content or "isaac_sim_physics" in content, \
            "Entrypoint must start physics process"

    def test_entrypoint_launches_python_bridge(self) -> None:
        content = _read_file(_ENTRYPOINT_SH)
        assert "python" in content.lower(), \
            "Entrypoint must launch the Python bridge"

    def test_entrypoint_references_bridge_script(self) -> None:
        content = _read_file(_ENTRYPOINT_SH)
        assert "bridge" in content.lower() or "BRIDGE_SCRIPT" in content, \
            "Entrypoint must reference the bridge script"

    def test_entrypoint_no_hardcoded_credentials(self) -> None:
        content = _read_file(_ENTRYPOINT_SH)
        forbidden = ["password=", "passwd=", "api_key=", "secret="]
        for f in forbidden:
            assert f.lower() not in content.lower(), \
                f"Hardcoded credential in entrypoint: '{f}'"


# ---------------------------------------------------------------------------
# 5. launch_isaac.sh and stop_isaac.sh (planned scripts)
# ---------------------------------------------------------------------------


class TestIsaacLaunchScripts:
    """launch_isaac.sh and stop_isaac.sh must exist and be executable.

    These scripts are planned additions to the scripts/ directory.
    Tests are marked xfail until the scripts are created.
    """

    @pytest.mark.xfail(
        not _LAUNCH_ISAAC_SH.exists(),
        reason="launch_isaac.sh not yet created",
        strict=False,
    )
    def test_launch_isaac_exists(self) -> None:
        assert _LAUNCH_ISAAC_SH.exists(), \
            f"launch_isaac.sh not found at {_LAUNCH_ISAAC_SH}"

    @pytest.mark.xfail(
        not _LAUNCH_ISAAC_SH.exists(),
        reason="launch_isaac.sh not yet created",
        strict=False,
    )
    def test_launch_isaac_is_executable(self) -> None:
        if _LAUNCH_ISAAC_SH.exists():
            assert _is_executable(_LAUNCH_ISAAC_SH), \
                "launch_isaac.sh must be executable"

    @pytest.mark.xfail(
        not _STOP_ISAAC_SH.exists(),
        reason="stop_isaac.sh not yet created",
        strict=False,
    )
    def test_stop_isaac_exists(self) -> None:
        assert _STOP_ISAAC_SH.exists(), \
            f"stop_isaac.sh not found at {_STOP_ISAAC_SH}"

    @pytest.mark.xfail(
        not _STOP_ISAAC_SH.exists(),
        reason="stop_isaac.sh not yet created",
        strict=False,
    )
    def test_stop_isaac_is_executable(self) -> None:
        if _STOP_ISAAC_SH.exists():
            assert _is_executable(_STOP_ISAAC_SH), \
                "stop_isaac.sh must be executable"

    def test_scripts_directory_exists(self) -> None:
        assert _SCRIPTS_DIR.exists(), \
            f"scripts/ directory not found at {_SCRIPTS_DIR}"

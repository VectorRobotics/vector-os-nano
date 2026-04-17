# Isaac Sim Docker image for Vector OS Nano
#
# Base: NVIDIA Isaac Sim 5.1.0 (Ubuntu 22.04, Python 3.11)
# Adds: ROS2 Humble + CycloneDDS + Vector bridge scripts
#
# Build:
#   docker build --network host -t vector-isaac-sim:latest docker/isaac-sim/
#
# Run via docker-compose:
#   docker compose -f docker/isaac-sim/docker-compose.yaml up

ARG ISAAC_SIM_TAG=5.1.0
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_TAG}

# Isaac Sim 5.1 runs rootless (uid 1234) by default.
# Switch to root for package installation, then switch back.
USER root

# ── System locale ──────────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ── ROS2 Jazzy ─────────────────────────────────────────────────────────────────
# Isaac Sim 5.1 ships on Ubuntu 24.04 (Noble), so Jazzy is the matching ROS2.
# This means container and host both run Jazzy — no DDS version mismatch!
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
       http://packages.ros.org/ros2/ubuntu noble main" \
       > /etc/apt/sources.list.d/ros2.list \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-base \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-nav-msgs \
    ros-jazzy-sensor-msgs \
    ros-jazzy-geometry-msgs \
    ros-jazzy-std-msgs \
    ros-jazzy-tf2-ros \
    ros-jazzy-tf2-msgs \
    python3-colcon-common-extensions \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ── CycloneDDS config ──────────────────────────────────────────────────────────
COPY cyclonedds.xml /opt/cyclonedds.xml

# ── Isaac Lab (RL training for Go2 locomotion) ─────────────────────────────────
ENV ISAACSIM_PATH=/isaac-sim
ENV ISAACSIM_PYTHON_EXE=/isaac-sim/python.sh
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && cd /opt && git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git \
    && cd /opt/IsaacLab && ./isaaclab.sh --install 2>&1 | tail -5 \
    && echo "Isaac Lab installed"

# ── Python bridge dependencies ─────────────────────────────────────────────────
COPY bridge/requirements.txt /tmp/bridge_requirements.txt
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r /tmp/bridge_requirements.txt 2>/dev/null || true

# ── Bridge scripts ─────────────────────────────────────────────────────────────
RUN mkdir -p /vector/bridge
COPY bridge/ /vector/bridge/
RUN chmod +x /vector/bridge/*.py

# ── Custom lidar config + RL policies ─────────────────────────────────────────
RUN mkdir -p /vector/bridge/lidar_configs /vector/bridge/policies
COPY bridge/lidar_configs/ /vector/bridge/lidar_configs/
COPY bridge/policies/ /vector/bridge/policies/

# ── Environment ────────────────────────────────────────────────────────────────
ENV ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    ROS_DOMAIN_ID=0 \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    CYCLONEDDS_URI=/opt/cyclonedds.xml \
    BRIDGE_SCRIPT=/vector/bridge/isaac_sim_bridge.py

# ── Entrypoint ─────────────────────────────────────────────────────────────────
COPY docker-entrypoint.sh /vector/docker-entrypoint.sh
RUN chmod +x /vector/docker-entrypoint.sh

# Run as root for volume write access and asset downloads.
# Isaac Sim 5.1 defaults to uid 1234 but needs write access to
# cache dirs and asset download paths.

ENTRYPOINT ["/vector/docker-entrypoint.sh"]
CMD []

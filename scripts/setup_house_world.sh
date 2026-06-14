#!/bin/bash
# Vector OS Nano — house-world one-click setup (N5).
#
# Idempotent + SELF-CHECKING: every step verifies its own result and an
# already-provisioned machine prints all [OK]/[SKIP] and exits 0. Heavy
# steps are license-gated and downloaded from upstream (datasets are NEVER
# vendored in this repo):
#   - habitat-sim 0.3.3 conda env (headless, EGL — needs an NVIDIA GPU)
#   - ReplicaCAD scene dataset (CC-BY-4.0, ~300 MB, Hugging Face git-LFS)
#   - Unitree G1 visible body (composed from MuJoCo Menagerie, BSD-3)
#
# Optional siblings (status-only — their licenses/repos are NOT fetched):
#   - vector_navigation_stack (sensor navigation; without it the sim-oracle
#     navigation still works)
#   - SysNav (semantic perception; PolyForm-NC — personal use, get it
#     yourself)
#
# Usage:
#   ./scripts/setup_house_world.sh           # provision + verify
#   ./scripts/setup_house_world.sh --check   # verify only, change nothing

set -u

CHECK_ONLY=""
for arg in "$@"; do
    case $arg in
        --check) CHECK_ONLY="1" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
DATA_ROOT="${VECTOR_HABITAT_DATA:-$HOME/sandbox/habitat-spike/data/scene_datasets}"
DATA_PARENT="$(dirname "$DATA_ROOT")"
CONDA_ENV="${VECTOR_HABITAT_CONDA_ENV:-habitat-spike}"
HAB_PY="${VECTOR_HABITAT_PYTHON:-$HOME/miniconda3/envs/$CONDA_ENV/bin/python}"
MENAGERIE="${VECTOR_MENAGERIE:-$HOME/Desktop/mujoco_menagerie}"
FAIL=0

ok()   { printf '  [OK]   %s\n' "$1"; }
skip() { printf '  [SKIP] %s\n' "$1"; }
miss() { printf '  [MISS] %s\n      -> %s\n' "$1" "$2"; FAIL=1; }
act()  { printf '  [....] %s\n' "$1"; }

echo "== Vector OS Nano house-world setup =="
echo "   data root: $DATA_ROOT"
if [ -n "$CHECK_ONLY" ]; then echo "   mode:      check-only"; else echo "   mode:      provision"; fi

# -- 1. conda + habitat env -------------------------------------------------
echo "[1/6] habitat conda env ($CONDA_ENV, python 3.9 + habitat-sim 0.3.3 headless)"
if [ -x "$HAB_PY" ] && "$HAB_PY" -c "import habitat_sim" 2>/dev/null; then
    ok "habitat-sim importable: $HAB_PY"
elif [ -n "$CHECK_ONLY" ]; then
    miss "habitat env" "run without --check, or: conda create -n $CONDA_ENV python=3.9 && conda install -n $CONDA_ENV habitat-sim=0.3.3 headless -c conda-forge -c aihabitat"
else
    # conda is usually a SHELL FUNCTION (bashrc) — invisible to scripts;
    # resolve the binary directly so non-interactive runs work too.
    CONDA_BIN="${CONDA_EXE:-}"
    [ -x "$CONDA_BIN" ] || CONDA_BIN="$(command -v conda 2>/dev/null || true)"
    [ -x "$CONDA_BIN" ] || CONDA_BIN="$HOME/miniconda3/bin/conda"
    if [ ! -x "$CONDA_BIN" ]; then
        miss "conda not found" "install miniconda first: https://docs.conda.io/en/latest/miniconda.html"
    else
        act "creating $CONDA_ENV (this downloads ~1 GB, takes minutes)"
        "$CONDA_BIN" create -y -n "$CONDA_ENV" python=3.9 >/dev/null \
          && "$CONDA_BIN" install -y -n "$CONDA_ENV" habitat-sim=0.3.3 headless \
               -c conda-forge -c aihabitat >/dev/null \
          && "$HAB_PY" -m pip install -q "numpy==1.26.4" "pillow==10.4.0" \
               "opencv-python==4.9.0.80"
        if "$HAB_PY" -c "import habitat_sim" 2>/dev/null; then
            ok "habitat env created"
        else
            miss "habitat env creation failed" "see conda output above"
        fi
    fi
fi

# -- 2. numpy pin re-verify (the silent-upgrade landmine) --------------------
echo "[2/6] numpy pin (habitat-sim is numpy-1 ABI; pip silently upgrades it)"
if [ -x "$HAB_PY" ]; then
    NUMPY_V="$("$HAB_PY" -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo none)"
    case "$NUMPY_V" in
        1.26.*) ok "numpy $NUMPY_V" ;;
        none)   miss "numpy not importable in $CONDA_ENV" "reinstall the env (step 1)" ;;
        *)
            if [ -n "$CHECK_ONLY" ]; then
                miss "numpy $NUMPY_V (must be 1.26.x)" "$HAB_PY -m pip install numpy==1.26.4 pillow==10.4.0"
            else
                act "re-pinning numpy ($NUMPY_V -> 1.26.4)"
                "$HAB_PY" -m pip install -q "numpy==1.26.4" "pillow==10.4.0"
                NUMPY_V="$("$HAB_PY" -c 'import numpy; print(numpy.__version__)')"
                case "$NUMPY_V" in 1.26.*) ok "numpy re-pinned to $NUMPY_V" ;;
                                   *) miss "numpy still $NUMPY_V" "fix manually" ;; esac
            fi
            ;;
    esac
fi

# -- 3. ReplicaCAD (CC-BY-4.0) ------------------------------------------------
echo "[3/6] ReplicaCAD scene dataset (CC-BY-4.0, ~300 MB)"
RC_CFG="$DATA_ROOT/replica_cad/replicaCAD.scene_dataset_config.json"
if [ -f "$RC_CFG" ]; then
    ok "ReplicaCAD present: $(dirname "$RC_CFG")"
elif [ -n "$CHECK_ONLY" ]; then
    miss "ReplicaCAD missing" "run without --check to download (license: CC-BY-4.0)"
else
    if ! command -v git-lfs >/dev/null 2>&1 && ! git lfs version >/dev/null 2>&1; then
        miss "git-lfs not installed" "sudo apt install git-lfs && git lfs install"
    elif [ ! -x "$HAB_PY" ]; then
        miss "ReplicaCAD needs the habitat env first" "fix step 1, then re-run"
    else
        act "downloading ReplicaCAD (HF git-LFS; license CC-BY-4.0 by Meta)"
        mkdir -p "$DATA_PARENT"
        (cd "$DATA_PARENT/.." 2>/dev/null || cd "$DATA_PARENT"; \
         "$HAB_PY" -m habitat_sim.utils.datasets_download \
            --uids replica_cad_dataset --data-path "$DATA_PARENT" >/dev/null 2>&1)
        mkdir -p "$DATA_ROOT"
        [ -e "$DATA_ROOT/replica_cad" ] || ln -sfn "$DATA_PARENT/versioned_data/replica_cad_dataset" "$DATA_ROOT/replica_cad"
        if [ -f "$RC_CFG" ]; then ok "ReplicaCAD downloaded"; else miss "ReplicaCAD download failed" "re-run; check network / git-lfs"; fi
    fi
fi

# -- 4. G1 visible body (Menagerie BSD-3) -------------------------------------
echo "[4/6] G1 robot body GLB (composed from MuJoCo Menagerie, BSD-3)"
G1_GLB="$DATA_ROOT/robots/unitree_g1/g1_rigid.glb"
if [ -f "$G1_GLB" ]; then
    ok "G1 body present: $G1_GLB"
elif [ -n "$CHECK_ONLY" ]; then
    miss "G1 body missing" "run without --check (clones Menagerie if needed, builds the GLB)"
else
    if [ ! -d "$MENAGERIE/unitree_g1" ]; then
        act "cloning MuJoCo Menagerie (sparse would do; full clone ~recent)"
        git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie "$MENAGERIE" >/dev/null 2>&1 \
            || miss "Menagerie clone failed" "git clone https://github.com/google-deepmind/mujoco_menagerie $MENAGERIE"
    fi
    if [ -d "$MENAGERIE/unitree_g1" ]; then
        BUILD_VENV="$DATA_PARENT/g1-build-venv"
        if [ ! -x "$BUILD_VENV/bin/python" ]; then
            act "creating GLB build venv (mujoco + trimesh, isolated)"
            python3 -m venv "$BUILD_VENV" && "$BUILD_VENV/bin/pip" install -q mujoco trimesh numpy
        fi
        act "composing g1_rigid.glb"
        "$BUILD_VENV/bin/python" "$REPO_DIR/scripts/build_g1_glb.py" \
            --mjcf "$MENAGERIE/unitree_g1/g1.xml" --out "$G1_GLB" >/dev/null \
            && ok "G1 body built" || miss "G1 GLB build failed" "run scripts/build_g1_glb.py manually"
    fi
fi

# -- 5. repo venv --------------------------------------------------------------
echo "[5/6] repo .venv (python 3.12 + vector-os-nano)"
if [ -x "$REPO_DIR/.venv/bin/python" ] && "$REPO_DIR/.venv/bin/python" -c "import vector_os_nano" 2>/dev/null; then
    ok "repo venv importable"
else
    miss "repo venv" "cd $REPO_DIR && uv sync (or: python3.12 -m venv .venv && .venv/bin/pip install -e .)"
fi

# -- 6. optional siblings (status only) ----------------------------------------
echo "[6/6] optional: ROS2 + sensor navigation + semantic perception"
if [ -f /opt/ros/jazzy/setup.bash ]; then ok "ROS2 jazzy present"; else skip "ROS2 jazzy absent — oracle navigation still works; sensor nav (N2) needs it"; fi
NAV_WS="${VECTOR_NAV_STACK:-$HOME/Desktop/vector_navigation_stack}"
if [ -f "$NAV_WS/install/setup.bash" ]; then ok "vector_navigation_stack built: $NAV_WS"; else skip "nav stack workspace absent — scripts/launch_habitat_nav.sh needs it (separate repo)"; fi
SYSNAV_WS="${VECTOR_SYSNAV_WS:-$HOME/Desktop/SysNav}"
if [ -f "$SYSNAV_WS/.venv-sysnav/bin/python" ]; then ok "SysNav workspace provisioned: $SYSNAV_WS"; else skip "SysNav absent — semantic goals need it (PolyForm-NC: obtain it yourself)"; fi

echo ""
if [ "$FAIL" = "1" ]; then
    echo "RESULT: INCOMPLETE — fix the [MISS] items above and re-run."
    exit 1
fi
echo "RESULT: READY — launch with: .venv/bin/vector-cli --scenario house"
echo "        (or plain 'vector-cli' then say: 启动habitat模拟)"
exit 0

"""Vector OS Nano — unified launcher.

Connects SO-101 arm + D405 camera + VLM (Moondream) + EdgeTAM tracker + LLM.

Usage::

    python run.py              # CLI mode (default)
    python run.py --dashboard  # TUI dashboard mode
    python run.py --cli        # CLI mode (explicit)
    python run.py -d           # TUI dashboard (short flag)

Set OPENROUTER_API_KEY in the environment (or in config/user.yaml) before
running if you want LLM-powered natural language control.

No ROS2 required — pure Python.
"""
import argparse
import os
import sys
import logging

# Suppress Qt font warnings from OpenCV
os.environ["QT_LOGGING_RULES"] = "*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Pre-import cv2 with stderr suppressed to catch Qt font warnings.
# Skip if cv2 is not installed (e.g. sim-only mode without perception).
_stderr = sys.stderr
try:
    sys.stderr = open(os.devnull, "w")
    import cv2 as _cv2_preload  # noqa: F401 — triggers Qt init
except ImportError:
    pass
finally:
    sys.stderr = _stderr

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
logger = logging.getLogger("vector_os")


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _load_calibration_yaml(cal_file: str) -> "Calibration | None":
    """Load a vector_ws-format workspace_calibration.yaml into a Calibration object.

    The YAML schema (produced by the vector_ws calibration wizard) contains:
        transform_matrix: [[...], [...], [...], [...]]
        points_camera: [[x, y, z], ...]
        points_base: [[x, y, z], ...]
        mean_error_mm: float
        num_points: int

    Falls back to Calibration() (identity) on any error.

    Args:
        cal_file: Path to the YAML calibration file.

    Returns:
        Loaded Calibration instance, or None on failure.
    """
    try:
        import yaml
        import numpy as np
        from vector_os_nano.perception.calibration import Calibration

        with open(cal_file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict):
            logger.warning("Calibration YAML does not contain a mapping: %s", cal_file)
            return None

        cal = Calibration()

        # Load 4x4 transform matrix
        raw_matrix = data.get("transform_matrix")
        if raw_matrix is not None:
            matrix = np.array(raw_matrix, dtype=np.float64)
            if matrix.shape == (4, 4):
                cal._matrix = matrix
            else:
                logger.warning(
                    "transform_matrix has unexpected shape %s in %s",
                    matrix.shape,
                    cal_file,
                )

        # Load calibration point correspondences so get_error_stats() works
        pts_cam_raw = data.get("points_camera")
        pts_base_raw = data.get("points_base")
        if pts_cam_raw is not None and pts_base_raw is not None:
            pts_cam = np.array(pts_cam_raw, dtype=np.float64)
            pts_base = np.array(pts_base_raw, dtype=np.float64)
            if pts_cam.ndim == 2 and pts_cam.shape[1] == 3:
                cal._cal_points_cam = pts_cam
                cal._cal_points_base = pts_base

        return cal

    except FileNotFoundError:
        logger.warning("Calibration file not found: %s", cal_file)
        return None
    except Exception as exc:
        logger.warning("Could not load calibration from %s: %s", cal_file, exc)
        return None


# ---------------------------------------------------------------------------
# Hardware initialisation — single code path shared by CLI and Dashboard
# ---------------------------------------------------------------------------

def _init_hardware(cfg: dict) -> tuple:
    """Initialise arm, camera, VLM, tracker, and calibration from config.

    Startup order mirrors perception.launch.py from vector_ws:
      1. Arm
      2. Camera
      3. VLM (Moondream)  — may download ~1.8 GB on first run
      4. Tracker (EdgeTAM)
      5. Calibration

    Args:
        cfg: Loaded config dict (from load_config).

    Returns:
        Tuple of (arm, gripper, perception, calibration).
        Any component that fails to start is returned as None.
    """
    # -------------------------------------------------------------------
    # Arm
    # -------------------------------------------------------------------
    arm = None
    gripper = None
    try:
        from vector_os_nano.hardware.so101 import SO101Arm, SO101Gripper
        port = cfg.get("arm", {}).get("port", "/dev/ttyACM0")
        arm = SO101Arm(port=port)
        print(f"Connecting arm on {port}...")
        arm.connect()
        gripper = SO101Gripper(arm._bus)
        joints = [round(j, 2) for j in arm.get_joint_positions()]
        print(f"Arm connected. Joints: {joints}")
    except Exception as exc:
        print(f"Arm not available: {exc}")

    # -------------------------------------------------------------------
    # Camera + Perception
    # -------------------------------------------------------------------
    perception = None
    try:
        from vector_os_nano.perception.realsense import RealSenseCamera
        from vector_os_nano.perception.vlm import VLMDetector
        from vector_os_nano.perception.tracker import EdgeTAMTracker
        from vector_os_nano.perception.pipeline import PerceptionPipeline

        # 1. Camera
        print("Connecting camera (RealSense D405)...")
        camera = RealSenseCamera()
        camera.connect()
        print("Camera connected.")

        # 2. VLM — first-run may download model weights (~1.8 GB)
        vlm_model = (
            cfg.get("perception", {}).get("vlm_model")
            or os.environ.get("MOONDREAM_MODEL", "vikhyatk/moondream2")
        )
        os.environ.setdefault("MOONDREAM_MODEL", vlm_model)
        print(f"Loading VLM ({vlm_model})... first run may download weights")
        vlm = VLMDetector()
        print("VLM loaded.")

        # 3. Tracker — first-run may download model weights
        tracker_model = cfg.get("perception", {}).get(
            "tracker_model", "yonigozlan/EdgeTAM-hf"
        )
        print(f"Loading tracker (EdgeTAM / {tracker_model})...")
        tracker = EdgeTAMTracker()
        print("Tracker loaded.")

        # 4. Pipeline
        perception = PerceptionPipeline(camera=camera, vlm=vlm, tracker=tracker)
        print("Perception pipeline ready.")

    except Exception as exc:
        import traceback
        print(f"Perception not available: {exc}")
        traceback.print_exc()

    # -------------------------------------------------------------------
    # Calibration
    # Prefer YAML format (from vector_ws wizard). The vector_os Calibration
    # class normally loads .npy files; we provide a YAML bridge here.
    # -------------------------------------------------------------------
    calibration = None
    cal_file: str = cfg.get("calibration", {}).get(
        "file", "config/workspace_calibration.yaml"
    )
    if cal_file and os.path.exists(cal_file):
        if cal_file.endswith((".yaml", ".yml")):
            calibration = _load_calibration_yaml(cal_file)
        else:
            # .npy path — use Calibration.load() directly
            try:
                from vector_os_nano.perception.calibration import Calibration
                calibration = Calibration.load(cal_file)
            except Exception as exc:
                print(f"Calibration load failed: {exc}")

        if calibration is not None:
            stats = calibration.get_error_stats()
            if stats and stats.get("mean_m") is not None:
                print(
                    f"Calibration loaded: {stats['num_points']} points, "
                    f"{stats['mean_m'] * 1000:.1f} mm mean error"
                )
            else:
                print("Calibration loaded (no error stats available).")
    else:
        print(f"No calibration file at {cal_file!r} — running without calibration.")

    return arm, gripper, perception, calibration


# ---------------------------------------------------------------------------
# Simulation initialisation — MuJoCo sim, no real hardware needed
# ---------------------------------------------------------------------------

def _init_sim(cfg: dict, gui: bool = False) -> tuple:
    """Initialise MuJoCo simulated arm, gripper, and perception.

    Perception uses MuJoCo ground-truth object positions (no camera/VLM).
    Calibration is identity (positions already in world frame).

    Args:
        cfg: Config dict (currently unused for sim, reserved for future options).
        gui: Open the MuJoCo viewer window for visual feedback.

    Returns:
        Tuple of (arm, gripper, perception, calibration).
    """
    from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm
    from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper
    from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception
    from vector_os_nano.perception.calibration import Calibration

    print("Starting MuJoCo simulation...")
    arm = MuJoCoArm(gui=gui)
    arm.connect()
    gripper = MuJoCoGripper(arm)

    # Start with gripper closed (rest state)
    gripper.close()

    joints = [round(j, 2) for j in arm.get_joint_positions()]
    print(f"Sim arm connected. Joints: {joints}")

    objs = arm.get_object_positions()
    if objs:
        print(f"Scene objects: {', '.join(objs.keys())}")

    # Simulated perception — ground truth from MuJoCo
    perception = MuJoCoPerception(arm)
    print("Sim perception ready (ground-truth mode).")

    # Identity calibration — sim positions are already in base frame
    calibration = Calibration()

    # Override config for sim mode:
    # 1. Disable real-robot hardware offsets (sim positions are world-frame)
    # 2. Set sim-appropriate z_offset and pre-grasp height
    # Real-robot config (in user.yaml) is NOT touched.
    import math as _math
    cfg.setdefault("skills", {}).setdefault("pick", {}).update({
        "z_offset": 0.0,
        "x_offset": 0.0,
        "pre_grasp_height": 0.04,
        "hardware_offsets": False,
        "wrist_roll_offset": _math.pi / 2,
    })
    cfg.setdefault("skills", {}).setdefault("home", {}).setdefault(
        "joint_values", [0.0, 0.0, 0.0, 0.0, 0.0]
    )

    # Sim motion speed: default duration for arm moves (seconds)
    cfg["sim_move_duration"] = 3.0

    return arm, gripper, perception, calibration


# ---------------------------------------------------------------------------
# Cleanup helper
# ---------------------------------------------------------------------------

def _shutdown(arm, perception) -> None:
    """Disconnect hardware gracefully."""
    print("Shutting down...")
    if perception is not None and hasattr(perception, "stop_continuous_tracking"):
        try:
            perception.stop_continuous_tracking()
        except Exception as exc:
            logger.warning("Error stopping background tracking: %s", exc)
    if arm is not None:
        try:
            arm.disconnect()
            print("Arm disconnected.")
        except Exception as exc:
            logger.warning("Error disconnecting arm: %s", exc)
    if perception is not None and hasattr(perception, "disconnect"):
        try:
            perception.disconnect()
            print("Perception disconnected.")
        except Exception as exc:
            logger.warning("Error disconnecting perception: %s", exc)
    # Force exit — MuJoCo viewer GL context can hang the process
    os._exit(0)


# ---------------------------------------------------------------------------
# Shared camera viewer (used by both CLI and Agent modes)
# ---------------------------------------------------------------------------

def _start_camera_viewer(perception):
    """Start background OpenCV camera viewer thread. Returns (thread, stop_event) or (None, None)."""
    if perception is None or not hasattr(perception, '_camera') or perception._camera is None:
        return None, None

    import threading
    import cv2
    import numpy as np
    from PIL import Image as PILImage, ImageDraw, ImageFont

    stop_event = threading.Event()

    _pil_font = None
    for font_path in [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]:
        if os.path.exists(font_path):
            try:
                _pil_font = ImageFont.truetype(font_path, 16)
                break
            except Exception:
                pass
    if _pil_font is None:
        try:
            _pil_font = ImageFont.load_default()
        except Exception:
            pass

    def _put_text_pil(img, text, pos, color=(0, 255, 0)):
        if _pil_font is None:
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            return img
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text(pos, text, font=_pil_font, fill=color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _viewer_loop():
        cam = perception._camera
        cv2.namedWindow("Vector OS", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Vector OS", 1280, 480)

        while not stop_event.is_set():
            try:
                color = cam.get_color_frame()
                depth = cam.get_depth_frame()
                if color is None or depth is None:
                    continue

                rgb_display = color.copy()
                if hasattr(perception, '_last_tracked') and perception._last_tracked:
                    for obj in perception._last_tracked:
                        if obj.bbox_2d:
                            x1, y1, x2, y2 = [int(v) for v in obj.bbox_2d]
                            cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            lbl = obj.label
                            if obj.pose:
                                lbl += f" ({obj.pose.x:.2f},{obj.pose.y:.2f},{obj.pose.z:.2f})"
                            rgb_display = _put_text_pil(rgb_display, lbl, (x1, max(y1 - 20, 0)))
                elif hasattr(perception, '_last_detections') and perception._last_detections:
                    for det in perception._last_detections:
                        x1, y1, x2, y2 = [int(v) for v in det.bbox]
                        cv2.rectangle(rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        rgb_display = _put_text_pil(rgb_display, det.label, (x1, max(y1 - 20, 0)))

                depth_f = np.clip(depth.astype(np.float32), 0, 500)
                depth_u8 = (depth_f / 500.0 * 255).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

                if hasattr(perception, '_last_tracked') and perception._last_tracked:
                    for obj in perception._last_tracked:
                        if obj.mask is not None and obj.mask.shape == depth_colored.shape[:2]:
                            contours, _ = cv2.findContours(
                                obj.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            cv2.drawContours(depth_colored, contours, -1, (255, 255, 255), 2)
                        if obj.pose and obj.bbox_2d:
                            cx = int((obj.bbox_2d[0] + obj.bbox_2d[2]) / 2)
                            cy = int((obj.bbox_2d[1] + obj.bbox_2d[3]) / 2)
                            cv2.circle(depth_colored, (cx, cy), 6, (0, 255, 0), -1)
                            cv2.circle(depth_colored, (cx, cy), 8, (255, 255, 255), 2)
                            info = f"{obj.pose.x:.3f},{obj.pose.y:.3f},{obj.pose.z:.3f}"
                            cv2.putText(depth_colored, info, (cx + 10, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                combined = np.hstack([rgb_display, depth_colored])
                cv2.imshow("Vector OS", combined)
                if cv2.waitKey(33) == 27:
                    break
            except Exception:
                pass

        cv2.destroyAllWindows()

    thread = threading.Thread(target=_viewer_loop, daemon=True)
    thread.start()
    print("Viewer started: RGB + Depth side-by-side (ESC to close)")
    return thread, stop_event


# ---------------------------------------------------------------------------
# Agent mode — LLM tool-calling conversation
# ---------------------------------------------------------------------------

def _run_agent_mode(agent, cfg, api_key, args) -> None:
    """Start the tool-calling agent mode with native function calling."""
    from rich.console import Console
    from rich.panel import Panel
    from prompt_toolkit import PromptSession

    if not api_key:
        print("ERROR: --agent mode requires OPENROUTER_API_KEY")
        return

    from vector_os_nano.core.tool_agent import ToolAgent

    model = args.agent_model or cfg.get("llm", {}).get("models", {}).get("agent", "openai/gpt-4o-mini")
    api_base = cfg.get("llm", {}).get("api_base", "https://openrouter.ai/api/v1")

    tool_agent = ToolAgent(
        agent_ref=agent,
        api_key=api_key,
        model=model,
        api_base=api_base,
    )

    console = Console()
    session = PromptSession()
    _teal = "#00b4b4"

    # Print the same logo as SimpleCLI
    import pathlib as _pl
    _logo_path = _pl.Path(__file__).parent / "vector_os_nano" / "cli" / "logo_braille.txt"
    if not _logo_path.exists():
        _logo_path = _pl.Path("vector_os_nano/cli/logo_braille.txt")

    console.clear()
    try:
        logo_lines = _logo_path.read_text().strip().splitlines()
        for line in logo_lines:
            console.print(f"[bold {_teal}]{line}[/]")
            import time as _t; _t.sleep(0.12)
    except FileNotFoundError:
        console.print(f"[bold {_teal}]VECTOR OS NANO[/]")

    from vector_os_nano.version import __version__ as _ver
    console.print(f"[dim]{'':>40}v{_ver}[/]")
    console.print(f"[dim]  Agent Mode | Model: {model} | Tools: {len(tool_agent._tools)} | 'q' to quit[/]")
    console.print()

    # Start camera viewer (same as CLI mode)
    perception = agent._perception
    camera_thread, stop_camera = _start_camera_viewer(perception)

    while True:
        try:
            user_input = session.prompt("you> ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            break

        def _on_tool_call(name, params):
            detail = ", ".join(f"{k}={v}" for k, v in params.items()) if params else ""
            console.print(f"  [yellow]TOOL[/] [{_teal}]{name}[/]({detail})")

        def _on_debug(stage, detail):
            if args.verbose:
                console.print(f"  [dim cyan][{stage}][/] [dim]{detail}[/]")

        response = tool_agent.chat(
            user_input,
            on_tool_call=_on_tool_call,
            on_debug=_on_debug,
        )

        console.print()
        console.print(Panel(
            response,
            title="[bold]V[/]",
            title_align="left",
            border_style=_teal,
            padding=(0, 1),
            width=min(console.width, 70),
        ))
        console.print()

    # Cleanup camera viewer
    if stop_camera is not None:
        stop_camera.set()
    if camera_thread is not None:
        camera_thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# CLI mode — readline shell + OpenCV camera viewer
# ---------------------------------------------------------------------------

def _run_cli(agent, perception, verbose: bool = False) -> None:
    """Start SimpleCLI and an OpenCV camera viewer (background thread)."""
    camera_thread, stop_camera = _start_camera_viewer(perception)

    from vector_os_nano.cli.simple import SimpleCLI
    cli = SimpleCLI(agent=agent, verbose=verbose)

    try:
        cli.run()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    finally:
        if stop_camera is not None:
            stop_camera.set()
        if camera_thread is not None:
            camera_thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Dashboard mode — Textual TUI (dashboard launches its own OpenCV viewer via F1)
# ---------------------------------------------------------------------------

def _run_dashboard(agent) -> None:
    """Start the Textual TUI dashboard with the given agent."""
    try:
        from vector_os_nano.cli.dashboard import DashboardApp, TEXTUAL_AVAILABLE
    except ImportError as exc:
        print(f"Dashboard import failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if not TEXTUAL_AVAILABLE:
        print("ERROR: textual not installed. pip install 'vector-os-nano[tui]'")
        sys.exit(1)

    app = DashboardApp(agent=agent)
    try:
        app.run()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")


# ---------------------------------------------------------------------------
# Web dashboard mode — FastAPI + WebSocket
# ---------------------------------------------------------------------------

def _run_web(agent, cfg: dict, port: int = 8000) -> None:
    """Start the FastAPI web dashboard."""
    try:
        import uvicorn
        from vector_os_nano.web.app import create_app
    except ImportError as exc:
        print(f"Web dashboard requires: pip install fastapi uvicorn\n{exc}")
        sys.exit(1)

    app = create_app(agent, cfg)

    # Auto-open browser
    import webbrowser
    import threading
    def _open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{port}")
    threading.Thread(target=_open_browser, daemon=True).start()

    print(f"\n  Web dashboard: http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Boot the full Vector OS Nano stack and start CLI or Dashboard."""
    parser = argparse.ArgumentParser(
        description="Vector OS Nano — unified launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python run.py              # CLI (default)\n"
            "  python run.py --dashboard  # TUI dashboard\n"
            "  python run.py -d           # TUI dashboard (short flag)\n"
            "  python run.py --cli        # CLI (explicit)\n"
            "  python run.py --sim              # MuJoCo sim with viewer\n"
            "  python run.py --sim-headless     # MuJoCo sim without viewer\n"
            "  python run.py --web --sim        # Web dashboard + sim\n"
            "  python run.py --sim -d           # Sim + TUI dashboard\n"
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Launch the Textual TUI dashboard instead of the CLI",
    )
    mode_group.add_argument(
        "--web", "-w",
        action="store_true",
        help="Launch web dashboard at http://localhost:8000",
    )
    mode_group.add_argument(
        "--cli",
        action="store_true",
        help="Launch the readline CLI (default when no flag given)",
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="MuJoCo simulation with viewer (no hardware needed)",
    )
    parser.add_argument(
        "--sim-headless",
        action="store_true",
        help="MuJoCo simulation without viewer (headless)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show agent thinking process (match, classify, route decisions)",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Use LLM tool-calling agent mode (smarter conversation, understands context)",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default=None,
        help="Model for agent mode (default: openai/gpt-4o-mini). Examples: openai/gpt-4o, anthropic/claude-sonnet-4-6",
    )
    args = parser.parse_args()

    dashboard_mode: bool = args.dashboard
    sim_mode: bool = args.sim or args.sim_headless

    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.config import load_config

    cfg = load_config("config/user.yaml")

    if sim_mode:
        arm, gripper, perception, calibration = _init_sim(cfg, gui=not args.sim_headless)
    else:
        arm, gripper, perception, calibration = _init_hardware(cfg)

    api_key = cfg.get("llm", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY")
    agent = Agent(
        arm=arm,
        gripper=gripper,
        perception=perception,
        llm_api_key=api_key,
        config=cfg,
    )

    # Inject calibration loaded from YAML (bypasses Calibration.load() .npy path)
    if calibration is not None:
        agent._calibration = calibration

    # Status summary
    print()
    if sim_mode:
        print(f"Mode      : MuJoCo simulation {'(headless)' if args.sim_headless else '(GUI)'}")
    print(f"Skills    : {', '.join(agent.skills)}")
    llm_label = (
        "configured (" + cfg.get("llm", {}).get("model", "unknown") + ")"
        if api_key else "none (set OPENROUTER_API_KEY)"
    )
    print(f"LLM       : {llm_label}")
    print(f"Perception: {'ready' if perception else 'not available'}")
    print(f"Calibration: {'loaded' if calibration else 'not loaded'}")
    print()

    try:
        if args.web:
            _run_web(agent, cfg)
        elif dashboard_mode:
            _run_dashboard(agent)
        elif args.agent:
            _run_agent_mode(agent, cfg, api_key, args)
        else:
            _run_cli(agent, perception, verbose=args.verbose)
    finally:
        _shutdown(arm, perception)


def main_dashboard() -> None:
    """Entry point that forces dashboard mode — used by vector-os-dashboard script."""
    # Inject --dashboard so main()'s argparse picks it up
    if "--dashboard" not in sys.argv and "-d" not in sys.argv:
        sys.argv.append("--dashboard")
    main()


if __name__ == "__main__":
    main()

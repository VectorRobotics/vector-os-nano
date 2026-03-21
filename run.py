"""Vector OS Nano — Full System.

Connects SO-101 arm + D405 camera + VLM (Moondream) + EdgeTAM tracker + LLM.

Usage::

    cd ~/Desktop/vector_os
    python run.py

Set OPENROUTER_API_KEY in the environment (or in config/user.yaml) before
running if you want LLM-powered natural language control.

No ROS2 required — pure Python.
"""
import os
import sys
import logging

# Suppress Qt font warnings from OpenCV
os.environ["QT_LOGGING_RULES"] = "*=false"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Pre-import cv2 with stderr suppressed to catch Qt font warnings
_stderr = sys.stderr
try:
    sys.stderr = open(os.devnull, "w")
    import cv2 as _cv2_preload  # noqa: F401 — triggers Qt init
finally:
    sys.stderr = _stderr

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Boot the full Vector OS Nano stack and start the interactive CLI."""
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.config import load_config

    cfg = load_config("config/user.yaml")

    # -----------------------------------------------------------------------
    # Arm
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Camera + Perception
    # Startup order mirrors perception.launch.py from vector_ws:
    #   1. Camera
    #   2. VLM (Moondream)   — delayed in ROS2 launch by 3 s after camera
    #   3. Tracker (EdgeTAM) — loaded after VLM
    #   4. Pipeline assembly
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Calibration
    # Prefer YAML format (from vector_ws wizard).  The vector_os Calibration
    # class normally loads .npy files; we provide a YAML bridge here.
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Agent
    # -----------------------------------------------------------------------
    api_key = cfg.get("llm", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY")

    agent = Agent(
        arm=arm,
        gripper=gripper,
        perception=perception,
        llm_api_key=api_key,
        config=cfg,
        # auto_perception=False: perception already wired above
    )

    # Inject calibration loaded from YAML (bypasses Calibration.load() .npy path)
    if calibration is not None:
        agent._calibration = calibration

    # -----------------------------------------------------------------------
    # Status summary
    # -----------------------------------------------------------------------
    print()
    print(f"Skills    : {', '.join(agent.skills)}")
    print(f"LLM       : {'configured (' + cfg.get('llm', {}).get('model', 'unknown') + ')' if api_key else 'none (set OPENROUTER_API_KEY)'}")
    print(f"Perception: {'ready' if perception else 'not available'}")
    print(f"Calibration: {'loaded' if calibration else 'not loaded'}")
    print()

    # -----------------------------------------------------------------------
    # Live camera viewer (background thread)
    # Shows raw feed + bounding box overlay in an OpenCV window
    # -----------------------------------------------------------------------
    camera_thread = None
    stop_camera = None
    if perception is not None and hasattr(perception, '_camera') and perception._camera is not None:
        import threading
        import cv2
        import numpy as np
        from PIL import Image as PILImage, ImageDraw, ImageFont

        stop_camera = threading.Event()

        # Load a font that supports Chinese
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
            """Draw text with PIL (supports Unicode/Chinese)."""
            if _pil_font is None:
                cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                return img
            pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text(pos, text, font=_pil_font, fill=color)
            return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        def _viewer_loop():
            """Single background thread: RGB + depth side by side."""
            cam = perception._camera
            cv2.namedWindow("Vector OS", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Vector OS", 1280, 480)

            while not stop_camera.is_set():
                try:
                    color = cam.get_color_frame()
                    depth = cam.get_depth_frame()
                    if color is None or depth is None:
                        continue

                    # --- Left: RGB + tracking overlay ---
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

                    # --- Right: Depth colormap + mask + centroid ---
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

                    # --- Combine side by side ---
                    combined = np.hstack([rgb_display, depth_colored])
                    cv2.imshow("Vector OS", combined)
                    key = cv2.waitKey(33)
                    if key == 27:
                        break
                except Exception:
                    pass

            cv2.destroyAllWindows()

        camera_thread = threading.Thread(target=_viewer_loop, daemon=True)
        camera_thread.start()
        print("Viewer started: RGB + Depth side-by-side (ESC to close)")

    # -----------------------------------------------------------------------
    # CLI
    # -----------------------------------------------------------------------
    from vector_os_nano.cli.simple import SimpleCLI
    cli = SimpleCLI(agent=agent, verbose=True)

    try:
        cli.run()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")
    finally:
        print("Shutting down...")
        if stop_camera is not None:
            stop_camera.set()
        if camera_thread is not None:
            camera_thread.join(timeout=2.0)
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


if __name__ == "__main__":
    main()

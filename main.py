"""
Real-Time Multi-Object Detection and Distance Estimation
Main application entry point.

Integrates:
- Arducam / UVC camera capture via OpenCV (fixed-focus, no autofocus)
- YOLOv8 multi-object detection
- Depth Anything V2 Metric depth estimation (real metres)
- Median depth extraction for object bounding boxes
- Simple IoU-based multi-object tracking
- Interactive camera intrinsics & metric depth calibration
- Distance validation and error reporting
- Live annotated display with side-by-side depth visualization
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np

import config
from camera_module import CameraManager
from detection_module import ObjectDetector
from depth_module import DepthEstimator
from distance_module import DistanceExtractor
from calibration_module import CalibrationManager
from tracker_module import SimpleTracker
from display_module import DisplayManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Multi-Object Detection and Distance Estimation Using Arducam Camera"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help=f"Camera index (default: {config.CAMERA_INDEX})",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=config.CAMERA_WIDTH,
        help=f"Capture width (default: {config.CAMERA_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=config.CAMERA_HEIGHT,
        help=f"Capture height (default: {config.CAMERA_HEIGHT})",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=config.CAMERA_FPS,
        help=f"Capture FPS (default: {config.CAMERA_FPS})",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default=config.YOLO_MODEL,
        help=f"YOLO model weight file (default: {config.YOLO_MODEL})",
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default=config.DEPTH_MODEL_TYPE,
        choices=["vits", "vitb", "vitl"],
        help=f"Depth Anything V2 model variant (default: {config.DEPTH_MODEL_TYPE})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=config.YOLO_CONFIDENCE_THRESHOLD,
        help=f"YOLO confidence threshold (default: {config.YOLO_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.YOLO_DEVICE,
        help=f"Compute device ('cuda:0' or 'cpu', default: {config.YOLO_DEVICE})",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable multi-object tracking",
    )
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Show depth map side-by-side with camera feed",
    )
    parser.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to custom calibration JSON file",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Launch interactive camera calibration mode using checkerboard",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Launch distance validation mode with objects at known distances",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional path to a test video or image file instead of live camera",
    )
    return parser.parse_args()


def run_calibration_mode(camera, calibration: CalibrationManager):
    """
    Interactive camera intrinsics calibration mode using a checkerboard pattern.
    Captures frames on keypress 'c' and runs cv2.calibrateCamera when finished ('q').
    """
    print("\n" + "=" * 60)
    print(" CAMERA INTRINSICS CALIBRATION MODE (Checkerboard)")
    print("=" * 60)
    print(f"Checkerboard inner corners: {calibration.checkerboard_size}")
    print(f"Square size: {calibration.square_size * 1000.0:.1f} mm")
    print("Controls:")
    print("  'c' - Capture frame when checkerboard is highlighted")
    print("  'q' - Finish capturing and compute calibration")
    print("=" * 60 + "\n")

    captured_frames = []

    while True:
        ret, frame = camera.read_frame()
        if not ret or frame is None:
            print("[Calibration] Failed to acquire camera frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_cb, corners = cv2.findChessboardCorners(gray, calibration.checkerboard_size, None)

        display_frame = frame.copy()
        if ret_cb:
            cv2.drawChessboardCorners(display_frame, calibration.checkerboard_size, corners, ret_cb)
            cv2.putText(
                display_frame,
                f"Checkerboard FOUND - Press 'c' to capture ({len(captured_frames)} captured)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                display_frame,
                f"Looking for checkerboard... ({len(captured_frames)} captured)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Camera Calibration", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            if ret_cb:
                captured_frames.append(frame.copy())
                print(f"[Calibration] Captured frame #{len(captured_frames)}")
            else:
                print("[Calibration] No checkerboard found in frame! Adjust camera or board.")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

    if len(captured_frames) >= 5:
        print(f"\n[Calibration] Processing {len(captured_frames)} frames...")
        success = calibration.calibrate_camera_intrinsics(captured_frames)
        if success:
            calibration.save_calibration()
            print("[Calibration] Intrinsics successfully saved to:", calibration.calibration_file)
        else:
            print("[Calibration] Intrinsics calculation failed.")
    else:
        print(f"[Calibration] Insufficient frames ({len(captured_frames)}/5 minimum). Calibration aborted.")


def run_validation_mode(
    camera,
    detector: ObjectDetector,
    depth_estimator: DepthEstimator,
    distance_extractor: DistanceExtractor,
    calibration: CalibrationManager,
):
    """
    Interactive metric distance validation mode.
    Allows user to place objects at known distances and record ground truth vs measured.
    Computes depth scale and offset correction.
    """
    print("\n" + "=" * 60)
    print(" METRIC DISTANCE VALIDATION & TUNING MODE")
    print("=" * 60)
    print("Place an object at a known distance from the camera.")
    print("Controls in display window:")
    print("  'v' - Record validation point for detected object (prompted in terminal)")
    print("  'r' - Print validation and error report")
    print("  'c' - Compute & save linear depth calibration (scale + offset)")
    print("  'q' - Quit validation mode")
    print("=" * 60 + "\n")

    while True:
        ret, frame = camera.read_frame()
        if not ret or frame is None:
            print("[Validation] Failed to acquire camera frame.")
            break

        # Object detection
        detections = detector.detect(frame)

        # Depth map
        depth_map = depth_estimator.estimate_depth(frame)
        if depth_map is not None:
            detections = distance_extractor.extract_all_distances(depth_map, detections)

            # Apply current calibration if active
            scale = calibration.calibration_params.get("scale", 1.0)
            offset = calibration.calibration_params.get("offset", 0.0)
            if scale != 1.0 or offset != 0.0:
                for det in detections:
                    if "distance_m" in det:
                        det["distance_m"] = distance_extractor.apply_calibration(
                            det["distance_m"], calibration.calibration_params
                        )
                        det["distance_cm"] = det["distance_m"] * 100.0

        display_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            dist_str = f"{det.get('distance_m', 0.0):.2f}m" if "distance_m" in det else "N/A"
            lbl = f"{det['class_name']} {dist_str}"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                lbl,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            display_frame,
            "VALIDATION: [v] Record  [r] Report  [c] Calibrate  [q] Quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Distance Validation Mode", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("v"):
            if not detections:
                print("[Validation] No objects currently detected to measure.")
                continue

            target_det = detections[0]
            obj_name = target_det.get("class_name", "object")
            measured_m = target_det.get("distance_m", 0.0)
            print(f"\nTarget detected: '{obj_name}' at estimated {measured_m:.2f} m.")
            try:
                actual_input = input("Enter actual measured ground-truth distance in metres (e.g. 1.5): ")
                actual_m = float(actual_input.strip())
                calibration.add_validation_point(actual_m, measured_m, obj_name)
                error_m = abs(measured_m - actual_m)
                rel_err = (error_m / actual_m) * 100.0 if actual_m > 0 else 0.0
                print(
                    f"-> Recorded: Actual = {actual_m:.2f}m | Measured = {measured_m:.2f}m | "
                    f"Error = {error_m:.3f}m ({rel_err:.1f}%)"
                )
            except ValueError:
                print("Invalid number entered, skipping.")

        elif key == ord("r"):
            print("\n" + calibration.get_validation_report())

        elif key == ord("c"):
            res = calibration.compute_depth_calibration()
            if res:
                print(f"\nCalibration computed: Scale = {res['scale']:.4f}, Offset = {res['offset']:.4f}")
                print(f"R² Fit = {res['r_squared']:.4f}, Mean Error = {res['mean_error_m']:.3f}m")
                calibration.save_calibration()
                print("Calibration parameters saved.")
            else:
                print("[Validation] Need at least 2 distinct validation points to compute calibration.")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


def main():
    """Main pipeline execution."""
    args = parse_args()

    print("=" * 70)
    print(" Real-Time Multi-Object Detection & Distance Estimation")
    print(" Arducam 12MP Fixed-Focus Camera System")
    print("=" * 70)

    # 1. Initialize Camera or Video Source
    if args.input:
        print(f"[Init] Using input source: {args.input}")
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"[Error] Failed to open input source: {args.input}")
            sys.exit(1)

        class VideoWrapper:
            def __init__(self, c):
                self.cap = c

            def read_frame(self):
                ret, f = self.cap.read()
                return ret, f

            def get_camera_info(self):
                return {
                    "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": self.cap.get(cv2.CAP_PROP_FPS),
                    "backend": "file",
                }

            def release(self):
                self.cap.release()

        camera = VideoWrapper(cap)
    else:
        print(f"[Init] Initializing camera index {args.camera} at {args.width}x{args.height} @ {args.fps} FPS...")
        camera = CameraManager(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        if not camera.initialize():
            print(f"[Error] Could not connect to camera at index {args.camera}.")
            print("Troubleshooting:")
            print("  - Check USB cable and port")
            print("  - Ensure no other app (e.g. Windows Camera, Zoom) is using it")
            print("  - Try testing with an image or video: python main.py --input test.mp4")
            sys.exit(1)

    cam_info = camera.get_camera_info()
    print(f"[Init] Camera Active: {cam_info['width']}x{cam_info['height']} @ {cam_info['fps']} FPS (Backend: {cam_info['backend']})")

    # 2. Load Calibration Data
    calibration = CalibrationManager()
    calib_file = args.calibration or config.CALIBRATION_FILE
    if os.path.exists(calib_file):
        if calibration.load_calibration(calib_file):
            print(f"[Init] Loaded calibration file: {calib_file}")
            print(f"       Scale: {calibration.calibration_params.get('scale', 1.0):.4f}, Offset: {calibration.calibration_params.get('offset', 0.0):.4f}")
    else:
        print("[Init] No existing calibration file found. Using default uncalibrated metric depth.")

    # 3. Handle Special Sub-Modes
    if args.calibrate:
        run_calibration_mode(camera, calibration)
        camera.release()
        return

    # 4. Load Models
    print(f"[Init] Loading YOLOv8 object detector ({args.yolo_model})...")
    detector = ObjectDetector(
        model_name=args.yolo_model,
        confidence=args.confidence,
        device=args.device,
    )
    if not detector.load_model():
        print("[Error] Failed to load YOLO model.")
        camera.release()
        sys.exit(1)

    print(f"[Init] Loading Depth Anything V2 Metric model ({args.depth_model})...")
    depth_estimator = DepthEstimator(
        model_type=args.depth_model,
        encoder=args.depth_model,
        device=args.device,
    )
    if not depth_estimator.load_model():
        print("[Error] Failed to load Depth Anything V2 Metric model.")
        camera.release()
        sys.exit(1)

    distance_extractor = DistanceExtractor()
    tracker = SimpleTracker() if not args.no_tracking else None
    display = DisplayManager()
    display.show_depth_map = args.show_depth

    # Check if user specifically requested validation mode
    if args.validate:
        run_validation_mode(camera, detector, depth_estimator, distance_extractor, calibration)
        camera.release()
        return

    print("\n" + "=" * 70)
    print(" LIVE STREAM STARTED")
    print(" Keyboard Controls:")
    print("   'q' : Quit application")
    print("   'c' : Switch active camera device (Cam 0 <-> Cam 1)")
    print("   'd' : Toggle depth map visualization side-by-side")
    print("   's' : Save current frame screenshot")
    print("=" * 70 + "\n")

    frame_count = 0
    cached_depth_map = None
    current_camera_idx = args.camera

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                print("\n[Stream] End of stream or failed to read frame.")
                break

            frame_count += 1

            # 1. Object Detection
            detections = detector.detect(frame)

            # 2. Depth Estimation (runs on configured cadence for smooth FPS)
            if frame_count % config.DEPTH_PROCESS_EVERY_N == 0 or cached_depth_map is None:
                new_depth = depth_estimator.estimate_depth(frame)
                if new_depth is not None:
                    cached_depth_map = new_depth

            # 3. Distance Extraction from Depth Map
            if cached_depth_map is not None and detections:
                detections = distance_extractor.extract_all_distances(cached_depth_map, detections)

                # 4. Calibration correction if active
                calib_scale = calibration.calibration_params.get("scale", 1.0)
                calib_offset = calibration.calibration_params.get("offset", 0.0)
                if calib_scale != 1.0 or calib_offset != 0.0:
                    for det in detections:
                        if "distance_m" in det:
                            det["distance_m"] = distance_extractor.apply_calibration(
                                det["distance_m"], calibration.calibration_params
                            )
                            det["distance_cm"] = det["distance_m"] * 100.0

            # 5. Multi-Object Tracking
            if tracker is not None and detections:
                detections = tracker.update(detections)

            # 6. Render Live Feed
            annotated = display.draw_detections(frame.copy(), detections)
            annotated = display.draw_fps(annotated)

            # Draw bottom summary panel
            info = {
                "num_objects": len(detections),
                "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
                "model_info": f"Tracking: {'ON' if tracker else 'OFF'}",
            }
            annotated = display.draw_info_panel(annotated, info)

            # Side-by-side depth visualization
            if display.show_depth_map and cached_depth_map is not None:
                depth_colormap = depth_estimator.get_depth_colormap(cached_depth_map)
                annotated = display.create_depth_visualization(annotated, depth_colormap)

            # Display window & handle keys
            key = display.show_frame(annotated)

            if key == ord("q"):
                print("\n[Exit] User quit requested.")
                break
            elif key == ord("c"):
                next_cam_idx = 1 if current_camera_idx == 0 else 0
                print(f"\n[Camera] Hot-switching active camera to index {next_cam_idx}...")
                if hasattr(camera, "switch_camera"):
                    success = camera.switch_camera(next_cam_idx)
                    if success:
                        current_camera_idx = next_cam_idx
                        print(f"[Camera] Active camera is now Index {current_camera_idx}")
                    else:
                        print(f"[Camera] Could not open camera {next_cam_idx}; reverted to {current_camera_idx}")
                else:
                    print("[Camera] Camera switching not supported for this source.")
            elif key == ord("d"):
                display.show_depth_map = not display.show_depth_map
                print(f"[Display] Depth map view: {'ON' if display.show_depth_map else 'OFF'}")
            elif key == ord("s"):
                filename = f"capture_{int(time.time())}.png"
                cv2.imwrite(filename, annotated)
                print(f"[Display] Saved screenshot: {filename}")

    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by keyboard.")
    finally:
        camera.release()
        display.destroy()
        print("[Exit] Cleaned up resources. Application terminated.")


if __name__ == "__main__":
    main()

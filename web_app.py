"""
Flask Web Application for Real-Time Multi-Object Detection & Distance Estimation.
Provides a modern browser UI to view:
1. Live camera feed with bounding boxes, object names, and metric distances.
2. Real-time Depth Anything V2 Metric colormap feed (Inferno).
3. Telemetry table of detected objects and distance estimates.
4. Interactive controls for resolution, confidence threshold, and streaming state.
"""

import argparse
import io
import math
import os
import sys
import threading
import time
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

import config
from camera_module import CameraManager
from detection_module import ObjectDetector
from depth_module import DepthEstimator
from distance_module import DistanceExtractor
from calibration_module import CalibrationManager
from tracker_module import SimpleTracker
from display_module import DisplayManager

app = Flask(__name__, template_folder="templates")

# Global state for streaming and pipeline
class VideoPipeline:
    def __init__(self, camera_index=0, width=1280, height=720, fps=30, device="cpu"):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.device = device
        self.confidence = config.YOLO_CONFIDENCE_THRESHOLD
        self.depth_cadence = config.DEPTH_PROCESS_EVERY_N
        self.is_streaming = True
        self.use_simulation = False

        self.lock = threading.Lock()
        self.latest_annotated_jpeg = None
        self.latest_depth_jpeg = None
        self.latest_detections = []
        self.current_fps = 0.0

        # Modules
        self.camera = None
        self.detector = None
        self.depth_estimator = None
        self.distance_extractor = None
        self.calibration = None
        self.tracker = None
        self.display = None

        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        """Initialize components and launch background capture thread."""
        print("[Pipeline] Initializing models and camera...")
        self.display = DisplayManager()
        self.distance_extractor = DistanceExtractor()
        self.calibration = CalibrationManager()
        self.tracker = SimpleTracker()

        # Load calibration if available
        if os.path.exists(config.CALIBRATION_FILE):
            self.calibration.load_calibration(config.CALIBRATION_FILE)

        # Initialize Camera
        self.camera = CameraManager(
            camera_index=self.camera_index,
            width=self.width,
            height=self.height,
            fps=self.fps,
        )
        if not self.camera.initialize():
            print(f"[Pipeline] Notice: Physical camera {self.camera_index} not found. Running in high-fidelity simulation mode.")
            self.use_simulation = True
        else:
            print("[Pipeline] Physical camera connected successfully.")

        # Initialize YOLO Detector
        try:
            self.detector = ObjectDetector(
                model_name=config.YOLO_MODEL,
                confidence=self.confidence,
                device=self.device,
            )
            loaded = self.detector.load_model()
            if not loaded:
                print("[Pipeline] Could not load YOLO model; running detection simulation fallback.")
                self.detector = None
        except Exception as e:
            print(f"[Pipeline] YOLO detector setup exception: {e}")
            self.detector = None

        # Initialize Depth Estimator
        try:
            self.depth_estimator = DepthEstimator(
                model_type=config.DEPTH_MODEL_TYPE,
                encoder=config.DEPTH_MODEL_TYPE,
                device=self.device,
            )
            loaded_depth = self.depth_estimator.load_model()
            if not loaded_depth:
                print("[Pipeline] Could not load Depth Anything V2 model; running depth simulation fallback.")
                self.depth_estimator = None
        except Exception as e:
            print(f"[Pipeline] Depth estimator setup exception: {e}")
            self.depth_estimator = None

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[Pipeline] Background inference worker started.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.camera:
            self.camera.release()

    def switch_camera(self, new_index):
        """Hot-switch active camera device on the fly."""
        new_index = int(new_index)
        print(f"[Pipeline] Request to switch camera to index {new_index}")
        with self.lock:
            self.camera_index = new_index
            if self.camera is None:
                self.camera = CameraManager(
                    camera_index=self.camera_index,
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                )
            success = self.camera.switch_camera(self.camera_index)
            if success:
                self.use_simulation = False
                print(f"[Pipeline] Switched to physical camera index {self.camera_index}")
            else:
                print(f"[Pipeline] Physical camera index {self.camera_index} not accessible. Using simulation.")
                self.use_simulation = True
            return success

    def _generate_synthetic_scene(self, t):
        """Generates a dynamic 3D simulated scene with objects at known physical distances."""
        h, w = self.height, self.width
        # Dark room gradient background
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (24, 28, 38)

        # Grid floor perspective
        for y in range(int(h * 0.55), h, 30):
            cv2.line(frame, (0, y), (w, y), (40, 48, 64), 1)

        depth_map = np.full((h, w), 8.0, dtype=np.float32)

        # Synthetic objects moving naturally
        sim_objects = [
            {
                "name": "person",
                "center_x": int(w * (0.35 + 0.15 * math.sin(t * 0.8))),
                "center_y": int(h * 0.60),
                "box_w": int(w * 0.14),
                "box_h": int(h * 0.45),
                "dist_m": 2.2 + 0.5 * math.sin(t * 0.8),
                "color": (210, 160, 60),
            },
            {
                "name": "chair",
                "center_x": int(w * 0.72),
                "center_y": int(h * 0.68),
                "box_w": int(w * 0.16),
                "box_h": int(h * 0.32),
                "dist_m": 1.5,
                "color": (160, 90, 200),
            },
            {
                "name": "bottle",
                "center_x": int(w * (0.50 + 0.05 * math.cos(t * 1.2))),
                "center_y": int(h * 0.75),
                "box_w": int(w * 0.06),
                "box_h": int(h * 0.15),
                "dist_m": 0.85 + 0.1 * math.cos(t * 1.2),
                "color": (80, 200, 120),
            },
        ]

        detections = []
        for obj in sim_objects:
            cx, cy = obj["center_x"], obj["center_y"]
            bw, bh = obj["box_w"], obj["box_h"]
            x1, y1 = max(0, cx - bw // 2), max(0, cy - bh // 2)
            x2, y2 = min(w, cx + bw // 2), min(h, cy + bh // 2)

            # Draw simulated object body
            cv2.rectangle(frame, (x1, y1), (x2, y2), obj["color"], -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # Fill synthetic depth map for object ROI
            depth_map[y1:y2, x1:x2] = obj["dist_m"]

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.88 + 0.08 * math.sin(t),
                "class_id": 0,
                "class_name": obj["name"],
            })

        return frame, depth_map, detections

    def _run_loop(self):
        frame_count = 0
        last_time = time.time()
        cached_depth_map = None

        while not self._stop_event.is_set():
            if not self.is_streaming:
                time.sleep(0.05)
                continue

            loop_start = time.time()
            frame_count += 1

            # 1. Acquire Frame
            if self.use_simulation:
                frame, sim_depth, sim_detections = self._generate_synthetic_scene(time.time())
                raw_frame = frame.copy()
            else:
                ret, raw_frame = self.camera.read_frame()
                if not ret or raw_frame is None:
                    time.sleep(0.03)
                    continue
                sim_depth = None
                sim_detections = None

            # 2. Object Detection
            if self.detector is not None:
                self.detector.confidence = self.confidence
                detections = self.detector.detect(raw_frame)
            elif sim_detections is not None:
                detections = sim_detections
            else:
                detections = []

            # 3. Depth Estimation
            if self.depth_estimator is not None:
                if frame_count % self.depth_cadence == 0 or cached_depth_map is None:
                    new_depth = self.depth_estimator.estimate_depth(raw_frame)
                    if new_depth is not None:
                        cached_depth_map = new_depth
            elif sim_depth is not None:
                cached_depth_map = sim_depth

            # 4. Extract Distances
            if cached_depth_map is not None and detections:
                detections = self.distance_extractor.extract_all_distances(cached_depth_map, detections)

                # Calibration correction
                if self.calibration:
                    scale = self.calibration.calibration_params.get("scale", 1.0)
                    offset = self.calibration.calibration_params.get("offset", 0.0)
                    if scale != 1.0 or offset != 0.0:
                        for d in detections:
                            if "distance_m" in d:
                                d["distance_m"] = self.distance_extractor.apply_calibration(
                                    d["distance_m"], self.calibration.calibration_params
                                )
                                d["distance_cm"] = d["distance_m"] * 100.0

            # 5. Object Tracking
            if self.tracker is not None and detections:
                detections = self.tracker.update(detections)

            # 6. Render Annotated Visuals
            annotated = self.display.draw_detections(raw_frame.copy(), detections)

            # Compute and draw FPS
            elapsed = time.time() - last_time
            last_time = time.time()
            if elapsed > 0:
                self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / elapsed)

            cv2.putText(
                annotated,
                f"FPS: {self.current_fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # Depth Colormap Frame
            if cached_depth_map is not None:
                if self.depth_estimator is not None:
                    depth_vis = self.depth_estimator.get_depth_colormap(cached_depth_map)
                else:
                    # Synthetic colormap normalization
                    norm = np.clip((cached_depth_map - 0.5) / 5.0, 0, 1)
                    grey = (norm * 255).astype(np.uint8)
                    depth_vis = cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO)
            else:
                depth_vis = np.zeros_like(annotated)

            # Encode JPEGs into memory buffers
            _, ann_jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _, dep_jpeg = cv2.imencode(".jpg", depth_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])

            with self.lock:
                self.latest_annotated_jpeg = ann_jpeg.tobytes()
                self.latest_depth_jpeg = dep_jpeg.tobytes()
                self.latest_detections = detections

            # Maintain sensible loop timing (cap at ~30 FPS)
            loop_duration = time.time() - loop_start
            sleep_needed = (1.0 / 30.0) - loop_duration
            if sleep_needed > 0:
                time.sleep(sleep_needed)


# Instantiate Pipeline
pipeline = None


@app.route("/")
def index():
    """Serves the main studio dashboard."""
    return render_template("index.html")


def gen_frames(stream_type="annotated"):
    """Video streaming generator function."""
    while True:
        if pipeline is None:
            time.sleep(0.1)
            continue

        with pipeline.lock:
            if stream_type == "annotated":
                frame_data = pipeline.latest_annotated_jpeg
            else:
                frame_data = pipeline.latest_depth_jpeg

        if frame_data is None:
            time.sleep(0.03)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
        )
        time.sleep(0.033)


@app.route("/video_feed")
def video_feed():
    """Live Object Detection MJPEG stream."""
    return Response(
        gen_frames("annotated"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/depth_feed")
def depth_feed():
    """Live Depth Anything V2 colormap MJPEG stream."""
    return Response(
        gen_frames("depth"),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/status")
def api_status():
    """Provides real-time telemetry: FPS, detections, camera state."""
    if pipeline is None:
        return jsonify({"status": "offline"})

    with pipeline.lock:
        dets = list(pipeline.latest_detections)
        fps = pipeline.current_fps
        is_sim = pipeline.use_simulation

    # Format telemetry for UI table
    clean_dets = []
    for d in dets:
        clean_dets.append({
            "track_id": d.get("track_id"),
            "class_name": d.get("class_name", "object"),
            "confidence": round(float(d.get("confidence", 0.0)), 2),
            "distance_m": round(float(d.get("distance_m", 0.0)), 2),
            "is_reliable": bool(d.get("is_reliable", True)),
        })

    cam_type = "SIMULATED" if is_sim else f"CAMERA {pipeline.camera_index} (UVC)"

    return jsonify({
        "status": "online",
        "camera_type": cam_type,
        "active_camera": pipeline.camera_index,
        "is_simulated": is_sim,
        "fps": round(fps, 1),
        "detections": clean_dets,
        "width": pipeline.width,
        "height": pipeline.height,
        "is_streaming": pipeline.is_streaming,
    })


@app.route("/api/switch_camera", methods=["POST"])
def api_switch_camera():
    """Hot-switch camera device to another index."""
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 500

    data = request.get_json(force=True)
    target_idx = int(data.get("camera_index", 0))
    success = pipeline.switch_camera(target_idx)

    return jsonify({
        "status": "ok",
        "success": success,
        "active_camera": pipeline.camera_index,
        "camera_type": "SIMULATED" if pipeline.use_simulation else f"CAMERA {pipeline.camera_index} (UVC)",
    })


@app.route("/api/scan_cameras")
def api_scan_cameras():
    """Scan and list connected camera devices."""
    detected = []
    # Test indices 0 to 4
    for idx in range(5):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            label = f"Camera {idx} (Default: {w}x{h})"
            if idx == 0:
                label += " - Primary"
            elif idx == 1:
                label += " - Secondary"
            detected.append({"index": idx, "label": label})
            cap.release()

    if not detected:
        detected = [
            {"index": 0, "label": "Camera 0 (Default)"},
            {"index": 1, "label": "Camera 1 (Secondary)"},
        ]

    return jsonify({
        "cameras": detected,
        "active_camera": pipeline.camera_index if pipeline else 0,
    })


@app.route("/api/controls", methods=["POST"])
def api_controls():
    """Update runtime parameters (confidence, resolution, streaming, cadence)."""
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 500

    data = request.get_json(force=True)

    if "is_streaming" in data:
        pipeline.is_streaming = bool(data["is_streaming"])

    if "confidence" in data:
        pipeline.confidence = float(data["confidence"])

    if "depth_cadence" in data:
        pipeline.depth_cadence = int(data["depth_cadence"])

    if "width" in data and "height" in data:
        pipeline.width = int(data["width"])
        pipeline.height = int(data["height"])
        if pipeline.camera and not pipeline.use_simulation:
            pipeline.camera.set_resolution(pipeline.width, pipeline.height)

    return jsonify({
        "status": "updated",
        "width": pipeline.width,
        "height": pipeline.height,
        "confidence": pipeline.confidence,
        "depth_cadence": pipeline.depth_cadence,
        "is_streaming": pipeline.is_streaming,
    })


@app.route("/api/snapshot")
def api_snapshot():
    """Download the latest annotated snapshot as a JPEG image."""
    if pipeline is None or pipeline.latest_annotated_jpeg is None:
        return "No frame available", 404

    with pipeline.lock:
        img_bytes = pipeline.latest_annotated_jpeg

    return Response(
        img_bytes,
        mimetype="image/jpeg",
        headers={"Content-Disposition": "attachment; filename=arducam_depth_snapshot.jpg"},
    )


def main():
    parser = argparse.ArgumentParser(description="Arducam Real-Time Vision Studio Web Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host binding (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5000, help="HTTP Port (default: 5000)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index")
    parser.add_argument("--width", type=int, default=config.CAMERA_WIDTH, help="Camera width")
    parser.add_argument("--height", type=int, default=config.CAMERA_HEIGHT, help="Camera height")
    parser.add_argument("--fps", type=int, default=config.CAMERA_FPS, help="Camera FPS")
    parser.add_argument("--device", type=str, default=config.YOLO_DEVICE, help="cuda:0 or cpu")
    args = parser.parse_args()

    global pipeline
    pipeline = VideoPipeline(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        device=args.device,
    )
    pipeline.start()

    print("\n" + "=" * 70)
    print(" 🔭 ARDUCAM REAL-TIME VISION STUDIO WEB APPLICATION")
    print("=" * 70)
    print(f" -> Open your browser at: http://localhost:{args.port}")
    print(f" -> Local network access:  http://127.0.0.1:{args.port}")
    print("=" * 70 + "\n")

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()


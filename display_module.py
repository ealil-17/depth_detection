"""
Live video display and rendering module.
Handles drawing bounding boxes, labels, distances, FPS, info panels,
and side-by-side depth map visualizations.
"""

import cv2
import numpy as np
import time
from config import (DISPLAY_WINDOW_NAME, BBOX_COLOR, TEXT_COLOR, TEXT_BG_COLOR,
                    FONT_SCALE, FONT_THICKNESS, SHOW_DEPTH_MAP, SHOW_FPS)


class DisplayManager:
    """Manages the live display window and rendering of detection results."""

    # Colour constants for reliability levels
    COLOR_RELIABLE = BBOX_COLOR           # green by default
    COLOR_UNRELIABLE = (0, 165, 255)      # orange (BGR)
    COLOR_NO_DISTANCE = (255, 255, 0)     # cyan (BGR)

    def __init__(self):
        self.fps_counter = FPSCounter()
        self.window_name = DISPLAY_WINDOW_NAME
        self.show_depth_map = SHOW_DEPTH_MAP

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def draw_detections(self, frame: np.ndarray, detections: list) -> np.ndarray:
        """
        Draw bounding boxes, labels and distances on the frame.

        Each detection dict has:
        - 'bbox': [x1, y1, x2, y2]
        - 'class_name': str
        - 'confidence': float
        - 'distance_m': float (may be missing)
        - 'is_reliable': bool (may be missing)
        - 'track_id': int (may be missing)

        For each detection:
        1. Draw rectangle bbox in BBOX_COLOR (or orange if unreliable)
        2. Build label  e.g. "[3] Person 0.85 | 2.40 m" or "Person 0.85 | 2.40 m (unreliable)"
        3. Draw filled rectangle behind text for readability
        4. Draw text in TEXT_COLOR

        Returns the annotated frame.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det.get('class_name', 'object')
            confidence = det.get('confidence', 0.0)
            distance_m = det.get('distance_m')
            is_reliable = det.get('is_reliable', True)
            track_id = det.get('track_id')

            # --- choose bbox colour ---
            if distance_m is None:
                box_color = self.COLOR_NO_DISTANCE
            elif not is_reliable:
                box_color = self.COLOR_UNRELIABLE
            else:
                box_color = self.COLOR_RELIABLE

            # --- draw bounding box ---
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                          box_color, 2)

            # --- build label string ---
            parts = []
            if track_id is not None:
                parts.append(f"[{track_id}]")
            parts.append(f"{class_name} {confidence:.2f}")

            if distance_m is not None:
                dist_str = f"{distance_m:.2f} m"
                if not is_reliable:
                    dist_str += " (unreliable)"
                parts.append(f"| {dist_str}")

            label = " ".join(parts)

            # --- draw label background + text ---
            self._draw_label(frame, label, int(x1), int(y1), font,
                             FONT_SCALE, FONT_THICKNESS, TEXT_COLOR,
                             TEXT_BG_COLOR)

        return frame

    def draw_fps(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS counter in top-left corner if SHOW_FPS is enabled."""
        if not SHOW_FPS:
            return frame

        fps_text = f"FPS: {self.fps_counter.get_fps():.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX

        self._draw_label(frame, fps_text, 10, 30, font,
                         FONT_SCALE, FONT_THICKNESS,
                         (0, 255, 0), TEXT_BG_COLOR)
        return frame

    def draw_info_panel(self, frame: np.ndarray, info: dict) -> np.ndarray:
        """
        Draw an info panel at the bottom of the frame.
        info dict may contain: 'resolution', 'num_objects', 'model_info'
        """
        if not info:
            return frame

        h, w = frame.shape[:2]
        panel_height = 30
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Semi-transparent dark bar at the bottom
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h),
                      TEXT_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Build info string
        parts = []
        if 'resolution' in info:
            parts.append(f"Res: {info['resolution']}")
        if 'num_objects' in info:
            parts.append(f"Objects: {info['num_objects']}")
        if 'model_info' in info:
            parts.append(info['model_info'])

        info_text = "  |  ".join(parts)
        cv2.putText(frame, info_text, (10, h - 10), font,
                    FONT_SCALE * 0.8, TEXT_COLOR, 1, cv2.LINE_AA)
        return frame

    def create_depth_visualization(self, frame: np.ndarray,
                                   depth_colormap: np.ndarray) -> np.ndarray:
        """
        Create side-by-side view of camera feed and depth map.
        Resize both to same height, concatenate horizontally.
        """
        h_frame, w_frame = frame.shape[:2]
        h_depth, w_depth = depth_colormap.shape[:2]

        # Use the camera frame's height as the reference
        target_h = h_frame

        if h_depth != target_h:
            scale = target_h / h_depth
            new_w = int(w_depth * scale)
            depth_colormap = cv2.resize(depth_colormap, (new_w, target_h))

        if h_frame != target_h:
            scale = target_h / h_frame
            new_w = int(w_frame * scale)
            frame = cv2.resize(frame, (new_w, target_h))

        # Ensure both are 3-channel BGR
        if len(depth_colormap.shape) == 2:
            depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_GRAY2BGR)

        return np.hstack((frame, depth_colormap))

    # ------------------------------------------------------------------
    # Window management
    # ------------------------------------------------------------------

    def show_frame(self, frame: np.ndarray) -> int:
        """
        Display frame in window. Returns the key pressed (for quit detection).
        Updates the FPS counter on every call.
        """
        self.fps_counter.update()
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key

    def destroy(self):
        """Destroy all OpenCV windows."""
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_label(frame, text, x, y, font, scale, thickness,
                    fg_color, bg_color):
        """Draw text with a filled background rectangle for readability."""
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        # Position: text is drawn with origin at bottom-left of the first character
        text_y = max(y, th + baseline + 4)  # keep inside the frame
        # Background rectangle
        cv2.rectangle(frame,
                      (x, text_y - th - baseline - 2),
                      (x + tw + 4, text_y + 2),
                      bg_color, -1)
        # Text
        cv2.putText(frame, text, (x + 2, text_y - baseline),
                    font, scale, fg_color, thickness, cv2.LINE_AA)


class FPSCounter:
    """Tracks and calculates frames per second using a sliding window."""

    def __init__(self, avg_frames: int = 30):
        self.avg_frames = avg_frames
        self.timestamps: list[float] = []
        self.fps: float = 0.0

    def update(self):
        """Record current timestamp and recalculate FPS."""
        now = time.perf_counter()
        self.timestamps.append(now)

        # Keep only the last avg_frames timestamps
        if len(self.timestamps) > self.avg_frames:
            self.timestamps = self.timestamps[-self.avg_frames:]

        # Need at least 2 timestamps to compute a rate
        if len(self.timestamps) >= 2:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed > 0:
                self.fps = (len(self.timestamps) - 1) / elapsed
            else:
                self.fps = 0.0
        else:
            self.fps = 0.0

    def get_fps(self) -> float:
        """Return current FPS."""
        return self.fps

"""Camera initialization and control module for the Real-Time Depth Detection System."""

import sys
import threading
import time
import numpy as np
import cv2

import config


class CameraManager:
    """Manages camera capture: open, read, configure, and release.
    Features hardware MJPEG streaming and asynchronous threaded frame acquisition
    for zero-latency locked 30 FPS.
    """

    def __init__(self, camera_index=None, width=None, height=None, fps=None, use_threading=True):
        """Initialize CameraManager with optional overrides; falls back to config defaults."""
        self.camera_index = camera_index if camera_index is not None else config.CAMERA_INDEX
        self.width = width if width is not None else config.CAMERA_WIDTH
        self.height = height if height is not None else config.CAMERA_HEIGHT
        self.fps = fps if fps is not None else config.CAMERA_FPS
        self.cap = None
        self._backend_name = "unknown"
        self._use_threading = use_threading

        # Asynchronous frame grabber state
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._has_frame = False
        self._grab_thread = None
        self._stop_grab = threading.Event()

    # ------------------------------------------------------------------
    # Core lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Open the camera and apply resolution/fps settings.

        On Windows, if the default backend fails the method retries with
        the DirectShow (CAP_DSHOW) backend.

        Returns:
            True if the camera was opened successfully, False otherwise.
        """
        # Determine which backend(s) to try
        backends_to_try = []
        if config.CAMERA_BACKEND is not None:
            backends_to_try.append(("config-specified", config.CAMERA_BACKEND))
        else:
            backends_to_try.append(("auto", cv2.CAP_ANY))
            # On Windows, fall back to DirectShow if auto fails
            if sys.platform == "win32":
                backends_to_try.append(("DirectShow", cv2.CAP_DSHOW))

        for backend_label, backend_flag in backends_to_try:
            try:
                self.cap = cv2.VideoCapture(self.camera_index, backend_flag)
                if self.cap.isOpened():
                    self._backend_name = backend_label
                    self._apply_settings()

                    # Start asynchronous background frame grabber for 30 FPS zero-latency capture
                    if self._use_threading:
                        self._stop_grab.clear()
                        self._has_frame = False
                        self._latest_frame = None
                        self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
                        self._grab_thread.start()
                        # Wait briefly for first frame
                        for _ in range(20):
                            if self._has_frame:
                                break
                            time.sleep(0.02)

                    print(
                        f"[CameraManager] Camera {self.camera_index} opened "
                        f"with {backend_label} backend (Hardware MJPEG 30FPS active)."
                    )
                    return True
                else:
                    self.cap.release()
                    self.cap = None
                    print(
                        f"[CameraManager] Backend '{backend_label}' could not "
                        f"open camera {self.camera_index}."
                    )
            except Exception as exc:
                print(
                    f"[CameraManager] Error opening camera {self.camera_index} "
                    f"with {backend_label}: {exc}"
                )
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None

        print("[CameraManager] Failed to open camera with any backend.")
        return False

    def _grab_loop(self):
        """Continuously drain camera USB buffers at maximum hardware rate (30 FPS)."""
        while not self._stop_grab.is_set():
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.02)
                continue
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._has_frame = True
            else:
                time.sleep(0.005)

    def release(self) -> None:
        """Release the camera resource and stop background grabber."""
        self._stop_grab.set()
        if self._grab_thread is not None:
            self._grab_thread.join(timeout=1.0)
            self._grab_thread = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as exc:
                print(f"[CameraManager] Error releasing camera: {exc}")
            finally:
                self.cap = None

    def switch_camera(self, new_index: int) -> bool:
        """Hot-switch to another camera index on the fly.
        
        Args:
            new_index: The camera index to connect to (0, 1, 2, ...).
            
        Returns:
            True if the new camera opened successfully, False otherwise.
        """
        old_index = self.camera_index
        print(f"[CameraManager] Switching camera from index {old_index} -> {new_index}...")
        self.release()
        self.camera_index = int(new_index)
        success = self.initialize()
        if not success:
            print(f"[CameraManager] Failed to switch to camera {new_index}. Attempting to restore {old_index}...")
            self.camera_index = old_index
            self.initialize()
            return False
        return True

    # ------------------------------------------------------------------
    # Frame I/O
    # ------------------------------------------------------------------

    def read_frame(self) -> tuple:
        """Read a single frame from the camera.
        Returns the newest acquired frame instantaneously from the background grabber.

        Returns:
            (success: bool, frame: np.ndarray | None)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        if self._use_threading:
            with self._frame_lock:
                if self._has_frame and self._latest_frame is not None:
                    return True, self._latest_frame.copy()
            # If thread hasn't picked up first frame yet, fallback to synchronous read once
            ret, frame = self.cap.read()
            return ret, frame

        try:
            ret, frame = self.cap.read()
            if not ret:
                print("[CameraManager] Failed to read frame.")
                return False, None
            return True, frame
        except Exception as exc:
            print(f"[CameraManager] Error reading frame: {exc}")
            return False, None

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_resolution(self, width: int, height: int) -> None:
        """Change the capture resolution on the fly."""
        self.width = width
        self.height = height
        if self.cap is not None and self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                print(f"[CameraManager] Resolution set to {width}x{height}.")
            except Exception as exc:
                print(f"[CameraManager] Error setting resolution: {exc}")

    def get_camera_info(self) -> dict:
        """Return a dict describing the current camera state."""
        if self.cap is None or not self.cap.isOpened():
            return {
                "width": None,
                "height": None,
                "fps": None,
                "backend": None,
            }
        try:
            return {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "backend": self._backend_name,
            }
        except Exception as exc:
            print(f"[CameraManager] Error getting camera info: {exc}")
            return {
                "width": None,
                "height": None,
                "fps": None,
                "backend": self._backend_name,
            }

    def is_opened(self) -> bool:
        """Return True if the underlying VideoCapture is open."""
        return self.cap is not None and self.cap.isOpened()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        """Enter the context manager; initializes the camera."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager; releases the camera."""
        self.release()
        return False  # do not suppress exceptions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_settings(self) -> None:
        """Apply hardware MJPEG encoding, zero-lag buffer, resolution and FPS."""
        try:
            # 1. Enable Hardware MJPEG streaming (Critical for Arducam 12MP to achieve 30/60/120 FPS on Windows)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            # 2. Minimize driver buffer to 1 to eliminate frame lag
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # 3. Apply resolution & framerate
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        except Exception as exc:
            print(f"[CameraManager] Error applying camera settings: {exc}")

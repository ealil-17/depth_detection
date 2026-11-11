"""Camera calibration and distance validation module."""
import json
import numpy as np
import cv2
from config import (CALIBRATION_FILE, CHECKERBOARD_SIZE, CHECKERBOARD_SQUARE_SIZE)


class CalibrationManager:
    """Camera calibration and metric depth validation."""

    def __init__(self, calibration_file=None, checkerboard_size=None, square_size=None):
        """Use config defaults. Initialize storage for calibration data."""
        self.calibration_file = calibration_file if calibration_file is not None else CALIBRATION_FILE
        self.checkerboard_size = checkerboard_size if checkerboard_size is not None else CHECKERBOARD_SIZE
        self.square_size = square_size if square_size is not None else CHECKERBOARD_SQUARE_SIZE
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_params = {'scale': 1.0, 'offset': 0.0}  # depth correction
        self.validation_records = []  # list of {actual_m, measured_m, error_m, object}

    def calibrate_camera_intrinsics(self, images: list) -> bool:
        """
        Calibrate camera using checkerboard images.
        Uses cv2.findChessboardCorners and cv2.calibrateCamera.
        Stores camera_matrix and dist_coeffs. Returns True on success.
        """
        obj_points = []  # 3D points in real-world space
        img_points = []  # 2D points in image plane

        # Prepare object points: (0,0,0), (1,0,0), ... scaled by square_size
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0],
                                0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size

        img_shape = None

        for img in images:
            if isinstance(img, str):
                frame = cv2.imread(img)
                if frame is None:
                    print(f"[CalibrationManager] Could not read image: {img}")
                    continue
            else:
                frame = img

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            img_shape = gray.shape[::-1]  # (width, height)

            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            if ret:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners_refined)

        if len(obj_points) < 3:
            print(f"[CalibrationManager] Not enough valid checkerboard images "
                  f"({len(obj_points)} found, need at least 3)")
            return False

        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points, img_points, img_shape, None, None
            )
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            print(f"[CalibrationManager] Camera calibration successful. RMS error: {ret:.4f}")
            return True
        except Exception as e:
            print(f"[CalibrationManager] Calibration failed: {e}")
            return False

    def calibrate_camera_from_video(self, camera_manager, num_frames=20) -> bool:
        """
        Interactive calibration from live video.
        Shows live feed, captures frames when checkerboard detected.
        Needs num_frames good captures. Returns True on success.
        """
        captured_images = []
        print(f"[CalibrationManager] Starting interactive calibration. "
              f"Need {num_frames} checkerboard captures.")
        print("[CalibrationManager] Press 'c' to capture, 'q' to quit early.")

        while len(captured_images) < num_frames:
            ret, frame = camera_manager.read_frame()
            if not ret or frame is None:
                print("[CalibrationManager] Failed to read frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, self.checkerboard_size, corners, found)
                cv2.putText(display, f"Checkerboard FOUND - Press 'c' to capture "
                            f"({len(captured_images)}/{num_frames})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, f"Searching for checkerboard... "
                            f"({len(captured_images)}/{num_frames})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c') and found:
                captured_images.append(frame.copy())
                print(f"[CalibrationManager] Captured frame {len(captured_images)}/{num_frames}")
            elif key == ord('q'):
                print("[CalibrationManager] Calibration cancelled by user.")
                break

        cv2.destroyWindow("Calibration")

        if len(captured_images) < 3:
            print(f"[CalibrationManager] Not enough captures ({len(captured_images)}). "
                  f"Need at least 3.")
            return False

        return self.calibrate_camera_intrinsics(captured_images)

    def save_calibration(self, filepath=None) -> bool:
        """Save camera_matrix, dist_coeffs, and calibration_params to JSON."""
        filepath = filepath or self.calibration_file
        data = {
            'calibration_params': self.calibration_params,
            'validation_records': self.validation_records,
        }
        if self.camera_matrix is not None:
            data['camera_matrix'] = self.camera_matrix.tolist()
        if self.dist_coeffs is not None:
            data['dist_coeffs'] = self.dist_coeffs.tolist()

        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[CalibrationManager] Calibration saved to {filepath}")
            return True
        except Exception as e:
            print(f"[CalibrationManager] Failed to save calibration: {e}")
            return False

    def load_calibration(self, filepath=None) -> bool:
        """Load calibration from JSON file. Returns True on success."""
        filepath = filepath or self.calibration_file
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            if 'camera_matrix' in data:
                self.camera_matrix = np.array(data['camera_matrix'])
            if 'dist_coeffs' in data:
                self.dist_coeffs = np.array(data['dist_coeffs'])
            if 'calibration_params' in data:
                self.calibration_params = data['calibration_params']
            if 'validation_records' in data:
                self.validation_records = data['validation_records']

            print(f"[CalibrationManager] Calibration loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"[CalibrationManager] Calibration file not found: {filepath}")
            return False
        except Exception as e:
            print(f"[CalibrationManager] Failed to load calibration: {e}")
            return False

    def add_validation_point(self, actual_distance_m: float, measured_distance_m: float,
                             object_name: str = ""):
        """Record a validation measurement for error analysis."""
        error_m = abs(actual_distance_m - measured_distance_m)
        record = {
            'actual_m': actual_distance_m,
            'measured_m': measured_distance_m,
            'error_m': error_m,
            'object': object_name,
        }
        self.validation_records.append(record)

    def compute_depth_calibration(self) -> dict:
        """
        Compute scale and offset from validation points using linear regression.
        actual = measured * scale + offset
        Returns {'scale': float, 'offset': float, 'r_squared': float,
                 'mean_error_m': float, 'max_error_m': float}
        Updates self.calibration_params.
        """
        if len(self.validation_records) < 2:
            print("[CalibrationManager] Need at least 2 validation points for calibration.")
            return {
                'scale': 1.0, 'offset': 0.0,
                'r_squared': 0.0, 'mean_error_m': 0.0, 'max_error_m': 0.0,
            }

        measured = np.array([r['measured_m'] for r in self.validation_records])
        actual = np.array([r['actual_m'] for r in self.validation_records])

        # Linear regression: actual = measured * scale + offset
        # Using numpy polyfit (degree 1)
        coeffs = np.polyfit(measured, actual, 1)
        scale = float(coeffs[0])
        offset = float(coeffs[1])

        # Compute R-squared
        predicted = measured * scale + offset
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Compute errors using calibrated values
        errors = np.abs(actual - predicted)
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))

        self.calibration_params = {'scale': scale, 'offset': offset}

        result = {
            'scale': scale,
            'offset': offset,
            'r_squared': r_squared,
            'mean_error_m': mean_error,
            'max_error_m': max_error,
        }
        print(f"[CalibrationManager] Depth calibration computed: "
              f"scale={scale:.4f}, offset={offset:.4f}, R²={r_squared:.4f}")
        return result

    def get_validation_report(self) -> str:
        """Return a human-readable report of validation measurements and errors."""
        if not self.validation_records:
            return "No validation records available."

        lines = ["=== Depth Validation Report ===", ""]
        lines.append(f"{'#':<4} {'Object':<20} {'Actual (m)':<12} {'Measured (m)':<14} {'Error (m)':<10}")
        lines.append("-" * 64)

        for i, r in enumerate(self.validation_records, 1):
            lines.append(
                f"{i:<4} {r['object']:<20} {r['actual_m']:<12.3f} "
                f"{r['measured_m']:<14.3f} {r['error_m']:<10.3f}"
            )

        errors = [r['error_m'] for r in self.validation_records]
        lines.append("-" * 64)
        lines.append(f"Total points: {len(self.validation_records)}")
        lines.append(f"Mean error:   {np.mean(errors):.4f} m")
        lines.append(f"Max error:    {np.max(errors):.4f} m")
        lines.append(f"Std error:    {np.std(errors):.4f} m")

        if self.calibration_params['scale'] != 1.0 or self.calibration_params['offset'] != 0.0:
            lines.append(f"\nCalibration: scale={self.calibration_params['scale']:.4f}, "
                         f"offset={self.calibration_params['offset']:.4f}")

        return "\n".join(lines)

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply lens distortion correction if calibration available."""
        if self.camera_matrix is None or self.dist_coeffs is None:
            return frame

        try:
            h, w = frame.shape[:2]
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
            )
            undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs,
                                        None, new_camera_matrix)
            # Crop to valid region
            x, y, rw, rh = roi
            if rw > 0 and rh > 0:
                undistorted = undistorted[y:y + rh, x:x + rw]
            return undistorted
        except Exception as e:
            print(f"[CalibrationManager] Undistort failed: {e}")
            return frame

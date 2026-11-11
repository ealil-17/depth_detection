"""
Unit and integration sanity tests for all depth detection modules.
Verifies module interfaces, bounding box median crop extraction,
linear calibration math, tracker state machines, and rendering functions.
"""

import unittest
import numpy as np
import cv2

from distance_module import DistanceExtractor
from calibration_module import CalibrationManager
from tracker_module import SimpleTracker
from display_module import DisplayManager


class TestDepthDetectionPipeline(unittest.TestCase):

    def setUp(self):
        self.distance_extractor = DistanceExtractor(center_crop_ratio=0.4)
        self.calibration_manager = CalibrationManager()
        self.tracker = SimpleTracker(max_age=5, min_hits=1, iou_threshold=0.3)
        self.display_manager = DisplayManager()

    def test_distance_extraction_clean(self):
        """Test median depth extraction for an object with uniform depth."""
        # Create a synthetic 720x1280 depth map with a background at 5.0m
        depth_map = np.full((720, 1280), 5.0, dtype=np.float32)
        # Place an object at [200, 300, 400, 500] with depth 2.0m
        depth_map[300:500, 200:400] = 2.0

        bbox = [200, 300, 400, 500]
        result = self.distance_extractor.extract_distance(depth_map, bbox)

        self.assertAlmostEqual(result["distance_m"], 2.0, places=2)
        self.assertAlmostEqual(result["distance_cm"], 200.0, places=1)
        self.assertTrue(result["is_reliable"])

    def test_distance_extraction_noisy_boundary(self):
        """Test that central 40% crop rejects background bleed at object edges."""
        depth_map = np.full((720, 1280), 6.0, dtype=np.float32)
        # Bbox is 100x100 pixels. Center 40% is 40x40 pixels.
        # Fill only the center 40% with the target object depth of 1.5m
        # The surrounding 60% of the bbox is background at 6.0m
        bbox = [100, 100, 200, 200]
        depth_map[120:180, 120:180] = 1.5

        result = self.distance_extractor.extract_distance(depth_map, bbox)
        # Should extract the true object distance 1.5m, ignoring the outer 6.0m edge
        self.assertAlmostEqual(result["distance_m"], 1.5, places=2)

    def test_calibration_linear_correction(self):
        """Test linear depth calibration application: d_cal = scale * d + offset."""
        params = {"scale": 1.10, "offset": -0.05}
        raw_dist = 2.0
        calibrated = self.distance_extractor.apply_calibration(raw_dist, params)
        expected = 2.0 * 1.10 - 0.05
        self.assertAlmostEqual(calibrated, expected, places=3)

    def test_calibration_linear_regression(self):
        """Test compute_depth_calibration solver."""
        # Simulated actual vs measured: actual = 1.05 * measured - 0.1
        measured = [1.0, 2.0, 3.0, 4.0]
        actual = [1.05 * m - 0.1 for m in measured]

        for a, m in zip(actual, measured):
            self.calibration_manager.add_validation_point(a, m, "test_target")

        res = self.calibration_manager.compute_depth_calibration()
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["scale"], 1.05, places=2)
        self.assertAlmostEqual(res["offset"], -0.1, places=2)
        self.assertGreater(res["r_squared"], 0.99)

    def test_tracker_lifecycle(self):
        """Test multi-object tracking ID assignment across frames."""
        detections_frame1 = [
            {"bbox": [100, 100, 200, 200], "confidence": 0.9, "class_name": "person"}
        ]
        tracked1 = self.tracker.update(detections_frame1)
        self.assertEqual(len(tracked1), 1)
        track_id = tracked1[0]["track_id"]

        # Frame 2: Slight movement
        detections_frame2 = [
            {"bbox": [105, 102, 205, 202], "confidence": 0.88, "class_name": "person"}
        ]
        tracked2 = self.tracker.update(detections_frame2)
        self.assertEqual(len(tracked2), 1)
        # Verify track ID is preserved across frames
        self.assertEqual(tracked2[0]["track_id"], track_id)

    def test_display_render(self):
        """Verify rendering does not crash and produces valid 3-channel frame."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dets = [{
            "bbox": [100, 100, 250, 300],
            "class_name": "person",
            "confidence": 0.87,
            "distance_m": 2.34,
            "is_reliable": True,
            "track_id": 1
        }]
        annotated = self.display_manager.draw_detections(frame, dets)
        self.assertEqual(annotated.shape, (720, 1280, 3))
        self.assertEqual(annotated.dtype, np.uint8)


if __name__ == "__main__":
    print("Running Depth Detection Pipeline automated test suite...\n")
    unittest.main()


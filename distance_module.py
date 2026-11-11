"""Distance extraction from depth maps for detected objects."""
import numpy as np
from config import (BBOX_CENTER_CROP_RATIO, MIN_DEPTH_CONFIDENCE,
                    MAX_DEPTH_CONFIDENCE, UNRELIABLE_DEPTH_THRESHOLD)


class DistanceExtractor:
    """Extracts distance measurements from depth maps for detected objects."""

    def __init__(self, center_crop_ratio=None, min_depth=None, max_depth=None,
                 unreliable_threshold=None):
        """Use config defaults if None."""
        self.center_crop_ratio = center_crop_ratio if center_crop_ratio is not None else BBOX_CENTER_CROP_RATIO
        self.min_depth = min_depth if min_depth is not None else MIN_DEPTH_CONFIDENCE
        self.max_depth = max_depth if max_depth is not None else MAX_DEPTH_CONFIDENCE
        self.unreliable_threshold = unreliable_threshold if unreliable_threshold is not None else UNRELIABLE_DEPTH_THRESHOLD

    def extract_distance(self, depth_map: np.ndarray, bbox: list) -> dict:
        """
        Extract distance for a single object from its bounding box region.

        Args:
            depth_map: 2D numpy array with depth values in metres
            bbox: [x1, y1, x2, y2] pixel coordinates

        Returns dict with:
            - 'distance_m': float, median depth in metres
            - 'distance_cm': float, in centimetres
            - 'is_reliable': bool, True if within valid range
            - 'reliability_note': str, explanation if unreliable
            - 'min_depth': float, min depth in ROI
            - 'max_depth': float, max depth in ROI
            - 'std_depth': float, std dev of depth in ROI
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Compute centre of bounding box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Compute crop dimensions using center_crop_ratio
        crop_w = int((x2 - x1) * self.center_crop_ratio)
        crop_h = int((y2 - y1) * self.center_crop_ratio)

        # Ensure minimum crop size of 1 pixel
        crop_w = max(crop_w, 1)
        crop_h = max(crop_h, 1)

        # Extract ROI, clamp to image bounds
        h, w = depth_map.shape[:2]
        roi_y1 = max(0, cy - crop_h // 2)
        roi_y2 = min(h, cy + crop_h // 2)
        roi_x1 = max(0, cx - crop_w // 2)
        roi_x2 = min(w, cx + crop_w // 2)

        roi = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]

        # Filter out invalid values (zeros, NaN, inf)
        valid_mask = np.isfinite(roi) & (roi > 0)
        valid_values = roi[valid_mask]

        # Default result for when no valid depth data exists
        if valid_values.size == 0:
            return {
                'distance_m': 0.0,
                'distance_cm': 0.0,
                'is_reliable': False,
                'reliability_note': 'No valid depth data in ROI',
                'min_depth': 0.0,
                'max_depth': 0.0,
                'std_depth': 0.0,
            }

        # Robust Outlier Rejection via Interquartile Range (IQR)
        # Prevents background walls or hollow gaps from corrupting foreground object distance
        q25, q75 = np.percentile(valid_values, 25), np.percentile(valid_values, 75)
        iqr = q75 - q25
        if iqr > 0:
            inliers = valid_values[(valid_values >= q25 - 1.5 * iqr) & (valid_values <= q75 + 1.5 * iqr)]
            if inliers.size >= 5:
                valid_values = inliers

        # The 35th percentile of inliers accurately captures the front-facing physical surface
        # of the object facing the camera, avoiding background bleed
        measured_depth = float(np.percentile(valid_values, 35))
        min_d = float(np.min(valid_values))
        max_d = float(np.max(valid_values))
        std_d = float(np.std(valid_values))

        # Determine reliability
        is_reliable = True
        reliability_note = 'OK'

        if measured_depth < self.min_depth:
            is_reliable = False
            reliability_note = f'Depth {measured_depth:.2f}m below minimum threshold {self.min_depth:.2f}m'
        elif measured_depth > self.max_depth:
            is_reliable = False
            reliability_note = f'Depth {measured_depth:.2f}m above maximum threshold {self.max_depth:.2f}m'
        elif measured_depth > self.unreliable_threshold:
            is_reliable = False
            reliability_note = f'Depth {measured_depth:.2f}m exceeds unreliable threshold {self.unreliable_threshold:.2f}m'

        return {
            'distance_m': measured_depth,
            'distance_cm': measured_depth * 100.0,
            'is_reliable': is_reliable,
            'reliability_note': reliability_note,
            'min_depth': min_d,
            'max_depth': max_d,
            'std_depth': std_d,
        }

    def extract_all_distances(self, depth_map: np.ndarray, detections: list) -> list:
        """
        Extract distances for all detected objects.

        Args:
            depth_map: depth map array
            detections: list of detection dicts with 'bbox' key

        Returns: list of detection dicts augmented with distance info:
            each dict gets 'distance_m', 'is_reliable', 'reliability_note' added
        """
        results = []
        for det in detections:
            det_copy = dict(det)
            bbox = det_copy.get('bbox')
            if bbox is not None:
                dist_info = self.extract_distance(depth_map, bbox)
                det_copy['distance_m'] = dist_info['distance_m']
                det_copy['distance_cm'] = dist_info['distance_cm']
                det_copy['is_reliable'] = dist_info['is_reliable']
                det_copy['reliability_note'] = dist_info['reliability_note']
                det_copy['min_depth'] = dist_info['min_depth']
                det_copy['max_depth'] = dist_info['max_depth']
                det_copy['std_depth'] = dist_info['std_depth']
            else:
                det_copy['distance_m'] = 0.0
                det_copy['distance_cm'] = 0.0
                det_copy['is_reliable'] = False
                det_copy['reliability_note'] = 'No bounding box available'
                det_copy['min_depth'] = 0.0
                det_copy['max_depth'] = 0.0
                det_copy['std_depth'] = 0.0
            results.append(det_copy)
        return results

    def apply_calibration(self, raw_distance: float, calibration_params: dict) -> float:
        """
        Apply calibration correction to a raw distance.
        calibration_params has 'scale' and 'offset' keys.
        corrected = raw_distance * scale + offset
        """
        scale = calibration_params.get('scale', 1.0)
        offset = calibration_params.get('offset', 0.0)
        return raw_distance * scale + offset

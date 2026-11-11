"""Simple IoU-based object tracker for maintaining IDs across frames."""
import numpy as np
from config import (TRACKER_MAX_AGE, TRACKER_MIN_HITS, TRACKER_IOU_THRESHOLD)


class SimpleTracker:
    """Simple IoU-based object tracker for maintaining object IDs across frames."""

    def __init__(self, max_age=None, min_hits=None, iou_threshold=None):
        """
        Initialize tracker with configurable parameters.

        Args:
            max_age: Frames to keep lost tracks before removal.
            min_hits: Minimum detections before a track is confirmed.
            iou_threshold: Minimum IoU to consider a match.
        """
        self.max_age = max_age if max_age is not None else TRACKER_MAX_AGE
        self.min_hits = min_hits if min_hits is not None else TRACKER_MIN_HITS
        self.iou_threshold = iou_threshold if iou_threshold is not None else TRACKER_IOU_THRESHOLD
        self.tracks = {}  # track_id -> track_info dict
        self.next_id = 1

    def update(self, detections: list) -> list:
        """
        Update tracks with new detections.

        Args:
            detections: list of dicts with 'bbox', 'class_name', 'confidence', etc.

        Returns:
            Updated detections list with 'track_id' added to each dict.

        Algorithm:
        1. Compute IoU matrix between existing tracks and new detections
        2. Match using greedy assignment (highest IoU first)
        3. Matched: update track position, reset age
        4. Unmatched detections: create new tracks
        5. Unmatched tracks: increment age, remove if age > max_age
        6. Only return tracks with hits >= min_hits (confirmed tracks)
        """
        if not detections:
            # Age all existing tracks, remove old ones
            to_remove = []
            for tid, track in self.tracks.items():
                track['age'] += 1
                if track['age'] > self.max_age:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.tracks[tid]
            return []

        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        num_dets = len(detections)

        if num_tracks == 0:
            # No existing tracks — create new ones for all detections
            results = []
            for det in detections:
                det_copy = dict(det)
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': list(det_copy['bbox']),
                    'class_name': det_copy.get('class_name', ''),
                    'hits': 1,
                    'age': 0,
                }
                if 1 >= self.min_hits:
                    det_copy['track_id'] = tid
                    results.append(det_copy)
            return results

        # Build IoU matrix: rows = existing tracks, cols = new detections
        iou_matrix = np.zeros((num_tracks, num_dets))
        for i, tid in enumerate(track_ids):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._compute_iou(self.tracks[tid]['bbox'], det['bbox'])

        # Greedy matching: pick highest IoU pairs first
        matched_tracks = set()
        matched_dets = set()
        matches = []  # list of (track_idx, det_idx)

        # Flatten and sort by IoU descending
        if iou_matrix.size > 0:
            flat_indices = np.argsort(-iou_matrix.ravel())
            for idx in flat_indices:
                i = int(idx // num_dets)
                j = int(idx % num_dets)
                if i in matched_tracks or j in matched_dets:
                    continue
                if iou_matrix[i, j] < self.iou_threshold:
                    break  # remaining are below threshold since sorted desc
                matches.append((i, j))
                matched_tracks.add(i)
                matched_dets.add(j)

        # Update matched tracks
        results = []
        for i, j in matches:
            tid = track_ids[i]
            det_copy = dict(detections[j])
            self.tracks[tid]['bbox'] = list(det_copy['bbox'])
            self.tracks[tid]['class_name'] = det_copy.get('class_name',
                                                          self.tracks[tid]['class_name'])
            self.tracks[tid]['hits'] += 1
            self.tracks[tid]['age'] = 0
            if self.tracks[tid]['hits'] >= self.min_hits:
                det_copy['track_id'] = tid
                results.append(det_copy)

        # Create new tracks for unmatched detections
        for j in range(num_dets):
            if j not in matched_dets:
                det_copy = dict(detections[j])
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': list(det_copy['bbox']),
                    'class_name': det_copy.get('class_name', ''),
                    'hits': 1,
                    'age': 0,
                }
                if 1 >= self.min_hits:
                    det_copy['track_id'] = tid
                    results.append(det_copy)

        # Age unmatched tracks and remove stale ones
        to_remove = []
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

        return results

    def _compute_iou(self, bbox1, bbox2) -> float:
        """Compute IoU between two [x1, y1, x2, y2] bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        intersection = inter_w * inter_h

        area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
        area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        if union <= 0:
            return 0.0
        return intersection / union

    def reset(self):
        """Clear all tracks."""
        self.tracks = {}
        self.next_id = 1

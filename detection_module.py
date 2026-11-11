"""YOLO-based object detection module for the Real-Time Depth Detection System."""

import numpy as np


class ObjectDetector:
    """YOLO-based object detection module.

    Wraps an Ultralytics YOLO model to provide a simple interface for running
    object detection on individual video frames.  Configuration defaults are
    pulled from ``config.py`` so the caller only needs to override what it
    wants to change.
    """

    def __init__(
        self,
        model_name: str | None = None,
        confidence: float | None = None,
        iou_threshold: float | None = None,
        device: str | None = None,
        img_size: int | None = None,
    ) -> None:
        """Initialise the detector, falling back to config defaults for any
        parameter left as ``None``.

        Args:
            model_name: YOLO model weight file (e.g. ``"yolov8n.pt"``).
            confidence: Minimum confidence threshold for detections.
            iou_threshold: IoU threshold for Non-Maximum Suppression.
            device: Compute device string (``"cuda:0"`` or ``"cpu"``).
            img_size: Input image size for the model.
        """
        from config import (
            YOLO_MODEL,
            YOLO_CONFIDENCE_THRESHOLD,
            YOLO_IOU_THRESHOLD,
            YOLO_DEVICE,
            YOLO_IMG_SIZE,
        )

        self.model_name: str = model_name or YOLO_MODEL
        self.confidence: float = (
            confidence if confidence is not None else YOLO_CONFIDENCE_THRESHOLD
        )
        self.iou_threshold: float = (
            iou_threshold if iou_threshold is not None else YOLO_IOU_THRESHOLD
        )
        self.device: str = device or YOLO_DEVICE
        self.img_size: int = img_size or YOLO_IMG_SIZE
        self.model = None

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """Load the YOLO model onto the configured device.

        If CUDA is requested but unavailable the method automatically falls
        back to CPU.

        Returns:
            ``True`` if the model was loaded successfully, ``False`` otherwise.
        """
        try:
            import torch
            from ultralytics import YOLO

            # Resolve device – try CUDA first, fall back to CPU.
            if "cuda" in self.device:
                if torch.cuda.is_available():
                    resolved_device = self.device
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"[ObjectDetector] ✓ NVIDIA GPU Detected: {gpu_name} (Using CUDA)")
                else:
                    print(
                        f"[ObjectDetector] ⚠ CUDA requested ({self.device}) but "
                        "torch.cuda.is_available() is False – falling back to CPU."
                    )
                    resolved_device = "cpu"
            else:
                resolved_device = self.device

            self.model = YOLO(self.model_name)
            self.model.to(resolved_device)
            self.device = resolved_device

            print(
                f"[ObjectDetector] Model '{self.model_name}' loaded on "
                f"'{self.device}'."
            )
            return True

        except Exception as exc:
            print(f"[ObjectDetector] Failed to load model: {exc}")
            self.model = None
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run object detection on a single BGR frame.

        Args:
            frame: A NumPy array representing a BGR image (H×W×3).

        Returns:
            A list of detection dictionaries, each containing:

            - ``'bbox'``: ``[x1, y1, x2, y2]`` in pixel coordinates (ints).
            - ``'confidence'``: Detection confidence as a float in [0, 1].
            - ``'class_id'``: Integer class index.
            - ``'class_name'``: Human-readable class label (e.g. ``'person'``).

            Returns an empty list when no objects are detected or if the model
            has not been loaded.
        """
        if self.model is None:
            print("[ObjectDetector] Model not loaded – call load_model() first.")
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
            )

            detections: list[dict] = []
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                class_names_map = result.names  # {int: str}

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = class_names_map.get(cls_id, str(cls_id))

                    detections.append(
                        {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf,
                            "class_id": cls_id,
                            "class_name": cls_name,
                        }
                    )

            return detections

        except Exception as exc:
            print(f"[ObjectDetector] Detection failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_class_names(self) -> dict:
        """Return the model's class-ID → class-name mapping.

        Returns:
            A ``dict[int, str]`` mapping each class index to its label.
            Returns an empty dict if the model is not loaded.
        """
        if self.model is None:
            print("[ObjectDetector] Model not loaded – call load_model() first.")
            return {}

        try:
            return dict(self.model.names)
        except Exception as exc:
            print(f"[ObjectDetector] Could not retrieve class names: {exc}")
            return {}

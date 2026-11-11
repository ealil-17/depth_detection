"""Depth Anything V2 Metric depth estimation module.

Uses Hugging Face transformers pipeline for Depth Anything V2 Metric models,
which output depth values in real metres.
"""

import numpy as np
import cv2
import torch
from PIL import Image

from config import (
    DEPTH_MODEL_TYPE,
    DEPTH_MODEL_ENCODER,
    DEPTH_MAX_DEPTH,
    DEPTH_INPUT_SIZE,
    DEPTH_DEVICE,
)

# Mapping from encoder / model key to Hugging Face Metric Model ID
# Top open-source metric depth models:
# 1. vitl: Depth Anything V2 Metric Large (335M params) - SOTA high-accuracy benchmark
# 2. vitb: Depth Anything V2 Metric Base (97M params) - balanced high-accuracy
# 3. vits: Depth Anything V2 Metric Small (25M params) - fast preview
# 4. zoedepth: Intel Labs ZoeDepth Metric - renowned zero-shot metric depth model
_MODEL_MAP = {
    "vitl": "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "vits": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "vitl-outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
    "vitb-outdoor": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
    "zoedepth": "Intel/zoedepth-nyu-kitti",
}


class DepthEstimator:
    """Depth Anything V2 Metric depth estimation module."""

    def __init__(
        self,
        model_type=None,
        encoder=None,
        max_depth=None,
        input_size=None,
        device=None,
    ):
        """Initialize depth estimator.

        All arguments fall back to ``config.py`` defaults when *None*.

        Args:
            model_type: Model variant key (``"vits"``, ``"vitb"``, ``"vitl"``).
            encoder: Encoder name (same keys as *model_type*; kept for compat).
            max_depth: Maximum reliable depth in metres.
            input_size: Model input resolution (pixels).
            device: Torch device string (``"cuda:0"`` or ``"cpu"``).
        """
        self.model_type = model_type or DEPTH_MODEL_TYPE
        self.encoder = encoder or DEPTH_MODEL_ENCODER
        self.max_depth = max_depth if max_depth is not None else DEPTH_MAX_DEPTH
        self.input_size = input_size if input_size is not None else DEPTH_INPUT_SIZE
        self.device = device or DEPTH_DEVICE
        self.pipe = None
        self._model_loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """Load the Depth Anything V2 Metric model via HF transformers pipeline.

        The model is selected from :pydata:`_MODEL_MAP` based on
        ``self.encoder``.  CUDA is attempted first; on failure the method
        retries on CPU.

        Returns:
            ``True`` if the model was loaded successfully, ``False`` otherwise.
        """
        model_id = _MODEL_MAP.get(self.encoder)
        if model_id is None:
            print(
                f"[DepthEstimator] Unknown encoder '{self.encoder}'. "
                f"Valid options: {list(_MODEL_MAP.keys())}"
            )
            return False

        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            print(
                "[DepthEstimator] 'transformers' package not installed. "
                "Install with: pip install transformers"
            )
            return False

        # Try GPU (cuda:0) first if requested and available, else fall back to CPU
        devices_to_try = []
        if "cuda" in self.device and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[DepthEstimator] ✓ NVIDIA GPU Detected: {gpu_name} — Initializing GPU Acceleration.")
            devices_to_try.append(self.device)
        elif "cuda" in self.device:
            print("[DepthEstimator] ⚠ CUDA was requested but torch.cuda.is_available() is False. Falling back to CPU.")
            devices_to_try.append("cpu")
        else:
            devices_to_try.append(self.device)

        if "cpu" not in devices_to_try:
            devices_to_try.append("cpu")

        for dev in devices_to_try:
            try:
                print(f"[DepthEstimator] Loading model '{model_id}' on device '{dev}' …")
                pipeline_kwargs = {
                    "task": "depth-estimation",
                    "model": model_id,
                    "device": dev,
                }
                # Enable FP16 on CUDA for 2x faster GPU throughput & lower VRAM
                if "cuda" in str(dev) and torch.cuda.is_available():
                    pipeline_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

                self.pipe = hf_pipeline(**pipeline_kwargs)
                self.device = dev  # record device that worked
                self._model_loaded = True
                print(f"[DepthEstimator] Model '{model_id}' successfully loaded on {dev} (GPU FP16 enabled: {'cuda' in str(dev)}).")
                return True
            except Exception as exc:
                print(f"[DepthEstimator] Failed to load on {dev}: {exc}")

        print("[DepthEstimator] Could not load model on any device.")
        return False

    # ------------------------------------------------------------------
    # Depth estimation
    # ------------------------------------------------------------------

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray | None:
        """Estimate metric depth for a full frame.

        Args:
            frame: BGR image from OpenCV (``np.ndarray``).

        Returns:
            A 2-D ``np.ndarray`` (float32) with the same height and width as
            *frame*, where each value is the estimated depth in **metres**.
            Returns ``None`` on failure.
        """
        if not self._model_loaded or self.pipe is None:
            print("[DepthEstimator] Model not loaded. Call load_model() first.")
            return None

        try:
            # 1. BGR → RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 2. NumPy → PIL
            pil_image = Image.fromarray(rgb)

            # 3. Run the pipeline
            result = self.pipe(pil_image)

            # 4. Extract the depth PIL Image
            depth_pil = result["depth"]

            # 5. Convert to numpy float32
            depth_map = np.array(depth_pil, dtype=np.float32)

            # 6. Resize to original frame dimensions if needed
            h, w = frame.shape[:2]
            if depth_map.shape[0] != h or depth_map.shape[1] != w:
                depth_map = cv2.resize(
                    depth_map, (w, h), interpolation=cv2.INTER_LINEAR
                )

            # 7. The metric model already outputs values in metres
            return depth_map

        except Exception as exc:
            print(f"[DepthEstimator] Depth estimation failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_depth_at_point(self, depth_map: np.ndarray, x: int, y: int) -> float:
        """Return the depth value (metres) at pixel coordinate (*x*, *y*).

        Args:
            depth_map: 2-D depth array produced by :meth:`estimate_depth`.
            x: Column index (horizontal pixel coordinate).
            y: Row index (vertical pixel coordinate).

        Returns:
            Depth in metres at the given point, or ``-1.0`` if the
            coordinate is out of bounds.
        """
        h, w = depth_map.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            return float(depth_map[y, x])
        print(
            f"[DepthEstimator] Coordinate ({x}, {y}) out of bounds "
            f"for depth map of size ({w}, {h})."
        )
        return -1.0

    def get_depth_colormap(self, depth_map: np.ndarray) -> np.ndarray:
        """Convert a depth map to a colourised BGR visualisation.

        The depth values are normalised to 0–255 and the
        ``cv2.COLORMAP_INFERNO`` colour map is applied.

        Args:
            depth_map: 2-D depth array produced by :meth:`estimate_depth`.

        Returns:
            A BGR ``np.ndarray`` (uint8) suitable for display with
            ``cv2.imshow``.
        """
        # Normalise to 0-255
        valid = depth_map[depth_map > 0]
        if valid.size == 0:
            # Nothing valid — return a black image
            return np.zeros((*depth_map.shape[:2], 3), dtype=np.uint8)

        min_val = float(valid.min())
        max_val = min(float(valid.max()), self.max_depth)
        if max_val <= min_val:
            max_val = min_val + 1.0  # avoid division by zero

        normalised = np.clip((depth_map - min_val) / (max_val - min_val), 0, 1)
        grey = (normalised * 255).astype(np.uint8)

        # Apply colour map (returns BGR)
        colourmap = cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO)
        return colourmap

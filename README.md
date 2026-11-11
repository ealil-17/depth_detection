# Real-Time Multi-Object Detection and Distance Estimation

A real-time computer vision system that combines **YOLOv8 object detection** with **Depth Anything V2 monocular depth estimation** to detect objects and estimate their distances from the camera. Designed for use with the Arducam 12MP 120FPS UVC USB camera on Windows.

## Features

- **Real-time object detection** using YOLOv8 (80 COCO classes)
- **Monocular depth estimation** using Depth Anything V2 Metric via Hugging Face Transformers
- **Per-object distance measurement** in metres, extracted from depth maps
- **Multi-object tracking** with persistent track IDs across frames
- **Live depth map visualisation** toggle
- **Camera calibration** workflow using a checkerboard pattern for improved accuracy
- **Validation mode** to measure and record accuracy against known distances
- **Configurable** resolution, model size, confidence thresholds, and processing frequency
- **GPU-accelerated** inference with CUDA (falls back to CPU automatically)

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| **Camera** | Arducam 12MP 120FPS UVC USB camera (or any UVC-compatible USB camera) |
| **PC** | Windows 10/11 with a dedicated NVIDIA GPU (recommended) |
| **GPU** | NVIDIA GPU with CUDA support (e.g., RTX 3060 or newer recommended) |
| **RAM** | 8 GB minimum, 16 GB recommended |
| **CPU** | CPU-only mode is supported but significantly slower |

## Software Requirements

- **Python** 3.9 or newer
- **CUDA Toolkit** 11.8+ and cuDNN (optional but strongly recommended for GPU acceleration)
- **pip** for package management

## Installation

### 1. Clone or download the project

```bash
cd depth_detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you have an NVIDIA GPU and want GPU acceleration, install PyTorch with CUDA support first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```
> Then install the remaining requirements.

### 4. First run — model downloads

The first time you run the application, it will automatically download:

- **YOLOv8 nano model** (`yolov8n.pt`) — ~6 MB
- **Depth Anything V2 Metric model** — ~500 MB

This is a one-time download. Subsequent launches load from the local cache.

## Quick Start

### Option A: Web Application Studio (Recommended)

Launch the interactive web application to monitor live camera feed, view depth maps, and configure settings directly in your browser:

```bash
python web_app.py
```
Then open: **`http://localhost:5000`**

**Web UI Features:**
- **Instant Camera Switcher**: Switch on the fly between Camera 0 and Camera 1 (or any connected UVC camera) directly from the top navigation bar or settings drawer.
- **Dual live video streams**: Real-time object detection feed and Depth Anything V2 metric colormap.
- **Side-by-side or single layout**: Switch between split-view, detection-only, or depth-only.
- **Live detection telemetry table**: Real-time listing of active tracked objects, IDs, metric distances ($m$ / $cm$), and confidence scores.
- **Interactive controls**: Resolution switching (720p, VGA, 1080p), confidence slider, depth cadence adjustment, pause/resume, and snapshot capture.
- **Simulation mode**: If no physical camera is attached, automatically runs an interactive 3D simulation with test objects.

### Option B: Native OpenCV Window

```bash
python main.py
```

Opens a native OpenCV window.
- Press **`c`** to hot-switch between connected cameras (Cam 0 ↔ Cam 1) on the fly!
- Press **`d`** to toggle depth view.
- Press **`s`** to save a screenshot.
- Press **`q`** to quit.

## Usage

### Basic usage

```bash
python main.py
```

### With custom settings

```bash
python main.py --camera 1 --width 1920 --height 1080
```

### All CLI arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--camera` | int | `0` | Camera index (0 = default, 1 = second camera, etc.) |
| `--width` | int | `1280` | Capture width in pixels |
| `--height` | int | `720` | Capture height in pixels |
| `--fps` | int | `30` | Target camera FPS |
| `--show-depth` | flag | off | Show depth map alongside the detection view |
| `--calibrate` | flag | off | Enter calibration mode using a checkerboard pattern |
| `--validate` | flag | off | Enter validation mode to test distance accuracy |
| `--cpu` | flag | off | Force CPU-only inference (ignore GPU) |
| `--yolo-model` | str | `yolov8n.pt` | YOLO model variant (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`) |
| `--confidence` | float | `0.5` | YOLO detection confidence threshold |

## Camera Calibration

Calibration improves distance estimation accuracy by computing a scale and offset correction factor.

### Running calibration

```bash
python main.py --calibrate
```

**Process:**

1. Print a **9×6 checkerboard pattern** (inner corners) with 25 mm square size.
2. Hold the checkerboard in front of the camera at various distances and angles.
3. The system captures calibration frames automatically when the checkerboard is detected.
4. Press **`c`** to compute calibration once enough samples are collected.
5. Calibration data is saved to `calibration_data.json`.

### Validating calibration

```bash
python main.py --validate
```

**Process:**

1. Place a known object at a measured distance from the camera.
2. Enter the actual distance (in metres) when prompted.
3. The system records the measured vs. actual distance.
4. Press **`r`** to view the validation report with error statistics.

## Architecture

```
depth_detection/
├── main.py               # Entry point — argument parsing, main loop, display
├── config.py             # Central configuration (all tuneable parameters)
├── camera_module.py      # Camera initialisation and frame capture
├── detection_module.py   # YOLOv8 object detection wrapper
├── depth_module.py       # Depth Anything V2 depth estimation wrapper
├── distance_extractor.py # Extracts per-object distances from depth maps
├── tracker.py            # IoU-based multi-object tracker
├── calibration.py        # Calibration and validation utilities
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── TESTING.md            # Testing and validation instructions
```

**Data flow:**

```
Camera → Frame → YOLO Detection → Bounding Boxes ─┐
                                                    ├─→ Distance Extractor → Annotated Frame
           Frame → Depth Anything → Depth Map ─────┘
                                                          ↕
                                                   Tracker (persistent IDs)
```

1. **`camera_module.py`** captures frames from the USB camera.
2. **`detection_module.py`** runs YOLOv8 on each frame and returns bounding boxes with class labels.
3. **`depth_module.py`** runs Depth Anything V2 on the frame and produces a metric depth map.
4. **`distance_extractor.py`** samples the depth map within each bounding box to compute a per-object distance.
5. **`tracker.py`** associates detections across frames using IoU matching and assigns persistent track IDs.
6. **`calibration.py`** provides calibration correction factors applied to raw depth estimates.
7. **`main.py`** ties everything together: parses arguments, runs the loop, and renders output.

## Key Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `d` | Toggle depth map display |
| `c` | Compute calibration (in calibration mode) |
| `r` | Show validation report (in validation mode) |

## Configuration

All tuneable parameters live in [`config.py`](config.py). You can modify them directly or override common ones via CLI arguments.

**Key settings:**

| Setting | Default | Description |
|---------|---------|-------------|
| `CAMERA_INDEX` | `0` | Which camera to open |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `1280` / `720` | Capture resolution |
| `YOLO_MODEL` | `yolov8n.pt` | YOLO variant — trade speed for accuracy |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.5` | Minimum detection confidence |
| `DEPTH_MODEL_TYPE` | `vits` | Depth model size: `vits` (fast), `vitb`, `vitl` (accurate) |
| `DEPTH_PROCESS_EVERY_N` | `2` | Run depth estimation every N frames to save GPU time |
| `PROCESS_EVERY_N_FRAMES` | `1` | Run detection every N frames |
| `BBOX_CENTER_CROP_RATIO` | `0.4` | Fraction of bounding box center used for depth sampling |
| `ENABLE_TRACKING` | `True` | Enable/disable multi-object tracking |
| `SHOW_DEPTH_MAP` | `False` | Show depth map side-by-side on startup |

## Troubleshooting

### Camera not found

- Verify the camera is connected and recognised by Windows (check Device Manager → Cameras).
- Try a different camera index: `python main.py --camera 1`.
- On Windows, the DSHOW backend is tried automatically if the default backend fails.

### CUDA / GPU errors

- Verify CUDA is installed: `python -c "import torch; print(torch.cuda.is_available())"`.
- If CUDA is not available, the system falls back to CPU automatically.
- Force CPU mode with `--cpu` if GPU issues persist.
- Ensure your PyTorch version matches your CUDA version.

### Slow FPS

- Use GPU acceleration (install PyTorch with CUDA).
- Reduce resolution: `python main.py --width 640 --height 480`.
- Use the nano YOLO model (`yolov8n.pt`) — this is the default.
- Increase `DEPTH_PROCESS_EVERY_N` in `config.py` (e.g., set to 3 or 4).
- Increase `PROCESS_EVERY_N_FRAMES` in `config.py`.

### Models fail to download

- Ensure you have an internet connection on the first run.
- YOLOv8 models are downloaded by the `ultralytics` package to `~/.cache/`.
- Depth Anything V2 models are downloaded by `transformers` to `~/.cache/huggingface/`.
- If behind a proxy, set `HTTP_PROXY` and `HTTPS_PROXY` environment variables.

### "No module named …" errors

- Make sure you activated the virtual environment: `venv\Scripts\activate`.
- Re-install requirements: `pip install -r requirements.txt`.

## Performance Tips

1. **Use a GPU** — CUDA-accelerated inference is 5–10× faster than CPU.
2. **Lower the resolution** — 640×480 is sufficient for detection and runs much faster.
3. **Increase `DEPTH_PROCESS_EVERY_N`** — Depth estimation is the heaviest step. Running it every 2–4 frames reduces GPU load while keeping distances reasonably up-to-date.
4. **Use the nano YOLO model** — `yolov8n.pt` is the fastest. Switch to `yolov8s.pt` or `yolov8m.pt` only if you need higher detection accuracy.
5. **Disable the depth map display** — Rendering the colourised depth map adds overhead. Keep it off unless debugging.
6. **Close other GPU-intensive applications** — Free up VRAM for the detection and depth models.

## Limitations

- **Monocular depth is approximate** — Depth Anything V2 estimates depth from a single image. It is not a replacement for stereo cameras, LIDAR, or structured-light sensors.
- **Accuracy varies with scene content** — Reflective surfaces, transparent objects, and uniform textures reduce depth estimation reliability.
- **Distance estimates are relative** — Without calibration, raw metric depth may have a systematic scale/offset error. Calibration significantly improves accuracy.
- **Small or distant objects** — Detection and depth accuracy degrade for small objects or distances beyond ~10 m.
- **Lighting conditions** — Extreme low-light or high-contrast scenes can affect both detection and depth estimation quality.
- **Single camera only** — The system processes one camera feed at a time.

## License

This project is provided as-is for educational and research purposes.

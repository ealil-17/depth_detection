# Testing Guide — Real-Time Depth Detection System

This document describes how to manually test and validate the system's functionality, accuracy, and performance.

---

## 1. Basic Functionality Test

Verify that the core pipeline works end-to-end.

### Steps

1. **Start the application:**
   ```bash
   python main.py
   ```
2. **Verify camera opens** — A window titled "Real-Time Depth Detection" should appear showing the live camera feed.
3. **Verify object detection** — Place common objects (person, bottle, cup, keyboard, phone) in view. Each detected object should have:
   - A green bounding box
   - A class label (e.g., "person", "bottle")
   - A distance estimate in metres (e.g., "1.23 m")
4. **Verify FPS counter** — An FPS value should be displayed on the frame (top-left corner).
5. **Verify depth map toggle** — Press **`d`** to show/hide the depth map visualisation alongside the main view.
6. **Verify clean exit** — Press **`q`** to quit. The application should exit without errors and release the camera.

### Expected result

| Check | Pass? |
|-------|-------|
| Camera window opens | ☐ |
| Objects detected with bounding boxes | ☐ |
| Distance estimates displayed | ☐ |
| FPS counter visible | ☐ |
| Depth map toggles with `d` | ☐ |
| Clean exit with `q` | ☐ |

---

## 2. Distance Accuracy Test

Measure how well the system estimates real-world distances.

### Setup

- Use a tape measure or ruler to mark known distances from the camera.
- Use a clearly detectable object (e.g., a water bottle, coffee mug, or book).
- Place the object directly in front of the camera, centred in the frame.

### Steps

1. **Launch in validation mode:**
   ```bash
   python main.py --validate
   ```
2. Place the object at **0.5 m** from the camera lens.
3. Enter the actual distance when prompted.
4. Record the measured distance reported by the system.
5. Repeat at **1.0 m**, **1.5 m**, **2.0 m**, and **3.0 m**.
6. Press **`r`** to view the validation report.

### Expected accuracy

| Condition | Expected error |
|-----------|---------------|
| **Uncalibrated** | ±20–30% |
| **After calibration** | ±10–15% |

### Results table

| Actual Distance (m) | Measured Distance (m) | Error (m) | Error (%) | Notes |
|---------------------:|----------------------:|----------:|----------:|-------|
| 0.50 | | | | |
| 1.00 | | | | |
| 1.50 | | | | |
| 2.00 | | | | |
| 3.00 | | | | |

---

## 3. Multi-Object Test

Verify that the system handles multiple simultaneous objects correctly.

### Steps

1. **Start the application:**
   ```bash
   python main.py
   ```
2. Place **3–5 different objects** at varying distances from the camera (e.g., a bottle at 0.5 m, a mug at 1.0 m, a book at 2.0 m).
3. Verify:
   - Each object has its **own bounding box** and class label.
   - Each object shows an **independent distance** estimate.
   - The **relative ordering** is correct (closer objects show smaller distances).
4. Move one object closer/further and verify its distance updates while others remain stable.

### Results table

| Object | Actual Distance (m) | Measured Distance (m) | Correct ordering? |
|--------|---------------------:|----------------------:|:-----------------:|
| Object 1 | | | ☐ |
| Object 2 | | | ☐ |
| Object 3 | | | ☐ |
| Object 4 | | | ☐ |
| Object 5 | | | ☐ |

---

## 4. Performance Test

Measure FPS at different resolutions and configurations.

### Steps

1. **Test at 480p:**
   ```bash
   python main.py --width 640 --height 480
   ```
   Record the FPS displayed on screen after ~10 seconds of stable operation.

2. **Test at 720p (default):**
   ```bash
   python main.py --width 1280 --height 720
   ```

3. **Test at 1080p:**
   ```bash
   python main.py --width 1920 --height 1080
   ```

4. **Test with depth map display:**
   ```bash
   python main.py --show-depth
   ```

### Expected performance

| Resolution | GPU (NVIDIA) | CPU-only |
|------------|:------------:|:--------:|
| 640×480 | 25–30 FPS | 8–12 FPS |
| 1280×720 | 15–25 FPS | 5–10 FPS |
| 1920×1080 | 10–15 FPS | 3–6 FPS |

### Results table

| Resolution | Depth display | Device | FPS observed |
|------------|:------------:|--------|-------------:|
| 640×480 | Off | GPU | |
| 640×480 | On | GPU | |
| 1280×720 | Off | GPU | |
| 1280×720 | On | GPU | |
| 1920×1080 | Off | GPU | |
| 1280×720 | Off | CPU (`--cpu`) | |

---

## 5. Calibration Test

Verify the calibration workflow works and improves accuracy.

### Setup

- Print a **9×6 checkerboard** (inner corners) with **25 mm squares**.
- Ensure good, even lighting.

### Steps

1. **Run calibration mode:**
   ```bash
   python main.py --calibrate
   ```
2. Hold the checkerboard in front of the camera at several distances (0.5 m, 1.0 m, 2.0 m) and angles.
3. Verify the system detects the checkerboard (visual feedback on screen).
4. Enter the actual distance for each capture when prompted.
5. Press **`c`** to compute calibration. Verify the console prints scale and offset values.
6. Verify that `calibration_data.json` is created in the project directory.
7. **Re-run in validation mode** to verify improved accuracy:
   ```bash
   python main.py --validate
   ```

### Results table

| Check | Pass? |
|-------|-------|
| Checkerboard detected on screen | ☐ |
| Calibration computes without error | ☐ |
| `calibration_data.json` saved | ☐ |
| Post-calibration accuracy improved | ☐ |

---

## 6. Tracking Test

Verify that object tracking assigns and maintains consistent IDs.

### Steps

1. **Start the application:**
   ```bash
   python main.py
   ```
2. Place a single object in view and note its **track ID** (displayed alongside the class label).
3. **Move the object slowly** across the frame (left → right, or towards/away from camera).
4. Verify the track ID **stays the same** throughout the movement.
5. **Briefly occlude** the object (cover it with your hand for 1–2 seconds), then reveal it again.
6. Check whether the track ID is reassigned or maintained (brief occlusions should maintain the ID if within `TRACKER_MAX_AGE` frames).
7. **Introduce a second object** and verify it gets a **different track ID**.

### Results table

| Scenario | Expected | Observed | Pass? |
|----------|----------|----------|-------|
| Slow movement across frame | ID stays constant | | ☐ |
| Brief occlusion (~1 s) | ID maintained | | ☐ |
| Long occlusion (~5 s) | ID may change | | ☐ |
| Two objects simultaneously | Different IDs | | ☐ |

---

## Summary Checklist

| Test | Status |
|------|--------|
| 1. Basic Functionality | ☐ |
| 2. Distance Accuracy | ☐ |
| 3. Multi-Object | ☐ |
| 4. Performance | ☐ |
| 5. Calibration | ☐ |
| 6. Tracking | ☐ |

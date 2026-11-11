"""Central configuration for the Real-Time Depth Detection System."""
import os

# --- Camera Settings ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAMERA_BACKEND = None  # None = auto, or cv2.CAP_DSHOW for Windows

# --- YOLO Detection Settings ---
YOLO_MODEL = "yolov8n.pt"  # nano model for speed; options: yolov8s.pt, yolov8m.pt
YOLO_CONFIDENCE_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_DEVICE = "cuda:0"  # "cuda:0" or "cpu"
YOLO_IMG_SIZE = 640

# --- Depth Estimation Settings ---
DEPTH_MODEL_TYPE = "vits"  # Options: "vits" (small/fast), "vitb" (base), "vitl" (large)
DEPTH_MODEL_ENCODER = "vits"
DEPTH_MAX_DEPTH = 20.0  # metres
DEPTH_INPUT_SIZE = 518  # model input resolution
DEPTH_DEVICE = "cuda:0"  # "cuda:0" or "cpu"

# --- Distance Extraction ---
BBOX_CENTER_CROP_RATIO = 0.4  # fraction of bbox center used for median depth
MIN_DEPTH_CONFIDENCE = 0.1  # minimum depth value considered valid (metres)
MAX_DEPTH_CONFIDENCE = 20.0  # maximum depth value considered valid (metres)
UNRELIABLE_DEPTH_THRESHOLD = 15.0  # beyond this, mark as "unreliable"

# --- Tracker Settings ---
ENABLE_TRACKING = True
TRACKER_MAX_AGE = 30  # frames to keep lost tracks
TRACKER_MIN_HITS = 3  # min detections before track is confirmed
TRACKER_IOU_THRESHOLD = 0.3

# --- Display Settings ---
DISPLAY_WINDOW_NAME = "Real-Time Depth Detection"
BBOX_COLOR = (0, 255, 0)  # green
TEXT_COLOR = (255, 255, 255)  # white
TEXT_BG_COLOR = (0, 0, 0)  # black
FONT_SCALE = 0.6
FONT_THICKNESS = 2
SHOW_DEPTH_MAP = False  # show depth map side-by-side
SHOW_FPS = True

# --- Calibration ---
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "calibration_data.json")
CHECKERBOARD_SIZE = (9, 6)  # inner corners
CHECKERBOARD_SQUARE_SIZE = 0.025  # metres per square

# --- Performance ---
PROCESS_EVERY_N_FRAMES = 1  # 1 = every frame, 2 = every other, etc.
DEPTH_PROCESS_EVERY_N = 2  # depth can run less frequently than detection

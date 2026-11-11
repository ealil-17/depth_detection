"""
Diagnostic and test utility to inspect connected USB/UVC cameras on Windows.
Detects available camera indices, backend support (DirectShow, MSMF),
and verifies supported resolutions without issuing autofocus commands.
"""

import sys
import cv2

RESOLUTIONS_TO_TEST = [
    (640, 480, "VGA"),
    (1280, 720, "720p HD (Default)"),
    (1920, 1080, "1080p Full HD"),
    (3840, 2160, "4K UHD (Arducam 12MP capable)"),
    (4056, 3040, "12MP Native Sensor Max"),
]

def scan_cameras(max_tested=5):
    print("=" * 65)
    print(" 📷 ARDUCAM & UVC CAMERA HARDWARE DIAGNOSTIC TOOL")
    print("=" * 65)
    print(f"Scanning camera indices 0 through {max_tested - 1}...\n")

    found_cameras = []

    for index in range(max_tested):
        # On Windows, try DirectShow first, then default
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        backend = "DirectShow"

        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
            backend = "Default (MSMF/V4L)"

        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)

                print(f"[FOUND] Camera Index {index} (Backend: {backend})")
                print(f"        Current Default: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")

                # Test supported resolutions
                print("        Testing resolution negotiation:")
                supported_res = []
                for w, h, label in RESOLUTIONS_TO_TEST:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    set_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    set_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if set_w == w and set_h == h:
                        print(f"          ✓ {label} ({w}x{h}) - Supported")
                        supported_res.append(f"{w}x{h}")
                    else:
                        print(f"          ✗ {label} ({w}x{h}) - Settled at {set_w}x{set_h}")

                found_cameras.append({
                    "index": index,
                    "backend": backend,
                    "default_res": f"{actual_w}x{actual_h}",
                    "supported": supported_res,
                })
            else:
                print(f"[WARN] Camera Index {index} opened but could not read test frame.")
            cap.release()
        else:
            print(f"[----] Camera Index {index}: Not connected.")

    print("\n" + "=" * 65)
    print(" DIAGNOSTIC SUMMARY")
    print("=" * 65)
    if found_cameras:
        print(f"Total operational cameras detected: {len(found_cameras)}")
        for cam in found_cameras:
            print(f"  -> Index {cam['index']}: Best default for project is {cam['supported'][1] if len(cam['supported']) > 1 else cam['default_res']}")
        print("\nTo launch with your selected camera index:")
        print(f"  python main.py --camera {found_cameras[0]['index']}")
        print(f"  python web_app.py --camera {found_cameras[0]['index']}")
    else:
        print("No physical cameras detected. You can still test the pipeline:")
        print("  python web_app.py (Will automatically launch in interactive simulation mode)")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    scan_cameras()


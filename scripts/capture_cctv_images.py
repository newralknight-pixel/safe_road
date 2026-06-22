from datetime import datetime
from pathlib import Path
import time

import cv2


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "datasets" / "cctv_yolo" / "images" / "train"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise SystemExit("Could not open CCTV camera index 0.")

    try:
        print("Capturing CCTV training images. Press Ctrl+C to stop.")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Skipped unreadable frame.")
                time.sleep(1)
                continue

            filename = datetime.now().strftime("cctv_%Y%m%d_%H%M%S_%f.jpg")
            output_path = OUTPUT_DIR / filename
            cv2.imwrite(str(output_path), frame)
            print(output_path)
            time.sleep(2)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        capture.release()


if __name__ == "__main__":
    main()

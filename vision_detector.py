from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import List

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_YOLO_MODEL_PATH = BASE_DIR / "models" / "cctv_best.pt"
DEFAULT_LITTER_MODEL_PATH = BASE_DIR / "models" / "external" / "litter_yolov8_best.pt"
DEFAULT_DEER_MODEL_PATH = BASE_DIR / "models" / "external" / "yolov8s-worldv2.pt"
DEER_WORLD_CLASSES = [
    "Korean water deer",
    "water deer",
    "deer",
    "wild animal",
    "animal on road",
]
YOLO_CLASS_MAP = {
    "alligator crack": "road_crack",
    "deer": "roe_deer",
    "longitudinal crack": "road_crack",
    "other corruption": "road_crack",
    "roe_deer": "roe_deer",
    "trash": "trash",
    "road_crack": "road_crack",
    "crack": "road_crack",
    "pothole": "pothole",
    "transverse crack": "road_crack",
}
YOLO_LABELS = {
    "roe_deer": "deer",
    "trash": "trash",
    "road_crack": "crack",
    "pothole": "pothole",
}
YOLO_MODEL = None
YOLO_MODEL_ERROR = None
LITTER_MODEL = None
LITTER_MODEL_ERROR = None
DEER_MODEL = None
DEER_MODEL_ERROR = None


@dataclass
class VisionDetection:
    type: str
    confidence: float
    severity: str
    detail: str
    bbox: tuple[int, int, int, int] | None = None

    def to_event_detection(self):
        detection = {
            "type": self.type,
            "confidence": self.confidence,
            "severity": self.severity,
            "detail": self.detail,
        }
        if self.bbox:
            detection["bbox"] = tuple(int(value) for value in self.bbox)
        return detection


def union_boxes(boxes):
    if not boxes:
        return None
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))


def load_cv_tools():
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as error:
        raise ValueError("OpenCV is not installed. Install dependencies with: pip install -r requirements.txt") from error
    return cv2, np


def load_yolo_model():
    global YOLO_MODEL, YOLO_MODEL_ERROR
    if YOLO_MODEL is not None:
        return YOLO_MODEL
    if YOLO_MODEL_ERROR is not None:
        return None

    model_path = Path(os.environ.get("SAFE_ROAD_YOLO_MODEL", DEFAULT_YOLO_MODEL_PATH))
    if not model_path.exists():
        YOLO_MODEL_ERROR = f"YOLO model not found: {model_path}"
        return None

    try:
        from ultralytics import YOLO

        YOLO_MODEL = YOLO(str(model_path))
        return YOLO_MODEL
    except Exception as error:
        YOLO_MODEL_ERROR = f"Could not load YOLO model: {error}"
        return None


def load_litter_model():
    global LITTER_MODEL, LITTER_MODEL_ERROR
    if LITTER_MODEL is not None:
        return LITTER_MODEL
    if LITTER_MODEL_ERROR is not None:
        return None

    model_path = Path(os.environ.get("SAFE_ROAD_LITTER_MODEL", DEFAULT_LITTER_MODEL_PATH))
    if not model_path.exists():
        LITTER_MODEL_ERROR = f"Litter model not found: {model_path}"
        return None

    try:
        from ultralytics import YOLO

        LITTER_MODEL = YOLO(str(model_path))
        return LITTER_MODEL
    except Exception as error:
        LITTER_MODEL_ERROR = f"Could not load litter model: {error}"
        return None


def load_deer_model():
    global DEER_MODEL, DEER_MODEL_ERROR
    if DEER_MODEL is not None:
        return DEER_MODEL
    if DEER_MODEL_ERROR is not None:
        return None

    model_path = Path(os.environ.get("SAFE_ROAD_DEER_MODEL", DEFAULT_DEER_MODEL_PATH))
    if not model_path.exists():
        DEER_MODEL_ERROR = f"Deer model not found: {model_path}"
        return None

    try:
        from ultralytics import YOLOWorld

        DEER_MODEL = YOLOWorld(str(model_path))
        DEER_MODEL.set_classes(DEER_WORLD_CLASSES)
        return DEER_MODEL
    except Exception as error:
        DEER_MODEL_ERROR = f"Could not load deer model: {error}"
        return None


def trained_model_exists() -> bool:
    model_paths = [
        Path(os.environ.get("SAFE_ROAD_YOLO_MODEL", DEFAULT_YOLO_MODEL_PATH)),
        Path(os.environ.get("SAFE_ROAD_LITTER_MODEL", DEFAULT_LITTER_MODEL_PATH)),
        Path(os.environ.get("SAFE_ROAD_DEER_MODEL", DEFAULT_DEER_MODEL_PATH)),
    ]
    return any(model_path.exists() for model_path in model_paths)


def detect_hazards_with_yolo(image, confidence_threshold=0.5) -> List[dict]:
    model = load_yolo_model()
    if model is None:
        return []

    results = model.predict(image, conf=confidence_threshold, verbose=False)
    detections = []
    for result in results:
        names = result.names
        for box in result.boxes:
            raw_name = names[int(box.cls[0])]
            hazard_type = YOLO_CLASS_MAP.get(str(raw_name).lower())
            if not hazard_type:
                continue
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf[0])
            severity = "high" if hazard_type in {"roe_deer", "road_crack"} else "medium"
            detections.append(
                {
                    "type": hazard_type,
                    "confidence": confidence,
                    "severity": severity,
                    "detail": f"YOLO {YOLO_LABELS[hazard_type]} detection",
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                }
            )
    return detections


def detect_hazards_with_litter_model(image, confidence_threshold=0.4) -> List[dict]:
    model = load_litter_model()
    if model is None:
        return []

    results = model.predict(image, conf=confidence_threshold, verbose=False)
    candidates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            candidates.append(
                {
                    "type": "trash",
                    "confidence": float(box.conf[0]),
                    "severity": "medium",
                    "detail": "YOLO litter detection",
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                }
            )
    return sorted(candidates, key=lambda detection: detection["confidence"], reverse=True)[:1]


def detect_hazards_with_deer_model(image, confidence_threshold=0.2) -> List[dict]:
    model = load_deer_model()
    if model is None:
        return []

    results = model.predict(image, conf=confidence_threshold, verbose=False)
    candidates = []
    for result in results:
        for box in result.boxes:
            raw_name = str(result.names[int(box.cls[0])])
            if raw_name not in DEER_WORLD_CLASSES:
                continue
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            candidates.append(
                {
                    "type": "roe_deer",
                    "confidence": float(box.conf[0]),
                    "severity": "high",
                    "detail": f"YOLO-World {raw_name} detection",
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                }
            )
    return sorted(candidates, key=lambda detection: detection["confidence"], reverse=True)[:1]


def decode_image(image_bytes: bytes):
    cv2, np = load_cv_tools()
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a readable image.")
    return image


def detect_road_crack(image) -> VisionDetection | None:
    cv2, np = load_cv_tools()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 180)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=70,
        maxLineGap=18,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    long_thin_contours = 0
    longest = 0.0
    line_count = 0 if lines is None else len(lines)
    candidate_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 25:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        longest = max(longest, float(max(width, height)))
        short_side = max(1, min(width, height))
        aspect_ratio = max(width, height) / short_side
        if aspect_ratio >= 4.0 and max(width, height) >= 60:
            long_thin_contours += 1
            candidate_boxes.append((x, y, width, height))

    if lines is not None:
        for line in lines[:20]:
            x1, y1, x2, y2 = line[0]
            x = min(x1, x2)
            y = min(y1, y2)
            candidate_boxes.append((x, y, abs(x2 - x1) + 1, abs(y2 - y1) + 1))

    if long_thin_contours == 0:
        if line_count == 0:
            return None

    confidence = min(0.95, 0.45 + long_thin_contours * 0.12 + line_count * 0.04 + longest / 900)
    severity = "high" if confidence >= 0.72 else "medium"
    return VisionDetection(
        type="road_crack",
        confidence=confidence,
        severity=severity,
        detail=f"{long_thin_contours} long edge patterns and {line_count} line segments found",
        bbox=union_boxes(candidate_boxes),
    )


def detect_road_trash(image) -> VisionDetection | None:
    cv2, np = load_cv_tools()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = cv2.inRange(saturation, 70, 255)
    mask = cv2.bitwise_and(mask, cv2.inRange(value, 80, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = image.shape[0] * image.shape[1]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if image_area * 0.0005 <= area <= image_area * 0.08:
            candidates.append((area, cv2.boundingRect(contour)))

    if not candidates:
        return None

    largest_area, best_box = max(candidates, key=lambda item: item[0])
    largest_ratio = largest_area / image_area
    confidence = min(0.9, 0.42 + len(candidates) * 0.08 + largest_ratio * 3)
    return VisionDetection(
        type="trash",
        confidence=confidence,
        severity="medium",
        detail=f"{len(candidates)} colored object candidates found",
        bbox=best_box,
    )


def detect_roe_deer(image) -> VisionDetection | None:
    cv2, np = load_cv_tools()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]
    image_area = height * width

    brown_mask = cv2.inRange(hsv, (5, 35, 35), (35, 210, 230))
    brown_mask[: int(height * 0.15), :] = 0
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    brown_mask = cv2.morphologyEx(brown_mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not image_area * 0.003 <= area <= image_area * 0.22:
            continue
        x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
        aspect_ratio = candidate_width / max(1, candidate_height)
        if not 0.45 <= aspect_ratio <= 2.8:
            continue
        fill_ratio = area / max(1, candidate_width * candidate_height)
        if fill_ratio < 0.22:
            continue
        lower_frame_bias = (y + candidate_height / 2) / height
        candidates.append((area, aspect_ratio, lower_frame_bias, (x, y, candidate_width, candidate_height)))

    if not candidates:
        return None

    largest_area, best_aspect_ratio, lower_frame_bias, best_box = max(candidates, key=lambda item: item[0])
    area_ratio = largest_area / image_area
    confidence = min(0.88, 0.38 + area_ratio * 6 + lower_frame_bias * 0.18)
    severity = "high" if confidence >= 0.62 else "medium"
    return VisionDetection(
        type="roe_deer",
        confidence=confidence,
        severity=severity,
        detail=(
            f"{len(candidates)} deer-like brown animal candidate(s) found "
            f"with aspect ratio {best_aspect_ratio:.2f}"
        ),
        bbox=best_box,
    )


def detect_pothole(image) -> VisionDetection | None:
    cv2, np = load_cv_tools()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    height, width = gray.shape[:2]
    image_area = height * width

    dark_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        7,
    )
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not image_area * 0.002 <= area <= image_area * 0.18:
            continue
        x, y, candidate_width, candidate_height = cv2.boundingRect(contour)
        if y < height * 0.25:
            continue
        aspect_ratio = candidate_width / max(1, candidate_height)
        if not 0.45 <= aspect_ratio <= 2.8:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        fill_ratio = area / max(1, candidate_width * candidate_height)
        if circularity >= 0.18 and fill_ratio >= 0.25:
            candidates.append((area, circularity, y + candidate_height / 2, (x, y, candidate_width, candidate_height)))

    if not candidates:
        return None

    largest_area, best_circularity, center_y, best_box = max(candidates, key=lambda item: item[0])
    area_ratio = largest_area / image_area
    lower_frame_bias = center_y / height
    confidence = min(0.92, 0.42 + area_ratio * 5 + best_circularity * 0.18 + lower_frame_bias * 0.12)
    severity = "high" if area_ratio >= 0.015 or confidence >= 0.7 else "medium"
    return VisionDetection(
        type="pothole",
        confidence=confidence,
        severity=severity,
        detail=f"{len(candidates)} pothole-like dark road surface candidate(s) found",
        bbox=best_box,
    )


def detect_hazards_from_image(image_bytes: bytes) -> List[dict]:
    image = decode_image(image_bytes)
    return detect_hazards_from_frame(image)


def detect_hazards_from_webcam(camera_index: int = 0) -> List[dict]:
    frame = read_webcam_frame(camera_index)
    return detect_hazards_from_frame(frame)


def read_webcam_frame(camera_index: int = 0):
    cv2, _ = load_cv_tools()
    capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise ValueError(f"Could not open webcam index {camera_index}.")

    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError("Webcam opened, but no frame could be read.")
        return frame
    finally:
        capture.release()


def capture_webcam_jpeg(camera_index: int = 0) -> bytes:
    cv2, _ = load_cv_tools()
    frame = read_webcam_frame(camera_index)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise ValueError("Webcam frame could not be encoded as JPEG.")
    return encoded.tobytes()


def detect_hazards_from_frame(image) -> List[dict]:
    model_detections = []
    model_detections.extend(detect_hazards_with_yolo(image))
    model_detections.extend(detect_hazards_with_litter_model(image))
    model_detections.extend(detect_hazards_with_deer_model(image))
    if model_detections or trained_model_exists():
        return model_detections

    detections = []

    deer = detect_roe_deer(image)
    if deer:
        detections.append(deer)

    crack = detect_road_crack(image)
    if crack:
        detections.append(crack)

    pothole = detect_pothole(image)
    if pothole:
        detections.append(pothole)

    trash = detect_road_trash(image)
    if trash:
        detections.append(trash)

    return [detection.to_event_detection() for detection in detections]


def merge_video_detections(frame_detections: List[tuple[int, dict]]) -> List[dict]:
    best_by_type = {}
    frame_counts = {}

    for frame_number, detection in frame_detections:
        hazard_type = detection["type"]
        frame_counts[hazard_type] = frame_counts.get(hazard_type, 0) + 1
        current = best_by_type.get(hazard_type)
        if current is None or detection["confidence"] > current["confidence"]:
            merged = detection.copy()
            merged["detail"] = (
                f"Video frame {frame_number}: {detection.get('detail', 'hazard pattern found')}"
            )
            best_by_type[hazard_type] = merged

    merged_detections = []
    for hazard_type, detection in best_by_type.items():
        detection = detection.copy()
        detection["detail"] = f"{detection['detail']} across {frame_counts[hazard_type]} sampled frame(s)"
        merged_detections.append(detection)

    return merged_detections


def detect_hazards_from_video(video_bytes: bytes, sample_count: int = 12) -> List[dict]:
    cv2, _ = load_cv_tools()
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError("Uploaded file is not a readable video.")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            frame_indexes = range(sample_count)
        else:
            step = max(1, total_frames // sample_count)
            frame_indexes = range(0, total_frames, step)

        frame_detections = []
        for sampled, frame_index in enumerate(frame_indexes):
            if sampled >= sample_count:
                break
            if total_frames > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            for detection in detect_hazards_from_frame(frame):
                frame_detections.append((frame_index + 1, detection))

        capture.release()
        if total_frames > 0 and not frame_detections:
            return []
        return merge_video_detections(frame_detections)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

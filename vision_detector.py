from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class VisionDetection:
    type: str
    confidence: float
    severity: str
    detail: str

    def to_event_detection(self):
        return {
            "type": self.type,
            "confidence": self.confidence,
            "severity": self.severity,
            "detail": self.detail,
        }


def decode_image(image_bytes: bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Uploaded file is not a readable image.")
    return image


def detect_road_crack(image) -> VisionDetection | None:
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
    )


def detect_road_trash(image) -> VisionDetection | None:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = cv2.inRange(saturation, 70, 255)
    mask = cv2.bitwise_and(mask, cv2.inRange(value, 80, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = image.shape[0] * image.shape[1]
    candidate_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if image_area * 0.0005 <= area <= image_area * 0.08:
            candidate_areas.append(area)

    if not candidate_areas:
        return None

    largest_ratio = max(candidate_areas) / image_area
    confidence = min(0.9, 0.42 + len(candidate_areas) * 0.08 + largest_ratio * 3)
    return VisionDetection(
        type="trash",
        confidence=confidence,
        severity="medium",
        detail=f"{len(candidate_areas)} colored object candidates found",
    )


def detect_hazards_from_image(image_bytes: bytes) -> List[dict]:
    image = decode_image(image_bytes)
    detections = []

    crack = detect_road_crack(image)
    if crack:
        detections.append(crack)

    trash = detect_road_trash(image)
    if trash:
        detections.append(trash)

    return [detection.to_event_detection() for detection in detections]

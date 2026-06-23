#!/usr/bin/env python3
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import cgi
import json
import random
import threading
import time
from datetime import datetime, timezone

from vision_detector import (
    detect_hazards_from_frame,
    detect_hazards_from_image,
    detect_hazards_from_video,
    detect_hazards_from_webcam,
    load_cv_tools,
)
from picoboard_alert import pico_status, send_picoboard_alert

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
DETECTIONS_FILE = DATA_DIR / "detections.json"
DETECTIONS_LOCK = threading.Lock()
WEBCAM_MONITOR = {
    "running": False,
    "camera_index": 0,
    "interval_seconds": 10,
    "last_capture_at": None,
    "last_error": None,
    "last_accepted": 0,
    "last_labels": [],
    "last_detections": [],
    "preview_ready": False,
}
WEBCAM_FRAME_LOCK = threading.Lock()
WEBCAM_LATEST_JPEG = None

HAZARD_CONFIG = {
    "trash": {
        "label": "Road trash",
        "severity": "medium",
        "message": "Trash detected on road. Notify road manager and warn nearby drivers.",
        "targets": ["road_manager", "navigation", "nearby_users"],
    },
    "roe_deer": {
        "label": "Roe deer",
        "severity": "high",
        "message": "Wild animal near road. Notify police and warn drivers to slow down.",
        "targets": ["police", "navigation", "nearby_users"],
    },
    "road_crack": {
        "label": "Road crack",
        "severity": "high",
        "message": "Road crack detected. Notify maintenance team and mark route as dangerous.",
        "targets": ["road_manager", "navigation", "nearby_users"],
    },
    "pothole": {
        "label": "Pothole",
        "severity": "high",
        "message": "Pothole detected. Notify maintenance team and warn nearby drivers.",
        "targets": ["road_manager", "navigation", "nearby_users"],
    },
}

SAMPLE_LOCATIONS = [
    {"road": "Route 37", "latitude": 37.5665, "longitude": 126.9780},
    {"road": "Mountain Road 12", "latitude": 37.7519, "longitude": 128.8761},
    {"road": "City Ring Road", "latitude": 35.1796, "longitude": 129.0756},
    {"road": "Coastal Highway", "latitude": 33.4996, "longitude": 126.5312},
]

PUBLIC_HAZARD_TYPES = {
    "pothole": "road_damage",
    "road_crack": "road_damage",
    "roe_deer": "wild_animal",
    "trash": "trash",
}

PUBLIC_ACTIONS = {
    "road_damage": "slow_down",
    "wild_animal": "caution",
    "trash": "avoid_object",
}

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}

SOURCE_DEFAULT_DRONE_IDS = {
    "blackbox": "blackbox-upload-01",
    "drone": "upload-drone-01",
    "roadside": "roadside-camera-01",
    "webcam": "cctv-camera-01",
}

SOURCE_LABELS = {
    "blackbox": "Blackbox camera",
    "drone": "Drone camera",
    "roadside": "Roadside camera",
    "webcam": "CCTV camera",
}


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def ensure_data_file():
    DATA_DIR.mkdir(exist_ok=True)
    if not DETECTIONS_FILE.exists():
        DETECTIONS_FILE.write_text("[]\n", encoding="utf-8")


def load_detections():
    ensure_data_file()
    with DETECTIONS_LOCK:
        with DETECTIONS_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)


def save_detections(detections):
    ensure_data_file()
    with DETECTIONS_LOCK:
        with DETECTIONS_FILE.open("w", encoding="utf-8") as file:
            json.dump(detections, file, indent=2)
            file.write("\n")


def append_detections(new_events):
    detections = load_detections()
    detections.extend(new_events)
    save_detections(detections)
    send_picoboard_alert(new_events)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_event(
    drone_id,
    hazard_type,
    latitude,
    longitude,
    confidence,
    severity=None,
    road="Unknown road",
    detail=None,
):
    config = HAZARD_CONFIG.get(hazard_type, {})
    event_id = f"evt-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}"
    return {
        "id": event_id,
        "created_at": utc_now(),
        "drone_id": drone_id,
        "type": hazard_type,
        "label": config.get("label", hazard_type.replace("_", " ").title()),
        "severity": severity or config.get("severity", "medium"),
        "confidence": round(float(confidence), 2),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "road": road,
        "message": config.get("message", "Road hazard detected."),
        "detail": detail,
        "alert_targets": config.get("targets", ["road_manager", "nearby_users"]),
        "status": "new",
    }


def simulate_event():
    location = random.choice(SAMPLE_LOCATIONS)
    hazard_type = random.choice(list(HAZARD_CONFIG.keys()))
    confidence = random.uniform(0.78, 0.98)
    return build_event(
        drone_id="sim-drone-01",
        hazard_type=hazard_type,
        latitude=location["latitude"] + random.uniform(-0.015, 0.015),
        longitude=location["longitude"] + random.uniform(-0.015, 0.015),
        confidence=confidence,
        road=location["road"],
    )


def parse_int_query(query, key, default=None, minimum=None, maximum=None):
    values = query.get(key)
    if not values:
        return default
    if len(values) > 1:
        raise ValueError(f"query parameter '{key}' must be provided once")
    try:
        value = int(values[0])
    except ValueError as error:
        raise ValueError(f"query parameter '{key}' must be an integer") from error
    if minimum is not None and value < minimum:
        raise ValueError(f"query parameter '{key}' must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"query parameter '{key}' must be at most {maximum}")
    return value


def parse_float_query(query, key, default=None, minimum=None, maximum=None):
    values = query.get(key)
    if not values:
        return default
    if len(values) > 1:
        raise ValueError(f"query parameter '{key}' must be provided once")
    try:
        value = float(values[0])
    except ValueError as error:
        raise ValueError(f"query parameter '{key}' must be a number") from error
    if minimum is not None and value < minimum:
        raise ValueError(f"query parameter '{key}' must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"query parameter '{key}' must be at most {maximum}")
    return value


def single_query_value(query, key):
    values = query.get(key)
    if not values:
        return None
    if len(values) > 1:
        raise ValueError(f"query parameter '{key}' must be provided once")
    return values[0]


def event_timestamp_ms(event):
    created_at = event.get("created_at", "")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        parsed = datetime.now(timezone.utc)
    return int(parsed.timestamp() * 1000)


def public_event(event, index):
    hazard_type = PUBLIC_HAZARD_TYPES.get(event.get("type"), "road_damage")
    return {
        "event_id": f"agg-{index:04d}",
        "road_segment_id": f"redacted-segment-{index % 999:03d}",
        "hazard_type": hazard_type,
        "severity": event.get("severity", "medium"),
        "confidence": round(float(event.get("confidence", 0)), 3),
        "timestamp_ms": event_timestamp_ms(event),
        "source": "private_aggregate_adapter",
        "action": PUBLIC_ACTIONS.get(hazard_type, "caution"),
    }


def risk_summary(events):
    if not events:
        return {"overall_level": "low", "top_hazard": "none", "incident_count": 0}

    top_hazard = "none"
    top_count = 0
    hazard_counts = {}
    for event in events:
        hazard_counts[event["hazard_type"]] = hazard_counts.get(event["hazard_type"], 0) + 1
    for hazard_type, count in hazard_counts.items():
        if count > top_count:
            top_hazard = hazard_type
            top_count = count

    overall_level = max(events, key=lambda event: SEVERITY_RANK[event["severity"]])["severity"]
    return {
        "overall_level": overall_level,
        "top_hazard": top_hazard,
        "incident_count": len(events),
    }


def build_public_event_feed(query):
    raw_events = sorted(load_detections(), key=lambda item: item["created_at"], reverse=True)
    events = [public_event(event, index) for index, event in enumerate(raw_events)]

    hazard_type = single_query_value(query, "hazard_type")
    if hazard_type:
        if hazard_type not in set(PUBLIC_HAZARD_TYPES.values()):
            raise ValueError("query parameter 'hazard_type' is not supported")
        events = [event for event in events if event["hazard_type"] == hazard_type]

    severity = single_query_value(query, "severity")
    if severity:
        if severity not in SEVERITY_RANK:
            raise ValueError("query parameter 'severity' is not supported")
        events = [event for event in events if event["severity"] == severity]

    min_confidence = parse_float_query(query, "min_confidence", minimum=0, maximum=1)
    if min_confidence is not None:
        events = [event for event in events if event["confidence"] >= min_confidence]

    limit = parse_int_query(query, "limit", default=50, minimum=1, maximum=100)
    events = events[:limit]

    return {
        "contract_version": "road-hazard-events.v1",
        "schema_version": "0.4.0",
        "feed_type": "road_hazard_events",
        "generated_at": utc_now(),
        "privacy": {
            "mode": "private_aggregate_only",
            "synthetic_or_aggregate_only": True,
            "contains_real_camera_pixels": False,
            "contains_private_paths": False,
            "publishable": True,
        },
        "risk_summary": risk_summary(events),
        "events": events,
    }


def is_video_upload(upload_field):
    filename = getattr(upload_field, "filename", "") or ""
    content_type = getattr(upload_field, "type", "") or ""
    return content_type.startswith("video/") or Path(filename).suffix.lower() in VIDEO_EXTENSIONS


def append_source_detail(detection, source_type, media_type):
    detail_parts = [f"{SOURCE_LABELS[source_type]} {media_type}"]
    if detection.get("detail"):
        detail_parts.append(detection["detail"])
    updated = detection.copy()
    updated["detail"] = ": ".join(detail_parts)
    return updated


def capture_webcam_events(camera_index=0, latitude=37.5665, longitude=126.9780, road="Local webcam road"):
    vision_detections = [
        append_source_detail(detection, "webcam", "frame")
        for detection in detect_hazards_from_webcam(camera_index)
    ]

    return [
        build_event(
            drone_id=SOURCE_DEFAULT_DRONE_IDS["webcam"],
            hazard_type=detection["type"],
            latitude=latitude,
            longitude=longitude,
            confidence=detection["confidence"],
            severity=detection["severity"],
            road=road,
            detail=detection.get("detail"),
        )
        for detection in vision_detections
    ]


def draw_detection_boxes(frame, detections):
    cv2, _ = load_cv_tools()
    colors = {
        "roe_deer": (42, 140, 255),
        "trash": (0, 220, 255),
        "road_crack": (0, 80, 255),
        "pothole": (255, 80, 0),
    }
    labels = {
        "roe_deer": "Deer",
        "trash": "Trash",
        "road_crack": "Crack",
        "pothole": "Pothole",
    }

    for detection in detections:
        bbox = detection.get("bbox")
        if not bbox:
            continue
        x, y, width, height = [int(value) for value in bbox]
        color = colors.get(detection.get("type"), (0, 255, 0))
        label = labels.get(detection.get("type"), detection.get("type", "Hazard"))
        confidence = int(float(detection.get("confidence", 0)) * 100)
        text = f"{label} {confidence}%"

        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 3)
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        text_width, text_height = text_size
        label_y = max(0, y - text_height - 10)
        cv2.rectangle(frame, (x, label_y), (x + text_width + 12, label_y + text_height + 10), color, -1)
        cv2.putText(
            frame,
            text,
            (x + 6, label_y + text_height + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


def webcam_monitor_loop():
    global WEBCAM_LATEST_JPEG
    cv2, _ = load_cv_tools()
    WEBCAM_MONITOR["running"] = True
    capture = None
    last_detection_at = 0

    while True:
        try:
            if capture is None or not capture.isOpened():
                capture = cv2.VideoCapture(WEBCAM_MONITOR["camera_index"], cv2.CAP_DSHOW)
                if not capture.isOpened():
                    capture.release()
                    capture = cv2.VideoCapture(WEBCAM_MONITOR["camera_index"])
                if not capture.isOpened():
                    raise ValueError(f"Could not open webcam index {WEBCAM_MONITOR['camera_index']}.")

            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                capture = None
                raise ValueError("Webcam opened, but no frame could be read.")

            now = time.monotonic()
            if now - last_detection_at >= WEBCAM_MONITOR["interval_seconds"]:
                vision_detections = [
                    append_source_detail(detection, "webcam", "frame")
                    for detection in detect_hazards_from_frame(frame)
                ]
                new_events = [
                    build_event(
                        drone_id=SOURCE_DEFAULT_DRONE_IDS["webcam"],
                        hazard_type=detection["type"],
                        latitude=37.5665,
                        longitude=126.9780,
                        confidence=detection["confidence"],
                        severity=detection["severity"],
                        road="CCTV road segment",
                        detail=detection.get("detail"),
                    )
                    for detection in vision_detections
                ]
                append_detections(new_events)
                WEBCAM_MONITOR["last_capture_at"] = utc_now()
                WEBCAM_MONITOR["last_accepted"] = len(new_events)
                WEBCAM_MONITOR["last_labels"] = [event["label"] for event in new_events]
                WEBCAM_MONITOR["last_detections"] = vision_detections
                last_detection_at = now

            preview_frame = draw_detection_boxes(frame.copy(), WEBCAM_MONITOR["last_detections"])
            ok, encoded = cv2.imencode(".jpg", preview_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                with WEBCAM_FRAME_LOCK:
                    WEBCAM_LATEST_JPEG = encoded.tobytes()
                WEBCAM_MONITOR["preview_ready"] = True

            WEBCAM_MONITOR["last_error"] = None
        except ValueError as error:
            WEBCAM_MONITOR["last_capture_at"] = utc_now()
            WEBCAM_MONITOR["last_error"] = str(error)
            WEBCAM_MONITOR["last_accepted"] = 0
            WEBCAM_MONITOR["last_labels"] = []
            WEBCAM_MONITOR["last_detections"] = []
            WEBCAM_MONITOR["preview_ready"] = False
            time.sleep(1)
            continue

        time.sleep(0.08)


def start_webcam_monitor():
    monitor = threading.Thread(target=webcam_monitor_loop, daemon=True)
    monitor.start()


class SafeRoadHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        parsed = urlparse(path)
        if parsed.path == "/":
            return str(STATIC_DIR / "index.html")
        if parsed.path.startswith("/static/"):
            return str(BASE_DIR / parsed.path.lstrip("/"))
        return str(STATIC_DIR / parsed.path.lstrip("/"))

    def send_json(self, payload, status=200):
        body = json.dumps(payload, default=json_default).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_jpeg(self, image_bytes, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Content-Length", str(len(image_bytes)))
        self.end_headers()
        self.wfile.write(image_bytes)

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw)

    def read_multipart_form(self):
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type"),
            },
        )
        return form

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/detections":
            detections = sorted(load_detections(), key=lambda item: item["created_at"], reverse=True)
            self.send_json({"detections": detections})
            return
        if parsed.path == "/api/v1/road-hazard-events":
            try:
                self.send_json(build_public_event_feed(parse_qs(parsed.query)))
            except ValueError as error:
                self.send_json({"error": str(error)}, status=400)
            return
        if parsed.path == "/api/health":
            self.send_json({"ok": True, "service": "safe-road", "time": utc_now()})
            return
        if parsed.path == "/api/vision/webcam/status":
            self.send_json(WEBCAM_MONITOR.copy())
            return
        if parsed.path == "/api/picoboard/status":
            self.send_json(pico_status())
            return
        if parsed.path == "/api/vision/webcam/frame.jpg":
            with WEBCAM_FRAME_LOCK:
                image_bytes = WEBCAM_LATEST_JPEG
            if image_bytes is None:
                self.send_json({"error": "Webcam preview frame is not ready yet."}, status=503)
                return
            self.send_jpeg(image_bytes)
            return
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/simulate":
            event = simulate_event()
            detections = load_detections()
            detections.append(event)
            save_detections(detections)
            send_picoboard_alert([event])
            self.send_json({"event": event}, status=201)
            return

        if parsed.path == "/api/picoboard/test":
            test_event = build_event(
                drone_id="picoboard-test",
                hazard_type="trash",
                latitude=37.5665,
                longitude=126.9780,
                confidence=0.99,
                severity="medium",
                road="PicoBoard demo",
                detail="Manual PicoBoard sound test",
            )
            sent = send_picoboard_alert([test_event])
            self.send_json({"sent": sent, "status": pico_status()}, status=200 if sent else 503)
            return

        if parsed.path == "/api/drone/frame":
            try:
                payload = self.read_json_body()
                drone_id = payload.get("drone_id", "unknown-drone")
                latitude = payload["latitude"]
                longitude = payload["longitude"]
                road = payload.get("road", "Unknown road")
                new_events = []
                for detection in payload.get("detections", []):
                    hazard_type = detection.get("type")
                    if not hazard_type:
                        continue
                    new_events.append(
                        build_event(
                            drone_id=drone_id,
                            hazard_type=hazard_type,
                            latitude=latitude,
                            longitude=longitude,
                            confidence=detection.get("confidence", 0.8),
                            severity=detection.get("severity"),
                            road=road,
                            detail=detection.get("detail"),
                        )
                    )
                detections = load_detections()
                detections.extend(new_events)
                save_detections(detections)
                send_picoboard_alert(new_events)
                self.send_json({"accepted": len(new_events), "events": new_events}, status=201)
            except (KeyError, json.JSONDecodeError, ValueError) as error:
                self.send_json({"error": str(error)}, status=400)
            return

        if parsed.path in {"/api/vision/image", "/api/vision/media"}:
            try:
                form = self.read_multipart_form()
                upload_field = form["media"] if "media" in form else form["image"]
                upload_bytes = upload_field.file.read()
                source_type = form.getfirst("source_type", "drone")
                if source_type not in SOURCE_DEFAULT_DRONE_IDS:
                    source_type = "drone"
                drone_id = form.getfirst("drone_id", SOURCE_DEFAULT_DRONE_IDS[source_type])
                latitude = float(form.getfirst("latitude", "37.5665"))
                longitude = float(form.getfirst("longitude", "126.9780"))
                road = form.getfirst("road", "Uploaded road media")
                if is_video_upload(upload_field):
                    vision_detections = detect_hazards_from_video(upload_bytes)
                    media_type = "video"
                else:
                    vision_detections = detect_hazards_from_image(upload_bytes)
                    media_type = "image"
                vision_detections = [
                    append_source_detail(detection, source_type, media_type)
                    for detection in vision_detections
                ]

                new_events = [
                    build_event(
                        drone_id=drone_id,
                        hazard_type=detection["type"],
                        latitude=latitude,
                        longitude=longitude,
                        confidence=detection["confidence"],
                        severity=detection["severity"],
                        road=road,
                        detail=detection.get("detail"),
                    )
                    for detection in vision_detections
                ]

                append_detections(new_events)
                self.send_json(
                    {
                        "accepted": len(new_events),
                        "events": new_events,
                        "message": f"OpenCV {media_type} analysis completed.",
                    },
                    status=201,
                )
            except (KeyError, ValueError) as error:
                self.send_json({"error": str(error)}, status=400)
            return

        if parsed.path == "/api/vision/webcam/capture":
            try:
                payload = self.read_json_body()
                camera_index = int(payload.get("camera_index", 0))
                latitude = float(payload.get("latitude", "37.5665"))
                longitude = float(payload.get("longitude", "126.9780"))
                road = payload.get("road", "Local webcam road")

                new_events = capture_webcam_events(camera_index, latitude, longitude, road)
                append_detections(new_events)
                self.send_json(
                    {
                        "accepted": len(new_events),
                        "events": new_events,
                        "message": "OpenCV webcam frame analysis completed.",
                    },
                    status=201,
                )
            except (ValueError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, status=400)
            return

        self.send_json({"error": "Not found"}, status=404)


def main():
    ensure_data_file()
    start_webcam_monitor()
    server = ThreadingHTTPServer(("0.0.0.0", 8000), SafeRoadHandler)
    print("Safe Road server running at http://localhost:8000")
    print("Webcam monitor running on camera index 0.")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()

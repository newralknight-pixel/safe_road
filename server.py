from __future__ import annotations

import base64
import ipaddress
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


ROOT = Path(__file__).resolve().parent
MODEL_SIZE = 640
WILDLIFE_CLASS_NAMES = [
    "Golden Eagle",
    "Pronghorn",
    "Bighorn Sheep",
    "American Bison",
    "Mule Deer",
    "Elk / Wapiti",
    "Coyote",
    "Grizzly Bear",
    "Gray Wolf",
    "Moose",
    "American Pika",
    "Swift Fox",
    "Mountain Lion",
    "North American River Otter",
    "American Black Bear",
    "Bald Eagle",
    "Red-tailed Hawk",
    "Osprey",
    "Greater Sage-Grouse",
    "Trumpeter Swan",
    "North American Beaver",
    "Common Raven",
    "Black-tailed Prairie Dog",
    "American Badger",
    "Bobcat",
    "Black-tailed Jackrabbit",
]
TRASH_CLASS_NAMES = ["glass", "paper", "plastic", "trash"]
POTHOLE_CLASS_NAMES = ["pothole"]
MODEL_CONFIGS = [
    {
        "name": "wildlife-north-american-yolo26s.onnx",
        "path": ROOT / "models" / "wildlife-north-american-yolo26s.onnx",
        "class_names": WILDLIFE_CLASS_NAMES,
        "target_class_ids": set(range(len(WILDLIFE_CLASS_NAMES))),
        "min_conf_by_class_id": {class_id: 0.65 for class_id in range(len(WILDLIFE_CLASS_NAMES))},
    },
    {
        "name": "trash-detection-yolo11n.onnx",
        "path": ROOT / "models" / "trash-detection-yolo11n.onnx",
        "class_names": TRASH_CLASS_NAMES,
        "target_class_ids": {0, 2, 3},
        "min_conf_by_class_id": {0: 0.80, 2: 0.80, 3: 0.85},
        "min_center_y_ratio": 0.45,
    },
    {
        "name": "pothole-yolov8s.onnx",
        "path": ROOT / "models" / "pothole-yolov8s.onnx",
        "class_names": POTHOLE_CLASS_NAMES,
        "target_class_ids": {0},
        "min_conf_by_class_id": {0: 0.60},
        "min_center_y_ratio": 0.45,
    },
]

app = Flask(__name__, static_folder=str(ROOT), static_url_path="")
CORS(app)

detectors = []
for config in MODEL_CONFIGS:
    session = ort.InferenceSession(str(config["path"]), providers=["CPUExecutionProvider"])
    detectors.append(
        {
            **config,
            "session": session,
            "input_name": session.get_inputs()[0].name,
            "output_name": session.get_outputs()[0].name,
        }
    )


def letterbox(image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(MODEL_SIZE / width, MODEL_SIZE / height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    pad_x = (MODEL_SIZE - new_width) // 2
    pad_y = (MODEL_SIZE - new_height) // 2

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((MODEL_SIZE, MODEL_SIZE, 3), 114, dtype=np.uint8)
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    return canvas, scale, pad_x, pad_y


def preprocess(image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    boxed, scale, pad_x, pad_y = letterbox(image)
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]
    return tensor, {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "width": image.shape[1],
        "height": image.shape[0],
    }


def rows_from_output(output: np.ndarray) -> np.ndarray:
    if output.ndim == 3:
        output = output[0]
    if output.ndim == 2 and output.shape[0] <= 16:
        output = output.T
    return output


def box_iou(a: dict, b: dict) -> float:
    ax2 = a["x"] + a["width"]
    ay2 = a["y"] + a["height"]
    bx2 = b["x"] + b["width"]
    by2 = b["y"] + b["height"]

    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a["width"] * a["height"] + b["width"] * b["height"] - inter
    return inter / max(union, 1e-6)


def nms(boxes: list[dict], iou_threshold: float) -> list[dict]:
    boxes = sorted(boxes, key=lambda item: item["score"], reverse=True)
    selected: list[dict] = []
    while boxes:
        current = boxes.pop(0)
        selected.append(current)
        boxes = [box for box in boxes if box_iou(current, box) <= iou_threshold]
    return selected


def decode(
    output: np.ndarray,
    meta: dict[str, float],
    conf: float,
    iou_threshold: float,
    class_names: list[str],
    target_class_ids: set[int],
    min_conf_by_class_id: dict[int, float] | None = None,
    min_center_y_ratio: float | None = None,
) -> list[dict]:
    rows = rows_from_output(output)
    boxes: list[dict] = []
    min_conf_by_class_id = min_conf_by_class_id or {}

    for row in rows:
        if row.shape[0] < 5:
            continue

        if row.shape[0] == 6:
            x1, y1, x2, y2 = map(float, row[:4])
            score = float(row[4])
            class_id = int(row[5])
            class_conf = max(conf, min_conf_by_class_id.get(class_id, conf))
            if score < class_conf or class_id not in target_class_ids:
                continue

            x1 = (x1 - meta["pad_x"]) / meta["scale"]
            y1 = (y1 - meta["pad_y"]) / meta["scale"]
            x2 = (x2 - meta["pad_x"]) / meta["scale"]
            y2 = (y2 - meta["pad_y"]) / meta["scale"]
        elif row.shape[0] == 5:
            class_id = 0
            score = float(row[4])
            class_conf = max(conf, min_conf_by_class_id.get(class_id, conf))
            if score < class_conf or class_id not in target_class_ids:
                continue

            cx, cy, width, height = map(float, row[:4])
            x1 = (cx - width / 2 - meta["pad_x"]) / meta["scale"]
            y1 = (cy - height / 2 - meta["pad_y"]) / meta["scale"]
            x2 = (cx + width / 2 - meta["pad_x"]) / meta["scale"]
            y2 = (cy + height / 2 - meta["pad_y"]) / meta["scale"]
        else:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])
            class_conf = max(conf, min_conf_by_class_id.get(class_id, conf))
            if score < class_conf or class_id not in target_class_ids:
                continue

            cx, cy, width, height = map(float, row[:4])
            x1 = (cx - width / 2 - meta["pad_x"]) / meta["scale"]
            y1 = (cy - height / 2 - meta["pad_y"]) / meta["scale"]
            x2 = (cx + width / 2 - meta["pad_x"]) / meta["scale"]
            y2 = (cy + height / 2 - meta["pad_y"]) / meta["scale"]

        x1 = max(0.0, min(float(meta["width"]), x1))
        y1 = max(0.0, min(float(meta["height"]), y1))
        x2 = max(0.0, min(float(meta["width"]), x2))
        y2 = max(0.0, min(float(meta["height"]), y2))

        if min_center_y_ratio is not None:
            center_y_ratio = ((y1 + y2) / 2) / max(float(meta["height"]), 1.0)
            if center_y_ratio < min_center_y_ratio:
                continue

        boxes.append(
            {
                "className": class_names[class_id],
                "score": score,
                "x": x1,
                "y": y1,
                "width": max(0.0, x2 - x1),
                "height": max(0.0, y2 - y1),
            }
        )

    return nms(boxes, iou_threshold)[:20]


def read_image() -> np.ndarray:
    if "image" in request.files:
        data = request.files["image"].read()
    else:
        payload = request.get_json(force=True, silent=True) or {}
        image_data = payload.get("image", "")
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        data = base64.b64decode(image_data)

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/health")
def health():
    return jsonify(
        {
            "ok": True,
            "model": ", ".join(detector["name"] for detector in detectors),
            "models": [
                {
                    "name": detector["name"],
                    "input": detector["input_name"],
                    "output": detector["output_name"],
                    "provider": detector["session"].get_providers()[0],
                }
                for detector in detectors
            ],
            "provider": detectors[0]["session"].get_providers()[0],
        }
    )


@app.get("/network")
def network():
    return jsonify(
        {
            "host": "0.0.0.0",
            "port": 8000,
            "localUrl": "http://127.0.0.1:8000",
            "phoneUrl": f"http://{get_lan_ip()}:8000",
        }
    )


@app.post("/detect")
def detect():
    started = time.perf_counter()
    conf = float(request.form.get("confidence", request.args.get("confidence", 0.45)))
    iou_threshold = float(request.form.get("iou", request.args.get("iou", 0.45)))
    image = read_image()
    tensor, meta = preprocess(image)
    detections = []
    for detector in detectors:
        output = detector["session"].run([detector["output_name"]], {detector["input_name"]: tensor})[0]
        detections.extend(
            decode(
                output,
                meta,
                conf,
                iou_threshold,
                detector["class_names"],
                detector["target_class_ids"],
                detector.get("min_conf_by_class_id"),
                detector.get("min_center_y_ratio"),
            )
        )
    detections = sorted(detections, key=lambda item: item["score"], reverse=True)[:20]
    return jsonify(
        {
            "detections": detections,
            "latencyMs": round((time.perf_counter() - started) * 1000, 2),
            "width": int(meta["width"]),
            "height": int(meta["height"]),
        }
    )


def get_lan_ip() -> str:
    hostnames = {socket.gethostname(), socket.getfqdn()}
    for hostname in hostnames:
        try:
            addresses = socket.getaddrinfo(hostname, None, socket.AF_INET)
        except OSError:
            continue
        for address in addresses:
            ip = address[4][0]
            parsed = ipaddress.ip_address(ip)
            if parsed.is_private and not parsed.is_loopback:
                return ip

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


if __name__ == "__main__":
    phone_url = f"http://{get_lan_ip()}:8000"
    print(f"Local browser: http://127.0.0.1:8000")
    print(f"Phone on same Wi-Fi: {phone_url}")
    app.run(host="0.0.0.0", port=8000, debug=False)

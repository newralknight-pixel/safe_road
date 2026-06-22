from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "pothole-yolov8s.onnx"
MODEL_SIZE = 640
CLASS_NAME = "pothole"

app = Flask(__name__, static_folder=str(ROOT), static_url_path="")
CORS(app)

session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


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


def decode(output: np.ndarray, meta: dict[str, float], conf: float, iou_threshold: float) -> list[dict]:
    rows = rows_from_output(output)
    boxes: list[dict] = []

    for row in rows:
        if row.shape[0] < 5:
            continue
        score = float(row[4]) if row.shape[0] == 5 else float(np.max(row[4:]))
        if score < conf:
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

        boxes.append(
            {
                "className": CLASS_NAME,
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
            "model": str(MODEL_PATH.name),
            "input": input_name,
            "output": output_name,
            "provider": session.get_providers()[0],
        }
    )


@app.post("/detect")
def detect():
    started = time.perf_counter()
    conf = float(request.form.get("confidence", request.args.get("confidence", 0.35)))
    iou_threshold = float(request.form.get("iou", request.args.get("iou", 0.45)))
    image = read_image()
    tensor, meta = preprocess(image)
    output = session.run([output_name], {input_name: tensor})[0]
    detections = decode(output, meta, conf, iou_threshold)
    return jsonify(
        {
            "detections": detections,
            "latencyMs": round((time.perf_counter() - started) * 1000, 2),
            "width": int(meta["width"]),
            "height": int(meta["height"]),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)

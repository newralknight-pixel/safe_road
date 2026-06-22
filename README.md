# Safe Road Detector

Browser-based road pothole detector using a local Flask + ONNX Runtime backend and a pretrained YOLOv8 model.

## Model

The app uses `models/pothole-yolov8s.onnx`, downloaded from:

https://huggingface.co/peterhdd/pothole-detection-yolov8

The model detects one class: `pothole`.

## Run

Create/install the virtual environment once:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the detector server from this folder:

```powershell
.\.venv\Scripts\python.exe server.py
```

Open:

```text
http://localhost:8000
```

Use **Start Webcam** for live camera detection, or use the upload button to test an image/video.

## GitHub

The target repository is currently empty:

https://github.com/newralknight-pixel/safe_road

After checking the app, push this local project to that repository.

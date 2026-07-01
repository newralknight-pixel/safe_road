# Safe Road Detector

Browser-based road watch detector using a local Flask + ONNX Runtime backend and pretrained YOLO models.

## Model

The app uses three ONNX models:

- `models/wildlife-north-american-yolo26s.onnx`, downloaded from https://huggingface.co/UWyo/wildlife-north-american-wildlife
- `models/trash-detection-yolo11n.onnx`, exported from https://huggingface.co/Alope/trash-detection-yolo11n
- `models/pothole-yolov8s.onnx`, an Ultralytics YOLOv8s pothole detector already included in this project

The app reports all 26 wildlife classes from the wildlife model. It also detects trash classes: `glass`, `paper`, `plastic`, and `trash`, plus `pothole`.

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

## Cellphone video upload

Start the server on the computer, then open the phone URL printed in the server window. It looks like:

```text
http://192.168.x.x:8000
```

The phone and computer must be on the same Wi-Fi network. On the phone, use **Record / Upload** to record or select a video. The browser plays the video locally and sends frames to the computer server for detection.

Note: live **Start Webcam** access from a phone may be blocked on plain HTTP. Use **Record / Upload** for the reliable phone workflow, or use the public tunnel option below for HTTPS camera access.

On Windows, you can also double-click:

```text
start_safe_road.bat
```

## Public temporary URL

To use the app from another laptop without moving this desktop, keep this desktop powered on and connected to the internet, then double-click:

```text
start_public_tunnel.bat
```

The script starts the local Flask server, downloads Cloudflare Tunnel if needed, and prints a public URL like:

```text
https://example-name.trycloudflare.com
```

Open that URL from the other laptop. Keep both command windows open while using the app. The URL is temporary and may change each time the tunnel is restarted.

Notes:

- Use Chrome or Edge on the remote laptop.
- Allow camera permission in the browser.
- Pico USB hardware only works on the computer it is plugged into.
- The desktop must stay on while the public URL is being used.

## GitHub

The target repository is currently empty:

https://github.com/newralknight-pixel/safe_road

After checking the app, push this local project to that repository.

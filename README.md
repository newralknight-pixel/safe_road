# Safe Road

Safe Road is a starter prototype for an AI road-safety system. It receives drone, roadside-camera, or blackbox video detection data, records road hazards, and shows alerts for police, road managers, navigation services, and nearby users.

## What it detects

- Road trash
- Roe deer / animals
- Road cracks
- Potholes

The current OpenCV media detector can analyze uploaded road images and videos for road-crack-like edge patterns, pothole-like dark road-surface candidates, and visible trash-like colored objects. Roe deer detection still needs a trained object model such as YOLO.

## YOLO CCTV model

The app automatically uses `models/cctv_best.pt` when it exists. Train this model with labeled CCTV data for:

- `deer`
- `trash`
- `crack`

Dataset layout:

```text
datasets/cctv_yolo/
  data.yaml
  images/train/
  images/val/
  labels/train/
  labels/val/
```

Capture CCTV training images:

```bash
python scripts/capture_cctv_images.py
```

Label those images in YOLO format, then train:

```bash
python scripts/train_yolo.py
```

After training, the best model is copied to:

```text
models/cctv_best.pt
```

Restart the server and CCTV detection will use YOLO first, then fall back to OpenCV if no YOLO model is available.

## Training data source

Safe Road is aligned with AI-Hub dataset 179, "Road obstacle/surface recognition videos (metropolitan area)", which includes road-surface labels such as potholes, repaired potholes, and cracks. The public AI-Hub page describes 100,000 pothole training images, 200,000 repaired-pothole images, and 300,000 crack segmentation images. The full dataset is about 1.01 TB and requires AI-Hub login/application before it can be downloaded for model training.

## Run on Ubuntu

```bash
cd /home/aa/safe_road
python3 safe_road_server.py
```

Then open:

```text
http://localhost:8000
```

## API

### Get detections

```bash
curl http://localhost:8000/api/detections
```

### Simulate one drone event

```bash
curl -X POST http://localhost:8000/api/simulate
```

### Send real drone detection data later

```bash
curl -X POST http://localhost:8000/api/drone/frame \
  -H "Content-Type: application/json" \
  -d '{
    "drone_id": "drone-01",
    "latitude": 37.5665,
    "longitude": 126.9780,
    "road": "Test Road",
    "detections": [
      {"type": "road_crack", "confidence": 0.91, "severity": "high"}
    ]
  }'
```

### Analyze an image or video with OpenCV

```bash
curl -X POST http://localhost:8000/api/vision/media \
  -F "media=@road.jpg" \
  -F "road=Test Road" \
  -F "latitude=37.5665" \
  -F "longitude=126.9780"
```

For video, use the same endpoint with a video file:

```bash
curl -X POST http://localhost:8000/api/vision/media \
  -F "media=@road.mp4" \
  -F "source_type=blackbox" \
  -F "road=Test Road" \
  -F "latitude=37.5665" \
  -F "longitude=126.9780"
```

Use `source_type=blackbox`, `source_type=drone`, or `source_type=roadside` to label where the uploaded media came from. Created alerts include communication targets such as road managers, navigation services, nearby users, and police when the hazard type requires it.

### Analyze a connected webcam

Open the dashboard at `http://localhost:8000`, use the **Webcam Detection** panel, and click **Start Camera**. Allow the browser camera permission, then click **Analyze Frame** to capture one still image from the connected webcam and run it through OpenCV.

Webcam captures are stored with `source_type=webcam` and `drone_id=local-webcam-01`.

### Export public road-hazard event feed

Inspired by the ABC_PROJECT road-hazard API contract, Safe Road can expose
privacy-safe aggregate events for navigation or road-management integrations.
The response does not include raw camera pixels, local file paths, or drone
camera images.

```bash
curl "http://localhost:8000/api/v1/road-hazard-events?limit=10&min_confidence=0.7"
```

Optional query filters:

- `limit=1..100`
- `hazard_type=trash|wild_animal|road_damage`
- `severity=low|medium|high`
- `min_confidence=0..1`

## Project structure

```text
safe_road/
  safe_road_server.py
  static/
    index.html
    style.css
    app.js
  data/
    detections.json
  README.md
```

## Next upgrades

- Connect a real drone camera stream
- Add YOLO animal detection model
- Add map display
- Send SMS/app/police-road-center alerts
- Store events in a database

# Safe Road

Safe Road is a starter prototype for an AI drone road-safety system. It receives drone-style detection data, records road hazards, and shows alerts for police, road managers, navigation services, and nearby users.

## What it detects

- Road trash
- Roe deer / animals
- Road cracks

The current OpenCV image detector can analyze uploaded road images for road-crack-like edge patterns and visible trash-like colored objects. Roe deer detection still needs a trained object model such as YOLO.

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

### Analyze an image with OpenCV

```bash
curl -X POST http://localhost:8000/api/vision/image \
  -F "image=@road.jpg" \
  -F "road=Test Road" \
  -F "latitude=37.5665" \
  -F "longitude=126.9780"
```

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

const eventsEl = document.querySelector("#events");
const simulateBtn = document.querySelector("#simulateBtn");
const refreshBtn = document.querySelector("#refreshBtn");
const totalCount = document.querySelector("#totalCount");
const highCount = document.querySelector("#highCount");
const policeCount = document.querySelector("#policeCount");
const lastUpdated = document.querySelector("#lastUpdated");
const visionForm = document.querySelector("#visionForm");
const visionStatus = document.querySelector("#visionStatus");
const dropZone = document.querySelector("#dropZone");
const dropZoneText = document.querySelector("#dropZoneText");
const mediaInput = document.querySelector("#mediaInput");
const mapCanvas = document.querySelector("#mapCanvas");
const mapList = document.querySelector("#mapList");
const webcamStatus = document.querySelector("#webcamStatus");
const webcamPreview = document.querySelector("#webcamPreview");
const webcamRecognition = document.querySelector("#webcamRecognition");
const webcamVideo = document.querySelector("#webcamVideo");
const webcamCanvas = document.querySelector("#webcamCanvas");
const webcamPlaceholder = document.querySelector("#webcamPlaceholder");
const webcamRoad = document.querySelector("#webcamRoad");
const webcamLatitude = document.querySelector("#webcamLatitude");
const webcamLongitude = document.querySelector("#webcamLongitude");
const webcamCameraIndex = document.querySelector("#webcamCameraIndex");
const startWebcamBtn = document.querySelector("#startWebcamBtn");
const captureWebcamBtn = document.querySelector("#captureWebcamBtn");
const serverWebcamBtn = document.querySelector("#serverWebcamBtn");
const stopWebcamBtn = document.querySelector("#stopWebcamBtn");
let webcamStream = null;
let connectedCameraBusy = false;
const PREVIEW_REFRESH_MS = 1000;

function isSupportedMedia(file) {
  return file && (file.type.startsWith("image/") || file.type.startsWith("video/"));
}

function setSelectedMedia(file) {
  if (!isSupportedMedia(file)) {
    visionStatus.textContent = "Choose an image or video file.";
    return;
  }

  const transfer = new DataTransfer();
  transfer.items.add(file);
  mediaInput.files = transfer.files;
  dropZoneText.textContent = file.name;
  visionStatus.textContent = "Ready";
}

function formatTime(value) {
  return new Date(value).toLocaleString();
}

function mapUrl(event) {
  return `https://www.openstreetmap.org/?mlat=${event.latitude}&mlon=${event.longitude}#map=16/${event.latitude}/${event.longitude}`;
}

function renderMap(detections) {
  const visibleEvents = detections.slice(0, 12);

  if (visibleEvents.length === 0) {
    mapCanvas.innerHTML = `<p class="empty">No locations yet.</p>`;
    mapList.innerHTML = "";
    return;
  }

  const latitudes = visibleEvents.map((event) => event.latitude);
  const longitudes = visibleEvents.map((event) => event.longitude);
  const minLat = Math.min(...latitudes);
  const maxLat = Math.max(...latitudes);
  const minLng = Math.min(...longitudes);
  const maxLng = Math.max(...longitudes);
  const latRange = Math.max(0.01, maxLat - minLat);
  const lngRange = Math.max(0.01, maxLng - minLng);

  mapCanvas.innerHTML = visibleEvents
    .map((event, index) => {
      const x = 8 + ((event.longitude - minLng) / lngRange) * 84;
      const y = 8 + (1 - (event.latitude - minLat) / latRange) * 84;
      return `
        <a
          class="map-marker ${event.severity}"
          href="${mapUrl(event)}"
          target="_blank"
          rel="noreferrer"
          style="left: ${x}%; top: ${y}%"
          title="${event.label} at ${event.road}"
          aria-label="${event.label} map location"
        >${index + 1}</a>
      `;
    })
    .join("");

  mapList.innerHTML = visibleEvents
    .map(
      (event, index) => `
        <a class="map-list-item" href="${mapUrl(event)}" target="_blank" rel="noreferrer">
          <span>${index + 1}</span>
          <strong>${event.label}</strong>
          <small>${event.road}</small>
        </a>
      `
    )
    .join("");
}

function renderEvents(detections) {
  totalCount.textContent = detections.length;
  highCount.textContent = detections.filter((event) => event.severity === "high").length;
  policeCount.textContent = detections.filter((event) => event.alert_targets.includes("police")).length;
  lastUpdated.textContent = `Updated ${new Date().toLocaleTimeString()}`;
  renderMap(detections);

  if (detections.length === 0) {
    eventsEl.innerHTML = `<p class="empty">No hazards yet. Simulate a drone detection to start.</p>`;
    return;
  }

  eventsEl.innerHTML = detections
    .map(
      (event) => `
        <article class="event ${event.severity}">
          <div>
            <h3>${event.label}</h3>
            <p>${event.message}</p>
            ${event.detail ? `<p class="detail">${event.detail}</p>` : ""}
            <div class="meta">
              <span class="badge">${event.severity.toUpperCase()}</span>
              <span class="badge">${Math.round(event.confidence * 100)}% confidence</span>
              <span class="badge">${event.drone_id}</span>
            </div>
            <div class="targets" aria-label="Alert targets">
              ${event.alert_targets.map((target) => `<span class="badge">${target.replace("_", " ")}</span>`).join("")}
            </div>
          </div>
          <div class="location">
            <strong>${event.road}</strong><br />
            ${event.latitude.toFixed(5)}, ${event.longitude.toFixed(5)}<br />
            ${formatTime(event.created_at)}<br />
            <a href="${mapUrl(event)}" target="_blank" rel="noreferrer">Open map</a>
          </div>
        </article>
      `
    )
    .join("");
}

async function loadEvents() {
  const response = await fetch("/api/detections");
  const payload = await response.json();
  renderEvents(payload.detections);
}

async function simulateDetection() {
  simulateBtn.disabled = true;
  try {
    await fetch("/api/simulate", { method: "POST" });
    await loadEvents();
  } finally {
    simulateBtn.disabled = false;
  }
}

function setWebcamControls(isRunning) {
  startWebcamBtn.disabled = isRunning;
  captureWebcamBtn.disabled = !isRunning;
  stopWebcamBtn.disabled = !isRunning;
  webcamPlaceholder.hidden = isRunning;
}

function getCameraErrorMessage(error) {
  const messages = {
    AbortError: "Camera startup was interrupted.",
    NotAllowedError: "Camera permission blocked. Allow camera access in the browser prompt.",
    NotFoundError: "No camera was found. Check the webcam connection.",
    NotReadableError: "Camera is busy or blocked by Windows. Close other camera apps and try again.",
    OverconstrainedError: "Camera settings are not supported. Trying default camera settings failed too.",
    SecurityError: "Camera access is blocked by browser security settings.",
  };
  return messages[error.name] || `Camera failed: ${error.name || error.message}`;
}

async function startWebcam() {
  await loadWebcamStatus();
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
  }
  webcamStream = null;
  webcamVideo.srcObject = null;
  webcamStatus.textContent = "CCTV idle";
  setWebcamControls(false);
}

function canvasToBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
      } else {
        reject(new Error("Could not capture webcam frame."));
      }
    }, "image/jpeg", 0.88);
  });
}

async function analyzeWebcamFrame() {
  if (!webcamStream || webcamVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    webcamStatus.textContent = "Camera frame is not ready yet.";
    return;
  }

  const width = webcamVideo.videoWidth || 1280;
  const height = webcamVideo.videoHeight || 720;
  webcamCanvas.width = width;
  webcamCanvas.height = height;
  webcamCanvas.getContext("2d").drawImage(webcamVideo, 0, 0, width, height);

  captureWebcamBtn.disabled = true;
  webcamStatus.textContent = "Analyzing frame...";
  try {
    const blob = await canvasToBlob(webcamCanvas);
    const formData = new FormData();
    formData.append("media", blob, `webcam-${Date.now()}.jpg`);
    formData.append("source_type", "webcam");
    formData.append("road", webcamRoad.value || "CCTV road segment");
    formData.append("latitude", webcamLatitude.value || "37.5665");
    formData.append("longitude", webcamLongitude.value || "126.9780");

    const response = await fetch("/api/vision/media", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Webcam analysis failed.");
    }
    webcamStatus.textContent = `Created ${payload.accepted} alert(s)`;
    await loadEvents();
  } catch (error) {
    webcamStatus.textContent = error.message;
  } finally {
    captureWebcamBtn.disabled = !webcamStream;
  }
}

async function analyzeConnectedCamera() {
  if (connectedCameraBusy) {
    return;
  }

  connectedCameraBusy = true;
  startWebcamBtn.disabled = true;
  serverWebcamBtn.disabled = true;
  webcamStatus.textContent = "Analyzing CCTV camera...";
  try {
    const response = await fetch("/api/vision/webcam/capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera_index: Number(webcamCameraIndex.value || 0),
        road: webcamRoad.value || "CCTV road segment",
        latitude: webcamLatitude.value || "37.5665",
        longitude: webcamLongitude.value || "126.9780",
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "CCTV camera analysis failed.");
    }
    webcamStatus.textContent = `CCTV running. Created ${payload.accepted} alert(s).`;
    await loadEvents();
  } catch (error) {
    webcamStatus.textContent = `CCTV retrying: ${error.message}`;
  } finally {
    connectedCameraBusy = false;
    startWebcamBtn.disabled = false;
    serverWebcamBtn.disabled = false;
  }
}

async function loadWebcamStatus() {
  try {
    const response = await fetch("/api/vision/webcam/status");
    const status = await response.json();
    if (status.last_error) {
      webcamStatus.textContent = `Server CCTV retrying: ${status.last_error}`;
      webcamRecognition.textContent = "CCTV connected, recognition retrying";
      return;
    }
    const accepted = status.last_accepted ?? 0;
    webcamStatus.textContent = `Server CCTV running. Last capture created ${accepted} alert(s).`;
    if (status.last_labels?.length) {
      webcamRecognition.textContent = `CCTV recognized: ${status.last_labels.join(", ")}`;
    } else if (status.preview_ready) {
      webcamRecognition.textContent = "CCTV connected. No deer, trash, or crack recognized in the last frame.";
    } else {
      webcamRecognition.textContent = "CCTV connecting...";
    }
  } catch (error) {
    webcamStatus.textContent = `CCTV status unavailable: ${error.message}`;
    webcamRecognition.textContent = "Recognition status unavailable";
  }
}

function refreshWebcamPreview() {
  const cameraIndex = Number(webcamCameraIndex.value || 0);
  webcamPreview.src = `/api/vision/webcam/frame.jpg?camera_index=${cameraIndex}&t=${Date.now()}`;
}

webcamPreview.addEventListener("load", () => {
  webcamPlaceholder.hidden = true;
});

webcamPreview.addEventListener("error", () => {
  webcamPlaceholder.hidden = false;
  webcamPlaceholder.textContent = "CCTV preview unavailable. Server will keep retrying.";
});

simulateBtn.addEventListener("click", simulateDetection);
refreshBtn.addEventListener("click", loadEvents);
startWebcamBtn.addEventListener("click", startWebcam);
captureWebcamBtn.addEventListener("click", analyzeWebcamFrame);
serverWebcamBtn.addEventListener("click", analyzeConnectedCamera);
stopWebcamBtn.addEventListener("click", stopWebcam);
dropZone.addEventListener("click", () => mediaInput.click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    mediaInput.click();
  }
});
dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});
dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("drag-over");
  setSelectedMedia(event.dataTransfer.files[0]);
});
mediaInput.addEventListener("change", () => {
  setSelectedMedia(mediaInput.files[0]);
});
visionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(visionForm);
  visionStatus.textContent = "Analyzing...";
  try {
    const response = await fetch("/api/vision/media", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Media analysis failed.");
    }
    visionStatus.textContent = `Created ${payload.accepted} alert(s)`;
    await loadEvents();
  } catch (error) {
    visionStatus.textContent = error.message;
  }
});
loadEvents();
setInterval(loadEvents, 10000);
loadWebcamStatus();
setInterval(loadWebcamStatus, 10000);
refreshWebcamPreview();
setInterval(refreshWebcamPreview, PREVIEW_REFRESH_MS);

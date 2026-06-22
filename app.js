const CLASS_NAME = "pothole";

const state = {
  running: false,
  source: null,
  stream: null,
  animationId: null,
  frame: 0,
  events: [],
  lastEventAt: 0,
  health: null,
};

const els = {
  video: document.querySelector("#video"),
  image: document.querySelector("#image"),
  overlay: document.querySelector("#overlay"),
  empty: document.querySelector("#emptyState"),
  cameraBtn: document.querySelector("#cameraBtn"),
  stopBtn: document.querySelector("#stopBtn"),
  snapshotBtn: document.querySelector("#snapshotBtn"),
  fileInput: document.querySelector("#fileInput"),
  confidence: document.querySelector("#confidence"),
  confidenceValue: document.querySelector("#confidenceValue"),
  iou: document.querySelector("#iou"),
  iouValue: document.querySelector("#iouValue"),
  modelLed: document.querySelector("#modelLed"),
  modelState: document.querySelector("#modelState"),
  modelDetail: document.querySelector("#modelDetail"),
  currentCount: document.querySelector("#currentCount"),
  bestScore: document.querySelector("#bestScore"),
  latency: document.querySelector("#latency"),
  eventCount: document.querySelector("#eventCount"),
  eventList: document.querySelector("#eventList"),
  eventTable: document.querySelector("#eventTable"),
  clearEventsBtn: document.querySelector("#clearEventsBtn"),
  riskBadge: document.querySelector("#riskBadge"),
  sourceTitle: document.querySelector("#sourceTitle"),
  sourceMeta: document.querySelector("#sourceMeta"),
  frameSkip: document.querySelector("#frameSkip"),
  eventCooldown: document.querySelector("#eventCooldown"),
  reloadModelBtn: document.querySelector("#reloadModelBtn"),
};

const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });
const overlayCtx = els.overlay.getContext("2d");

function setModelStatus(kind, title, detail) {
  els.modelLed.className = `led ${kind}`;
  els.modelState.textContent = title;
  els.modelDetail.textContent = detail || "";
}

async function checkBackend() {
  setModelStatus("loading", "Connecting backend", "Flask / ONNX Runtime");
  const response = await fetch("/health", { cache: "no-store" });
  if (!response.ok) throw new Error(`Backend returned ${response.status}`);
  state.health = await response.json();
  setModelStatus("ready", "Backend ready", `${state.health.model} · ${state.health.provider}`);
}

function setView(name) {
  document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
  document.querySelector(`#${name}View`).classList.add("active");
  document.querySelectorAll(".nav-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.view === name);
  });
  const titles = {
    detect: ["Live Road Detection", "Run local webcam inference with a pretrained YOLOv8 pothole model."],
    events: ["Detection Events", "Logged pothole detections from webcam, images, and videos."],
    settings: ["Detector Settings", "Adjust thresholds, logging, and backend model path."],
  };
  document.querySelector("#viewTitle").textContent = titles[name][0];
  document.querySelector("#viewSubtitle").textContent = titles[name][1];
}

function updateThresholdLabels() {
  els.confidenceValue.textContent = `${els.confidence.value}%`;
  els.iouValue.textContent = `${els.iou.value}%`;
}

function sourceElement() {
  if (state.source === "image") return els.image;
  if (state.source === "video" || state.source === "webcam") return els.video;
  return null;
}

function mediaDimensions(media) {
  if (media === els.image) return { width: media.naturalWidth || 1, height: media.naturalHeight || 1 };
  return { width: media.videoWidth || 1, height: media.videoHeight || 1 };
}

function resizeOverlay() {
  const rect = document.querySelector("#stage").getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  els.overlay.width = Math.max(1, Math.round(rect.width * dpr));
  els.overlay.height = Math.max(1, Math.round(rect.height * dpr));
  overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function displayedMediaRect(media) {
  const stage = document.querySelector("#stage").getBoundingClientRect();
  const { width, height } = mediaDimensions(media);
  const scale = Math.min(stage.width / width, stage.height / height);
  const drawWidth = width * scale;
  const drawHeight = height * scale;
  return {
    x: (stage.width - drawWidth) / 2,
    y: (stage.height - drawHeight) / 2,
    width: drawWidth,
    height: drawHeight,
    scale,
  };
}

async function mediaToBlob(media) {
  const { width, height } = mediaDimensions(media);
  captureCanvas.width = width;
  captureCanvas.height = height;
  captureCtx.drawImage(media, 0, 0, width, height);
  return new Promise((resolve) => captureCanvas.toBlob(resolve, "image/jpeg", 0.88));
}

async function detectOnce() {
  const media = sourceElement();
  if (!media) return [];
  if (media === els.video && (!media.videoWidth || media.readyState < 2)) return [];
  if (media === els.image && !media.naturalWidth) return [];

  const blob = await mediaToBlob(media);
  if (!blob) return [];

  const form = new FormData();
  form.append("image", blob, "frame.jpg");
  form.append("confidence", String(Number(els.confidence.value) / 100));
  form.append("iou", String(Number(els.iou.value) / 100));

  const response = await fetch("/detect", { method: "POST", body: form });
  if (!response.ok) throw new Error(`Detection failed: ${response.status}`);
  const result = await response.json();
  const detections = result.detections || [];
  drawDetections(media, detections);
  updateMetrics(detections, result.latencyMs || 0);
  logEvent(detections);
  return detections;
}

function drawDetections(media, detections) {
  resizeOverlay();
  const rect = displayedMediaRect(media);
  const stage = document.querySelector("#stage").getBoundingClientRect();
  overlayCtx.clearRect(0, 0, stage.width, stage.height);
  overlayCtx.lineWidth = 3;
  overlayCtx.font = "700 14px Segoe UI, Arial";
  overlayCtx.textBaseline = "top";

  for (const detection of detections) {
    const x = rect.x + detection.x * rect.scale;
    const y = rect.y + detection.y * rect.scale;
    const width = detection.width * rect.scale;
    const height = detection.height * rect.scale;
    const label = `${detection.className || CLASS_NAME} ${Math.round(detection.score * 100)}%`;
    const labelWidth = overlayCtx.measureText(label).width + 14;
    const labelY = Math.max(0, y - 26);

    overlayCtx.strokeStyle = "#facc15";
    overlayCtx.fillStyle = "rgba(250, 204, 21, 0.10)";
    overlayCtx.strokeRect(x, y, width, height);
    overlayCtx.fillRect(x, y, width, height);
    overlayCtx.fillStyle = "#facc15";
    overlayCtx.fillRect(x, labelY, labelWidth, 24);
    overlayCtx.fillStyle = "#161616";
    overlayCtx.fillText(label, x + 7, labelY + 4);
  }
}

function updateMetrics(detections, latencyMs) {
  const best = detections.reduce((max, item) => Math.max(max, item.score), 0);
  els.currentCount.textContent = detections.length;
  els.bestScore.textContent = `${Math.round(best * 100)}%`;
  els.latency.textContent = `${Math.round(latencyMs)}ms`;
  els.riskBadge.textContent = detections.length ? "Pothole" : "Clear";
  els.riskBadge.className = `badge ${detections.length ? "danger" : "ok"}`;
}

function logEvent(detections) {
  if (!detections.length) return;
  const now = Date.now();
  if (now - state.lastEventAt < Number(els.eventCooldown.value)) return;
  state.lastEventAt = now;
  const best = detections[0];
  state.events.unshift({
    time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    source: state.source || "unknown",
    className: best.className || CLASS_NAME,
    confidence: Math.round(best.score * 100),
    box: `${Math.round(best.x)}, ${Math.round(best.y)}, ${Math.round(best.width)}×${Math.round(best.height)}`,
  });
  state.events = state.events.slice(0, 200);
  renderEvents();
}

function renderEvents() {
  els.eventCount.textContent = state.events.length;
  els.eventList.innerHTML = state.events.slice(0, 6).map((event) => `
    <article class="event-card">
      <div>
        <strong>${event.className} · ${event.confidence}%</strong>
        <span>${event.time} · ${event.source}</span>
      </div>
      <span>${event.box}</span>
    </article>
  `).join("") || `<article class="event-card"><div><strong>No events yet</strong><span>Detections will appear here.</span></div></article>`;

  els.eventTable.innerHTML = state.events.map((event) => `
    <tr>
      <td>${event.time}</td>
      <td>${event.source}</td>
      <td>${event.className}</td>
      <td>${event.confidence}%</td>
      <td>${event.box}</td>
    </tr>
  `).join("");
}

async function detectionLoop() {
  if (!state.running) return;
  state.frame += 1;
  if (state.frame % Number(els.frameSkip.value) === 0) {
    try {
      await detectOnce();
    } catch (error) {
      console.error(error);
      setModelStatus("error", "Detection error", error.message);
      stopDetection(false);
      return;
    }
  }
  state.animationId = requestAnimationFrame(detectionLoop);
}

async function startWebcam() {
  await checkBackend();
  stopDetection(false);
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
    audio: false,
  });
  state.stream = stream;
  state.source = "webcam";
  els.video.srcObject = stream;
  els.video.style.display = "block";
  els.image.style.display = "none";
  els.empty.style.display = "none";
  await els.video.play();
  state.running = true;
  els.sourceTitle.textContent = "Webcam";
  els.sourceMeta.textContent = `${els.video.videoWidth || "camera"} × ${els.video.videoHeight || "stream"}`;
  detectionLoop();
}

function stopDetection(clearSource = true) {
  state.running = false;
  if (state.animationId) cancelAnimationFrame(state.animationId);
  state.animationId = null;
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  els.video.pause();
  els.video.srcObject = null;
  if (clearSource) {
    state.source = null;
    els.video.removeAttribute("src");
    els.video.style.display = "none";
    els.image.style.display = "none";
    els.empty.style.display = "grid";
    overlayCtx.clearRect(0, 0, els.overlay.width, els.overlay.height);
    els.sourceTitle.textContent = "Camera Source";
    els.sourceMeta.textContent = "Waiting for webcam or file";
    updateMetrics([], 0);
    els.riskBadge.textContent = "Idle";
    els.riskBadge.className = "badge idle";
  }
}

async function loadFile(file) {
  await checkBackend();
  stopDetection(false);
  const url = URL.createObjectURL(file);
  els.empty.style.display = "none";
  if (file.type.startsWith("video/")) {
    state.source = "video";
    els.image.style.display = "none";
    els.video.style.display = "block";
    els.video.src = url;
    els.video.loop = true;
    await els.video.play();
    state.running = true;
    els.sourceTitle.textContent = file.name;
    els.sourceMeta.textContent = "Local video";
    detectionLoop();
  } else {
    state.source = "image";
    els.video.style.display = "none";
    els.image.style.display = "block";
    els.image.src = url;
    els.sourceTitle.textContent = file.name;
    els.sourceMeta.textContent = "Local image";
    await new Promise((resolve) => {
      els.image.onload = resolve;
    });
    await detectOnce();
  }
}

function saveSnapshot() {
  const media = sourceElement();
  if (!media) return;
  const out = document.createElement("canvas");
  const { width, height } = mediaDimensions(media);
  out.width = width;
  out.height = height;
  const outCtx = out.getContext("2d");
  outCtx.drawImage(media, 0, 0, width, height);
  const link = document.createElement("a");
  link.download = `safe-road-${Date.now()}.png`;
  link.href = out.toDataURL("image/png");
  link.click();
}

document.querySelectorAll(".nav-item").forEach((button) => {
  button.addEventListener("click", () => setView(button.dataset.view));
});

els.cameraBtn.addEventListener("click", () => {
  startWebcam().catch((error) => {
    console.error(error);
    setModelStatus("error", "Webcam error", error.message);
  });
});
els.stopBtn.addEventListener("click", () => stopDetection(true));
els.snapshotBtn.addEventListener("click", saveSnapshot);
els.fileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) loadFile(file).catch((error) => setModelStatus("error", "File error", error.message));
});
els.clearEventsBtn.addEventListener("click", () => {
  state.events = [];
  renderEvents();
});
els.confidence.addEventListener("input", updateThresholdLabels);
els.iou.addEventListener("input", updateThresholdLabels);
els.reloadModelBtn.addEventListener("click", () => checkBackend().catch((error) => setModelStatus("error", "Backend error", error.message)));
window.addEventListener("resize", resizeOverlay);

updateThresholdLabels();
renderEvents();
resizeOverlay();
checkBackend().catch((error) => setModelStatus("error", "Backend offline", error.message));

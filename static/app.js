const eventsEl = document.querySelector("#events");
const simulateBtn = document.querySelector("#simulateBtn");
const refreshBtn = document.querySelector("#refreshBtn");
const totalCount = document.querySelector("#totalCount");
const highCount = document.querySelector("#highCount");
const policeCount = document.querySelector("#policeCount");
const lastUpdated = document.querySelector("#lastUpdated");
const visionForm = document.querySelector("#visionForm");
const visionStatus = document.querySelector("#visionStatus");

function formatTime(value) {
  return new Date(value).toLocaleString();
}

function renderEvents(detections) {
  totalCount.textContent = detections.length;
  highCount.textContent = detections.filter((event) => event.severity === "high").length;
  policeCount.textContent = detections.filter((event) => event.alert_targets.includes("police")).length;
  lastUpdated.textContent = `Updated ${new Date().toLocaleTimeString()}`;

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
            ${formatTime(event.created_at)}
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

simulateBtn.addEventListener("click", simulateDetection);
refreshBtn.addEventListener("click", loadEvents);
visionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const formData = new FormData(visionForm);
  visionStatus.textContent = "Analyzing...";
  try {
    const response = await fetch("/api/vision/image", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.error || "Image analysis failed.");
    }
    visionStatus.textContent = `Created ${payload.accepted} alert(s)`;
    await loadEvents();
  } catch (error) {
    visionStatus.textContent = error.message;
  }
});
loadEvents();
setInterval(loadEvents, 10000);

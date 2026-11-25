"use strict";

const CLASS_LABELS = {
  person: "Человек",
  person_without_helmet: "Без каски",
  vehicle: "Транспорт",
  smoke_fire: "Дым/огонь",
  spill: "Разлив",
  obstacle: "Препятствие",
};

const $ = (id) => document.getElementById(id);

async function api(path, options) {
  const response = await fetch(path, options);
  const type = response.headers.get("content-type") || "";
  const payload = type.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "object" && payload.detail ? payload.detail : response.statusText;
    throw new Error(detail);
  }
  return payload;
}

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleTimeString("ru-RU");
}

function setState(state, error) {
  const badge = $("state-badge");
  badge.textContent = state;
  badge.className = "badge " + state;
  const banner = $("error-banner");
  if (error) {
    banner.textContent = "Ошибка: " + error;
    banner.classList.remove("hidden");
  } else {
    banner.classList.add("hidden");
  }
  const overlay = $("video-overlay");
  if (state === "running" || state === "starting") overlay.classList.add("hidden");
  else overlay.classList.remove("hidden");
}

async function refreshStatus() {
  try {
    const s = await api("/status");
    $("fps").textContent = Number(s.fps || 0).toFixed(1);
    $("latency").textContent = Number(s.latency_ms || 0).toFixed(0);
    $("frames").textContent = s.frames_processed || 0;
    $("events-count").textContent = s.events_created || 0;
    $("source-info").textContent = s.source || "(источник не задан)";
    $("mode-info").textContent = s.mode || "—";
    if (s.model) {
      const file = s.model.model_file ? " / " + s.model.model_file : "";
      $("model-info").textContent = (s.model.backend || "—") + file;
    }
    setState(s.state, s.error);
  } catch (err) {
    setState("offline", err.message);
  }
}

async function refreshMetrics() {
  try {
    const m = await api("/metrics");
    $("latency-avg").textContent = Number(m.latency_avg_ms || 0).toFixed(0);
    $("latency-p95").textContent = Number(m.latency_p95_ms || 0).toFixed(0);
    $("frames-dropped").textContent = m.frames_dropped || 0;
    $("errors").textContent = m.errors || 0;
  } catch (err) { /* статус уже покажет ошибку */ }
}

function buildRow(event) {
  const tr = document.createElement("tr");
  const cells = [
    fmtTime(event.timestamp),
    event.event_type,
    CLASS_LABELS[event.object_class] || event.object_class,
    Number(event.confidence || 0).toFixed(2),
  ];
  for (const text of cells) {
    const td = document.createElement("td");
    td.textContent = text;
    tr.appendChild(td);
  }

  const sevTd = document.createElement("td");
  const sev = document.createElement("span");
  sev.className = "sev sev-" + (event.severity || "medium");
  sev.textContent = event.severity || "—";
  sevTd.appendChild(sev);
  tr.appendChild(sevTd);

  const frameTd = document.createElement("td");
  if (event.frame_path) {
    const link = document.createElement("a");
    link.className = "frame-link";
    link.href = "#";
    link.textContent = "открыть";
    link.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      openModal(event);
    });
    frameTd.appendChild(link);
  } else {
    frameTd.textContent = "—";
  }
  tr.appendChild(frameTd);

  tr.addEventListener("click", () => openModal(event));
  return tr;
}

async function refreshEvents() {
  const filter = $("event-filter").value;
  const query = filter ? `?object_class=${encodeURIComponent(filter)}&limit=100` : "?limit=100";
  try {
    const events = await api("/events" + query);
    const body = $("events-body");
    body.innerHTML = "";
    if (!events.length) {
      const tr = document.createElement("tr");
      tr.className = "empty";
      const td = document.createElement("td");
      td.colSpan = 6;
      td.textContent = "Событий пока нет";
      tr.appendChild(td);
      body.appendChild(tr);
      return;
    }
    for (const event of events) body.appendChild(buildRow(event));
  } catch (err) { /* пропускаем */ }
}

function openModal(event) {
  if (!event.frame_path) return;
  $("modal-img").src = `/events/${encodeURIComponent(event.id)}/frame`;
  $("modal-meta").textContent =
    `${event.id} · ${event.event_type} · ${event.object_class} · conf ${Number(event.confidence).toFixed(2)} · ${event.timestamp}`;
  $("modal").classList.remove("hidden");
}

function collectSource() {
  const body = { source_type: $("source-type").value };
  const path = $("source-path").value.trim();
  if (path) body.path = path;
  const camera = $("source-camera").value;
  if (camera !== "") body.camera_index = Number(camera);
  return body;
}

function reloadStream() {
  const img = $("stream");
  img.src = "/stream.mjpg?t=" + Date.now();
}

async function doStart() {
  try {
    const result = await api("/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectSource()),
    });
    $("action-message").textContent = result.message;
    reloadStream();
  } catch (err) {
    $("action-message").textContent = "Ошибка запуска: " + err.message;
  }
  refreshStatus();
}

async function doStop() {
  try {
    const result = await api("/stop", { method: "POST" });
    $("action-message").textContent = result.message;
  } catch (err) {
    $("action-message").textContent = "Ошибка остановки: " + err.message;
  }
  refreshStatus();
}

async function applyConfidence() {
  const value = Number($("conf-range").value);
  try {
    const result = await api("/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confidence_threshold: value }),
    });
    $("action-message").textContent = result.message;
  } catch (err) {
    $("action-message").textContent = "Ошибка конфигурации: " + err.message;
  }
}

async function clearEvents() {
  try {
    const result = await api("/events", { method: "DELETE" });
    $("action-message").textContent = result.message;
  } catch (err) {
    $("action-message").textContent = "Ошибка очистки: " + err.message;
  }
  refreshEvents();
  refreshStatus();
}

function bind() {
  $("btn-start").addEventListener("click", doStart);
  $("btn-stop").addEventListener("click", doStop);
  $("btn-apply-conf").addEventListener("click", applyConfidence);
  $("btn-clear").addEventListener("click", clearEvents);
  $("event-filter").addEventListener("change", refreshEvents);
  $("conf-range").addEventListener("input", (e) => {
    $("conf-value").textContent = Number(e.target.value).toFixed(2);
  });
  $("conf-range").addEventListener("change", applyConfidence);
  $("modal-close").addEventListener("click", () => $("modal").classList.add("hidden"));
  $("modal").addEventListener("click", (e) => {
    if (e.target.id === "modal") $("modal").classList.add("hidden");
  });
}

async function init() {
  bind();
  try {
    const status = await api("/status");
    $("conf-range").value = 0.35;
    $("source-type").value = status.source_type || "file";
    if (status.source) $("source-path").value = status.source;
  } catch (err) { /* ignore */ }
  await refreshStatus();
  await refreshMetrics();
  await refreshEvents();
  setInterval(refreshStatus, 1000);
  setInterval(refreshMetrics, 1500);
  setInterval(refreshEvents, 1500);
}

document.addEventListener("DOMContentLoaded", init);

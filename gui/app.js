/* FIRMAMENT viewer — vanilla JS, no dependencies. Attach/detach leaves no residue. */
"use strict";
const $ = (id) => document.getElementById(id);
let ws = null, meta = null, lastFrame = null, currentLayer = "elevation";
let zoom = 1, panX = 0, panY = 0, dragging = null;

/* ---------- colormap (viridis-ish) ---------- */
function colormap(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(255 * Math.min(1, Math.max(0, 4 * t - 1.6)));
  const g = Math.round(255 * Math.min(1, Math.max(0, 1.6 * t)));
  const b = Math.round(255 * Math.min(1, Math.max(0, 1.2 - Math.abs(2.4 * t - 1.2)) + 0.25 * (1 - t)));
  return [r, g, b];
}

/* ---------- run picker ---------- */
async function loadRuns() {
  const runs = await (await fetch("/api/runs")).json();
  const tb = $("run-table").querySelector("tbody");
  tb.innerHTML = "";
  for (const r of runs) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${r.run_id}</td><td class="st-${r.status}">${r.status}</td>` +
      `<td>${r.tick ?? "—"}</td><td>${r.tps ?? "—"}</td><td>${r.last_milestone ?? "—"}</td>` +
      `<td>${r.touched ? "⚠ TOUCHED" : ""}</td><td>${r.snapshots}</td>` +
      `<td>${r.status === "running" ? "<button>attach</button>" : "<span class=hint>resumable (firmament resume)</span>"}</td>`;
    if (r.status === "running") tr.querySelector("button").onclick = () => {
      if (r.mine) attach(r.run_id);
      else window.location.href = `http://${location.hostname}:${r.port}/`;   // that sim's own bridge
    };
    tb.appendChild(tr);
  }
}

async function attach(runId) {
  meta = await (await fetch("/api/meta")).json();
  $("run-name").textContent = meta.run_id;
  $("touched-banner").classList.toggle("hidden", !meta.touched);
  $("picker").classList.add("hidden");
  $("main").classList.remove("hidden");
  buildLayerSelect();
  connectWS();
  refreshEvents(); refreshLineage(); refreshRunMgr(); buildMetricSelect(); buildEventParams();
}

function detach() {
  if (ws) { ws.close(); ws = null; }
  $("main").classList.add("hidden");
  $("picker").classList.remove("hidden");
  loadRuns();
}

/* ---------- layers & map ---------- */
function buildLayerSelect() {
  const sel = $("layer-select");
  sel.innerHTML = "";
  const layers = ["elevation", "water", "temp", "light", "polymer_density", "compartments", "lineage"];
  for (const s of meta.species) layers.push("species:" + s);
  meta.motifs.forEach((m, i) => layers.push("motif:" + i));
  for (const l of layers) {
    const o = document.createElement("option");
    o.value = l; o.textContent = l.startsWith("motif:") ? "motif:" + meta.motifs[+l.split(":")[1]] : l;
    sel.appendChild(o);
  }
  sel.value = currentLayer;
  sel.onchange = () => {
    currentLayer = sel.value;
    if (ws) ws.send(JSON.stringify({ layers: ["elevation", "water", "temp", "light", "polymer_density", currentLayer] }));
  };
}

function connectWS() {
  ws = new WebSocket(`ws://${location.host}/ws/state`);
  ws.onmessage = (ev) => { lastFrame = JSON.parse(ev.data); drawMap(); drawClocks(); };
  ws.onopen = () => ws.send(JSON.stringify({ layers: ["elevation", "water", "temp", "light", "polymer_density", currentLayer] }));
  ws.onclose = () => { if (ws) setTimeout(connectWS, 2000); };
}

function decodeLayer(l) {
  const bytes = Uint8Array.from(atob(l.data), (c) => c.charCodeAt(0));
  return { arr: new Float32Array(bytes.buffer), h: l.shape[0], w: l.shape[1], min: l.min, max: l.max };
}

function drawMap() {
  if (!lastFrame) return;
  const l = lastFrame.layers[currentLayer] || lastFrame.layers.elevation;
  if (!l) return;
  const { arr, h, w, min, max } = decodeLayer(l);
  const cv = $("map"), ctx = cv.getContext("2d");
  const img = ctx.createImageData(w, h);
  const range = max - min || 1;
  for (let i = 0; i < h * w; i++) {
    const [r, g, b] = colormap((arr[i] - min) / range);
    img.data[4 * i] = r; img.data[4 * i + 1] = g; img.data[4 * i + 2] = b; img.data[4 * i + 3] = 255;
  }
  const off = new OffscreenCanvas(w, h);
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.translate(panX, panY); ctx.scale(zoom, zoom);
  ctx.drawImage(off, 0, 0, cv.width, cv.height);
  ctx.restore();
}

function mapCell(ev) {
  if (!lastFrame) return null;
  const l = lastFrame.layers[currentLayer] || lastFrame.layers.elevation;
  if (!l) return null;
  const cv = $("map"), rect = cv.getBoundingClientRect();
  const cx = (ev.clientX - rect.left - panX) / zoom, cy = (ev.clientY - rect.top - panY) / zoom;
  const { h, w } = { h: l.shape[0], w: l.shape[1] };
  const ds = meta.world[0] / h;                    // downsample factor
  const x = Math.floor((cx / cv.width) * w) * ds, y = Math.floor((cy / cv.height) * h) * ds;
  if (x < 0 || y < 0 || x >= meta.world[1] || y >= meta.world[0]) return null;
  return { x, y };
}

function setupMap() {
  const cv = $("map");
  cv.addEventListener("wheel", (e) => {
    e.preventDefault();
    const f = e.deltaY < 0 ? 1.2 : 1 / 1.2;
    zoom = Math.max(1, Math.min(32, zoom * f));
    drawMap();
  });
  cv.addEventListener("mousedown", (e) => { dragging = { x: e.clientX - panX, y: e.clientY - panY, moved: false }; });
  window.addEventListener("mouseup", () => { dragging = null; });
  cv.addEventListener("mousemove", (e) => {
    if (dragging) {
      panX = e.clientX - dragging.x; panY = e.clientY - dragging.y; dragging.moved = true; drawMap();
    }
    const c = mapCell(e);
    if (c && lastFrame) {
      const l = lastFrame.layers[currentLayer];
      if (l) {
        const d = decodeLayer(l);
        const ds = meta.world[0] / d.h;
        const v = d.arr[Math.floor(c.y / ds) * d.w + Math.floor(c.x / ds)];
        $("hover-readout").textContent = `(${c.x},${c.y}) ${currentLayer}=${v.toPrecision(4)}`;
      }
    }
  });
  cv.addEventListener("click", async (e) => {
    if (dragging && dragging.moved) return;
    const c = mapCell(e);
    if (c) inspectCell(c.x, c.y);
  });
}

/* ---------- clocks & time control ---------- */
function drawClocks() {
  if (!lastFrame) return;
  const s = lastFrame.sim_seconds, d = Math.floor(s / 86400), y = Math.floor(d / 365);
  $("clocks").textContent = `tick ${lastFrame.tick} · ${y}y ${d % 365}d ${Math.floor((s % 86400) / 3600)}h · mode ${lastFrame.mode}`;
  $("tps").textContent = `${(lastFrame.tps || 0).toFixed(1)} tps`;
  $("mode-label").textContent = lastFrame.mode;
}

function setupTime() {
  document.querySelectorAll("#time-panel button[data-mode]").forEach((b) => {
    b.onclick = () => fetch("/api/time", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ mode: b.dataset.mode, steps: 1 }) });
  });
  $("tps-slider").onchange = () => fetch("/api/time", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ mode: "throttled", target_tps: +$("tps-slider").value }) });
}

/* ---------- inspector ---------- */
async function inspectCell(x, y) {
  const d = await (await fetch(`/api/inspect/cell?x=${x}&y=${y}`)).json();
  const sp = Object.entries(d.species).filter(([, v]) => v > 0).map(([k, v]) => `${k}:${v}`).join(" ");
  $("inspector").innerHTML =
    `<b>cell (${x},${y})</b> elev ${d.elevation.toFixed(2)} m · water ${d.water_depth.toFixed(3)} m · ` +
    `T ${d.temp.map((t) => t.toFixed(1)).join("/")} K · light ${d.light.toFixed(0)} · ` +
    `compartment ${d.compartment} · membrane L ${d.membrane_L}<br><span class=mono>${sp || "(no species)"}</span>` +
    `<br>polymers: ${d.polymers.map((p) => `<a href="#" onclick="inspectPolymer(${p});return false">#${p}</a>`).join(" ") || "none"}`;
}

async function inspectPolymer(id) {
  const r = await fetch(`/api/inspect/polymer?id=${id}`);
  if (!r.ok) { $("inspector").innerHTML += `<br>polymer ${id}: not alive`; return; }
  const d = await r.json();
  let seq = d.sequence;
  for (const m of [...d.motifs].sort((a, b) => b.pos - a.pos)) {
    const start = m.pos * 2, end = (m.pos + m.len) * 2;  // "M1" = 2 chars per monomer
    seq = seq.slice(0, start * 1) + `<mark title="${m.name}">` + seq.slice(start, end) + "</mark>" + seq.slice(end);
  }
  $("inspector").innerHTML =
    `<b>polymer #${d.id}</b> len ${d.length} · born tick ${d.born_tick} · state ${d.state} · muts ${d.mutations}` +
    `<br>parent #${d.parent} · lineage to seed: ${d.lineage_path.slice(0, 12).join(" → ")}${d.lineage_path.length > 12 ? " …" : ""}` +
    `<br><span class=mono>${seq}</span>`;
}
window.inspectPolymer = inspectPolymer;

/* ---------- charts ---------- */
async function buildMetricSelect() {
  const data = await (await fetch(`/api/runs/${meta.run_id}/metrics?tail=1`)).json();
  const sel = $("metric-select");
  sel.innerHTML = "";
  for (const k of Object.keys(data).filter((k) => k !== "tick")) {
    const o = document.createElement("option");
    o.value = o.textContent = k;
    if (k === "n_polymers") o.selected = true;
    sel.appendChild(o);
  }
}

async function drawChart() {
  const names = [...$("metric-select").selectedOptions].map((o) => o.value);
  if (!names.length) return;
  const data = await (await fetch(`/api/runs/${meta.run_id}/metrics?names=${names.join(",")}`)).json();
  const cv = $("chart"), ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const ticks = data.tick || [];
  if (!ticks.length) { ctx.fillStyle = "#888"; ctx.fillText("no metrics yet", 20, 20); return; }
  const log = $("log-scale").checked;
  const colors = ["#6cf", "#fc6", "#6f9", "#f6c", "#c9f", "#ff9"];
  names.forEach((name, ni) => {
    const raw = (data[name] || []).map((v) => (typeof v === "number" ? v : 0));
    const ys = log ? raw.map((v) => Math.log10(Math.max(1e-12, v))) : raw;
    const ymin = Math.min(...ys), ymax = Math.max(...ys), yr = ymax - ymin || 1;
    ctx.strokeStyle = colors[ni % colors.length];
    ctx.beginPath();
    ys.forEach((v, i) => {
      const x = (i / (ys.length - 1 || 1)) * (cv.width - 40) + 30;
      const y = cv.height - 15 - ((v - ymin) / yr) * (cv.height - 30);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = colors[ni % colors.length];
    ctx.fillText(`${name} [${raw[raw.length - 1]}]`, 35, 12 + 12 * ni);
  });
}

/* ---------- lineage tree ---------- */
async function refreshLineage() {
  const rows = await (await fetch("/api/lineage/tree")).json();
  const byParent = new Map();
  for (const r of rows) {
    if (!byParent.has(r.parent)) byParent.set(r.parent, []);
    byParent.get(r.parent).push(r);
  }
  const render = (pid, depth) => {
    const kids = byParent.get(pid) || [];
    if (!kids.length || depth > 8) return "";
    return kids.slice(0, 50).map((k) =>
      `<details ${depth < 2 ? "open" : ""}><summary>#${k.id} len ${k.len} <a href="#" onclick="inspectPolymer(${k.id});return false">inspect</a></summary>${render(k.id, depth + 1)}</details>`).join("");
  };
  const roots = rows.filter((r) => !rows.some((q) => q.id === r.parent));
  $("lineage").innerHTML = rows.length
    ? roots.slice(0, 20).map((r) => `<details open><summary>#${r.id} len ${r.len}</summary>${render(r.id, 1)}</details>`).join("")
    : "<span class=hint>no living polymers</span>";
}

/* ---------- events timeline ---------- */
async function refreshEvents() {
  const evs = await (await fetch("/api/events?tail=100")).json();
  $("events").innerHTML = evs.length ? evs.reverse().map((e) => {
    const cell = e.cell ? ` <a href="#" onclick="inspectCell(${e.cell[0]},${e.cell[1]});return false">@${e.cell}</a>` : "";
    return `<div class="ev ev-${e.component}">t${e.tick} <b>${e.event}</b>${cell}</div>`;
  }).join("") : "<span class=hint>no events yet</span>";
}
window.inspectCell = inspectCell;

/* ---------- god console ---------- */
const EVENT_PARAMS = {
  rain: { cx: 128, cy: 128, radius: 60, intensity_mm: 10 },
  drought: { cx: 128, cy: 128, radius: 60, strength: 0.5 },
  flood: { cx: 128, cy: 128, radius: 60, rise_m: 0.5 },
  earthquake: { cx: 128, cy: 128, radius: 40, magnitude_m: 1.0 },
  volcano: { cx: 128, cy: 128, radius: 20, heat_J_m2: 5e7, cone_m: 3, mineral_counts: 10000 },
  meteor: { cx: 128, cy: 128, energy_J: 1e12, crater_m: 2, dust: 0.3 },
  climate: { multiplier: 0.9 },
  solar: { multiplier: 1.05 },
};

function buildEventParams() {
  const t = $("event-type").value;
  $("event-params").innerHTML = Object.entries(EVENT_PARAMS[t]).map(
    ([k, v]) => `<label>${k} <input data-param="${k}" value="${v}" size="8"></label>`).join(" ");
}

function setupConsole() {
  $("event-type").onchange = buildEventParams;
  $("event-submit").onclick = async () => {
    const t = $("event-type").value;
    const body = {};
    document.querySelectorAll("#event-params input").forEach((i) => (body[i.dataset.param] = +i.value));
    const r = await (await fetch(`/api/console/${t}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) })).json();
    if (r.confirm_token) {
      $("confirm-box").classList.remove("hidden");
      $("confirm-text").textContent = `Trigger ${t} with ${JSON.stringify(body)}?`;
      $("confirm-yes").onclick = async () => {
        body.confirm_token = r.confirm_token;
        await fetch(`/api/console/${t}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
        $("confirm-box").classList.add("hidden");
        setTimeout(refreshEvents, 500);
      };
      $("confirm-no").onclick = () => $("confirm-box").classList.add("hidden");
    }
  };
  $("dev-toggle").onchange = async () => {
    const on = $("dev-toggle").checked;
    await fetch("/api/developer/enable", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ on }) });
    $("dev-forms").classList.toggle("hidden", !on);
    document.body.classList.toggle("dev-mode", on);
  };
  $("dev-submit").onclick = async () => {
    const body = JSON.parse($("dev-json").value);
    const action = body.action;
    delete body.action;
    await fetch(`/api/developer/${action}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    const m = await (await fetch("/api/meta")).json();
    $("touched-banner").classList.toggle("hidden", !m.touched);
  };
}

/* ---------- run manager ---------- */
async function refreshRunMgr() {
  const runs = await (await fetch("/api/runs")).json();
  $("runmgr").innerHTML = runs.map((r) =>
    `<div>${r.run_id} <span class=hint>${r.snapshots} snaps</span> ` +
    `<button onclick="forkRun('${r.run_id}')">fork</button> ` +
    `<button onclick="compareRun('${r.run_id}')">compare</button></div>`).join("");
}
window.forkRun = async (rid) => {
  const r = await (await fetch(`/api/runs/${rid}/fork`, { method: "POST", headers: { "content-type": "application/json" }, body: "{}" })).json();
  alert(`forked → ${r.run_id}`);
  refreshRunMgr();
};
window.compareRun = async (rid) => {
  const names = [...$("metric-select").selectedOptions].map((o) => o.value);
  const mine = await (await fetch(`/api/runs/${meta.run_id}/metrics?names=${names.join(",")}`)).json();
  const theirs = await (await fetch(`/api/runs/${rid}/metrics?names=${names.join(",")}`)).json();
  const cv = $("chart"), ctx = cv.getContext("2d");
  drawChart().then(() => {
    const name = names[0];
    const raw = (theirs[name] || []);
    if (!raw.length) return;
    const ymin = Math.min(...raw), yr = Math.max(...raw) - ymin || 1;
    ctx.strokeStyle = "#f44"; ctx.setLineDash([4, 3]); ctx.beginPath();
    raw.forEach((v, i) => {
      const x = (i / (raw.length - 1 || 1)) * (cv.width - 40) + 30;
      const y = cv.height - 15 - ((v - ymin) / yr) * (cv.height - 30);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#f44"; ctx.fillText(`${rid}:${name}`, 35, cv.height - 4);
  });
};

/* ---------- boot ---------- */
$("back").onclick = detach;
$("chart-refresh").onclick = drawChart;
$("lineage-refresh").onclick = refreshLineage;
$("events-refresh").onclick = refreshEvents;
setupMap(); setupTime(); setupConsole();
loadRuns();
setInterval(() => { if (!$("main").classList.contains("hidden")) refreshEvents(); }, 10000);

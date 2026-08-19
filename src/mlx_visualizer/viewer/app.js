/* MLX Visualizer — WebGL2 viewer.
 *
 * Rendering strategy: every tensor lives on one layer of a single R32F
 * TEXTURE_2D_ARRAY; all panels are drawn with ONE instanced draw call
 * (colormapping happens in the fragment shader via a LUT atlas), so the
 * scene cost is independent of how many matrices are on screen. Labels
 * and architecture edges are lightweight DOM/SVG positioned in screen
 * space each frame.
 */
"use strict";

// ---------------------------------------------------------------- utilities
const $ = (id) => document.getElementById(id);

function fmt(x) {
  if (x === null || x === undefined || !isFinite(x)) return String(x);
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return x.toExponential(3);
  return +x.toFixed(4) + "";
}

// ------------------------------------------------------------------ colormaps
const CMAP_STOPS = {
  viridis: [[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],[31,158,137],[53,183,121],[109,205,89],[180,222,44],[253,231,37]],
  magma:   [[0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],[229,80,100],[251,135,97],[254,194,135],[252,253,191]],
  turbo:   [[48,18,59],[70,107,227],[40,187,235],[32,229,181],[122,252,82],[218,227,25],[253,158,9],[239,71,17],[122,4,3]],
  coolwarm:[[59,76,192],[144,178,254],[220,220,220],[245,156,125],[180,4,38]],
  gray:    [[0,0,0],[255,255,255]],
};
const CMAP_NAMES = Object.keys(CMAP_STOPS);
const cmapIndex = (name) => Math.max(0, CMAP_NAMES.indexOf(name));

function buildLUT() {
  const rows = CMAP_NAMES.length, data = new Uint8Array(256 * rows * 4);
  CMAP_NAMES.forEach((name, r) => {
    const stops = CMAP_STOPS[name];
    for (let i = 0; i < 256; i++) {
      const t = (i / 255) * (stops.length - 1);
      const k = Math.min(Math.floor(t), stops.length - 2), f = t - k;
      const o = (r * 256 + i) * 4;
      for (let c = 0; c < 3; c++)
        data[o + c] = Math.round(stops[k][c] * (1 - f) + stops[k + 1][c] * f);
      data[o + 3] = 255;
    }
  });
  return { data, rows };
}

// ------------------------------------------------------------------- GL setup
const canvas = $("gl");
const metricCanvas = $("metrics");
const metric2d = metricCanvas.getContext("2d");
const gl = canvas.getContext("webgl2", { antialias: false, alpha: false, preserveDrawingBuffer: true });
if (!gl) $("hud").textContent = "WebGL2 unavailable";

const VS = `#version 300 es
layout(location=0) in vec4 iRect;
layout(location=1) in vec4 iTex;   // layer, uMax, vMax, cmapRow
layout(location=2) in vec2 iRange; // vmin, vmax
uniform vec2 uPan; uniform float uZoom; uniform vec2 uView;
uniform float uOutline; // 1 = LINE_LOOP border pass (perimeter vertex order)
out vec2 vUV;
flat out float vLayer; flat out float vCmap; flat out vec2 vRange;
void main(){
  vec2 corner;
  if (uOutline > 0.5) {
    // perimeter order: (0,0) (1,0) (1,1) (0,1)
    corner = vec2(
      (gl_VertexID == 1 || gl_VertexID == 2) ? 1.0 : 0.0,
      (gl_VertexID >= 2) ? 1.0 : 0.0);
  } else {
    corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1));
  }
  vec2 world = iRect.xy + corner * iRect.zw;
  vec2 screen = world * uZoom + uPan;
  vec2 clip = screen / uView * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  vUV = corner * iTex.yz;
  vLayer = iTex.x; vCmap = iTex.w; vRange = iRange;
}`;

const FS = `#version 300 es
precision highp float; precision highp sampler2DArray;
uniform sampler2DArray uData; uniform sampler2D uLUT; uniform float uLUTRows;
uniform float uSolid;
in vec2 vUV;
flat in float vLayer; flat in float vCmap; flat in vec2 vRange;
out vec4 frag;
void main(){
  if (uSolid > 0.5) { frag = vec4(0.35, 0.42, 0.58, 1.0); return; }
  float v = texture(uData, vec3(vUV, vLayer)).r;
  if (isnan(v)) { frag = vec4(1.0, 0.0, 0.8, 1.0); return; }
  float t = clamp((v - vRange.x) / max(vRange.y - vRange.x, 1e-30), 0.0, 1.0);
  float row = (vCmap + 0.5) / uLUTRows;
  frag = texture(uLUT, vec2(t * 0.99609375 + 0.001953125, row));
}`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
  throw new Error(gl.getProgramInfoLog(prog));
gl.useProgram(prog);
const U = {};
for (const n of ["uPan","uZoom","uView","uData","uLUT","uLUTRows","uSolid","uOutline"])
  U[n] = gl.getUniformLocation(prog, n);

// Colormap LUT atlas.
const lut = buildLUT();
const lutTex = gl.createTexture();
gl.activeTexture(gl.TEXTURE1);
gl.bindTexture(gl.TEXTURE_2D, lutTex);
gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, lut.rows, 0, gl.RGBA, gl.UNSIGNED_BYTE, lut.data);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.uniform1i(U.uLUT, 1);
gl.uniform1f(U.uLUTRows, lut.rows);

// Texture array pool — one layer per tensor, grown geometrically.
let LAYER_SIZE = 1024;               // matches server maxSide (hello updates it)
let dataTex = null, layerCapacity = 0;
const freeLayers = [];

function allocTextureArray(capacity) {
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
  gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.R32F, LAYER_SIZE, LAYER_SIZE, capacity);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

function uploadLayer(layer, w, h, f32) {
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, dataTex);
  gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, layer, w, h, 1, gl.RED, gl.FLOAT, f32);
}

function allocLayer() {
  if (freeLayers.length) return freeLayers.pop();
  const maxLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS);
  const next = Math.min(Math.max(8, layerCapacity * 2), maxLayers);
  if (next <= layerCapacity) return -1;
  const old = dataTex;
  dataTex = allocTextureArray(next);
  const used = layerCapacity;
  layerCapacity = next;
  if (old) gl.deleteTexture(old);
  // Re-upload resident panels into the fresh storage.
  for (const p of panels.values())
    if (p.layer >= 0 && p.image) uploadLayer(p.layer, p.texW, p.texH, p.image);
  for (let i = used; i < next; i++) freeLayers.push(i);
  return freeLayers.pop();
}
gl.uniform1i(U.uData, 0);

// Instance buffer: 10 floats per panel (rect4 + tex4 + range2).
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
const instBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
const STRIDE = 10 * 4;
gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 4, gl.FLOAT, false, STRIDE, 0);  gl.vertexAttribDivisor(0, 1);
gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, STRIDE, 16); gl.vertexAttribDivisor(1, 1);
gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 2, gl.FLOAT, false, STRIDE, 32); gl.vertexAttribDivisor(2, 1);
let instData = new Float32Array(0);
let instCount = 0, instDirty = true;

// ---------------------------------------------------------------------- state
const panels = new Map();   // id -> panel
let panelOrder = [];        // ids in stable draw/layout order
let edges = [];             // [srcName, dstName]
let mode = "grid";
let paused = false;
let cmapOverride = "";
const cam = { x: 40, y: 40, zoom: 1 };
let needsLayout = true, needsFit = true;
let bytesReceived = 0, frames = 0, lastHud = performance.now(), fps = 0;

// -------------------------------------------------------------------- layout
function panelReady(p) {
  return p.kind === "metric" ? p.historyData.length > 0 : p.texW > 0;
}

function panelDisplaySize(p) {
  if (p.kind === "metric") return { w: 280, h: 120 };
  const sz = (n) => Math.max(36, Math.min(380, 30 + 46 * Math.log10(Math.max(n, 1) + 1)));
  if (p.rows <= 1) return { w: Math.max(140, sz(p.cols) * 1.6), h: 26 }; // vector strip
  let w = sz(p.cols), h = sz(p.rows);
  const ar = w / h;
  if (ar > 8) w = h * 8;
  if (ar < 1 / 8) h = w * 8;
  return { w, h };
}

function layoutGrid() {
  const ids = panelOrder.filter((id) => panelReady(panels.get(id)));
  const items = ids.map((id) => ({ p: panels.get(id), ...panelDisplaySize(panels.get(id)) }));
  items.sort((a, b) => b.h - a.h);
  const total = items.reduce((s, it) => s + (it.w + 26) * (it.h + 46), 0);
  const maxW = Math.max(420, Math.sqrt(total) * 1.35);
  let x = 0, y = 0, shelf = 0;
  for (const it of items) {
    if (x > 0 && x + it.w > maxW) { x = 0; y += shelf + 46; shelf = 0; }
    it.p.x = x; it.p.y = y; it.p.dw = it.w; it.p.dh = it.h;
    x += it.w + 26;
    shelf = Math.max(shelf, it.h);
  }
}

function layoutGraph() {
  const byName = new Map();
  for (const p of panels.values()) byName.set(p.name, p);
  const adj = new Map(), indeg = new Map();
  for (const p of panels.values()) { adj.set(p.name, []); indeg.set(p.name, 0); }
  for (const [s, d] of edges) {
    if (byName.has(s) && byName.has(d)) {
      adj.get(s).push(d);
      indeg.set(d, indeg.get(d) + 1);
    }
  }
  // Longest-path layering (Kahn order).
  const depth = new Map(), queue = [];
  for (const [n, deg] of indeg) { depth.set(n, 0); if (deg === 0) queue.push(n); }
  const indegWork = new Map(indeg);
  while (queue.length) {
    const n = queue.shift();
    for (const m of adj.get(n)) {
      depth.set(m, Math.max(depth.get(m), depth.get(n) + 1));
      indegWork.set(m, indegWork.get(m) - 1);
      if (indegWork.get(m) === 0) queue.push(m);
    }
  }
  // Panels that participate in no edge (e.g. biases) sit in the same
  // column as their group's connected representative instead of piling
  // into column 0.
  const connected = new Set();
  for (const [s, d] of edges) { connected.add(s); connected.add(d); }
  const groupDepth = new Map();
  for (const p of panels.values())
    if (p.group && connected.has(p.name))
      groupDepth.set(p.group, Math.max(groupDepth.get(p.group) ?? 0, depth.get(p.name) || 0));
  const cols = new Map();
  for (const id of panelOrder) {
    const p = panels.get(id);
    if (!panelReady(p)) continue;
    const d = connected.has(p.name)
      ? (depth.get(p.name) || 0)
      : (groupDepth.get(p.group) ?? 0);
    if (!cols.has(d)) cols.set(d, []);
    cols.get(d).push(p);
  }
  let x = 0;
  for (const d of [...cols.keys()].sort((a, b) => a - b)) {
    const column = cols.get(d);
    let colW = 0, y = 0;
    const heights = column.map((p) => panelDisplaySize(p));
    const totalH = heights.reduce((s, sz) => s + sz.h + 56, 0);
    y = -totalH / 2;
    column.forEach((p, i) => {
      const sz = heights[i];
      p.x = x; p.y = y; p.dw = sz.w; p.dh = sz.h;
      y += sz.h + 56;
      colW = Math.max(colW, sz.w);
    });
    x += colW + 150;
  }
}

function relayout() {
  (mode === "grid" ? layoutGrid : layoutGraph)();
  instDirty = true;
  needsLayout = false;
  syncLabels();
}

// ------------------------------------------------------------ DOM labels/edges
const labelsEl = $("labels"), edgesEl = $("edges");
const LABEL_HIDE_ZOOM = 0.18;
const LABEL_FADE_ZOOM = 0.48;
const LABEL_MIN_SCALE = 0.4;
const LABEL_MAX_SCALE = 1.75;
const LABEL_READABLE_SCALE = 0.48;
edgesEl.innerHTML =
  '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" ' +
  'markerWidth="7" markerHeight="7" orient="auto-start-reverse">' +
  '<path d="M0,0 L10,5 L0,10 z" fill="#5ad0b1" fill-opacity="0.6"/></marker></defs>';
const edgePaths = new Map();

function breakableLabel(text) {
  return text.replaceAll("/", "/\u200b").replaceAll(".", ".\u200b");
}

function labelStyleForZoom(zoom) {
  const scale = Math.max(
    LABEL_MIN_SCALE,
    Math.min(LABEL_MAX_SCALE, Math.pow(zoom, 0.65)),
  );
  const fade = Math.max(0, Math.min(
    1,
    (zoom - LABEL_HIDE_ZOOM) / (LABEL_FADE_ZOOM - LABEL_HIDE_ZOOM),
  ));
  // Smoothstep keeps labels from popping in as the user zooms.
  const opacity = fade * fade * (3 - 2 * fade);
  return { scale, opacity, hidden: zoom <= LABEL_HIDE_ZOOM };
}

function syncLabels() {
  for (const p of panels.values()) {
    if (!p.labelEl) {
      p.labelEl = document.createElement("div");
      p.labelEl.className = "label";
      p.labelEl.innerHTML =
        '<div><span class="name"></span> <span class="group"></span></div>' +
        '<div class="meta"></div><div class="stats"></div>';
      labelsEl.appendChild(p.labelEl);
    }
    const groupText = p.group ? "· " + p.group : "";
    let metaText, statsText;
    if (p.kind === "metric") {
      const finite = p.historyData.filter(Number.isFinite);
      const lo = finite.length ? Math.min(...finite) : NaN;
      const hi = finite.length ? Math.max(...finite) : NaN;
      metaText = `${p.historyData.length}/${p.history} samples`;
      statsText = `latest ${fmt(p.latest)}  range ${fmt(lo)}…${fmt(hi)}`;
    } else {
      metaText = p.shape ? "(" + p.shape.join("×") + ")" +
        (p.texW < p.cols || p.texH < p.rows ? "  LOD " + p.texW + "×" + p.texH : "") : "";
      statsText = p.shape
        ? `min ${fmt(p.vmin)}  max ${fmt(p.vmax)}  μ ${fmt(p.mean)}  σ ${fmt(p.std)}` +
          (p.nan ? `  NaN ${p.nan}` : "")
        : "";
    }
    const signature = `${p.name}\0${groupText}\0${metaText}\0${statsText}`;
    if (p.labelSignature === signature) continue;
    p.labelEl.querySelector(".name").textContent = breakableLabel(p.name);
    p.labelEl.querySelector(".group").textContent = groupText;
    p.labelEl.querySelector(".meta").textContent = metaText;
    p.labelEl.querySelector(".stats").textContent = statsText;
    p.labelSignature = signature;
    p.labelSizeDirty = true;
  }
}

function positionOverlay() {
  const labelStyle = labelStyleForZoom(cam.zoom);
  const viewWidth = canvas.clientWidth, viewHeight = canvas.clientHeight;
  for (const p of panels.values()) {
    if (!p.labelEl) continue;
    const sx = p.x * cam.zoom + cam.x, sy = p.y * cam.zoom + cam.y;
    const panelWidth = p.dw * cam.zoom, panelHeight = p.dh * cam.zoom;
    const visibleWidth = Math.max(
      0, Math.min(sx + panelWidth, viewWidth) - Math.max(sx, 0),
    );
    const visibleHeight = Math.max(
      0, Math.min(sy + panelHeight, viewHeight) - Math.max(sy, 0),
    );
    // Do not pin the label of a mostly off-screen panel over its neighbor.
    // It appears as soon as the user pans that panel substantially into view.
    const panelVisible = visibleWidth >= panelWidth * 0.9 &&
      visibleHeight >= panelHeight * 0.5;
    if (!panelReady(p) || labelStyle.hidden || !panelVisible) {
      p.labelEl.style.display = "none";
      continue;
    }
    p.labelEl.style.display = "";
    const layoutMode = cam.zoom >= 2.25 ? "detailed" : "compact";
    const labelWidth = Math.max(48, panelWidth / labelStyle.scale);
    const layoutKey = `${layoutMode}:${labelWidth.toFixed(2)}`;
    if (p.labelLayoutKey !== layoutKey) {
      p.labelEl.classList.toggle("compact", layoutMode === "compact");
      p.labelEl.style.width = `${labelWidth}px`;
      p.labelLayoutKey = layoutKey;
      p.labelSizeDirty = true;
    }
    if (p.labelSizeDirty || !p.labelWidth || !p.labelHeight) {
      p.labelWidth = p.labelEl.scrollWidth;
      p.labelHeight = p.labelEl.scrollHeight;
      p.labelSizeDirty = false;
    }
    // Compact labels show the complete, wrapped name while fitting inside
    // the row gap. Details return once zoom provides enough room for them.
    const rowGap = (mode === "grid" ? 46 : 56) * cam.zoom;
    const verticalFit = layoutMode === "compact"
      ? Math.max(0.05, (rowGap - 8) / Math.max(p.labelHeight, 1))
      : labelStyle.scale;
    const scale = Math.min(labelStyle.scale, verticalFit);
    if (scale < LABEL_READABLE_SCALE) {
      p.labelEl.style.display = "none";
      continue;
    }
    const renderedWidth = p.labelWidth * scale;
    const renderedHeight = p.labelHeight * scale;
    const gap = 8 * scale;
    const labelX = Math.max(8, Math.min(sx, viewWidth - renderedWidth - 8));
    let labelY = sy - renderedHeight - gap;
    if (labelY < 8) labelY = sy + panelHeight + gap;
    labelY = Math.max(8, Math.min(labelY, viewHeight - renderedHeight - 8));
    p.labelEl.style.opacity = labelStyle.opacity.toFixed(3);
    p.labelEl.style.transform = `translate(${labelX}px, ${labelY}px) scale(${scale})`;
  }
  const wantEdges = mode === "graph";
  edgesEl.style.display = wantEdges ? "" : "none";
  if (!wantEdges) return;
  const byName = new Map();
  for (const p of panels.values()) byName.set(p.name, p);
  const seen = new Set();
  for (const [s, d] of edges) {
    const a = byName.get(s), b = byName.get(d);
    if (!a || !b || !panelReady(a) || !panelReady(b)) continue;
    const key = s + "→" + d;
    seen.add(key);
    let path = edgePaths.get(key);
    if (!path) {
      path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      edgesEl.appendChild(path);
      edgePaths.set(key, path);
    }
    const x1 = (a.x + a.dw) * cam.zoom + cam.x, y1 = (a.y + a.dh / 2) * cam.zoom + cam.y;
    const x2 = b.x * cam.zoom + cam.x, y2 = (b.y + b.dh / 2) * cam.zoom + cam.y;
    const dx = Math.max(40, (x2 - x1) / 2);
    path.setAttribute("d", `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`);
  }
  for (const [key, path] of edgePaths)
    if (!seen.has(key)) { path.remove(); edgePaths.delete(key); }
}

// ------------------------------------------------------------------ instances
function rebuildInstances() {
  const live = panelOrder.map((id) => panels.get(id))
    .filter((p) => p.kind !== "metric" && p.texW > 0 && p.layer >= 0);
  instCount = live.length;
  if (instData.length < instCount * 10) instData = new Float32Array(instCount * 10 * 2);
  live.forEach((p, i) => {
    const o = i * 10;
    instData[o] = p.x; instData[o + 1] = p.y; instData[o + 2] = p.dw; instData[o + 3] = p.dh;
    instData[o + 4] = p.layer;
    instData[o + 5] = p.texW / LAYER_SIZE; instData[o + 6] = p.texH / LAYER_SIZE;
    instData[o + 7] = cmapIndex(cmapOverride || p.cmap);
    instData[o + 8] = p.vmin; instData[o + 9] = p.vmax;
  });
  gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
  gl.bufferData(gl.ARRAY_BUFFER, instData.subarray(0, instCount * 10), gl.DYNAMIC_DRAW);
  instDirty = false;
}

// --------------------------------------------------------------------- render
function resize() {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr; canvas.height = h * dpr;
    metricCanvas.width = w * dpr; metricCanvas.height = h * dpr;
  }
}

function renderMetrics() {
  const dpr = window.devicePixelRatio || 1;
  metric2d.setTransform(dpr, 0, 0, dpr, 0, 0);
  metric2d.clearRect(0, 0, metricCanvas.width / dpr, metricCanvas.height / dpr);
  for (const p of panels.values()) {
    if (p.kind !== "metric" || !panelReady(p)) continue;
    const values = p.historyData;
    const finite = values.filter(Number.isFinite);
    if (!finite.length) continue;
    let lo = Math.min(...finite), hi = Math.max(...finite);
    if (lo === hi) {
      const pad = Math.max(Math.abs(lo) * 0.05, 1e-6);
      lo -= pad; hi += pad;
    }
    const x = p.x * cam.zoom + cam.x;
    const y = p.y * cam.zoom + cam.y;
    const w = p.dw * cam.zoom;
    const h = p.dh * cam.zoom;
    if (w < 2 || h < 2) continue;
    metric2d.fillStyle = "rgba(19, 24, 38, 0.92)";
    metric2d.fillRect(x, y, w, h);
    metric2d.strokeStyle = "rgba(124, 136, 161, 0.20)";
    metric2d.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const gy = y + h * i / 4;
      metric2d.beginPath(); metric2d.moveTo(x, gy); metric2d.lineTo(x + w, gy); metric2d.stroke();
    }
    const stops = CMAP_STOPS[p.cmap] || CMAP_STOPS.turbo;
    const color = stops[Math.floor(stops.length * 0.72)];
    metric2d.strokeStyle = `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
    metric2d.lineWidth = Math.max(1.25, Math.min(2.5, cam.zoom * 1.5));
    metric2d.beginPath();
    let drawing = false;
    values.forEach((value, i) => {
      if (!Number.isFinite(value)) { drawing = false; return; }
      const px = x + (values.length === 1 ? w : i * w / (values.length - 1));
      const py = y + h - (value - lo) / (hi - lo) * h;
      if (drawing) metric2d.lineTo(px, py);
      else { metric2d.moveTo(px, py); drawing = true; }
    });
    metric2d.stroke();
    metric2d.strokeStyle = "rgba(90, 208, 177, 0.45)";
    metric2d.strokeRect(x, y, w, h);
  }
}

function fitView() {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of panels.values()) {
    if (!panelReady(p)) continue;
    minX = Math.min(minX, p.x); minY = Math.min(minY, p.y - 50);
    maxX = Math.max(maxX, p.x + p.dw); maxY = Math.max(maxY, p.y + p.dh);
  }
  if (!isFinite(minX)) return;
  const vw = canvas.clientWidth, vh = canvas.clientHeight;
  const zoom = Math.min(2.5, Math.min(vw / (maxX - minX + 80), vh / (maxY - minY + 80)));
  cam.zoom = Math.max(0.05, zoom);
  cam.x = (vw - (maxX - minX) * cam.zoom) / 2 - minX * cam.zoom;
  cam.y = (vh - (maxY - minY) * cam.zoom) / 2 - minY * cam.zoom;
  needsFit = false;
}

function render() {
  requestAnimationFrame(render);
  resize();
  if (needsLayout) relayout();
  if (needsFit && instCountEstimate() > 0) { relayout(); fitView(); }
  if (instDirty) rebuildInstances();
  const dpr = window.devicePixelRatio || 1;
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.043, 0.055, 0.078, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  renderMetrics();
  gl.useProgram(prog);
  gl.bindVertexArray(vao);
  gl.uniform2f(U.uPan, cam.x, cam.y);
  gl.uniform1f(U.uZoom, cam.zoom);
  gl.uniform2f(U.uView, canvas.width / dpr, canvas.height / dpr);
  if (instCount > 0 && dataTex) {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, dataTex);
    gl.uniform1f(U.uSolid, 0); gl.uniform1f(U.uOutline, 0);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, instCount);  // ← all panels, one call
    gl.uniform1f(U.uSolid, 1); gl.uniform1f(U.uOutline, 1);
    gl.drawArraysInstanced(gl.LINE_LOOP, 0, 4, instCount);       // borders
  }
  positionOverlay();
  frames++;
  const now = performance.now();
  if (now - lastHud > 500) {
    fps = Math.round(frames * 1000 / (now - lastHud));
    frames = 0;
    const mb = bytesReceived / (now - lastHud) * 1000 / 1e6;
    bytesReceived = 0;
    lastHud = now;
    let cells = 0, tensorCount = 0, metricCount = 0;
    for (const p of panels.values()) {
      if (p.kind === "metric") metricCount++;
      else { tensorCount++; cells += (p.rows || 0) * (p.cols || 0); }
    }
    $("hud").textContent =
      `${tensorCount} tensors · ${metricCount} metrics · ${cells.toLocaleString()} cells · ${mb.toFixed(1)} MB/s · ${fps} fps`;
  }
  $("empty").hidden = instCountEstimate() > 0;
}
function instCountEstimate() {
  let n = 0;
  for (const p of panels.values()) if (panelReady(p)) n++;
  return n;
}
requestAnimationFrame(render);

// ------------------------------------------------------------------ websocket
let ws = null;

function connect() {
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.binaryType = "arraybuffer";
  ws.onopen = () => $("conn").className = "dot on";
  ws.onclose = () => {
    $("conn").className = "dot off";
    setTimeout(connect, 1000);
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") handleJSON(JSON.parse(ev.data));
    else handleBinary(ev.data);
  };
}

function handleJSON(msg) {
  if (msg.type === "hello" || msg.type === "structure") {
    if (msg.maxSide && msg.maxSide !== LAYER_SIZE && panels.size === 0)
      LAYER_SIZE = msg.maxSide;
    const alive = new Set();
    for (const w of msg.watches) {
      alive.add(w.id);
      let p = panels.get(w.id);
      if (!p) {
        p = { id: w.id, name: w.name, group: w.group, cmap: w.colormap,
              kind: w.kind || "tensor", history: w.history || 512,
              historyData: [], latest: null,
              layer: -1, image: null, texW: 0, texH: 0, rows: 0, cols: 0,
              shape: null, vmin: 0, vmax: 1, mean: 0, std: 0, nan: 0,
              x: 0, y: 0, dw: 0, dh: 0, labelEl: null,
              labelWidth: 0, labelHeight: 0, labelSizeDirty: true,
              labelSignature: "", labelLayoutKey: "" };
        panels.set(w.id, p);
        panelOrder.push(w.id);
      } else {
        p.name = w.name; p.group = w.group; p.cmap = w.colormap;
        p.kind = w.kind || "tensor"; p.history = w.history || 512;
        if (p.historyData.length > p.history)
          p.historyData = p.historyData.slice(-p.history);
      }
    }
    for (const [id, p] of [...panels]) {
      if (!alive.has(id)) {
        if (p.layer >= 0) freeLayers.push(p.layer);
        if (p.labelEl) p.labelEl.remove();
        panels.delete(id);
      }
    }
    panelOrder = panelOrder.filter((id) => panels.has(id));
    edges = msg.edges || [];
    needsLayout = true;
  } else if (msg.type === "pickResult") {
    showPickResult(msg);
  }
}

function handleBinary(buf) {
  bytesReceived += buf.byteLength;
  if (paused) return;
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x4D4C5856) return;
  const kind = dv.getUint16(6, true);
  if (kind !== 1) return;
  const metaLen = dv.getUint32(8, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 12, metaLen)));
  const off = 12 + metaLen;
  const values = (off % 4 === 0)
    ? new Float32Array(buf, off, meta.w * meta.h)
    : new Float32Array(buf.slice(off, off + meta.w * meta.h * 4));
  let p = panels.get(meta.id);
  if (!p) return; // structure message will (re)introduce it
  const firstData = !panelReady(p);
  const resized = p.texW !== meta.w || p.texH !== meta.h;
  Object.assign(p, {
    texW: meta.w, texH: meta.h, rows: meta.rows, cols: meta.cols,
    shape: meta.shape, vmin: meta.vmin, vmax: meta.vmax,
    mean: meta.mean, std: meta.std, nan: meta.nan, cmap: meta.cmap,
    group: meta.group,
  });
  if (p.kind === "metric") {
    const value = values.length ? Number(values[0]) : NaN;
    p.latest = value;
    p.historyData.push(value);
    if (p.historyData.length > p.history) p.historyData.shift();
    if (firstData) needsLayout = true;
    syncLabels();
    return;
  }
  p.image = values.slice(); // keep CPU copy for hover + pool re-allocation
  if (p.layer < 0) p.layer = allocLayer();
  if (p.layer >= 0) uploadLayer(p.layer, meta.w, meta.h, p.image);
  if (firstData || resized) needsLayout = true;
  instDirty = true;
  syncLabels();
}
connect();

// ------------------------------------------------------------------- controls
$("mode-grid").onclick = () => setMode("grid");
$("mode-graph").onclick = () => setMode("graph");
function setMode(m) {
  mode = m;
  $("mode-grid").classList.toggle("active", m === "grid");
  $("mode-graph").classList.toggle("active", m === "graph");
  needsLayout = true; needsFit = true;
}
$("cmap-override").onchange = (e) => { cmapOverride = e.target.value; instDirty = true; };
$("pause").onclick = () => {
  paused = !paused;
  $("pause").textContent = paused ? "Resume" : "Pause";
};
$("fit").onclick = () => { needsFit = true; };

// pan/zoom
let panning = false, lastX = 0, lastY = 0;
canvas.addEventListener("mousedown", (e) => {
  panning = true; lastX = e.clientX; lastY = e.clientY;
  canvas.classList.add("panning");
});
window.addEventListener("mouseup", () => {
  panning = false;
  canvas.classList.remove("panning");
});
window.addEventListener("mousemove", (e) => {
  if (panning) {
    cam.x += e.clientX - lastX; cam.y += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
  }
  hover(e);
});
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  const factor = Math.exp(-e.deltaY * 0.0012);
  const z = Math.min(40, Math.max(0.02, cam.zoom * factor));
  cam.x = mx - (mx - cam.x) * (z / cam.zoom);
  cam.y = my - (my - cam.y) * (z / cam.zoom);
  cam.zoom = z;
}, { passive: false });

// ------------------------------------------------------------------- tooltip
const tooltip = $("tooltip");
let lastPickSent = 0, pickPending = null;

function hover(e) {
  const rect = canvas.getBoundingClientRect();
  const wx = (e.clientX - rect.left - cam.x) / cam.zoom;
  const wy = (e.clientY - rect.top - cam.y) / cam.zoom;
  let hit = null;
  for (const p of panels.values()) {
    if (panelReady(p) && wx >= p.x && wx <= p.x + p.dw && wy >= p.y && wy <= p.y + p.dh) {
      hit = p; break;
    }
  }
  if (!hit || panning) { tooltip.hidden = true; pickPending = null; return; }
  if (hit.kind === "metric") {
    tooltip.hidden = false;
    tooltip.style.left = (e.clientX - rect.left + 14) + "px";
    tooltip.style.top = (e.clientY - rect.top + 14) + "px";
    tooltip.innerHTML =
      `<div>${hit.name}</div><div class="v">${fmt(hit.latest)}</div>`;
    pickPending = null;
    return;
  }
  const fu = (wx - hit.x) / hit.dw, fv = (wy - hit.y) / hit.dh;
  const row = Math.min(hit.rows - 1, Math.floor(fv * hit.rows));
  const col = Math.min(hit.cols - 1, Math.floor(fu * hit.cols));
  const tx = Math.min(hit.texW - 1, Math.floor(fu * hit.texW));
  const ty = Math.min(hit.texH - 1, Math.floor(fv * hit.texH));
  const approx = hit.image ? hit.image[ty * hit.texW + tx] : NaN;
  tooltip.hidden = false;
  tooltip.style.left = (e.clientX - rect.left + 14) + "px";
  tooltip.style.top = (e.clientY - rect.top + 14) + "px";
  const lod = hit.texW < hit.cols || hit.texH < hit.rows;
  tooltip.innerHTML =
    `<div>${hit.name} [${row}, ${col}]</div>` +
    `<div class="v" id="tt-value">${lod ? "block μ " : ""}${fmt(approx)}</div>`;
  if (lod && ws && ws.readyState === 1) {
    pickPending = { id: hit.id, row, col };
    const now = performance.now();
    if (now - lastPickSent > 80) {
      lastPickSent = now;
      ws.send(JSON.stringify({ type: "pick", ...pickPending }));
    }
  } else {
    pickPending = null;
  }
}

function showPickResult(msg) {
  if (!pickPending || msg.id !== pickPending.id ||
      msg.row !== pickPending.row || msg.col !== pickPending.col) return;
  const el = document.getElementById("tt-value");
  if (el) el.textContent = fmt(msg.value);
}

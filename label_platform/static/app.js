/* Firebrand labeling app — canvas viewer + assisted verify/add. */
"use strict";

const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const tl = document.getElementById("tl");
const tctx = tl.getContext("2d");

const S = {
  cfg: null, segs: [], seg: 1,            // default to ignition segment
  frame: 250,
  labels: null, labelCache: new Map(),
  imgCache: new Map(),
  playing: false, speed: 1.0, lastTick: 0,
  zoom: 1, ox: 0, oy: 0,                  // view transform
  hover: null, selectedHid: null,
  mode: "idle", drag: null,               // idle|draw|move|resize|pan
  showProposals: true, showRejected: false, showScene: false,
  pendingOps: 0,
};

const COLORS = {
  rule: "#22dd66", model: "#ee44ee", human: "#33ccee",
  accepted: "#4488ff", rejected: "#883333",
  flame: "#ff5533", fuel: "#ffaa00", burner: "#8888ff",
};

// ---------------------------------------------------------------- helpers
async function api(path, opts) {
  const r = await fetch(path, opts);
  return r.json();
}

function saveState(busy) {
  const el = document.getElementById("save-state");
  el.className = busy ? "busy" : "ok";
  el.textContent = busy ? "saving…" : "saved";
}

async function sendOp(op, payload) {
  S.pendingOps++; saveState(true);
  const r = await api("/api/op", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({op, payload}),
  });
  S.pendingOps--; if (!S.pendingOps) saveState(false);
  S.labelCache.clear();          // server state changed
  await loadLabels(S.frame, true);
  draw();
  return r;
}

function segOf(frame) {
  return S.segs.find(s => frame >= s.start && frame < s.end);
}

// ---------------------------------------------------------------- data
async function loadLabels(frame, force) {
  if (!force && S.labelCache.has(frame)) { S.labels = S.labelCache.get(frame); return; }
  const data = await api(`/api/labels/${frame}`);
  S.labelCache.set(frame, data);
  if (frame === S.frame) S.labels = data;
}

function getImage(frame) {
  if (S.imgCache.has(frame)) return S.imgCache.get(frame);
  const img = new Image();
  img.src = `/frame/${frame}.jpg`;
  img.onload = () => { if (frame === S.frame) draw(); };
  S.imgCache.set(frame, img);
  if (S.imgCache.size > 120) {
    const first = S.imgCache.keys().next().value;
    S.imgCache.delete(first);
  }
  return img;
}

function prefetch() {
  for (let d = 1; d <= 4; d++) {
    getImage(S.frame + d);
    loadLabels(S.frame + d).catch(() => {});
  }
}

// ---------------------------------------------------------------- view
function fitView() {
  const w = cv.clientWidth, h = cv.clientHeight;
  cv.width = w; cv.height = h;
  const fw = S.cfg.frame_width, fh = S.cfg.frame_height;
  S.zoom = Math.min(w / fw, h / fh);
  S.ox = (w - fw * S.zoom) / 2;
  S.oy = (h - fh * S.zoom) / 2;
}

function toImg(px, py) { return [(px - S.ox) / S.zoom, (py - S.oy) / S.zoom]; }
function toScr(ix, iy) { return [ix * S.zoom + S.ox, iy * S.zoom + S.oy]; }

// ---------------------------------------------------------------- boxes
function allBoxes() {
  if (!S.labels) return [];
  const out = [];
  if (S.showProposals) {
    for (const p of S.labels.proposals) {
      if (p.status === "rejected" && !S.showRejected) continue;
      out.push({kind: "prop", ...p});
    }
  }
  for (const hbox of S.labels.human) out.push({kind: "human", ...hbox});
  return out;
}

function hitBox(ix, iy) {
  let best = null, bestArea = 1e18;
  for (const b of allBoxes()) {
    const [x1, y1, x2, y2] = b.bbox;
    const m = 4 / S.zoom;
    if (ix >= x1 - m && ix <= x2 + m && iy >= y1 - m && iy <= y2 + m) {
      const a = (x2 - x1) * (y2 - y1);
      if (a < bestArea) { best = b; bestArea = a; }
    }
  }
  return best;
}

function cornerOf(b, ix, iy) {
  const [x1, y1, x2, y2] = b.bbox, m = 6 / S.zoom;
  const corners = [[x1, y1, "tl"], [x2, y1, "tr"], [x1, y2, "bl"], [x2, y2, "br"]];
  for (const [cx2, cy2, name] of corners)
    if (Math.abs(ix - cx2) < m && Math.abs(iy - cy2) < m) return name;
  return null;
}

// ---------------------------------------------------------------- draw
function drawBox(b) {
  const [x1, y1] = toScr(b.bbox[0], b.bbox[1]);
  const [x2, y2] = toScr(b.bbox[2], b.bbox[3]);
  let color, dash = [], label;
  if (b.kind === "human") {
    color = COLORS.human; label = `human #${b.hid}` + (b.keyframe ? " ●" : "");
    if (b.hid === S.selectedHid) { dash = []; ctx.lineWidth = 2.5; }
  } else {
    if (b.status === "accepted") { color = COLORS.accepted; label = `✓ ${b.source} #${b.track_id}`; }
    else if (b.status === "rejected") { color = COLORS.rejected; label = `✗ ${b.source} #${b.track_id}`; }
    else { color = COLORS[b.source]; dash = [5, 4]; label = `${b.source} #${b.track_id}`; }
  }
  const hov = S.hover && sameBox(S.hover, b);
  ctx.strokeStyle = color;
  ctx.lineWidth = hov ? 2.5 : 1.4;
  ctx.setLineDash(dash);
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  ctx.setLineDash([]);
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "#000";
  ctx.fillText(label, x1 + 1, y1 - 4 + 1);
  ctx.fillStyle = color;
  ctx.fillText(label, x1, y1 - 5);
  if (hov || (b.kind === "human" && b.hid === S.selectedHid)) {
    ctx.fillStyle = color;
    for (const [cx2, cy2] of [[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
      ctx.fillRect(cx2 - 3, cy2 - 3, 6, 6);
  }
}

function sameBox(a, b) {
  if (a.kind !== b.kind) return false;
  if (a.kind === "human") return a.hid === b.hid;
  return a.source === b.source && a.track_id === b.track_id;
}

function draw() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, cv.width, cv.height);
  const img = getImage(S.frame);
  if (img.complete && img.naturalWidth) {
    ctx.imageSmoothingEnabled = S.zoom < 1.5;
    const [x, y] = toScr(0, 0);
    ctx.drawImage(img, x, y, S.cfg.frame_width * S.zoom, S.cfg.frame_height * S.zoom);
  }
  if (S.labels) {
    if (S.showScene) {
      for (const [name, bbox] of Object.entries(S.labels.scene || {})) {
        if (!bbox) continue;
        const [x1, y1] = toScr(bbox[0], bbox[1]);
        const [x2, y2] = toScr(bbox[2], bbox[3]);
        ctx.strokeStyle = COLORS[name]; ctx.setLineDash([2, 6]);
        ctx.lineWidth = 1;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.setLineDash([]);
        ctx.fillStyle = COLORS[name];
        ctx.fillText(name, x1 + 2, y2 - 4);
      }
    }
    for (const b of allBoxes()) drawBox(b);
  }
  if (S.drag && S.mode === "draw") {
    const [x1, y1] = toScr(S.drag.x0, S.drag.y0);
    const [x2, y2] = toScr(S.drag.x1, S.drag.y1);
    ctx.strokeStyle = COLORS.human; ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
  updateBar();
  drawTimeline();
}

function updateBar() {
  const seg = segOf(S.frame);
  document.getElementById("seg-name").textContent = seg ? seg.name : "outside segments";
  const t = (S.frame / S.cfg.fps).toFixed(2);
  const done = S.labels && S.labels.done ? " ✔ done" : "";
  document.getElementById("frame-info").textContent = `f${S.frame}  t=${t}s${done}`;
  if (S.labels) {
    const n = {pending: 0, accepted: 0, rejected: 0};
    for (const p of S.labels.proposals) n[p.status]++;
    document.getElementById("counts").textContent =
      `proposals: ${n.pending} pending · ${n.accepted} ✓ · ${n.rejected} ✗ · human: ${S.labels.human.length}`;
  }
}

function drawTimeline() {
  const w = tl.clientWidth, h = 38;
  tl.width = w; tl.height = h;
  tctx.fillStyle = "#15151a"; tctx.fillRect(0, 0, w, h);
  const total = 25467;
  for (const seg of S.segs) {
    const x1 = seg.start / total * w, x2 = seg.end / total * w;
    tctx.fillStyle = "#2a3a4a";
    tctx.fillRect(x1, 8, Math.max(x2 - x1, 2), 22);
  }
  const cur = S.frame / total * w;
  tctx.fillStyle = "#ffcc33";
  tctx.fillRect(cur - 1, 2, 2, 34);
}

// ---------------------------------------------------------------- navigation
async function gotoFrame(f) {
  const seg = segOf(f);
  if (!seg) {  // snap into nearest segment
    let best = null, bd = 1e18;
    for (const s of S.segs) {
      for (const cand of [s.start, s.end - 1]) {
        const d = Math.abs(cand - f);
        if (d < bd) { bd = d; best = cand; }
      }
    }
    f = best;
  }
  S.frame = f;
  await loadLabels(f);
  S.labels = S.labelCache.get(f);
  draw();
  prefetch();
}

function step(d) {
  let f = S.frame + d;
  const seg = segOf(S.frame);
  if (seg) {
    if (f >= seg.end || f < seg.start) {
      // hop to next/prev segment
      const i = S.segs.indexOf(seg);
      const tgt = d > 0 ? S.segs[i + 1] : S.segs[i - 1];
      if (!tgt) return;
      f = d > 0 ? tgt.start : tgt.end - 1;
    }
  }
  gotoFrame(f);
}

function playLoop(ts) {
  if (S.playing) {
    if (!S.lastTick) S.lastTick = ts;
    const interval = 1000 / (S.cfg.fps * S.speed);
    if (ts - S.lastTick >= interval) {
      S.lastTick = ts;
      step(1);
    }
  } else S.lastTick = 0;
  requestAnimationFrame(playLoop);
}

// ---------------------------------------------------------------- actions
function hoveredProposal() {
  return S.hover && S.hover.kind === "prop" ? S.hover : null;
}

async function actAccept(track) {
  const p = hoveredProposal(); if (!p) return;
  await sendOp(track ? "accept_track" : "accept_box",
               {source: p.source, track_id: p.track_id, frame: S.frame});
}
async function actReject(track) {
  const p = hoveredProposal(); if (!p) return;
  await sendOp(track ? "reject_track" : "reject_box",
               {source: p.source, track_id: p.track_id, frame: S.frame});
}

async function addKeyframe(hid, bbox) {
  const r = await sendOp("add_keyframe", {hid, frame: S.frame, bbox});
  S.selectedHid = hid || r.next_hid - 1;
}

// ---------------------------------------------------------------- input
cv.addEventListener("mousemove", e => {
  const [ix, iy] = toImg(e.offsetX, e.offsetY);
  if (S.mode === "pan" && S.drag) {
    S.ox += e.offsetX - S.drag.px; S.oy += e.offsetY - S.drag.py;
    S.drag.px = e.offsetX; S.drag.py = e.offsetY;
    draw(); return;
  }
  if (S.mode === "draw" && S.drag) { S.drag.x1 = ix; S.drag.y1 = iy; draw(); return; }
  if ((S.mode === "move" || S.mode === "resize") && S.drag) {
    const b = S.drag.box, bb = S.drag.bbox0.slice();
    const dx = ix - S.drag.ix0, dy = iy - S.drag.iy0;
    if (S.mode === "move") { bb[0]+=dx; bb[1]+=dy; bb[2]+=dx; bb[3]+=dy; }
    else {
      if (S.drag.corner.includes("l")) bb[0]+=dx;
      if (S.drag.corner.includes("r")) bb[2]+=dx;
      if (S.drag.corner.includes("t")) bb[1]+=dy;
      if (S.drag.corner.includes("b")) bb[3]+=dy;
    }
    S.drag.live = bb;
    b.bbox = bb;   // optimistic
    draw(); return;
  }
  const h = hitBox(ix, iy);
  if (!sameBox(h || {}, S.hover || {})) { S.hover = h; draw(); }
});

cv.addEventListener("mousedown", e => {
  const [ix, iy] = toImg(e.offsetX, e.offsetY);
  if (e.altKey || e.button === 1) {
    S.mode = "pan"; S.drag = {px: e.offsetX, py: e.offsetY}; return;
  }
  if (S.mode === "draw-armed") {
    S.mode = "draw"; S.drag = {x0: ix, y0: iy, x1: ix, y1: iy}; return;
  }
  const b = hitBox(ix, iy);
  if (b) {
    if (b.kind === "human") S.selectedHid = b.hid;
    const corner = cornerOf(b, ix, iy);
    S.mode = corner ? "resize" : "move";
    S.drag = {box: b, bbox0: b.bbox.slice(), ix0: ix, iy0: iy, corner};
  }
});

cv.addEventListener("mouseup", async () => {
  if (S.mode === "draw" && S.drag) {
    const {x0, y0, x1, y1} = S.drag;
    const bbox = [Math.min(x0,x1), Math.min(y0,y1), Math.max(x0,x1), Math.max(y0,y1)];
    S.mode = "idle"; S.drag = null;
    if (bbox[2]-bbox[0] > 2 && bbox[3]-bbox[1] > 2) await addKeyframe(null, bbox);
    return;
  }
  if ((S.mode === "move" || S.mode === "resize") && S.drag && S.drag.live) {
    const b = S.drag.box, bb = S.drag.live;
    S.mode = "idle"; const drag = S.drag; S.drag = null;
    if (b.kind === "human") await sendOp("add_keyframe", {hid: b.hid, frame: S.frame, bbox: bb});
    else await sendOp("move_box", {source: b.source, track_id: b.track_id, frame: S.frame, bbox: bb});
    return;
  }
  S.mode = "idle"; S.drag = null;
});

cv.addEventListener("wheel", e => {
  e.preventDefault();
  const f = Math.exp(-e.deltaY * 0.0015);
  const [ix, iy] = toImg(e.offsetX, e.offsetY);
  S.zoom *= f;
  S.ox = e.offsetX - ix * S.zoom;
  S.oy = e.offsetY - iy * S.zoom;
  draw();
}, {passive: false});

tl.addEventListener("click", e => {
  const f = Math.round(e.offsetX / tl.clientWidth * 25467);
  gotoFrame(f);
});

document.addEventListener("keydown", async e => {
  if (e.target.tagName === "SELECT") return;
  const k = e.key;
  if (k === "ArrowRight") { e.preventDefault(); step(e.shiftKey ? 10 : 1); }
  else if (k === "ArrowLeft") { e.preventDefault(); step(e.shiftKey ? -10 : -1); }
  else if (k === " ") { e.preventDefault(); S.playing = !S.playing; }
  else if (k === "[") S.speed = Math.max(0.1, S.speed / 1.5);
  else if (k === "]") S.speed = Math.min(8, S.speed * 1.5);
  else if (k === "a") await actAccept(false);
  else if (k === "A") await actAccept(true);
  else if (k === "x") await actReject(false);
  else if (k === "X") await actReject(true);
  else if (k === "n") { S.mode = "draw-armed"; }
  else if (k === "i" && S.selectedHid) {
    const cur = S.labels.human.find(h => h.hid === S.selectedHid);
    if (cur) await addKeyframe(S.selectedHid, cur.bbox);
  }
  else if (k === "Backspace") {
    if (S.hover && S.hover.kind === "human") {
      e.preventDefault();
      if (e.shiftKey) await sendOp("delete_human_track", {hid: S.hover.hid});
      else await sendOp("delete_keyframe", {hid: S.hover.hid, frame: S.frame});
    }
  }
  else if (k === "Enter") {
    await sendOp("mark_frame_done", {frame: S.frame, done: true});
    step(1);
  }
  else if (k === "u") await sendOp("mark_frame_done", {frame: S.frame, done: false});
  else if (k === "z") await sendOp("undo", {});
  else if (k === "h") { S.showProposals = !S.showProposals; draw(); }
  else if (k === "j") { S.showRejected = !S.showRejected; draw(); }
  else if (k === "s") { S.showScene = !S.showScene; draw(); }
  else if (k === "0") { fitView(); draw(); }
});

// ---------------------------------------------------------------- boot
async function boot() {
  S.cfg = await api("/api/config");
  S.segs = S.cfg.segments;
  const sel = document.getElementById("seg-select");
  S.segs.forEach((s, i) => {
    const o = document.createElement("option");
    o.value = i; o.textContent = `${s.name} (f${s.start}–${s.end})`;
    sel.appendChild(o);
  });
  sel.value = 1;
  sel.addEventListener("change", () => gotoFrame(S.segs[+sel.value].start));
  document.getElementById("btn-export").addEventListener("click", async () => {
    const r = await api("/api/export", {method: "POST"});
    alert(`Exported gold set:\n${JSON.stringify(r, null, 1)}`);
  });
  document.getElementById("btn-help").addEventListener("click", () =>
    document.getElementById("help").classList.toggle("hidden"));
  window.addEventListener("resize", () => { fitView(); draw(); });
  fitView();
  await gotoFrame(S.segs[1].start);
  requestAnimationFrame(playLoop);
}
boot();

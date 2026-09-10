"use strict";

const HUES = [0, 35, 60, 100, 160, 190, 220, 260, 290, 330];

const state = { data: null, key: "", checked: new Set(), filter: "" };

const $ = (id) => document.getElementById(id);
const canvas = $("plot");
const ctx = canvas.getContext("2d");
const tooltip = $("tooltip");

const PALETTE = new Map();
function colorOf(b) {
  if (!PALETTE.has(b)) PALETTE.set(b, HUES[PALETTE.size % HUES.length]);
  return PALETTE.get(b);
}

// selection key: aggregate is "dz|tag", a year cohort appends "|y<year>".
function keyOf() {
  const y = $("year").value;
  return `${$("dz").value}|${$("tag").value}${y === "all" ? "" : `|y${y}`}`;
}

// years that exist for the current dz x tag; aggregate ("all") always offered.
function yearOptions() {
  const dz = $("dz").value, tag = $("tag").value;
  const years = Object.keys(state.data)
    .filter((k) => k.startsWith(`${dz}|${tag}|y`) && state.data[k].year !== undefined)
    .map((k) => String(state.data[k].year))
    .sort();
  return ["all", ...years];
}

function current() {
  return state.data?.[state.key] ?? null;
}

function setOptions(sel, values, keep) {
  sel.textContent = "";
  for (const v of values) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = v;
    sel.appendChild(o);
  }
  if (keep !== undefined && values.includes(keep)) sel.value = keep;
}

function rebuild() {
  const d = current();
  if (!d) return;
  const f = state.filter.toLowerCase();
  const list = $("list");
  list.textContent = "";
  for (const b of d.benchmarks) {
    if (f && !b.toLowerCase().includes(f)) continue;
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = state.checked.has(b);
    cb.value = b;
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(b));
    list.appendChild(lab);
  }
  const tagDesc = d.tag === "pa" ? "PA factor count" : "forced 2 factors";
  const cohort = d.year === undefined ? "all years" : `cohort ${d.year}`;
  $("meta").textContent =
    `${cohort} · ${d.benchmarks.length} benchmarks · averaged over ${d.n_cells} cells · ${tagDesc}`;
  draw();
}

function extent(d) {
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const p of d.points) {
    if (p.x < xmin) xmin = p.x;
    if (p.x > xmax) xmax = p.x;
    if (p.y < ymin) ymin = p.y;
    if (p.y > ymax) ymax = p.y;
  }
  return { xmin, xmax, ymin, ymax };
}

function draw() {
  const d = current();
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (w === 0 || h === 0) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  if (!d) return;

  const m = 40; // margin covering the largest dot radius
  const e = extent(d);
  const sx = (w - 2 * m) / Math.max(e.xmax - e.xmin, 1e-9);
  const sy = (h - 2 * m) / Math.max(e.ymax - e.ymin, 1e-9);
  const s = Math.min(sx, sy);

  // map at raw scale, then fit the actual mapped bounding box onto the
  // canvas: shrink if it overflows, then center it.
  let pts = d.points.map((p) => ({ b: p.benchmark, x: p.x * s, y: p.y * s }));
  let bxmin = Infinity, bxmax = -Infinity, bymin = Infinity, bymax = -Infinity;
  for (const p of pts) {
    if (p.x < bxmin) bxmin = p.x;
    if (p.x > bxmax) bxmax = p.x;
    if (p.y < bymin) bymin = p.y;
    if (p.y > bymax) bymax = p.y;
  }
  const bw = bxmax - bxmin, bh = bymax - bymin;
  const s2 = Math.min(1, (w - 2 * m) / Math.max(bw, 1e-9), (h - 2 * m) / Math.max(bh, 1e-9));
  if (s2 < 1) {
    for (const p of pts) { p.x *= s2; p.y *= s2; }
    bxmin *= s2; bxmax *= s2; bymin *= s2; bymax *= s2;
  }
  const ox = w / 2 - (bxmin + bxmax) / 2;
  const oy = h / 2 - (bymin + bymax) / 2;

  d.screen = pts.map((p) => ({ b: p.b, x: p.x + ox, y: p.y + oy }));

  const anyHighlight = state.checked.size > 0;

  // dim base layer: one uniform color
  ctx.fillStyle = anyHighlight ? "rgba(120, 120, 135, 0.30)" : "#9a9aa8";
  for (const p of d.screen) {
    if (anyHighlight && state.checked.has(p.b)) continue;
    ctx.beginPath();
    ctx.arc(p.x, p.y, anyHighlight ? 2.5 : 5, 0, 2 * Math.PI);
    ctx.fill();
  }

  // highlighted layer: colored + labeled
  for (const p of d.screen) {
    if (!state.checked.has(p.b)) continue;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, 2 * Math.PI);
    ctx.fillStyle = `hsl(${colorOf(p.b)} 65% 45%)`;
    ctx.fill();
    ctx.fillStyle = "rgba(34, 34, 42, 0.92)";
    ctx.font = "12px system-ui";
    ctx.fillText(p.b, p.x + 9, p.y - 8);
  }
}

canvas.addEventListener("mousemove", (ev) => {
  const d = current();
  if (!d || !d.screen) {
    tooltip.hidden = true;
    return;
  }
  const r = canvas.getBoundingClientRect();
  const mx = ev.clientX - r.left;
  const my = ev.clientY - r.top;
  let best = null;
  let bestD = 100;
  for (const p of d.screen) {
    const dx = p.x - mx;
    const dy = p.y - my;
    const dist = dx * dx + dy * dy;
    if (dist < bestD) {
      bestD = dist;
      best = p;
    }
  }
  if (best) {
    tooltip.hidden = false;
    tooltip.textContent = best.b;
    tooltip.style.left = `${ev.clientX + 12}px`;
    tooltip.style.top = `${ev.clientY - 20}px`;
  } else {
    tooltip.hidden = true;
  }
});
canvas.addEventListener("mouseleave", () => (tooltip.hidden = true));
window.addEventListener("resize", draw);

$("list").addEventListener("change", (ev) => {
  const b = ev.target.value;
  if (!b) return;
  ev.target.checked ? state.checked.add(b) : state.checked.delete(b);
  draw();
});
$("filter").addEventListener("input", () => {
  state.filter = $("filter").value;
  rebuild();
});
$("clear").addEventListener("click", () => {
  state.checked.clear();
  rebuild();
});

function init(data) {
  state.data = data;
  const keys = Object.keys(data);
  const dzs = [...new Set(keys.map((k) => k.split("|")[0]))];
  const tags = [...new Set(keys.map((k) => k.split("|")[1]))];
  state.key = keys[0];
  setOptions($("dz"), dzs, dzs[0]);
  setOptions($("tag"), tags, tags[0]);
  setOptions($("year"), ["all"], "all");

  const apply = () => {
    state.key = keyOf();
    if (!state.data[state.key]) return; // stale selection: keep last valid view
    rebuild();
  };
  $("dz").addEventListener("change", () => {
    setOptions($("year"), yearOptions(), "all");
    apply();
  });
  $("tag").addEventListener("change", () => {
    setOptions($("year"), yearOptions(), "all");
    apply();
  });
  $("year").addEventListener("change", apply);
  setOptions($("year"), yearOptions(), "all");
  rebuild();
  requestAnimationFrame(draw);
  window.addEventListener("load", draw);
}

if (window.POSITIONS) {
  init(window.POSITIONS);
} else {
  $("meta").textContent = "positions.js missing — run compute_positions.py";
}

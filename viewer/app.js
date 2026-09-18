"use strict";

// Slots 1-8 are the dataviz reference categorical palette in its validated
// order; 9-14 extend it because cluster count is driven by the data, not by
// how many colors exist. Ordered so the ADJACENT pairlist still passes every
// gate at 14 slots (worst CVD dE 9.1 protan, worst normal-vision dE 19.6) --
// identical to the 8-slot baseline, so the extension costs nothing there.
// A scatter is really an all-pairs form, and at 14 slots all-pairs FAILS:
// periwinkle/blue dE 1.6 (protan) and red/orange dE 7.1 (normal vision) can
// read as one group. Accepted deliberately rather than capping clusters at 8.
// Identity is therefore never color-alone: legend, tooltip, and click-to-
// highlight all name the group. Re-check with the dataviz validator:
//   node scripts/validate_palette.js "<these 14>" --mode light --surface "#ffffff"
const CLUSTER = [
  "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
  "#e34948", "#0e8fa8", "#b35a00", "#9d4bc7", "#7d8f00", "#8a6fe0", "#d4457f",
];
const NOISE = "#9a9aa8";
const DIM_ALPHA = 0.22;

const dim = (hex) =>
  `rgba(${parseInt(hex.slice(1, 3), 16)},${parseInt(hex.slice(3, 5), 16)},${parseInt(
    hex.slice(5, 7),
    16
  )},${DIM_ALPHA})`;
const CLUSTER_DIM = CLUSTER.map(dim);
const NOISE_DIM = dim(NOISE);

const state = {
  data: null,
  key: "",
  checked: new Set(),
  filter: "",
  algo: "hac",
  k: null,
  colorBy: "cluster",   // "cluster" or an axis name from the payload
  variant: "with_g",
  highlights: new Set(),  // multi-select highlight: label ids and/or NO_LABEL
};

const $ = (id) => document.getElementById(id);
const canvas = $("plot");
const ctx = canvas.getContext("2d");
const tooltip = $("tooltip");

// selection key: aggregate is "dz|tag", a year cohort appends "|y<year>", and a
// per-imputer view inserts "|m<method>" before the year segment.
function keyOf() {
  const y = $("year").value;
  const imp = $("imputer").value;
  return (
    `${$("dz").value}|${$("tag").value}` +
    (imp === "aggregate" ? "" : `|m${imp}`) +
    (y === "all" ? "" : `|y${y}`)
  );
}

// Imputation methods available for the current dz x tag x cohort. Derived from
// the key set: per-method keys carry a "|m<method>" segment.
function imputerOptions() {
  const prefix = `${$("dz").value}|${$("tag").value}`;
  const y = $("year").value;
  const methods = new Set(["aggregate"]);
  for (const k of Object.keys(state.data)) {
    if (!k.startsWith(prefix)) continue;
    let method = "aggregate";
    let year = "all";
    for (const seg of k.split("|").slice(2)) {
      if (seg.startsWith("y")) year = seg.slice(1);
      else method = seg.slice(1);
    }
    if (year === y) methods.add(method);
  }
  return [...methods].sort();
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

function variantOf(d) {
  return d.clusters?.[state.variant] ?? null;
}

// A benchmark's category labels. Tolerates a bare string (single-label
// payloads) or an array (multi-label), so promoting `category` to multi-valued
// needs no viewer change.
function labelsOf(d, i) {
  const v = d.categories?.[i];
  if (v === null || v === undefined) return [];
  return typeof v === "string" ? [v] : v;
}

function clusterLabels(d) {
  const v = variantOf(d);
  if (!v) return null;
  return state.algo === "hdbscan"
    ? v.hdbscan?.labels ?? null
    : v.hac?.by_k?.[String(state.k)]?.labels ?? null;
}

// Per-point group value. Cluster mode: the cluster id (-1 = noise). Category
// mode: the color index of the FIRST selected label the point carries (legend
// order), -1 if none. Multi-highlight deliberately reuses the cluster palette
// slots only up to the number of selected labels; a multi-label point has no
// single colour, so overlaps resolve to the first match and the tooltip names
// every selected label the point carries -- colour is never the sole carrier
// of identity.
function groupValues(d) {
  if (!d.clusters && !d.categories) return null;
  if (inCategoryMode()) {
    if (!d.categories) return null;
    if (!state.highlights.size) return d.categories.map(() => -1);
    const idx = highlightMap(d).idx;
    return d.categories.map((_, i) => {
      const labs = axisLabelsAt(d, i);
      if (!labs.length) return idx.get(NO_LABEL) ?? -1;
      for (const l of labs) {
        const c = idx.get(l);
        if (c !== undefined) return c;
      }
      return -1;
    });
  }
  const cl = d.clusters ? clusterLabels(d) : null;
  // Cluster mode: selecting the noise row promotes noise to the highlighted set.
  if (cl && state.highlights.has(NO_LABEL)) return cl.map((v) => (v < 0 ? 0 : -1));
  return cl;
}

// Color index per selected label, in legend order, plus the NO_LABEL sentinel
// when selected. One shared map keeps the legend swatches and the point colors
// consistent.
function highlightMap(d) {
  const [rows] = groupRows(d);
  const idx = new Map();
  let next = 0;
  for (const r of rows) {
    if (state.highlights.has(r.value)) idx.set(r.value, next++);
  }
  if (state.highlights.has(NO_LABEL)) idx.set(NO_LABEL, next++);
  return { idx, labels: [...idx.keys()] };
}

const colorFor = (value, dimmed) => {
  if (value === null || value === undefined || value < 0) {
    return dimmed ? NOISE_DIM : NOISE;
  }
  return dimmed ? CLUSTER_DIM[value % CLUSTER.length] : CLUSTER[value % CLUSTER.length];
};

// Label ids are "<axis>:<label>", so the same display name (e.g. miscellaneous)
// can live on several axes and still be a distinct, separately scored set.
function labelSep(d) {
  return d.clusters?.params?.label_sep ?? null;
}

function axisOf(d, label) {
  const sep = labelSep(d);
  if (sep && label.includes(sep)) return label.slice(0, label.indexOf(sep));
  return d.clusters?.params?.label_axis?.[label] ?? "other";   // pre-qualified payloads
}

function displayOf(d, label) {
  const sep = labelSep(d);
  return sep && label.includes(sep) ? label.slice(label.indexOf(sep) + sep.length) : label;
}

function axisOrder(d) {
  return d.clusters?.params?.axis_order ?? [];
}

// Labels on the currently selected axis, ordered by descending frequency.
function axisLabels(d, axis) {
  return d.clusters?.params?.axis_labels?.[axis] ?? [];
}

const inCategoryMode = () => state.colorBy !== "cluster";

// Selecting the residual row is a real query -- "which benchmarks does this axis
// say nothing about" -- so it gets a sentinel rather than being unclickable.
const NO_LABEL = "__unlabelled__";

// A point's labels restricted to the axis currently being coloured by.
function axisLabelsAt(d, i) {
  return labelsOf(d, i).filter((l) => axisOf(d, l) === state.colorBy);
}

function cohesionOf(d, category) {
  const rows = d.category_cohesion?.[state.variant] ?? [];
  return rows.find((r) => r.category === category) ?? null;
}

// Legend rows. Cluster mode: one row per cluster, id + size only (naming them
// after a plurality member category over-claimed -- no base-rate correction,
// and singletons scored 100%). Category mode: one row per category present,
// single-select, with its cohesion score on hover.
function groupRows(d) {
  if (inCategoryMode()) {
    if (!d.categories) return [[], 0];
    const axis = state.colorBy;
    const counts = new Map();
    let unlabelled = 0;
    d.categories.forEach((_, i) => {
      const labs = labelsOf(d, i).filter((l) => axisOf(d, l) === axis);
      if (!labs.length) unlabelled += 1;
      for (const l of labs) {
        if (!counts.has(l)) counts.set(l, []);
        counts.get(l).push(i);
      }
    });
    const rows = [...counts.entries()].map(([label, members]) => {
      const c = cohesionOf(d, label);
      const stat = c
        ? `${c.z <= 0 ? "tighter" : "looser"} than chance: z=${c.z}, p=${c.p}`
        : "no cohesion score (too few members)";
      const name = displayOf(d, label);
      return {
        value: label,
        members,
        label: name,
        size: members.length,
        title: `${name}: ${members.length} benchmarks — ${stat}`,
      };
    });
    rows.sort((a, b) => b.size - a.size || a.label.localeCompare(b.label));
    return [rows, unlabelled];
  }
  const labels = clusterLabels(d);
  if (!labels) return [[], 0];
  const byValue = new Map();
  labels.forEach((v, i) => {
    const key = v === null || v === undefined ? -1 : v;
    if (!byValue.has(key)) byValue.set(key, []);
    byValue.get(key).push(i);
  });
  const rows = [];
  let noise = 0;
  for (const [value, members] of byValue) {
    if (value === -1) {
      noise = members.length;
      continue;
    }
    rows.push({
      value,
      members,
      label: `cluster ${value}`,
      size: members.length,
      title: `cluster ${value}: ${members.length} benchmarks`,
    });
  }
  rows.sort((a, b) => b.size - a.size || a.value - b.value);
  return [rows, noise];
}

// Cluster mode: a row toggles its members into the checkbox highlight.
// Category mode: a row toggles in/out of the multi-select highlight, each
// selected row keeping its own color.
function onLegendClick(d, value) {
  if (value === NO_LABEL) {
    state.highlights.has(NO_LABEL)
      ? state.highlights.delete(NO_LABEL)
      : state.highlights.add(NO_LABEL);
    rebuild();
    return;
  }
  if (inCategoryMode()) {
    state.highlights.has(value)
      ? state.highlights.delete(value)
      : state.highlights.add(value);
    rebuild();
    return;
  }
  const [rows] = groupRows(d);
  const row = rows.find((r) => String(r.value) === value);
  if (!row) return;
  const names = row.members.map((i) => d.benchmarks[i]);
  const allOn = names.every((b) => state.checked.has(b));
  for (const b of names) allOn ? state.checked.delete(b) : state.checked.add(b);
  rebuild();
}

function drawLegend(d) {
  if (!d) return;
  const legend = $("legend");
  legend.textContent = "";
  const cat = inCategoryMode();
  const has = cat ? Boolean(d.categories) : groupValues(d) !== null;
  $("legend-head").hidden = !has;
  legend.hidden = !has;
  if (!has) return;

  const [rows, unassigned] = groupRows(d);
  // Category mode groups rows under Topic / Task / Axis so a subject-matter
  // label is never read as though it were a capability.
  const groups = [[null, rows]];
  const hidx = cat ? highlightMap(d).idx : null;
  for (const [, group] of groups) {
  for (const row of group) {
    const el = document.createElement("div");
    el.dataset.value = String(row.value);
    el.title = row.title;
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = cat
      ? hidx.has(row.value) ? colorFor(hidx.get(row.value), false) : NOISE
      : colorFor(row.value, false);
    const name = document.createElement("span");
    name.textContent = row.label;
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = row.size;
    el.append(sw, name, n);
    const on = cat ? hidx.has(row.value) : row.members.every((i) => state.checked.has(d.benchmarks[i]));
    if (on) el.classList.add("on");
    legend.appendChild(el);
  }
  }
  if (unassigned) {
    const el = document.createElement("div");
    el.className = "unassigned";
    el.dataset.value = NO_LABEL;
    const on = state.highlights.has(NO_LABEL);
    if (on) el.classList.add("on");
    el.title = cat
      ? `${unassigned} benchmarks carry no ${state.colorBy} label — click to highlight`
      : `${unassigned} benchmarks the clusterer declined to place — click to highlight`;
    const sw = document.createElement("span");
    sw.className = "swatch";
    sw.style.background = on ? colorFor(cat ? hidx.get(NO_LABEL) : 0, false) : NOISE;
    const name = document.createElement("span");
    name.textContent = cat ? "unlabelled" : "noise";
    const n = document.createElement("span");
    n.className = "n";
    n.textContent = unassigned;
    el.append(sw, name, n);
    legend.appendChild(el);
  }
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
  syncColorBy(d);
  syncClusterControls(d);
  drawLegend(d);
  const tagDesc =
    d.tag === "pa" ? "PA factor count" : d.tag === "2f" ? "forced 2 factors" : d.tag;
  const cohort = d.year === undefined ? "all years" : `cohort ${d.year}`;
  $("meta").textContent =
    `${cohort} · ${d.benchmarks.length} benchmarks · averaged over ${d.n_cells} cells · ` +
    (d.method ? `imputer ${d.method} · ` : "") +
    tagDesc +
    clusterMeta(d);
  draw();
}

// The color-by dropdown is the axis tier: "cluster", then one entry per axis
// present in the payload. Axes come from the data, so a new label column in
// benchmarks.csv appears here with no code change.
function syncColorBy(d) {
  const axes = axisOrder(d);
  const opts = ["cluster", ...axes];
  if (!opts.includes(state.colorBy)) state.colorBy = "cluster";
  setOptions($("colorby"), opts, state.colorBy);
  const labels = inCategoryMode() ? axisLabels(d, state.colorBy) : [];
  // A label from the old axis is meaningless on the new one; "carries no label
  // here" survives every axis.
  for (const h of [...state.highlights]) {
    if (h !== NO_LABEL && !labels.includes(h)) state.highlights.delete(h);
  }
}

function syncClusterControls(d) {
  const ks = d.clusters?.params?.k_values ?? [];
  if (ks.length) {
    if (!ks.includes(state.k)) {
      const v = variantOf(d);
      state.k = v?.hac?.best_k ?? ks[0];
    }
    setOptions($("k"), ks.map(String), String(state.k));
  }
  $("k").disabled = state.algo !== "hac" || inCategoryMode();
  $("algo").disabled = inCategoryMode();
  $("variant").disabled = false;  // cohesion is reported per g-variant too
}

const pct = (x) => `${Math.round(x * 100)}%`;

function clusterMeta(d) {
  const c = d.clusters;
  if (!c) return "";
  const parts = [];
  if (inCategoryMode()) {
    const [rows, unlabelled] = groupRows(d);
    parts.push(
      `${state.colorBy}: ${rows.length} labels here`,
      `${unlabelled} with none`
    );
    if (state.highlights.has(NO_LABEL)) {
      parts.push(`highlighting the ${unlabelled} with no ${state.colorBy} label (not scored)`);
    }
    if (!state.highlights.size) {
      parts.push("pick a category to highlight");
    } else {
      const g = state.variant === "with_g" ? "with g" : "without g";
      for (const h of state.highlights) {
        if (h === NO_LABEL) continue;
        const co = cohesionOf(d, h);
        const name = displayOf(d, h);
        if (co) {
          const verdict =
            co.p < 0.05 ? "tighter than chance" : "not tighter than chance";
          parts.push(
            `${name}: n=${co.n} · ${verdict} (z=${co.z}, p=${co.p}, ${g}, coverage-matched)`
          );
        } else if (d.category_cohesion) {
          parts.push(`${name}: too few members to score`);
        } else {
          parts.push(`${name}: no cohesion score in this payload`);
        }
      }
    }
  } else {
    const v = variantOf(d);
    const g = state.variant === "with_g" ? "with g" : "without g";
    if (state.algo === "hdbscan" && v?.hdbscan) {
      const h = v.hdbscan;
      parts.push(
        `HDBSCAN · ${g}`,
        `${h.n_clusters} clusters`,
        `${pct(h.noise_frac)} noise`,
        `sil ${h.silhouette ?? "n/a"}`
      );
    } else if (v?.hac) {
      const k = v.hac.by_k?.[String(state.k)];
      const best = state.k === v.hac.best_k ? " (silhouette-max)" : "";
      parts.push(
        `${c.params.linkage} linkage · ${g}`,
        `k=${state.k}${best}`,
        `sil ${k?.silhouette ?? "n/a"}`
      );
    }
  }
  const dg = c.diagnostics;
  if (dg?.n_fabricated_pairs) {
    parts.push(`${pct(1 - dg.pair_coverage)} of pairs imputed`);
  }
  return parts.length ? ` · ${parts.join(" · ")}` : "";
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
  const labels = groupValues(d);
  let pts = d.points.map((p, i) => ({
    b: p.benchmark,
    g: labels ? labels[i] : null,
    cats: labelsOf(d, i),
    x: p.x * s,
    y: p.y * s,
  }));
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

  d.screen = pts.map((p) => ({ ...p, x: p.x + ox, y: p.y + oy }));

  // Base layer, batched one path per color: a per-point fillStyle change would
  // cost a state flush per dot at 820 points. Checkbox highlighting never
  // dims or recolors anything; every dot keeps its group color.
  const r = 5;
  const buckets = new Map();
  for (const p of d.screen) {
    const col = labels ? colorFor(p.g, false) : NOISE;
    if (!buckets.has(col)) buckets.set(col, []);
    buckets.get(col).push(p);
  }
  for (const [col, pts] of buckets) {
    ctx.fillStyle = col;
    ctx.beginPath();
    for (const p of pts) {
      ctx.moveTo(p.x + r, p.y);
      ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
    }
    ctx.fill();
  }

  // Highlight layer: the dot's already-decided group color, enlarged, plus a
  // white ring and its name. Ring and size carry "selected", never hue --
  // hue is already spent on group identity.
  const showLabels = state.checked.size <= 25;
  ctx.font = "12px system-ui";
  ctx.lineJoin = "round";
  for (const p of d.screen) {
    if (!state.checked.has(p.b)) continue;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, 2 * Math.PI);
    ctx.fillStyle = labels ? colorFor(p.g, false) : NOISE;
    ctx.fill();
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 2;
    ctx.stroke();
    if (showLabels) {
      const tw = ctx.measureText(p.b).width;
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(p.x + 7, p.y - 8 - 11, tw + 4, 14);
      ctx.fillStyle = "rgba(34, 34, 42, 0.92)";
      ctx.fillText(p.b, p.x + 9, p.y - 8);
    }
  }

  // In-plot legend for color-by highlight colors, drawn on the canvas itself
  // (not the side panel) so right-click "Copy image" retains the mapping.
  // Swatches read the same highlightMap the dots and panel swatches use.
  if (inCategoryMode() && d.categories && state.highlights.size) {
    const hmap = highlightMap(d);
    const [groupRows_] = groupRows(d);
    const sizeOf = (label) =>
      label === NO_LABEL
        ? d.categories.filter((_, i) => !axisLabelsAt(d, i).length).length
        : groupRows_.find((r) => r.value === label)?.size ?? 0;
    const rows = [...hmap.idx.entries()].map(([label, ci]) => ({
      col: colorFor(ci, false),
      text: `${label === NO_LABEL ? "unlabelled" : displayOf(d, label)} (${sizeOf(label)})`,
    }));
    if (rows.length) {
      const pad = 8;
      const lineH = 18;
      const sw = 10;
      ctx.font = "12px system-ui";
      const tw = Math.max(...rows.map((row) => ctx.measureText(row.text).width));
      const bw = pad * 2 + sw + 6 + tw;
      const bh = pad * 2 + rows.length * lineH - 6;
      ctx.fillStyle = "rgba(255, 255, 255, 0.9)";
      ctx.fillRect(8, 8, bw, bh);
      ctx.strokeStyle = "#d9d9e0";
      ctx.lineWidth = 1;
      ctx.strokeRect(8.5, 8.5, bw - 1, bh - 1);
      rows.forEach((row, i) => {
        const y = 8 + pad + i * lineH + 9;
        ctx.fillStyle = row.col;
        ctx.fillRect(8 + pad, y - 9, sw, sw);
        ctx.fillStyle = "#22222a";
        ctx.fillText(row.text, 8 + pad + sw + 6, y);
      });
    }
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
    // textContent, never innerHTML: these strings come from CSV data.
    const labs = best.cats.length
      ? best.cats.map((l) => `${axisOf(d, l)}: ${displayOf(d, l)}`).join(", ")
      : "no category";
    tooltip.textContent = [best.b, groupText(d, best), labs].join("\n");
    tooltip.style.left = `${ev.clientX + 12}px`;
    tooltip.style.top = `${ev.clientY - 20}px`;
  } else {
    tooltip.hidden = true;
  }
});

function groupText(d, p) {
  if (inCategoryMode()) {
    if (!state.highlights.size) return "no category highlighted";
    // Name EVERY selected label the point carries; the dot only shows the
    // first match's colour, so the tooltip carries the overlap.
    const hmap = highlightMap(d);
    const carried = p.cats.filter((l) => hmap.idx.has(l)).map((l) => displayOf(d, l));
    const unl =
      hmap.idx.has(NO_LABEL) &&
      !p.cats.some((l) => axisOf(d, l) === state.colorBy);
    if (unl && !carried.length) return `no ${state.colorBy} label`;
    if (unl) carried.push(`no ${state.colorBy} label`);
    return carried.length ? carried.join(" · ") : "not highlighted";
  }
  if (state.highlights.has(NO_LABEL)) {
    return p.g >= 0 ? "noise" : "has a cluster";
  }
  if (p.g === null || p.g === undefined) return "unassigned";
  if (p.g === -1) return "noise";
  const [rows] = groupRows(d);
  const row = rows.find((r) => r.value === p.g);
  return row ? `${row.label} · ${row.size}` : `cluster ${p.g}`;
}

canvas.addEventListener("mouseleave", () => (tooltip.hidden = true));
window.addEventListener("resize", draw);

$("list").addEventListener("change", (ev) => {
  const b = ev.target.value;
  if (!b) return;
  ev.target.checked ? state.checked.add(b) : state.checked.delete(b);
  drawLegend(current());
  draw();
});
$("legend").addEventListener("click", (ev) => {
  const row = ev.target.closest("div[data-value]");
  const d = current();
  if (row && d) onLegendClick(d, row.dataset.value);
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
  // Prefer the pa tag as the default view; fall back to insertion order.
  const defaultDz = dzs.includes("C") ? "C" : dzs[0];
  const defaultTag = tags.includes("pa") ? "pa" : tags[0];
  state.key = `${defaultDz}|${defaultTag}`;
  setOptions($("dz"), dzs, defaultDz);
  setOptions($("tag"), tags, defaultTag);
  setOptions($("year"), ["all"], "all");

  const apply = () => {
    // The imputer list depends on dz x tag x cohort, so it is resynced here;
    // an imputer absent from the new view falls back to the aggregate.
    const cur = $("imputer").value;
    const opts = imputerOptions();
    setOptions($("imputer"), opts, opts.includes(cur) ? cur : "aggregate");
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

  $("algo").addEventListener("change", () => {
    state.algo = $("algo").value;
    rebuild();
  });
  $("k").addEventListener("change", () => {
    state.k = Number($("k").value);
    rebuild();
  });
  $("variant").addEventListener("change", () => {
    state.variant = $("variant").value;
    rebuild();
  });
  $("colorby").addEventListener("change", () => {
    state.colorBy = $("colorby").value;
    // Stale labels are dropped inside syncColorBy; NO_LABEL survives there.
    rebuild();
  });
  $("imputer").addEventListener("change", apply);

  // An old positions.js has no clusters; hide the whole apparatus rather than
  // showing dead controls. The imputer dropdown is hidden when the payload has
  // no per-method keys (a pre-method positions.js).
  const hasClusters = Object.values(data).some((c) => c.clusters);
  for (const id of ["algo", "k", "variant", "colorby"]) {
    $(id).closest("label").hidden = !hasClusters;
  }
  const hasMethods = Object.keys(data).some((k) =>
    k.split("|").some((seg) => /^m./.test(seg))
  );
  $("imputer").closest("label").hidden = !hasMethods;
  $("legend-head").hidden = !hasClusters;
  $("legend").hidden = !hasClusters;

  apply();   // not rebuild(): apply() seeds the imputer dropdown on first load
  requestAnimationFrame(draw);
  window.addEventListener("load", draw);
}

if (window.POSITIONS) {
  init(window.POSITIONS);
} else {
  $("meta").textContent = "positions.js missing — run compute_positions.py";
}

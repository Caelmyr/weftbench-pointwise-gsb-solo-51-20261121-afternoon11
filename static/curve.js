import {escape} from "./api.js";

// Station band geometry shared by the workbench chart and the evidence desk.
const PALETTE = {warp: "#2f8f83", weft: "#c9803f", blank: "#9ba1d7", grid: "#d4d8cc"};

function seriesFor(analysis, axis) {
  return (analysis?.[axis] || []).map(row => ({
    station: row.station,
    value: row.count_per_100_mm,
    status: row.status || "",
    pitch_mm: row.pitch_mm,
    strength: row.strength,
  }));
}

export function densityChart(analysis, {onSelect, selected = null, compact = false} = {}) {
  const stations = analysis.stations;
  const warp = seriesFor(analysis, "warp");
  const weft = seriesFor(analysis, "weft");
  const known = [...warp, ...weft].filter(row => row.value !== null && Number.isFinite(row.value));
  if (!known.length) {
    return `<div class="empty">Every station was left blank: the image support or peaks were not separable.</div>`;
  }
  const values = known.map(row => row.value);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = Math.max(hi - lo, hi * 0.04, 0.5);
  const yMin = Math.floor((lo - span * 0.18) * 10) / 10;
  const yMax = Math.ceil((hi + span * 0.18) * 10) / 10;
  const W = compact ? 460 : 560;
  const H = compact ? 220 : 268;
  const pad = {l: 52, r: 14, t: 16, b: 42};
  const x = i => pad.l + ((i + 0.5) / stations) * (W - pad.l - pad.r);
  const y = v => pad.t + (1 - (v - yMin) / (yMax - yMin)) * (H - pad.t - pad.b);
  const bands = Array.from({length: stations}, (_, i) => {
    const left = pad.l + (i / stations) * (W - pad.l - pad.r);
    const width = (W - pad.l - pad.r) / stations;
    return {left, width, index: i};
  });
  const line = rows => {
    const points = rows.filter(row => row.value !== null && Number.isFinite(row.value));
    let d = "";
    let previous = null;
    for (const row of points) {
      const px = x(row.station - 1);
      const py = y(row.value);
      d += previous === null || row.station - previous > 1 ? `M${px.toFixed(1)},${py.toFixed(1)} `
        : `L${px.toFixed(1)},${py.toFixed(1)} `;
      previous = row.station;
    }
    return d.trim();
  };
  const dots = (rows, color) => rows.filter(row => row.value !== null && Number.isFinite(row.value)).map(row => {
    const px = x(row.station - 1);
    const py = y(row.value);
    const tip = `${row.station} · ${row.value} threads/100mm · ${row.status}`;
    return `<circle cx="${px.toFixed(1)}" cy="${py.toFixed(1)}" r="3.4" fill="${color}"><title>${escape(tip)}</title></circle>`;
  }).join("");
  const blankMarks = rows => rows.filter(row => row.value === null).map(row => {
    const px = x(row.station - 1);
    const py = H - pad.b + 14;
    const tip = `${row.station} · blank · ${row.status}`;
    return `<g class="blank-mark"><title>${escape(tip)}</title>
      <line x1="${px - 5}" y1="${py - 5}" x2="${px + 5}" y2="${py + 5}" stroke="${PALETTE.blank}" stroke-width="2"></line>
      <line x1="${px - 5}" y1="${py + 5}" x2="${px + 5}" y2="${py - 5}" stroke="${PALETTE.blank}" stroke-width="2"></line></g>`;
  }).join("");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map(f => {
    const value = yMin + f * (yMax - yMin);
    const py = y(value);
    return `<line x1="${pad.l}" y1="${py}" x2="${W - pad.r}" y2="${py}" stroke="${PALETTE.grid}"></line>
      <text x="${pad.l - 7}" y="${py + 4}" text-anchor="end" class="axis-text">${value.toFixed(1)}</text>`;
  }).join("");
  const xticks = warp.map(row =>
    `<text x="${x(row.station - 1)}" y="${H - pad.b + 32}" text-anchor="middle" class="axis-text">${row.station}</text>`
  ).join("");
  return `<div class="density-curve${onSelect ? " selectable" : ""}">
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Yarn density curve across width stations" data-density-chart>
      ${ticks}
      ${bands.map(band => `<rect class="station-band${selected === band.index ? " selected" : ""}"
        x="${band.left.toFixed(1)}" y="${pad.t}" width="${band.width.toFixed(1)}"
        height="${H - pad.t - pad.b}" data-station="${band.index}"
        ${onSelect ? 'role="button" tabindex="0"' : ""}></rect>`).join("")}
      <path d="${line(warp)}" fill="none" stroke="${PALETTE.warp}" stroke-width="2.2"></path>
      <path d="${line(weft)}" fill="none" stroke="${PALETTE.weft}" stroke-width="2.2"></path>
      ${dots(warp, PALETTE.warp)}${dots(weft, PALETTE.weft)}
      ${blankMarks(warp)}${blankMarks(weft)}
      <text x="${pad.l}" y="${H - 6}" class="axis-text">Station across cloth width</text>
      <text transform="translate(14,${(pad.t + H - pad.b) / 2}) rotate(-90)" text-anchor="middle" class="axis-text">
        Threads / 100 mm</text>
      ${xticks}
    </svg>
    <div class="curve-legend">
      <span><i style="background:${PALETTE.warp}"></i>Warp density</span>
      <span><i style="background:${PALETTE.weft}"></i>Weft density</span>
      <span><i class="blank-key"></i>Blank station</span>
    </div>
    ${onSelect ? `<p class="muted curve-hint">Click a station band to locate it on the original photograph.</p>` : ""}
  </div>`;
}

export function stationBox(analysis, index) {
  const bounds = analysis.station_bounds?.[index];
  const image = analysis.source_dimensions;
  if (!bounds || !image) return null;
  const [x0, y0, x1, y1] = bounds;
  const [width, height] = image;
  return [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height];
}

export function bindDensityChart(root, onSelect) {
  const chart = root.querySelector("[data-density-chart]");
  if (!chart) return;
  const activate = target => {
    const index = Number(target.dataset.station);
    chart.querySelectorAll(".station-band").forEach(band =>
      band.classList.toggle("selected", Number(band.dataset.station) === index));
    onSelect(index);
  };
  chart.querySelectorAll(".station-band").forEach(band => {
    band.addEventListener("click", () => activate(band));
    band.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate(band);
      }
    });
  });
}

import {api, asset, checkbox, choices, escape, field, fields, notes, png, post, readBoxes, select} from "./api.js";
import {lineTypes, pairRows, pane, ribbon, studyTools, toolFields} from "./bench.js";
import {archivePage, handleArchiveForm, loadRecord, renderEvidence, swatchPair} from "./evidence.js";

const app = document.querySelector("#app");
const dialog = document.querySelector("#dialog");
const notice = document.querySelector("#notice");
const state = {
  tab:"library", options:null, fabrics:[], fabric:null, capture:null, plane:null, mode:"measure", tool:"pitch",
  drawings:true, gesture:null, points:[], anchor:null, cursor:null, imageURL:null, archiveRecord:null,
  detailRegions:[], preflightURL:null,
};
let noticeTimer;

function notify(message, error = false) {
  window.clearTimeout(noticeTimer);
  notice.classList.toggle("error", error);
  notice.textContent = message;
  noticeTimer = window.setTimeout(() => { notice.textContent = ""; }, error ? 8500 : 4100);
}

async function action(element, work) {
  if (element?.matches("button")) element.disabled = true;
  try {
    await work();
  } catch (error) {
    notify(error.message, true);
  } finally {
    if (element?.matches("button")) element.disabled = false;
  }
}

function registered() {
  return state.plane !== null;
}
const capturePath = () => "/captures/" + state.capture.id;
const context = () => ({...state, marks:visibleMarks()});
const planeArgs = () => ({plane:state.plane?.id ?? null, registered:registered()});
const roleName = id => state.options.roles[id] || id;

function visibleMarks() {
  if (!state.fabric || !state.capture) return [];
  return state.fabric.marks.filter(row => !row.hidden && (row.basis === "fabric"
    ? state.plane && row.fabric_extent_mm?.join(",") === [state.plane.width_mm,state.plane.height_mm].join(",")
    : row.capture === state.capture.id));
}

function clearGesture() {
  state.gesture = null;
  state.points = [];
  state.anchor = null;
  state.cursor = null;
}

function cleanupResult() {
  if (state.imageURL) URL.revokeObjectURL(state.imageURL);
  state.imageURL = null;
}

function setTab(name, updateHash = true) {
  const valid = ["library","workbench","archive"];
  state.tab = valid.includes(name) ? name : "library";
  if (updateHash && location.hash !== "#" + state.tab) history.replaceState(null, "", "#" + state.tab);
  clearGesture();
  cleanupResult();
  for (const button of document.querySelectorAll("[data-tab]")) {
    const active = button.dataset.tab === state.tab;
    button.classList.toggle("selected", active);
    if (active) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  if (state.tab === "library") renderLibrary();
  else if (state.tab === "workbench") renderBench();
  else renderArchive();
}

async function refreshFabrics() {
  state.fabrics = (await api("/fabrics")).fabrics;
}

function chooseCapture(row, planeId) {
  state.capture = row || null;
  const planes = state.fabric?.planes.filter(item => item.capture === row?.id) || [];
  state.plane = planeId === "" ? null : planes.find(item => item.id === planeId) || planes[0] || null;
  state.mode = "measure";
  state.tool = defaultTool(row);
  clearGesture();
}
function defaultTool(row) {
  if (row?.role === "backlight") return "openings";
  if (row?.role === "rake") return "relief";
  return "pitch";
}

async function openFabric(id) {
  state.fabric = await api("/fabrics/" + id);
  sessionStorage.setItem("weftbench-fabric",id);
  const sources = state.fabric.captures;
  chooseCapture(sources.find(row => row.role === "face" && !/finish|current/i.test(row.title)) ||
    sources.find(row => row.role === "face") || sources[0]);
  state.archiveRecord = null;
  setTab("workbench");
}

async function refreshActive() {
  if (!state.fabric) return;
  const current = state.capture?.id;
  state.fabric = await api("/fabrics/" + state.fabric.id);
  state.capture = state.fabric.captures.find(row => row.id === current) || state.fabric.captures[0] || null;
  if (state.plane) state.plane = state.fabric.planes.find(row => row.id === state.plane.id) || null;
}

function renderLibrary() {
  app.innerHTML = `<section class="hero">
    <div class="hero-copy"><h1>Fabric inspection</h1>
      <p>Read woven detail, record observations, and retain the source images behind each decision.</p>
      <div class="actions"><button data-new-fabric class="primary">Add fabric</button>
      ${state.fabrics.some(row => row.training) ? "" : '<button data-training class="ghost">Load training fabrics</button>'}</div>
    </div>
    <div class="hero-photo"><img src="/api/training-example.png" alt="Close image of woven Juniper twill">
      <small>Synthetic training photograph</small>
    </div>
  </section>
  <div class="section-title"><h2>Fabric library</h2><span class="muted">${state.fabrics.length} ${state.fabrics.length === 1 ? "fabric" : "fabrics"}</span></div>
  ${state.fabrics.length ? `<div class="cards">${state.fabrics.map(row => `<button class="card fabric-card" data-fabric="${row.id}">
    ${row.cover_capture ? `<img src="${asset(row.cover_capture,"thumbnail")}" alt="">` : ""}
    <div><strong>${escape(row.title)}</strong><small>${escape(row.roll_code)} · ${escape(state.options.weaves[row.weave])}</small>
    <small>${row.counts.captures} photographs · ${row.counts.readings} saved readings</small></div>
  </button>`).join("")}</div>` : '<p class="empty">Add a fabric or load the synthetic training set to explore the inspection tools.</p>'}`;
}

function renderBench() {
  const {fabric,capture,plane,drawings} = state;
  if (!fabric) {
    app.innerHTML = '<div class="heading"><h1>Inspection bench</h1></div><p class="empty">Choose a fabric in the library to begin.</p>';
    return;
  }
  const planes = fabric.planes.filter(item => item.capture === capture?.id);
  const planeNames = {"":"Original photo coordinates", ...Object.fromEntries(planes.map(row => [row.id,row.title]))};
  const query = new URLSearchParams({registered:String(registered()),marks:String(drawings)});
  if (plane) query.set("plane",plane.id);
  const sources = fabric.captures.slice().sort((a,b) => a.title.localeCompare(b.title));
  app.innerHTML = `<div class="heading"><div><h1>${escape(fabric.title)}</h1>
    <p class="muted">${escape(fabric.roll_code)} · ${escape(state.options.weaves[fabric.weave])} · ${fabric.width_mm} × ${fabric.length_mm} mm${fabric.training ? ' · Synthetic training fabric' : ''}</p></div>
    <div class="actions"><button data-edit-fabric>Edit fabric</button><button class="primary" data-upload>Import photo</button></div>
  </div>
  ${capture ? `<div class="grid">
    <aside class="panel rail"><h3>Source photographs</h3>${sources.map(row => `<button class="capture ${row.id === capture.id ? "current" : ""}" data-capture="${row.id}">
      <img src="${asset(row.id,"thumbnail")}" alt=""><span>${escape(row.title)}<small>${escape(roleName(row.role))}</small></span>
    </button>`).join("")}</aside>
    <section class="card viewer">
      <div class="viewer-tools"><strong>${escape(capture.title)}</strong>
        <select class="action-end" id="planeSelect" aria-label="Image coordinates">${choices(planeNames,plane?.id || "")}</select>
        <label class="inline"><input id="showMarks" type="checkbox" ${drawings ? "checked" : ""}> Outlines</label>
      </div>
      <div class="stage"><div class="image-frame"><img id="captureImage" src="${"/api" + capturePath()}/display.png?${query}" alt="${escape(capture.title)}">
        <canvas id="paint" aria-label="Image drawing surface"></canvas></div></div>
      ${ribbon(capture,plane)}<div class="viewer-note" id="viewerNote">Select a measurement, mark an observation, or sample the image.</div>
    </section>
    <section class="panel controls" aria-label="Inspection controls">
      <div id="modePane">${pane(state.mode,context())}</div><div class="result" id="result" aria-live="polite"></div>
    </section>
  </div>` : '<p class="empty">Import a fabric photograph to inspect it.</p>'}`;
  canvasEvents();
}

function renderArchive() {
  app.innerHTML = archivePage(state.fabric,state.options);
  if (state.archiveRecord) {
    const target = document.querySelector("#evidenceResult");
    if (target) target.innerHTML = renderEvidence(state.archiveRecord);
  }
}

function showDialog(title, body) {
  if (state.preflightURL) URL.revokeObjectURL(state.preflightURL);
  state.preflightURL = null;
  dialog.innerHTML = `<div class="title-row"><h2>${escape(title)}</h2><button type="button" data-close aria-label="Close dialog">Close</button></div>${body}`;
  if (!dialog.open) dialog.showModal();
}

function fabricDialog(fabric = {}) {
  showDialog(fabric.id ? "Edit fabric" : "Add fabric", `<form id="fabricForm">
    ${field("Fabric name","title","text",fabric.title,"required maxlength=95")}
    ${field("Roll or swatch reference","roll_code","text",fabric.roll_code,"required maxlength=48")}
    ${field("Batch","lot","text",fabric.lot,"maxlength=48")}
    ${select("Weave","weave",state.options.weaves,fabric.weave || "unassigned")}
    <div class="two">${field("Width (mm)","width_mm","number",fabric.width_mm,'required min="30" max="2800" step="any"')}
      ${field("Length (mm)","length_mm","number",fabric.length_mm,'required min="30" max="4800" step="any"')}</div>
    ${notes("Notes","memo",fabric.memo)}
    <button type="submit" class="primary" value="${fabric.id || ""}">Save fabric</button>
  </form>`);
}

function uploadDialog() {
  showDialog("Import photograph", `<form id="uploadForm">
    ${field("Source image","image","file","",'required accept="image/jpeg,image/png,image/webp,image/tiff"')}
    ${field("Photo label","title","text","","maxlength=95")}
    ${select("Light role","role",state.options.roles,"face")}${select("Fabric side","face",state.options.faces,"face")}
    ${field("Captured on","captured_day","date")}${field("Session","session","text","","maxlength=56")}
    ${notes("Notes","memo")}<div class="actions"><button type="submit" value="preflight">Check source first</button>
    <button type="submit" value="save" class="primary">Retain source photo</button></div><div id="preflightResult" class="result"></div>
  </form>`);
}

function result(record, options = {}) {
  const target = document.querySelector("#result");
  if (!target) return;
  const values = record.analysis ?? record;
  const short = Object.entries(values).filter(([,value]) => ["number","boolean","string"].includes(typeof value) && String(value).length < 62).slice(0,8);
  const metrics = short.length ? `<div class="metrics">${short.map(([name,value]) =>
    `<div><small>${escape(name.replaceAll("_"," "))}</small><strong>${escape(value)}</strong></div>`).join("")}</div>` : "";
  target.innerHTML = `<h3>${escape(options.title || record.title || "Image result")}</h3>${metrics}
    ${options.url ? `<img class="result-visual" src="${escape(options.url)}" alt="${escape(options.title || record.title || 'Inspection image')}">` : ""}
    ${record.id ? `<div class="actions"><button data-record="${record.id}">Open retained evidence</button>
      ${record.tool === "color" ? `<button data-approve="${record.id}" data-version="${record.version}">Approve swatch</button>` : ""}</div>` : ""}
    <details><summary>Result details</summary><pre>${escape(JSON.stringify(values,null,2))}</pre></details>`;
}

function previewImage(blob) {
  cleanupResult();
  state.imageURL = URL.createObjectURL(blob);
  return state.imageURL;
}

function paneTo(mode) {
  if (!state.capture) return;
  clearGesture();
  cleanupResult();
  state.detailRegions = [];
  state.mode = mode === "tone" ? "measure" : mode;
  if (mode === "align" && registered()) {
    state.plane = null;
    renderBench();
  } else {
    document.querySelector("#modePane").innerHTML = pane(state.mode,context());
  }
  document.querySelector("#result").innerHTML = "";
  draw();
  if (mode === "tone") showDialog("Display tones", `<form id="toneForm">
    ${field("Black point","black","number",0,'min="0" max=".95" step=".01"')}
    ${field("White point","white","number",1,'min=".05" max="1" step=".01"')}
    ${field("Gamma","gamma","number",1,'min=".2" max="4" step=".1"')}
    ${checkbox("Grayscale","gray")}
    <button type="submit" class="primary">Preview display tones</button>
  </form>`);
}

function messageForGesture(mode) {
  const messages = {probe:"Click a pixel to sample it.",mark:"Click points on the photo to draw your observation.",
    corners:"Click the corners clockwise, starting at the top left.",region:"Drag on the photo to select the analysis area.",
    lit:"Drag a box around the uncovered light reference.",landmarks:"Click a beginning and end for each measured span."};
  return messages[mode];
}

function startGesture(mode) {
  if (!state.capture) return;
  clearGesture();
  state.gesture = mode;
  document.querySelector("#viewerNote").innerHTML = `${escape(messageForGesture(mode))}
    <button type="button" data-undo>Undo point</button> <button type="button" data-clear>Clear</button>`;
  draw();
}

function imagePoint(event) {
  const bounds = event.currentTarget.getBoundingClientRect();
  const unit = (delta,size) => Math.max(0, Math.min(1, Number((delta / size).toFixed(6))));
  return [unit(event.clientX - bounds.left,bounds.width),unit(event.clientY - bounds.top,bounds.height)];
}

function draw() {
  const canvas = document.querySelector("#paint");
  if (!canvas) return;
  const bounds = canvas.getBoundingClientRect();
  if (bounds.width <= 0 || bounds.height <= 0) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(bounds.width * ratio);
  canvas.height = Math.round(bounds.height * ratio);
  const pen = canvas.getContext("2d");
  pen.scale(ratio,ratio);
  const xy = point => [point[0] * bounds.width, point[1] * bounds.height];
  pen.strokeStyle = "#ffe3a1";
  pen.fillStyle = "#ffe3a1";
  pen.lineWidth = 2;
  pen.setLineDash([6,4]);
  if (state.anchor && state.cursor) {
    const a = xy(state.anchor), b = xy(state.cursor);
    pen.strokeRect(...a,b[0]-a[0],b[1]-a[1]);
  }
  if (state.points.length) {
    pen.beginPath();
    state.points.forEach((point,index) => {
      if (!index || (state.gesture === "landmarks" && index % 2 === 0)) pen.moveTo(...xy(point));
      else pen.lineTo(...xy(point));
    });
    if (state.gesture === "corners" && state.points.length === 4) pen.closePath();
    pen.stroke();
    pen.setLineDash([]);
    state.points.forEach((point,index) => {
      const [x,y] = xy(point);
      pen.beginPath();
      pen.arc(x,y,4,0,Math.PI*2);
      pen.fill();
      pen.fillText(String(index + 1),x + 7,y - 5);
    });
  }
}

function placeBox() {
  const selector = state.gesture === "lit" ? "lit" : "box";
  const target = document.querySelector(`[data-box="${selector}"]`);
  if (!target || !state.anchor || !state.cursor) return;
  const [[x,y],[a,b]] = [state.anchor,state.cursor];
  const shape = [Math.min(x,a),Math.min(y,b),Math.abs(x-a),Math.abs(y-b)];
  if (shape[2] < .005 || shape[3] < .005) return;
  Array.from(target.querySelectorAll("input")).forEach((node,i) => { node.value = shape[i].toFixed(6); });
  document.querySelector("#viewerNote").textContent = "Area selected. Run the tool to preview the result.";
}

function canvasEvents() {
  const canvas = document.querySelector("#paint");
  if (!canvas) return;
  document.querySelector("#captureImage").addEventListener("load", draw);
  canvas.addEventListener("pointerdown", event => {
    if (event.button !== 0 || !state.gesture) return;
    if (["region","lit"].includes(state.gesture)) {
      state.anchor = imagePoint(event);
      state.cursor = state.anchor;
      canvas.setPointerCapture(event.pointerId);
    }
  });
  canvas.addEventListener("pointermove", event => {
    if (!state.anchor || !canvas.hasPointerCapture(event.pointerId)) return;
    state.cursor = imagePoint(event);
    draw();
  });
  canvas.addEventListener("pointerup", event => {
    const position = imagePoint(event);
    if (["region","lit"].includes(state.gesture)) {
      state.cursor = position;
      placeBox();
    } else if (state.gesture === "probe") {
      state.points = [position];
      action(null,async () => result(await post(capturePath() + "/probe",{...planeArgs(),point:position}),{title:"Pixel sample"}));
    } else if (["corners","mark","landmarks"].includes(state.gesture)) {
      const limit = state.gesture === "corners" ? 4 : state.gesture === "landmarks" ? 72 : 80;
      if (state.points.length < limit) state.points.push(position);
    }
    draw();
  });
}
window.addEventListener("resize", draw);

async function measure(form, save) {
  const measurement = readBoxes(fields(form),form);
  const tool = measurement.tool;
  delete measurement.tool;
  for (const key of ["mark","seam","cut","trace","reference"]) {
    if (!measurement[key]) delete measurement[key];
  }
  if (tool === "checkpoints") {
    if (!state.plane) throw new Error("Select a calibrated fabric view first.");
    const payload = {
      plane:state.plane.id,
      checkpoints:pairRows(measurement.pairs).map(row => ({name:row.name,measured:row.before,expected:row.after})),
    };
    if (save) {
      const report = await post("/planes/checkpoints",payload);
      await refreshActive();
      result(report,{url:asset(report.id,"map")});
      notify("Calibration review saved.");
    } else {
      result({}, {title:studyTools[tool],url:previewImage(await png("/planes/checkpoints/preview.png",payload))});
    }
    return;
  }
  if (studyTools[tool]) {
    const endpoint = capturePath() + "/study/" + tool;
    const payload = {...planeArgs(),...measurement};
    if (tool === "gauge") {
      if (state.gesture !== "landmarks" || !state.points.length || state.points.length % 2) {
        throw new Error("Draw a beginning and end for every landmark pair.");
      }
      payload.spans = state.points.filter((_,index) => index % 2 === 0).map((begin,index) =>
        ({name:`Landmark span ${index + 1}`,begin,end:state.points[2*index+1]}));
    } else if (tool === "relaxation") {
      payload.pairs = pairRows(measurement.pairs);
      payload.reference_plane = state.fabric.planes.find(row => row.capture === measurement.reference)?.id;
    }
    if (save) {
      const report = await post(endpoint,payload);
      await refreshActive();
      result(report,{url:asset(report.id,report.proofs[0])});
      notify("Reading saved to evidence.");
    } else {
      const report = await post(endpoint + "/preview",payload);
      result(report,{title:studyTools[tool],url:previewImage(await png(endpoint + "/preview.png",payload))});
    }
    return;
  }
  const body = {...planeArgs(),tool,settings:measurement};
  const endpoint = capturePath() + "/reading";
  if (save) {
    const report = await post(endpoint,body);
    await refreshActive();
    const role = ["map","pair","chart","card"].find(name => report.proofs.includes(name));
    result(report,{url:role ? asset(report.id,role) : null});
    notify("Reading saved to evidence.");
  } else {
    const report = await post(endpoint + "/preview",body);
    const role = ["map","pair","chart","card"].find(name => report.proofs.includes(name));
    const image = role ? previewImage(await png(endpoint + "/preview/" + role + ".png",body)) : null;
    result(report,{url:image});
  }
}

async function showRecord(id, version) {
  state.archiveRecord = await loadRecord(id,version);
  setTab("archive");
}

const submitters = {
  async fabricForm(form, button) {
    const payload = fields(form);
    const active = state.fabric;
    const saved = button?.value && active?.id === button.value
      ? await api("/fabrics/" + active.id,{method:"PUT",body:{...payload,version:active.version}})
      : await post("/fabrics",payload);
    dialog.close();
    await refreshFabrics();
    await openFabric(saved.id);
    notify("Fabric saved.");
  },
  async uploadForm(form,button) {
    const upload = new FormData(form);
    for (const [key,value] of upload) if (value === "") upload.delete(key);
    if (button?.value === "preflight") {
      const path = "/fabrics/" + state.fabric.id + "/captures/preflight";
      const checked = await post(path,upload);
      if (state.preflightURL) URL.revokeObjectURL(state.preflightURL);
      state.preflightURL = URL.createObjectURL(await png(path + ".png",upload));
      const matches = checked.visual_matches.map(row => `<li>${escape(row.title)}</li>`).join("");
      document.querySelector("#preflightResult").innerHTML = `<h3>Source preview</h3>
        <img class="result-visual" src="${state.preflightURL}" alt="Source luminance and inspection grid">
        <p>${checked.inspection.extent.join(" × ")} working pixels · ${escape(checked.inspection.color_profile)}</p>
        ${checked.exact_source ? `<p>Already retained as ${escape(checked.exact_source.title)}.</p>` : ""}
        ${matches ? `<p>Visually similar photographs</p><ul>${matches}</ul>` : ""}`;
      return;
    }
    const photo = await post("/fabrics/" + state.fabric.id + "/captures",upload);
    dialog.close();
    await refreshActive();
    chooseCapture(state.fabric.captures.find(row => row.id === photo.id),"");
    renderBench();
    notify("Original photo retained.");
  },
  async measureForm(form, button) { await measure(form,button?.value === "save"); },
  async markForm(form) {
    const payload = fields(form);
    const shape = lineTypes.includes(payload.mark_type) ? "line" : "polygon";
    if (state.gesture !== "mark" || state.points.length < (shape === "line" ? 2 : 3)) {
      throw new Error(shape === "line" ? "Place at least two points on the photo." : "Place at least three outline points on the photo.");
    }
    const found = await post("/fabrics/" + state.fabric.id + "/observations",{
      ...payload,capture:state.capture.id,plane:state.plane?.id ?? null,shape,points:state.points,
    });
    await refreshActive();
    clearGesture();
    renderBench();
    result(found.reading,{title:found.title});
    notify("Observation saved.");
  },
  async planeForm(form, button) {
    if (state.gesture !== "corners" || state.points.length !== 4) throw new Error("Place all four corners before calibrating.");
    const payload = {...fields(form),corners:state.points,fabric_version:state.fabric.version};
    if (button?.value === "save") {
      state.plane = await post(capturePath() + "/plane",payload);
      await refreshActive();
      clearGesture();
      renderBench();
      notify("Physical calibration saved.");
    } else {
      result({}, {title:"Calibrated image preview",url:previewImage(await png(capturePath() + "/plane/preview.png",payload))});
    }
  },
  async proofForm(form, button) {
    const payload = {...readBoxes(fields(form),form),plane:state.plane?.id ?? null};
    if (payload.print_scale) payload.print_scale = Number(payload.print_scale);
    else delete payload.print_scale;
    if (button?.value === "save") {
      const proof = await post(capturePath() + "/proof",payload);
      await refreshActive();
      result(proof,{url:asset(proof.id,"page")});
      document.querySelector("#result .actions").insertAdjacentHTML("beforeend",`<a class="button" href="${asset(proof.id,"pdf",true)}" download>Download PDF</a>`);
      notify("Print proof saved.");
    } else {
      result({}, {title:"Print proof preview",url:previewImage(await png(capturePath() + "/proof/preview.png",payload))});
    }
  },
  async toneForm(form) {
    const parameters = fields(form);
    if (parameters.gray) parameters.channel = "gray";
    const preview = await png(capturePath() + "/tone.png",{...planeArgs(),settings:parameters});
    dialog.close();
    result({}, {title:"Temporary tonal preview",url:previewImage(preview)});
  },
  async detailForm(form,button) {
    const payload = readBoxes(fields(form),form);
    const regions = state.detailRegions.length ? state.detailRegions : [{title:payload.cropTitle || "Detail 1",box:payload.box}];
    const options = {...planeArgs(),title:payload.title,marks:payload.marks,regions};
    if (button?.value === "save") {
      const page = await post(capturePath() + "/details",options);
      await refreshActive();
      result(page,{url:asset(page.id,"map")});
      notify("Native detail sheet saved.");
    } else {
      result({}, {title:"Detail sheet preview",url:previewImage(await png(capturePath() + "/details/preview.png",options))});
    }
  },
};

for (const element of [app,dialog]) element.addEventListener("submit", event => {
  event.preventDefault();
  const form = event.target;
  const button = event.submitter;
  action(button,async () => {
    if (submitters[form.id]) await submitters[form.id](form,button);
    else await handleArchiveForm(form,button,{
      fabric:state.fabric,
      refresh:refreshActive, showRecord, notice:notify,
      showImage:(blob,title) => {
        const box = document.querySelector("#evidenceResult");
        box.innerHTML = `<h3>${escape(title)}</h3><img class="result-visual" src="${previewImage(blob)}" alt="${escape(title)}">`;
      },
    });
  });
});

document.addEventListener("change", event => {
  const element = event.target;
  if (element.id === "planeSelect") {
    state.plane = state.fabric.planes.find(row => row.id === element.value) || null;
    clearGesture();
    renderBench();
  } else if (element.id === "showMarks") {
    state.drawings = element.checked;
    renderBench();
  } else if (element.name === "tool" && element.form?.id === "measureForm") {
    state.tool = element.value;
    clearGesture();
    document.querySelector("#toolFields").innerHTML = toolFields(state.tool,context());
    document.querySelector("#result").innerHTML = "";
    draw();
  }
});

document.addEventListener("click", event => {
  const node = event.target.closest("button");
  if (!node) return;
  const actionKey = node.dataset;
  if (actionKey.tab) setTab(actionKey.tab);
  else if ("close" in actionKey) dialog.close();
  else if ("newFabric" in actionKey) fabricDialog();
  else if ("editFabric" in actionKey) fabricDialog(state.fabric);
  else if ("upload" in actionKey) uploadDialog();
  else if (actionKey.mode) paneTo(actionKey.mode);
  else if (actionKey.draw) startGesture(actionKey.draw);
  else if ("undo" in actionKey) { state.points.pop(); draw(); }
  else if ("clear" in actionKey) { state.points = []; state.anchor = null; state.cursor = null; draw(); }
  else if (actionKey.capture) {
    chooseCapture(state.fabric.captures.find(row => row.id === actionKey.capture));
    cleanupResult();
    renderBench();
  } else if (actionKey.fabric) action(node, () => openFabric(actionKey.fabric));
  else if ("training" in actionKey) action(node,async () => {
    const seed = await post("/training",{});
    await refreshFabrics();
    await openFabric(seed.training[0].fabric);
    notify("Synthetic training fabrics are ready.");
  });
  else if (actionKey.record) action(node, () => showRecord(actionKey.record,actionKey.version));
  else if (actionKey.auditDetail) action(node,async () => {
    const audit = await api("/details/" + actionKey.auditDetail + "/audit");
    document.querySelector("#detailAudit").innerHTML = `<p>${audit.verified ? "Saved crops and coverage maps verified." : "Review the flagged detail files."}</p>
      <p class="muted">${audit.revised_source_count} source revisions since capture.</p>
      <details><summary>Verification details</summary><pre>${escape(JSON.stringify(audit,null,2))}</pre></details>`;
  });
  else if ("addDetail" in actionKey) action(node,async () => {
    if (state.detailRegions.length >= 6) throw new Error("A detail sheet holds up to six crops.");
    const body = readBoxes(fields(node.form),node.form);
    if (body.box.some((value,index) => !Number.isFinite(value) || value < (index < 2 ? 0 : .002)) || body.box[0] + body.box[2] > 1.0000001 || body.box[1] + body.box[3] > 1.0000001) {
      throw new Error("Draw a crop inside the image before adding it.");
    }
    const title = body.cropTitle || `Detail ${state.detailRegions.length + 1}`;
    state.detailRegions.push({title,box:body.box});
    document.querySelector("#detailCrops").innerHTML = state.detailRegions.map((row,index) =>
      `<button class="chip" data-remove-detail="${index}" type="button" aria-label="Remove ${escape(row.title)}">${escape(row.title)} ×</button>`).join("");
  });
  else if ("removeDetail" in actionKey) {
    state.detailRegions.splice(Number(actionKey.removeDetail),1);
    node.remove();
    for (const [index,button] of Array.from(document.querySelectorAll("[data-remove-detail]")).entries()) button.dataset.removeDetail = index;
  }
  else if (actionKey.approve) action(node,async () => {
    const approved = await post("/readings/" + actionKey.approve + "/swatch",{version:Number(actionKey.version)});
    await refreshActive();
    await showRecord(approved.id);
    notify("Recorded image color approved.");
  });
  else if ("pairPhoto" in actionKey) {
    const id = node.form?.elements.reference.value;
    if (!id) return notify("Choose an earlier photograph first.",true);
    document.querySelector("#pairImage").innerHTML = `<img class="result-visual" src="/api/captures/${id}/display.png?marks=false" alt="Earlier calibrated photograph">`;
  } else if ("swatchPair" in actionKey) action(node,async () => {
    const box = await swatchPair(node.form);
    document.querySelector("#evidenceResult").innerHTML = box;
  });
});
window.addEventListener("hashchange", () => setTab(location.hash.slice(1),false));

action(null,async () => {
  try {
    [state.options] = await Promise.all([api("/options"),refreshFabrics()]);
    const tab = location.hash.slice(1);
    const saved = sessionStorage.getItem("weftbench-fabric");
    if (["workbench","archive"].includes(tab) && state.fabrics.some(row => row.id === saved)) await openFabric(saved);
    setTab(tab,false);
  } finally {
    app.setAttribute("aria-busy","false");
  }
});

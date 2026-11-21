import {asset, boxes, checkbox, escape, field, notes, select} from "./api.js";

export const lineTypes = ["seam","cut","selvedge"];
export const studyTools = {
  findings:"Mapped observation inventory",
  gauge:"Physical landmark gauge",
  relaxation:"Before/after dimension check",
  checkpoints:"Independent calibration check",
};

export const checkboxMarks = (label, mode) => `<button type="button" data-draw="${mode}">${label}</button>`;
const number = (label, key, value, min, max, step = "any") =>
  field(label, key, "number", value, `min="${min}" max="${max}" step="${step}"`);
const few = output => `<div class="two">${output.join("")}</div>`;
const region = () => `<label class="field">Analysis area · normalized</label>${boxes()}${checkboxMarks("Draw area on image","region")}`;
const defaultBox = () => `<details><summary>Analysis area</summary>${region()}</details>`;

export function captureChoices(fabric, exclude, predicate = () => true) {
  return {"":"Choose a photograph", ...Object.fromEntries(fabric.captures
    .filter(item => item.id !== exclude && predicate(item)).map(item => [item.id, item.title]))};
}

function markChoices(marks, kinds, includeNone = true) {
  return {...(includeNone ? {"":"Choose an observation"} : {}),
    ...Object.fromEntries(marks.filter(row => kinds.includes(row.mark_type)).map(row => [row.id,row.title]))};
}

const settingsFor = {
  audit: () => few([number("Low signal", "low", 0.045, 0.005, 0.30), number("Highlight threshold", "high", 0.94, 0.60, 0.999)]),
  pitch: () => few([number("Shortest spacing (px)","min_period",3,2.2,24), number("Longest spacing (px)","max_period",36,3,100)]),
  skew: () => "",
  bands: () => select("Trace direction","axis",{"cross-roll":"Across the width","along-roll":"Along the roll"},"cross-roll"),
  openings: () => few([number("Local contrast","contrast",0.145,0.015,0.85), number("Minimum area (px²)","min_area",20,2,900,1)]),
  mesh: () => `<p class="muted">Draw one box over intact mesh as the lattice reference, then a separate box over the region to inspect.</p>
    <label class="field">Intact mesh reference · normalized</label>${boxes("reference",[0.06,0.08,0.24,0.22])}${checkboxMarks("Draw intact reference","reference")}
    ${few([number("Opening contrast","contrast",0.12,0.02,0.8), number("Enlarged ratio","growth",1.35,1.05,4)])}`,
  passage: () => `<p class="muted">Use a transmitted-light photo. Place the reference over uncovered light.</p>
    <label class="field">Light reference · normalized</label>${boxes("lit",[0.715,0.08,0.18,0.10])}${checkboxMarks("Draw light reference","lit")}`,
  relief: () => "",
  abrasion: () => select("Strip direction","direction",{horizontal:"Horizontal",vertical:"Vertical"},"horizontal"),
  color: ({marks}) => select("Optional color region","mark",{"":"Use analysis area", ...markChoices(marks,["area","proof","stain"],false)},""),
  allowance: ({marks}) => select("Sewn trace","seam",markChoices(marks,["seam"])) +
    select("Cut trace","cut",markChoices(marks,["cut","selvedge"])) +
    select("Search side","side",{nearest:"Nearest",positive:"Positive normal",negative:"Negative normal"},"nearest"),
  stitches: ({marks}) => select("Sewn trace","trace",markChoices(marks,["seam"])) +
    select("Stitch contrast","polarity",{light:"Light",dark:"Dark",either:"Either"},"light"),
  section: ({marks}) => select("Profile trace","trace",markChoices(marks,lineTypes.slice())),
  compare: ({fabric,capture}) => select("Reference photograph","reference",captureChoices(fabric,capture.id, row => row.role === capture.role)) +
    select("Difference","metric",{light:"Luminance",color:"Image color"},"light"),
  coverage: ({fabric,capture}) => select("Reference photograph","reference",captureChoices(fabric,capture.id)),
  findings: ({options}) => select("Observation category","category",{any:"All categories",...options.mark_types},"any") +
    select("Observation state","state",{any:"Any state",...options.observation_states},"any"),
  gauge: () => `<p class="muted">Click landmarks in pairs on the calibrated image.</p>${checkboxMarks("Draw landmark pairs","landmarks")}`,
  relaxation: ({fabric,capture}) => select("Earlier photograph","reference",captureChoices(fabric,capture.id, row => row.role === capture.role)) +
    `<p class="muted">Enter matching landmark positions in the earlier and current calibrated views.</p>
    <div class="actions"><button type="button" data-pair-photo>Show earlier image</button></div>
    <div id="pairImage"></div>
    <label class="field">Landmark pairs, one row per earlier x, earlier y, current x, current y
    <textarea name="pairs" placeholder="0.20, 0.25, 0.21, 0.23"></textarea></label>`,
  checkpoints: () => `<p class="muted">Use recognizable points inside the crop to check a saved calibration.</p>
    <label class="field">Checkpoints, one row per original x, original y, expected fabric x, expected fabric y
    <textarea name="pairs" placeholder="0.20, 0.25, 0.21, 0.23"></textarea></label>`,
};

export function toolFields(tool, context) {
  const options = settingsFor[tool]?.(context) || "";
  const useArea = !["allowance","stitches","section","gauge","relaxation","checkpoints"].includes(tool);
  return options + (useArea ? defaultBox() : "");
}

export function pane(mode, context) {
  const {fabric,capture,plane,options,marks,tool} = context;
  const top = `<div class="action-tabs" aria-label="Image actions">
    ${Object.entries({measure:"Measure",mark:"Observe",align:"Calibrate",proof:"Print",detail:"Details"}).map(([key,name]) =>
      `<button data-mode="${key}" class="${key === mode ? "active" : ""}">${name}</button>`).join("")}
  </div>`;
  const readings = Object.fromEntries(Object.entries(options.tools).map(([id,item]) => [id,item.name]));
  if (mode === "measure") return top + `<form id="measureForm">
    ${select("Inspection instrument","tool",{...readings,...studyTools},tool)}
    <div id="toolFields">${toolFields(tool,context)}</div>
    <div class="actions"><button type="submit" value="preview">Run inspection</button>
      <button type="submit" value="save" class="primary">Save reading</button></div>
  </form>`;
  if (mode === "mark") return top + `<form id="markForm">
    ${field("Observation title","title","text","","required maxlength=80")}
    ${select("Mark type","mark_type",options.mark_types,"area")}
    ${select("Review state","state",options.observation_states,"new")}
    ${notes("Note","memo")}
    <div class="actions">${checkboxMarks("Place points","mark")}<button type="submit" class="primary">Save observation</button></div>
  </form>${marks.length ? `<details><summary>${marks.length} existing observations</summary>
    <div class="record-list">${marks.map(row => `<button type="button" class="record-row" data-record="${row.id}">${escape(row.title)}</button>`).join("")}</div></details>` : ""}`;
  if (mode === "align") return top + `<form id="planeForm">
    <p class="muted">Use the raw view. Place four corners in order from the top left, clockwise.</p>
    <div class="chips"><span class="chip">${fabric.width_mm} × ${fabric.length_mm} mm</span></div>
    ${field("Calibration name","title","text","Aligned fabric face","required maxlength=95")}
    ${number("Output pitch (mm per pixel)","pitch",0.71,0.06,8)}
    <div class="actions">${checkboxMarks("Place corners","corners")}
      <button type="submit" value="preview">Preview</button>
      <button type="submit" value="save" class="primary">Save calibration</button></div>
  </form>`;
  if (mode === "detail") return top + `<form id="detailForm">
    ${field("Sheet name","title","text","Fabric detail sheet","required maxlength=90")}
    ${field("Detail name","cropTitle","text","","maxlength=55")}
    ${region()}
    ${checkbox("Show outlines in the overview","marks",true)}
    <div class="actions"><button type="button" data-add-detail>Add this crop</button></div>
    <div id="detailCrops" class="chips"></div>
    <div class="actions"><button type="submit" value="preview">Preview details</button>
      <button type="submit" value="save" class="primary">Save detail sheet</button></div>
  </form>`;
  return top + `<form id="proofForm">
    ${field("Proof title","title","text","Fabric proof","required maxlength=90")}
    ${select("Paper size","paper",{a4:"A4",a3:"A3",letter:"US Letter"},"a4")}
    ${checkbox("Landscape","landscape")}${checkbox("Show observation outlines","marks",true)}
    ${select("Proof scale","print_scale",{"":"Fit to page","0.5":"1:2","1":"1:1"}, "", plane ? "" : "disabled")}
    <details><summary>Crop for print</summary>${region()}</details>
    <div class="actions"><button type="submit" value="preview">Preview proof</button>
      <button type="submit" value="save" class="primary">Save PDF</button></div>
  </form>`;
}

export function ribbon(capture, plane) {
  return `<div class="viewer-tools">
    ${capture?.id ? `<button type="button" data-draw="probe">Sample pixel</button><button type="button" data-mode="tone">Display tones</button>
      <a class="button" href="${asset(capture.id,"original",true)}" download>Original</a>` : ""}
    ${plane ? '<span class="chip">Physical scale</span>' : '<span class="chip">Camera coordinates</span>'}
  </div>`;
}

export function pairRows(text) {
  return text.split(/\r?\n/).filter(line => line.trim()).map((line,index) => {
    const values = line.split(",").map(value => value.trim()).map(Number);
    if (values.length !== 4 || values.some(value => !Number.isFinite(value) || value < 0 || value > 1)) {
      throw new Error(`Enter four normalized coordinates on line ${index + 1}.`);
    }
    return {name:`Landmark ${index + 1}`,before:values.slice(0,2),after:values.slice(2)};
  });
}

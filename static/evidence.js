import {api, asset, checkbox, escape, field, fields, notes, png, post, select} from "./api.js";

function list(rows, empty) {
  return rows.length ? `<div class="record-list">${rows.map(row =>
    `<button type="button" class="record-row" data-record="${row.id}">${escape(row.title)}<small>${escape(row.kind || row.state || "")}</small></button>`
  ).join("")}</div>` : `<p class="muted">${escape(empty)}</p>`;
}

function selectable(rows) {
  return `<div class="checklist">${rows.map(row =>
    `<label><input type="checkbox" name="entries" value="${row.id}"><span>${escape(row.title)} <small>· ${escape(row.kind)}</small></span></label>`
  ).join("")}</div>`;
}

export function archivePage(fabric, options) {
  const intake = `<section class="panel"><h3>Verify a portable evidence bundle</h3>
    <form id="receiveForm">${field("Evidence ZIP","archive","file","",'required accept=".zip,application/zip"')}
    <button type="submit">Verify and retain</button></form></section>`;
  const heading = `<div class="heading"><div><h1>Evidence desk</h1>${fabric ? `<p class="muted">${escape(fabric.title)}</p>` : ""}</div></div>`;
  if (!fabric) {
    return heading + `<div class="archive-grid"><div>${intake}</div>
      <section class="panel" id="evidenceResult"><p class="muted">Choose a fabric to retrieve its retained evidence.</p></section></div>`;
  }
  const groups = [
    ["Saved readings",fabric.readings],["Observed marks",fabric.marks],
    ["Approved color samples",fabric.swatches],["Print proofs",fabric.proofs || []],["Native detail sheets",fabric.details || []],
    ["Inspection visits",fabric.visits],["Portable bundles",fabric.packets],
  ];
  const photos = [...fabric.captures,...fabric.readings,...fabric.swatches,...(fabric.proofs || []),...(fabric.details || [])];
  const swatches = Object.fromEntries(fabric.swatches.filter(row => row.state === "approved").map(row => [row.id,row.title]));
  return heading + `<div class="archive-grid"><div>
    ${groups.map(([name,rows]) => `<section class="panel"><h3>${escape(name)}</h3>${list(rows,"None retained.")}</section>`).join("")}
  </div><div>
    <section class="panel" id="evidenceResult"><h3>Retained record</h3><p class="muted">Open a record to see its images, sources, and revisions.</p></section>
    <section class="panel"><h3>Contact sheet or portable bundle</h3>
      ${photos.length ? `<form id="packetForm">
        ${field("Report title","title","text","","maxlength=90")}
        <p class="muted">Select retained image evidence for the report.</p>${selectable(photos)}
        <div class="two">${select("Paper","paper",{a4:"A4",a3:"A3",letter:"US Letter"},"a4")}
          ${select("Panel layout","layout",{large:"Four large panels",grid:"Six panels",strip:"Three rows"},"large")}</div>
        ${checkbox("Landscape","landscape")}
        <div class="actions"><button type="submit" value="preview">Preview contact sheet</button>
          <button type="submit" value="pdf">Save contact PDF</button><button type="submit" value="zip" class="primary">Create evidence ZIP</button></div>
      </form>` : '<p class="muted">Retain a source photograph to assemble a report.</p>'}
    </section>
    ${Object.keys(swatches).length >= 2 ? `<section class="panel"><h3>Compare approved samples</h3>
      <form id="swatchPair">${select("First swatch","first",swatches)}${select("Second swatch","second",swatches)}
        <button type="button" data-swatch-pair>Compare image colors</button></form></section>` : ""}
    <section class="panel"><h3>Record an inspection visit</h3><form id="visitForm">
      ${field("Visit title","title","text","","required maxlength=94")}
      ${field("Date","observed_day","date",new Date().toISOString().slice(0,10),"required")}
      ${select("Visit type","visit_type",options.visit_types)}${select("Decision","outcome",options.visit_outcomes)}
      ${notes("Notes","memo")}
      ${selectable([...fabric.marks,...fabric.readings,...(fabric.details || [])])}
      <button type="submit" class="primary">Save visit</button>
    </form></section>
    ${intake}
  </div></div>`;
}

export async function loadRecord(id, version) {
  return api("/records/" + id + (version ? "?version=" + encodeURIComponent(version) : ""));
}

function media(record) {
  const files = record.assets || [];
  const artwork = ["map","pair","page","card","chart","working","plane","image-001"]
    .map(role => files.find(file => file.role === role)).find(Boolean);
  return `${artwork ? `<img class="result-visual" src="${asset(record.id,artwork.role)}" alt="${escape(record.title)} · ${escape(artwork.role)}">` : ""}
    ${record.kind === "visits" ? `<img class="result-visual" src="/api/visits/${record.id}/proof.png?version=${record.version}" alt="Visit review sheet">` : ""}
    ${files.length > 0 ? `<div class="actions">${files.map(file =>
      `<a class="button" href="${asset(record.id,file.role,true)}" download>↓ ${escape(file.role)}</a>`
    ).join("")}</div>` : ""}`;
}

export function renderEvidence(record) {
  return `<h2>${escape(record.title)}</h2>
    <div class="chips"><span class="chip">Revision ${record.version}</span><span class="chip">${escape(record.kind)}</span>
    ${record.review_state ? `<span class="chip">${escape(record.review_state)}</span>` : ""}</div>${media(record)}
    ${record.sources?.length ? `<details open><summary>Source citations</summary><div class="record-list">${record.sources.map(row => `
       <button type="button" class="record-row" data-record="${row.source}" data-version="${row.version}">
         ${escape(row.role)} <small>revision ${row.version}</small></button>`).join("")}</div></details>` : ""}
    ${record.history?.length ? `<details><summary>Revision history</summary><div class="record-list">${record.history.map(row => `
       <button type="button" class="record-row" data-record="${record.id}" data-version="${row.version}">
         ${escape(row.action)} <small>revision ${row.version}</small></button>`).join("")}</div></details>` : ""}
    ${record.kind === "readings" ? `<form id="decisionForm" data-record-id="${record.id}" data-version="${record.version}">
      ${select("Review decision","state",{unreviewed:"Unreviewed","look again":"Look again",accepted:"Accepted",retired:"Retired"},record.review_state)}
      ${notes("Review note","note",record.review_note)}<div class="actions"><button type="submit">Update decision</button>
      ${record.tool === "color" ? `<button type="button" data-approve="${record.id}" data-version="${record.version}">Approve swatch</button>` : ""}</div>
    </form>` : ""}
    ${record.kind === "details" ? `<button type="button" data-audit-detail="${record.id}">Verify native crops</button><div id="detailAudit"></div>` : ""}
    <details><summary>Record details</summary><pre>${escape(JSON.stringify(record.analysis || record,null,2))}</pre></details>`;
}

export async function handleArchiveForm(form, button, controls) {
  if (form.id === "packetForm") {
    const payload = fields(form);
    payload.entries = new FormData(form).getAll("entries");
    if (!payload.title) delete payload.title;
    const base = "/fabrics/" + controls.fabric.id;
    if (button?.value === "preview") {
      controls.showImage(await png(base + "/contact/preview.png",payload),"Contact sheet preview");
      return;
    }
    const output = await post(base + (button?.value === "zip" ? "/pack" : "/contact"),payload);
    await controls.refresh();
    await controls.showRecord(output.id);
    controls.notice(button?.value === "zip" ? "Portable evidence bundle saved." : "Contact sheet saved.");
  } else if (form.id === "receiveForm") {
    const output = await post("/received",new FormData(form));
    await controls.showRecord(output.id);
    controls.notice(output.duplicate ? "This bundle was already verified." : "Bundle verified and retained.");
  } else if (form.id === "visitForm") {
    const payload = fields(form);
    delete payload.entries;
    payload.observations = new FormData(form).getAll("entries");
    const output = await post("/fabrics/" + controls.fabric.id + "/visits",payload);
    await controls.refresh();
    await controls.showRecord(output.id);
    controls.notice("Inspection visit saved.");
  } else if (form.id === "decisionForm") {
    const payload = {...fields(form),version:Number(form.dataset.version)};
    const output = await api("/readings/" + form.dataset.recordId + "/decision",{method:"PUT",body:payload});
    await controls.refresh();
    await controls.showRecord(output.id);
    controls.notice("Review decision updated.");
  }
}

export async function swatchPair(form) {
  const query = new URLSearchParams(new FormData(form));
  const pair = await api("/swatches/pair?" + query);
  return `<h3>Image color comparison</h3><p>Recorded color distance · ${escape(pair.raw_image_distance)}</p>
    <img class="result-visual" src="/api/swatches/pair.png?${query}" alt="Side-by-side approved swatches">
    <p class="muted">${escape(pair.note)}</p>`;
}

export async function api(path, {method = "GET", body, image = false} = {}) {
  const serialized = body === undefined || body instanceof FormData ? body : JSON.stringify(body);
  const response = await fetch("/api" + path, {
    method, body: serialized,
    headers: typeof serialized === "string" ? {"Content-Type": "application/json"} : undefined,
  });
  if (!response.ok) {
    const failure = await response.json().catch(() => ({}));
    throw new Error(failure.error?.message || `Request failed (${response.status}).`);
  }
  return image ? response.blob() : response.json();
}

export const post = (path, body) => api(path, {method:"POST", body});
export const png = (path, body) => api(path, {method:"POST", body, image:true});
export const asset = (record, role, download = false) =>
  `/api/assets/${encodeURIComponent(record)}/${encodeURIComponent(role)}${download ? "?download=true" : ""}`;

export function escape(value = "") {
  return String(value ?? "").replace(/[&<>"']/g, char => ({
    "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;",
  }[char]));
}

export function choices(entries, selected = "") {
  return Object.entries(entries).map(([key, name]) =>
    `<option value="${escape(key)}"${String(key) === String(selected) ? " selected" : ""}>${escape(name)}</option>`
  ).join("");
}

export function field(label, name, type = "text", value = "", attrs = "") {
  return `<label class="field">${escape(label)}<input name="${escape(name)}" type="${type}" value="${escape(value)}" ${attrs}></label>`;
}

export function select(label, name, entries, selected, attrs = "") {
  return `<label class="field">${escape(label)}<select name="${escape(name)}" ${attrs}>${choices(entries, selected)}</select></label>`;
}

export function notes(label, name, value = "") {
  return `<label class="field">${escape(label)}<textarea name="${escape(name)}">${escape(value)}</textarea></label>`;
}

export function checkbox(label, name, checked = false) {
  return `<label class="inline"><input type="checkbox" name="${name}" ${checked ? "checked" : ""}> ${escape(label)}</label>`;
}

export function fields(form) {
  const built = Object.fromEntries(new FormData(form));
  for (const input of form.querySelectorAll('input[type="number"],input[type="range"],input[type="checkbox"]')) {
    if (input.type === "checkbox") built[input.name] = input.checked;
    else if (input.value === "") delete built[input.name];
    else built[input.name] = Number(input.value);
  }
  return built;
}

export function boxes(id = "box", coords = [0,0,1,1]) {
  return `<div class="four" data-box="${id}">${["Left","Top","Width","Height"].map((label, index) =>
    field(label, `${id}-${index}`, "number", coords[index], 'min="0" max="1" step="any" required')
  ).join("")}</div>`;
}

export function readBoxes(body, form) {
  for (const section of form.querySelectorAll("[data-box]")) {
    const id = section.dataset.box;
    body[id] = Array.from(section.querySelectorAll("input"), input => Number(input.value));
    body[id].forEach((_, i) => delete body[`${id}-${i}`]);
  }
  return body;
}

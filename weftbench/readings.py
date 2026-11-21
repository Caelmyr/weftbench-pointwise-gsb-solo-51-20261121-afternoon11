"""Inspection dispatch, immutable evidence, and frozen color approval."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import comparison, inspection, seams, surface, weave
from .catalog import detached, version, view
from .contracts import (
    MAX_RESULTS, WorkbenchError, asset_label, choice, conflict, flag, identifier,
    limit, mapping, text,
)
from .raster import png, to_srgb
from .visual import neutral_card


@dataclass(frozen=True)
class InspectionTool:
    name: str
    family: str
    handler: object
    paired: bool = False

    def public(self):
        return {"name": self.name, "family": self.family, "paired": self.paired}


TOOLS = {
    "audit": InspectionTool("Exposure and image support", "Image review", inspection.audit),
    "passage": InspectionTool("Relative light passage", "Transmitted light", inspection.passage),
    "openings": InspectionTool("Opening candidates", "Transmitted light", surface.openings),
    "pitch": InspectionTool("Warp and weft pitch", "Weave reading", weave.yarn_density),
    "skew": InspectionTool("Thread direction across width", "Weave reading", weave.local_skew),
    "bands": InspectionTool("Periodic shading bands", "Weave reading", weave.banding),
    "color": InspectionTool("Cloth color and variation", "Color observation", inspection.swatch_reading),
    "relief": InspectionTool("Surface fibre relief", "Raking light", surface.relief),
    "abrasion": InspectionTool("Optical abrasion strips", "Raking light", surface.abrasion_profile),
    "allowance": InspectionTool("Seam allowance", "Sewn finish", seams.allowance),
    "stitches": InspectionTool("Stitch rhythm", "Sewn finish", seams.stitches),
    "section": InspectionTool("Three-way surface profile", "Sewn finish", seams.edge_profile),
    "compare": InspectionTool("Calibrated image difference", "Paired evidence", comparison.compare, True),
    "coverage": InspectionTool("Photo coverage ledger", "Paired evidence", comparison.coverage, True),
}


def inventory():
    return {key: tool.public() for key, tool in TOOLS.items()}


def parse_context(lib, frame, form):
    selected = identifier(form.get("plane"), "calibration", optional=True)
    use_registration = flag(form.get("registered"), "use calibrated fabric", default=True)
    if selected is not None and not use_registration:
        raise WorkbenchError("The chosen plane requires calibrated viewing.", "registered")
    return view(lib, frame, plane_key=selected, registered=True if selected else None if use_registration else False)


def input_receipt(ctx, settings):
    used = settings.copy()
    used.pop("title", None)
    used.pop("name", None)
    return {
        "source_dimensions": list(ctx.surface.extent),
        "capture": ctx.capture["id"], "capture_version": ctx.capture["version"],
        "capture_role": ctx.capture["role"], "capture_day": ctx.capture["captured_day"],
        "fabric": ctx.fabric["id"], "fabric_version": ctx.fabric["version"],
        "plane": ctx.plane["id"] if ctx.plane else None, "coordinate_basis": ctx.basis,
        "scale": ctx.scale.tolist() if ctx.scale is not None else None,
        "settings": used,
    }


def analyze(lib, key, raw):
    form = mapping(raw)
    tool_id = choice(form.get("tool"), "inspection tool", TOOLS)
    tool = TOOLS[tool_id]
    args = mapping(form.get("settings", {}), "inspection settings")
    context = parse_context(lib, key, form)
    if tool.paired:
        report, images, secondary = tool.handler(context, args, lib)
    else:
        report, images = tool.handler(context, args)
        secondary = []
    sources = context.references() + secondary
    used = []
    for item, role in sources:
        item_key = (item["id"], role)
        if item_key not in [(x["id"], y) for x, y in used]:
            used.append((item, role))
    crucial = (args.get("mark"), args.get("seam"), args.get("cut"), args.get("trace"))
    for mark in context.marks:
        if mark["mark_type"] in {"ignore", "fold"} or mark["id"] in crucial:
            used.append((lib.get(mark["id"], "marks"), "observation in this reading"))
    output = {
        "tool": tool_id, "title": text(form.get("title"), "reading name", 95, default=tool.name),
        "parameters": input_receipt(context, args), "analysis": report,
        "proofs": sorted(images), "notes": text(form.get("notes"), "reading note", 1200, default="", multiline=True),
    }
    return context, output, images, used


def preview(store, key, raw):
    with store.open() as lib:
        _, result, _, _ = analyze(lib, key, raw)
        return result


def rendered_preview(store, key, raw, role):
    role = asset_label(role)
    with store.open() as lib:
        _, _, pictures, _ = analyze(lib, key, raw)
        if role not in pictures:
            raise WorkbenchError("This tool has no image with that role.", status=404)
        return pictures[role]


def save_reading(store, frame, form):
    with store.open() as lib:
        context, record, pictures, references = analyze(lib, frame, form)
    parent = context.fabric["id"]
    encoded = {role: png(artwork) for role, artwork in pictures.items()}
    with store.open(write=True) as lib:
        if lib.count("readings", parent=parent) >= MAX_RESULTS:
            raise limit("This fabric has reached its limit of retained readings.")
        for source, _ in references:
            lib.require_version(lib.get(source["id"], source["kind"]), source["version"])
        saved = lib.add("readings", record | {"review_state": "unreviewed"},
                        parent=parent, action="inspection and imagery retained")
        for source, role in references:
            lib.cite(saved["id"], source, role)
        for role, artwork in encoded.items():
            lib.attach(saved["id"], role, artwork, "image/png")
        return lib.public(saved, links=True)


def review_state(store, key, form):
    mark = choice(form.get("state"), "review decision", {"unreviewed", "look again", "accepted", "retired"})
    note = text(form.get("note"), "review note", 1600, default="", multiline=True)
    with store.open(write=True) as lib:
        previous = lib.get(key, "readings")
        new = lib.revise(previous, version(form.get("version")), detached(previous) | {
            "review_state": mark, "review_note": note,
        }, action="inspection review decision")
        return lib.public(new, links=True)


def approve_swatch(store, key, form):
    label_ = text(form.get("title"), "approved swatch name", 80, default="")
    with store.open(write=True) as lib:
        reading = lib.get(key, "readings")
        if reading["tool"] != "color":
            raise WorkbenchError("Approve a swatch from a retained color observation.", "reading")
        if form.get("version") is not None:
            lib.require_version(reading, version(form["version"]))
        used = lib.list("swatches", parent=reading["parent"], limit=80)
        if len(used) >= 80:
            raise limit("This fabric has reached its limit of approved swatches.")
        if any(row["reading"] == key and row["state"] == "approved" for row in used):
            raise conflict("A swatch from this color reading is already approved.")
        current = reading["analysis"]
        capture = lib.get(reading["parameters"]["capture"], "captures")
        sample = lib.add("swatches", {
            "title": label_ or current["name"], "reading": key,
            "measurement_version": reading["version"], "capture": capture["id"],
            "capture_role": reading["parameters"]["capture_role"], "color": current["color"],
            "variation": current["color_variability"], "pixels": current["pixel_count"],
            "state": "approved",
        }, parent=reading["parent"], action="image sample approved")
        lib.cite(sample["id"], reading, "frozen reported color")
        for link in lib.citations(reading["id"]):
            if link["source"] == capture["id"]:
                capture_version = lib.revision(capture["id"], link["version"])
                lib.cite(sample["id"], capture_version, "supporting photo")
                break
        palette = neutral_card([{"label": sample["title"], "color": sample["color"]}], "Approved image swatch")
        lib.attach(sample["id"], "card", png(palette), "image/png")
        return lib.public(sample, links=True)


def swatch_book(store, parent, *, retired=False):
    with store.open() as lib:
        lib.get(parent, "fabrics")
        rows = lib.list("swatches", parent=parent, limit=80)
        return [lib.public(row, links=True) for row in rows if retired or row["state"] == "approved"]


def retire_swatch(store, key, form):
    with store.open(write=True) as lib:
        swatch = lib.get(key, "swatches")
        new = lib.revise(swatch, version(form.get("version")), detached(swatch) | {"state": "retired"},
                         action="removed from active swatch book")
        return lib.public(new, links=True)


def compare_swatches(store, first, second):
    if first == second:
        raise WorkbenchError("Choose a second approved color sample.", "swatches")
    with store.open() as lib:
        left, right = [lib.get(key, "swatches") for key in (first, second)]
        if left["state"] != "approved" or right["state"] != "approved":
            raise WorkbenchError("Select two currently approved swatches.", "swatches")
        first_lab = np.asarray(left["color"]["lab_d65"], dtype=np.float64)
        second_lab = np.asarray(right["color"]["lab_d65"], dtype=np.float64)
        delta = round(float(np.linalg.norm(first_lab - second_lab)), 5)
        possible = left["capture_role"] == right["capture_role"]
        result = {
            "first": lib.public(left, links=True), "second": lib.public(right, links=True),
            "raw_image_distance": delta, "same_light_role": possible,
            "note": "The distance describes stored image colors, including lighting and source exposure.",
        }
        paired = neutral_card([
            {"label": left["title"], "color": left["color"]},
            {"label": right["title"], "color": right["color"], "lab_distance": delta},
        ], "Recorded swatches in their respective light")
        return result, paired


def tone_reference_from_meter(r, g, b):
    colors = np.asarray([r, g, b], dtype=np.float32)
    return np.rint(255 * to_srgb(colors)).astype(int).tolist()

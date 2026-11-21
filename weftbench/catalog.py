"""Fabric records, source image receipts, calibrations and observation geometry."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256

import numpy as np

from .contracts import (
    ACTIONS, FACES, LINE_TYPES, LOCKED_FIELDS, MARK_TYPES, MASK_TYPES, MAX_CAPTURES,
    MAX_FABRICS, MAX_FILE_BYTES, MAX_MARKS, ROLES, STATES, WEAVES, WorkbenchError,
    choice, conflict, date, flag, identifier, integer, limit, mapping, real, text,
)
from .geometry import (
    Calibration, hole_polygons, line_points, masks_for, move_geometry,
    physical_span, polygon, polygon_mask, rectify,
)
from .raster import decode_image, fit_image, open_work, png, read_fraction, rgba

FABRIC_FIELDS = ("title", "roll_code", "lot", "weave", "width_mm", "length_mm", "memo", "state")
CAPTURE_FIELDS = ("title", "role", "face", "session", "captured_day", "memo", "state")


def retained(record, keys):
    if record is None:
        return {}
    return {key: record[key] for key in keys if key in record}


def detached(record):
    return {key: val for key, val in record.items() if key not in LOCKED_FIELDS}


def version(value):
    return integer(value, "record version", 1, 99999999)


def fabric_values(form, former=None):
    form = retained(former, FABRIC_FIELDS) | form
    return {
        "title": text(form.get("title"), "fabric name", 95),
        "roll_code": text(form.get("roll_code"), "roll or swatch reference", 48),
        "lot": text(form.get("lot"), "batch label", 48, default=""),
        "weave": choice(form.get("weave"), "weave", WEAVES, "unassigned"),
        "width_mm": real(form.get("width_mm"), "measured width", 30, 2800),
        "length_mm": real(form.get("length_mm"), "measured length", 30, 4800),
        "memo": text(form.get("memo"), "fabric note", 1200, default="", multiline=True),
        "state": choice(form.get("state"), "fabric state", ACTIONS, "active"),
    }


def save_fabric(store, raw, key=None):
    form = mapping(raw, "fabric")
    with store.open(write=True) as lib:
        previous = lib.get(key, "fabrics") if key else None
        definition = fabric_values(form, previous)
        if previous is None:
            if lib.count("fabrics") >= MAX_FABRICS:
                raise limit(f"The library supports up to {MAX_FABRICS} fabrics.")
            item = lib.add("fabrics", definition, action="fabric entered")
        else:
            item = lib.revise(previous, version(form.get("version")), detached(previous) | definition,
                              action="fabric details reviewed")
    return item


def list_fabrics(store, *, archived=False):
    with store.open() as lib:
        found = lib.list("fabrics", limit=MAX_FABRICS)
        result = []
        for fabric in found:
            if not archived and fabric["state"] != "active":
                continue
            key = fabric["id"]
            counts = lib.scoped_totals(key, ["captures", "marks", "readings", "visits"])
            captures = lib.list("captures", parent=key, limit=MAX_CAPTURES) if counts["captures"] else []
            active = [photo for photo in captures if photo["state"] == "active"]
            cover = next((photo for photo in active if photo["role"] == "face"), active[0] if active else None)
            result.append(fabric | {"counts": counts, "cover_capture": cover["id"] if cover else None})
        return result


def capture_values(form, default_title, former=None):
    form = retained(former, CAPTURE_FIELDS) | form
    return {
        "title": text(form.get("title"), "image description", 95, default=default_title),
        "role": choice(form.get("role"), "light role", ROLES, "face"),
        "face": choice(form.get("face"), "fabric side", FACES, "face"),
        "session": text(form.get("session"), "capture session", 56, default=""),
        "captured_day": date(form.get("captured_day"), "capture day", optional=True),
        "memo": text(form.get("memo"), "capture note", 1000, default="", multiline=True),
        "state": choice(form.get("state"), "capture state", ACTIONS, "active"),
    }


def readable_stats(surface):
    valid = surface.valid
    return {
        "visible": read_fraction(valid),
        "low_signal": read_fraction(surface.dark, denominator=int(valid.sum())),
        "clipped": read_fraction(surface.bright, denominator=int(valid.sum())),
    }


def install_capture(lib, fabric, raw, supplied, name, *, training=False):
    image = decode_image(supplied)
    digest = sha256(supplied).hexdigest()
    record = capture_values(raw, name)
    existing = lib.includes_blob("captures", fabric["id"], digest)
    if existing:
        raise conflict("This file is already kept for this fabric.")
    if lib.count("captures", parent=fabric["id"]) >= MAX_CAPTURES:
        raise limit(f"Keep at most {MAX_CAPTURES} source captures on one fabric.")
    working = png(image.working)
    direct = lib.add("captures", record | image.receipt() | {
        "source_sha256": digest, "original_name": name, "training": training,
    }, parent=fabric["id"], action="source photo retained")
    lib.attach(direct["id"], "original", supplied, image.original_type)
    lib.attach(direct["id"], "working", working, "image/png")
    lib.attach(direct["id"], "thumbnail", png(fit_image(image.working, 240, 200)), "image/png")
    lib.cite(direct["id"], fabric, "fabric at intake")
    return lib.public(direct, links=True)


def upload_source(file):
    if file is None:
        raise WorkbenchError("Select a source image first.", "image")
    source = file.read(MAX_FILE_BYTES + 1)
    if len(source) > MAX_FILE_BYTES:
        raise limit("Keep each source photograph under 24 MiB.")
    original = file.filename.replace("\\", "/").split("/")[-1]
    name = text(original[:80], "filename", 80, default="capture")
    return source, name


def intake(store, parent, form, file):
    source, name = upload_source(file)
    with store.open(write=True) as lib:
        fabric = lib.get(parent, "fabrics")
        if fabric["state"] != "active":
            raise conflict("Reactivate the fabric before importing another photo.")
        return install_capture(lib, fabric, form, source, name)


def edit_capture(store, key, form):
    with store.open(write=True) as lib:
        record = lib.get(key, "captures")
        update = capture_values(form, record["original_name"], record)
        revised = lib.revise(record, version(form.get("version")), detached(record) | update,
                             action="photo details reviewed")
        return lib.public(revised, links=True)


def latest_plane(lib, capture):
    planes = lib.list("planes", parent=capture["parent"], limit=MAX_CAPTURES * 4)
    return next((item for item in planes if item["capture"] == capture["id"]), None)


def calibration_inputs(lib, capture, form):
    fabric = lib.get(capture["parent"], "fabrics")
    if form.get("fabric_version") is not None:
        lib.require_version(fabric, version(form["fabric_version"]))
    surface = open_work(lib.asset(capture["id"], "working").path)
    mapping_ = Calibration.parse(form, fabric["width_mm"], fabric["length_mm"])
    rectified = rectify(surface, mapping_)
    stats = readable_stats(rectified)
    if stats["visible"] < 0.24:
        raise WorkbenchError("Too little of the proposed fabric plane is visible.", "corners")
    details = mapping_.receipt(surface.extent) | {"coverage": stats}
    return fabric, rectified, details


def approve_plane(store, key, form, *, preview=False):
    with store.open() as lib:
        source = lib.get(key, "captures")
        fabric, plane, details = calibration_inputs(lib, source, form)
    image = rgba(plane)
    if preview:
        return details, image
    with store.open(write=True) as lib:
        lib.require_version(lib.get(key, "captures"), source["version"])
        lib.require_version(lib.get(fabric["id"], "fabrics"), fabric["version"])
        calibrations = lib.list("planes", parent=fabric["id"], limit=MAX_CAPTURES * 4)
        if sum(record["capture"] == key for record in calibrations) >= 4:
            raise limit("Keep at most four retained calibrations per source photo.")
        item = lib.add("planes", details | {
            "capture": key, "title": text(form.get("title"), "calibration name", 95, default="Aligned fabric face"),
            "physical_version": fabric["version"],
        }, parent=fabric["id"], action="physical plane measured")
        lib.cite(item["id"], source, "calibrated photo")
        lib.cite(item["id"], fabric, "physical dimensions")
        lib.attach(item["id"], "plane", png(image), "image/png")
        return lib.public(item, links=True)


@dataclass(frozen=True)
class View:
    fabric: dict
    capture: dict
    surface: object
    plane: dict | None
    marks: list

    @property
    def scale(self):
        return None if self.plane is None else np.asarray(self.plane["millimetres_per_pixel"])

    @property
    def basis(self):
        return "fabric" if self.plane else "capture"

    @cached_property
    def obscured(self):
        return masks_for(self.surface.extent, self.marks, MASK_TYPES)

    @cached_property
    def visible(self):
        return self.surface.valid & ~self.obscured

    @cached_property
    def readable(self):
        return self.surface.readable & ~self.obscured

    def references(self):
        return [(self.capture, "input photo"), (self.fabric, "fabric dimensions")] + (
            [(self.plane, "selected physical plane")] if self.plane else []
        )


def view(lib, key, *, plane_key=None, registered=None):
    capture = lib.get(key, "captures")
    fabric = lib.get(capture["parent"], "fabrics")
    plane = lib.get(plane_key, "planes", parent=fabric["id"]) if plane_key else None
    if plane is not None and plane["capture"] != key:
        raise WorkbenchError("This calibration belongs to another photo.", "plane")
    if plane is None and registered is not False:
        plane = latest_plane(lib, capture)
    if plane is None and registered:
        raise WorkbenchError("Save a physical calibration for this photo before starting.", "plane")
    working = open_work(lib.asset(key, "working").path)
    if plane is not None:
        working = rectify(working, Calibration.recorded(plane))
    marks = view_marks(lib, fabric, capture, plane)
    return View(fabric, capture, working, plane, marks)


def view_marks(lib, fabric, capture, plane):
    found = []
    marks = lib.list("marks", parent=fabric["id"], oldest=True, limit=MAX_MARKS)
    for item in marks:
        if item.get("hidden"):
            continue
        if item["basis"] == "fabric":
            if plane is not None and item["fabric_extent_mm"] == [plane["width_mm"], plane["height_mm"]]:
                found.append(item)
        elif item["capture"] == capture["id"]:
            found.append(move_geometry(item, np.linalg.inv(plane["matrix"])) if plane else item)
    return found


def area_readout(ctx, geometry, holes):
    selected = polygon_mask(ctx.surface.extent, geometry, holes)
    count = int(selected.sum())
    visible = int((selected & ctx.visible).sum())
    clear = int((selected & ctx.readable).sum())
    result = {"area_px": count, "visible_px": visible, "readable_px": clear}
    if ctx.scale is not None:
        scale = float(np.prod(ctx.scale))
        result |= {
            "outline_area_mm2": round(count * scale, 4), "visible_area_mm2": round(visible * scale, 4),
            "readable_area_mm2": round(clear * scale, 4),
        }
    return result


def line_readout(ctx, geometry):
    summary = {"span_px": round(physical_span(geometry, ctx.surface.extent), 4)}
    if ctx.scale is not None:
        summary["span_mm"] = round(physical_span(geometry, ctx.surface.extent, ctx.scale), 4)
    return summary


def mark_values(form, ctx, prior=None):
    prior = prior or {}
    shape = choice(form.get("shape", prior.get("shape")), "shape", {"polygon", "line"}, "polygon")
    kind = choice(form.get("mark_type", prior.get("mark_type")), "mark type", MARK_TYPES, "area")
    if (shape == "line") != (kind in LINE_TYPES):
        raise WorkbenchError("Use a trace for a sewn line, cut line or selvedge; outline other observations.", "shape")
    vertices = form.get("points", prior.get("points"))
    if shape == "line":
        outline, holes = line_points(vertices), []
        measures = line_readout(ctx, outline)
    else:
        outline = polygon(vertices)
        holes = hole_polygons(outline, form.get("holes", prior.get("holes")))
        measures = area_readout(ctx, outline, holes)
    return {
        "title": text(form.get("title", prior.get("title")), "observation title", 80),
        "memo": text(form.get("memo", prior.get("memo")), "observation note", 1200, default="", multiline=True),
        "mark_type": kind, "shape": shape, "points": outline, "holes": holes,
        "basis": ctx.basis, "capture": ctx.capture["id"], "plane": ctx.plane["id"] if ctx.plane else None,
        "fabric_extent_mm": [ctx.plane["width_mm"], ctx.plane["height_mm"]] if ctx.plane else None,
        "state": choice(form.get("state", prior.get("state")), "observation state", STATES, "new"),
        "hidden": flag(form.get("hidden", prior.get("hidden")), "hide observation"),
        "reading": measures,
    }


def save_mark(store, parent, form, key=None):
    with store.open(write=True) as lib:
        prior = lib.get(key, "marks", parent=parent) if key else None
        if prior is None:
            frame = identifier(form.get("capture"), "capture")
            requested = identifier(form.get("plane"), "calibration", optional=True)
        else:
            frame, requested = prior["capture"], prior["plane"]
        ctx = view(lib, frame, plane_key=requested, registered=bool(requested))
        if ctx.fabric["id"] != parent:
            raise WorkbenchError("The photo belongs to another fabric.", "capture")
        definition = mark_values(form, ctx, prior)
        if prior:
            changed = lib.revise(prior, version(form.get("version")), definition,
                                 action="observation reviewed")
        else:
            if lib.count("marks", parent=parent) >= MAX_MARKS:
                raise limit("This fabric already has the maximum number of observations.")
            changed = lib.add("marks", definition, parent=parent, action="observation measured")
            lib.cite(changed["id"], ctx.capture, "captured observation")
            if ctx.plane:
                lib.cite(changed["id"], ctx.plane, "measured observation")
        return lib.public(changed, links=True)


def fabric_detail(store, key):
    with store.open() as lib:
        record = lib.get(key, "fabrics")
        entries = {}
        for name, count in (("captures", MAX_CAPTURES), ("planes", MAX_CAPTURES * 4), ("marks", MAX_MARKS),
                            ("readings", 120), ("swatches", 80), ("visits", 80), ("packets", 70),
                            ("proofs", 55), ("details", 48), ("cut_plans", 40)):
            found = lib.list(name, parent=key, limit=count)
            entries[name] = [lib.public(item) if name in {"captures", "swatches", "packets"} else item
                             for item in found]
        return record | entries


def review_case(store, key, *, version_=None):
    with store.open() as lib:
        record = lib.get(key) if version_ is None else lib.revision(key, version_)
        return lib.public(record, links=True) | {
            "history": lib.history(key),
            "cited_by": lib.cited_by(key),
        }

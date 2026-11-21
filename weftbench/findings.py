"""Summarize, locate and retain review conclusions on fabric photographs."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from io import StringIO
import csv

import numpy as np
from PIL import ImageDraw

from .catalog import view
from .contracts import (
    MARK_TYPES, STATES, WorkbenchError, box, choice, csv_cell, flag, identifier,
    integer, limit, mapping, text,
)
from .geometry import box_mask, cut_rectangle, pixel_points, polygon_mask
from .raster import png, rgba
from .visual import LILAC, MARK_COLORS, OCHRE, TEAL, blended_overlay, label, legend

SHOW_STATES = dict(STATES) | {"any": "Any observation state"}
SHOW_CATEGORIES = dict(MARK_TYPES) | {"any": "All categories"}


@dataclass(frozen=True)
class FindingRequest:
    selection: list
    category: str
    state: str
    include_lines: bool
    include_empty: bool
    cell: int

    @classmethod
    def parse(cls, raw):
        form = mapping(raw)
        display = box(form.get("box"), default=[0, 0, 1, 1])
        return cls(
            display,
            choice(form.get("category"), "observation category", SHOW_CATEGORIES, "any"),
            choice(form.get("state"), "observation state", SHOW_STATES, "any"),
            flag(form.get("lines"), "include route marks", default=True),
            flag(form.get("empty"), "show empty regions", default=False),
            integer(form.get("cell"), "inventory cell", 24, 400, default=96),
        )


def wants(mark, criteria):
    if mark["shape"] == "line" and not criteria.include_lines:
        return False
    type_ok = criteria.category in {"any", mark["mark_type"]}
    state_ok = criteria.state in {"any", mark["state"]}
    return type_ok and state_ok


def decoration(draw, scope, vertices, category, title, route):
    coords = pixel_points(vertices, scope)
    if len(coords) <= 1:
        return
    track = coords if route else coords + [coords[0]]
    width, height = scope
    anchor = min(coords, key=lambda item: (item[1], item[0]))
    shade = MARK_COLORS[category]
    draw.line(track, fill=(*shade, 255), width=max(2, width // 640))
    x = max(0, min(width - 45, anchor[0]))
    y = max(0, min(height - 22, anchor[1] - 16))
    label(draw, (int(x), int(y)), title, fill=shade, size=10, maximum=max(26, min(width - int(x) - 12, 150)))


def mask_line_at(points, allowed, extent):
    width, height = extent
    packed = np.array(pixel_points(points, extent))
    length = 0.0
    visible = 0.0
    included = 0
    for one, two in zip(packed, packed[1:]):
        span = float(np.linalg.norm(two - one))
        if span < 0.01:
            continue
        length += span
        checks = np.linspace(one, two, max(2, int(np.ceil(span / 2))))
        xx, yy = np.floor(checks).astype(int).T
        inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        local = allowed[yy[inside], xx[inside]]
        included += int(local.sum())
        visible += span * local.sum() / len(checks)
    return length, visible, included


def outline_support(ctx, selection, mark):
    whole = polygon_mask(ctx.surface.extent, mark["points"], mark.get("holes", []))
    chosen = whole & selection
    on_image = int(chosen.sum())
    if not on_image:
        return None, None
    included = int((chosen & ctx.visible).sum())
    invalid = on_image - included
    area = ctx.scale.prod() if ctx.scale is not None else None
    return {
        "unit": "mm²" if area is not None else "px²",
        "visible_area": round(float(included * area), 5) if area is not None else included,
        "unread_area": round(float(invalid * area), 5) if area is not None else invalid,
        "pixel_area": on_image, "visible_pix": included,
        "coverage": round(included / on_image, 5), "is_cut_by_window": on_image < int(whole.sum()),
    }, chosen


def review_rows(ctx, criteria, region):
    evidence = []
    extents = ctx.surface.extent
    masks = []
    for item in ctx.marks:
        if not wants(item, criteria):
            continue
        if item["shape"] == "line":
            _, selected_distance, chosen = mask_line_at(item["points"], region, extents)
            if not chosen:
                continue
            _, visible, _ = mask_line_at(item["points"], region & ctx.visible, extents)
            result = {"unit": "px", "trace_in_window_px": round(selected_distance, 5),
                      "trace_supported_px": round(visible, 5), "visible_area": None,
                      "coverage": round(visible / max(selected_distance, 1e-5), 5)}
        else:
            result, chosen = outline_support(ctx, region, item)
            if result is None:
                continue
            masks.append(chosen)
        evidence.append({
            "id": item["id"], "title": item["title"], "mark_type": item["mark_type"],
            "state": item["state"], "reading": result,
            "version": item["version"], "shape": item["shape"], "points": item["points"],
        })
    return evidence, masks


def inventory_tiles(region, visible, flagged, step, add_empty):
    height, width = region.shape
    grid = []
    for y in range(0, height, step):
        for x in range(0, width, step):
            end_x, end_y = min(x + step, width), min(y + step, height)
            square = np.s_[y:end_y, x:end_x]
            total = int(region[square].sum())
            if not total:
                continue
            found = int(flagged[square].sum())
            if not found and not add_empty:
                continue
            valid = int(visible[square].sum())
            grid.append({
                "bounds": [x, y, end_x, end_y], "visible_fraction": round(valid / total, 5),
                "marked_fraction": round(found / total, 5), "marked_pixels": found,
            })
    return grid


def finding_overlay(ctx, records, mask, target, region):
    classes = np.zeros(mask.shape, dtype=np.uint8)
    classes[region & ~ctx.visible] = 1
    classes[mask & ctx.visible] = 2
    workspace = blended_overlay(ctx.surface, classes, {1: LILAC, 2: TEAL})
    draw = ImageDraw.Draw(workspace)
    x0, y0, x1, y1 = cut_rectangle(target, ctx.surface.extent)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=OCHRE, width=2)
    for i, item in enumerate(records, 1):
        decoration(draw, workspace.size, item["points"], item["mark_type"], str(i), item["shape"] == "line")
    return legend(workspace, "Mapped observation evidence", [("Selected window", OCHRE), ("Visible observation area", TEAL),
                                                              ("Missing or excluded", LILAC)])


def summarize(ctx, raw):
    request = FindingRequest.parse(raw)
    searchable = box_mask(ctx.surface.extent, request.selection)
    visible = searchable & ctx.visible
    included, pieces = review_rows(ctx, request, searchable)
    combined = np.logical_or.reduce(pieces) if pieces else np.zeros_like(visible)
    measured_count = int((combined & visible).sum())
    area_mm2 = measured_count * np.prod(ctx.scale) if ctx.scale is not None else None
    art = finding_overlay(ctx, included, combined, request.selection, searchable)
    state_counts = Counter(item["state"] for item in included)
    category_counts = Counter(item["mark_type"] for item in included)
    detail = {
        "box": request.selection,
        "categories": dict(category_counts),
        "states": dict(state_counts),
        "selected_observations": included,
        "visible_window_pixels": int(visible.sum()),
        "visible_marked_pixels": measured_count,
        "visible_marked_mm2": round(float(area_mm2), 5) if area_mm2 is not None else None,
        "grid": inventory_tiles(searchable, visible, combined, request.cell, request.include_empty),
        "note": "Union area is reported once when observation outlines overlap.",
    }
    return detail, art


def finding_sheet(rows):
    out = StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(["label", "category", "state", "record", "version", "geometry",
                     "visible_area", "unit", "visible_fraction"])
    for row in rows:
        report = row["reading"]
        writer.writerow([csv_cell(row["title"]), row["mark_type"], row["state"], row["id"],
                         row["version"], row["shape"], report["visible_area"], report["unit"], report["coverage"]])
    return out.getvalue().encode("utf-8")


def saved_findings(store, capture, raw, *, save):
    form = mapping(raw)
    plane = identifier(form.get("plane"), "plane", optional=True)
    with store.open() as lib:
        ctx = view(lib, capture, plane_key=plane)
        report, image = summarize(ctx, form)
        chosen = [lib.get(item["id"], "marks") for item in report["selected_observations"]]
    if not save:
        return report, image
    with store.open(write=True) as lib:
        if lib.count("readings", parent=ctx.fabric["id"]) >= 120:
            raise limit("This fabric has reached its retained inspection limit.")
        depends = ctx.references() + [(entry, "observation in mapped review") for entry in chosen]
        for row, _ in depends:
            lib.require_version(lib.get(row["id"], row["kind"]), row["version"])
        stored = lib.add("readings", {
            "title": text(form.get("title"), "finding review title", 95, default="Fabric observation inventory"),
            "tool": "findings", "analysis": report, "proofs": ["map", "csv"], "review_state": "unreviewed",
            "parameters": {"capture": capture, "capture_role": ctx.capture["role"],
                           "plane": ctx.plane["id"] if ctx.plane else None, "coordinate_basis": ctx.basis},
        }, parent=ctx.fabric["id"], action="mapped observation evidence retained")
        for source, role in depends:
            lib.cite(stored["id"], source, role)
        lib.attach(stored["id"], "map", png(image), "image/png")
        lib.attach(stored["id"], "csv", finding_sheet(report["selected_observations"]), "text/csv")
        return lib.public(stored, links=True)

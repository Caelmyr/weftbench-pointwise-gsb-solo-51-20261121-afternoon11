"""Measure checkpoints on retained calibration maps without changing their fit."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
from PIL import ImageDraw

from .catalog import view
from .contracts import MAX_RESULTS, WorkbenchError, choice, identifier, integer, limit, mapping, point, real, sequence, text
from .geometry import Calibration, point_pixels, transform
from .raster import open_work, quantiles, rgba
from .visual import INK, LILAC, OCHRE, PINK, TEAL, label, legend, panel_pair


@dataclass(frozen=True)
class Checkpoint:
    name: str
    measured: list
    expected: list

    @classmethod
    def parse(cls, data, index):
        form = mapping(data, "independent checkpoint")
        return cls(
            text(form.get("name"), "checkpoint label", 45, default=f"Point {index}"),
            point(form.get("measured"), "point on original photograph"),
            point(form.get("expected"), "point on fabric plane"),
        )


@dataclass(frozen=True)
class ReviewOptions:
    plane: str
    points: list
    cutoff: float
    subdivisions: int
    mode: str

    @classmethod
    def parse(cls, data):
        form = mapping(data, "calibration review")
        readings = [Checkpoint.parse(row, i) for i, row in enumerate(
            sequence(form.get("checkpoints"), "checkpoints", 2, 24), 1)]
        if len(set(row.name for row in readings)) != len(readings):
            raise WorkbenchError("Use different names for the independent checkpoints.", "checkpoints")
        return cls(
            identifier(form.get("plane"), "calibration"), readings,
            real(form.get("tolerance"), "alignment tolerance", 0.1, 80, default=3),
            integer(form.get("cells"), "display grid", 2, 12, default=5),
            choice(form.get("unit"), "alignment tolerance unit", {"mm", "px"}, "mm"),
        )


def original_point(surface, position):
    width, height = surface.extent
    if min(position) < 0 or max(position) > 1:
        return False
    x, y = point_pixels(position, surface.extent)
    return bool(surface.valid[min(height - 1, int(y)), min(width - 1, int(x))])


def check_points(context, camera, options):
    model = Calibration.recorded(context.plane)
    image_points = [row.measured for row in options.points]
    plan_points = [row.expected for row in options.points]
    try:
        expected_camera = transform(model.matrix, plan_points)
        measured_plan = transform(np.linalg.inv(model.matrix), image_points)
    except WorkbenchError:
        raise WorkbenchError("A checkpoint is too close to the original calibration horizon.", "checkpoints") from None
    raw_px = np.asarray(camera.extent, dtype=np.float64)
    physical = np.array([model.width_mm, model.height_mm])
    records = []
    for row, estimate, footprint in zip(options.points, expected_camera, measured_plan, strict=True):
        image_delta = (np.asarray(row.measured) - estimate) * raw_px
        physical_delta = (footprint - row.expected) * physical
        distance_px = float(np.linalg.norm(image_delta))
        distance_mm = float(np.linalg.norm(physical_delta))
        usable = original_point(camera, row.measured) and original_point(camera, estimate)
        usable &= original_point(context.surface, row.expected) and original_point(context.surface, footprint)
        score = distance_mm if options.mode == "mm" else distance_px
        category = "held" if not usable else "inside tolerance" if score <= options.cutoff else "outside tolerance"
        records.append({
            "name": row.name, "observed_original": row.measured, "expected_plane": row.expected,
            "fit_original": np.round(estimate, 7).tolist(), "fit_plane": np.round(footprint, 7).tolist(),
            "source_distance_px": round(distance_px, 5) if usable else None,
            "plane_distance_mm": round(distance_mm, 5) if usable else None,
            "source_error_px": np.round(image_delta, 5).tolist() if usable else None,
            "plane_error_mm": np.round(physical_delta, 5).tolist() if usable else None,
            "status": category,
        })
    return records


def audit_grid(mask, steps):
    height, width = mask.shape
    x = np.linspace(0, width, steps + 1).round().astype(int)
    y = np.linspace(0, height, steps + 1).round().astype(int)
    cells = []
    for row, (top, bottom) in enumerate(zip(y, y[1:]), 1):
        for col, (left, right) in enumerate(zip(x, x[1:]), 1):
            sub = mask[top:bottom, left:right]
            cells.append({
                "row": row, "column": col, "area_px": int(sub.size), "visible_px": int(sub.sum()),
                "box": [int(left), int(top), int(right), int(bottom)],
                "support": round(float(sub.mean()), 5) if sub.size else 0,
            })
    return cells


def source_residuals(surface, records):
    board = rgba(surface)
    draw = ImageDraw.Draw(board)
    for index, row in enumerate(records, 1):
        original = point_pixels(row["observed_original"], board.size)
        expected = point_pixels(row["fit_original"], board.size)
        shade = TEAL if row["status"] == "inside tolerance" else PINK if row["status"] == "outside tolerance" else LILAC
        x, y = original
        if row["status"] != "held":
            draw.line([original, expected], width=2, fill=(*shade, 255))
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(*shade, 255), width=2)
        label(draw, (int(x) + 5, int(y) + 5), str(index), fill=shade)
    return board


def registration_board(ctx, entries, cells):
    artwork = rgba(ctx.surface)
    paint = ImageDraw.Draw(artwork)
    for grid in cells:
        left, top, right, bottom = grid["box"]
        support = grid["support"]
        shade = TEAL if support >= 0.75 else OCHRE if support >= 0.40 else LILAC
        paint.rectangle((left, top, right - 1, bottom - 1), outline=(*shade, 255), width=1)
    for i, item in enumerate(entries, 1):
        x, y = point_pixels(item["expected_plane"], artwork.size)
        color = PINK if item["status"] == "outside tolerance" else LILAC if item["status"] == "held" else TEAL
        paint.line((x - 5, y, x + 5, y), fill=(*color, 255), width=2)
        paint.line((x, y - 5, x, y + 5), fill=(*color, 255), width=2)
        label(paint, (x - 9, y + 9), str(i), fill=color)
    return artwork


def review_measure(lib, form):
    options = ReviewOptions.parse(form)
    plane = lib.get(options.plane, "planes")
    context = view(lib, plane["capture"], plane_key=options.plane, registered=True)
    original = open_work(lib.asset(context.capture["id"], "working").path)
    verified = check_points(context, original, options)
    usable = [row for row in verified if row["status"] != "held"]
    comparison = [row["plane_distance_mm"] if options.mode == "mm" else row["source_distance_px"]
                  for row in usable]
    above = sum(row["status"] == "outside tolerance" for row in usable)
    areas = audit_grid(context.visible, options.subdivisions)
    result = {
        "capture": context.capture["id"], "plane": plane["id"], "unit": options.mode,
        "tolerance": options.cutoff, "tested_points": len(usable), "held_points": len(verified) - len(usable),
        "over_tolerance": above, "readings": verified,
        "distance_percentiles": quantiles(comparison),
        "rms": round(sqrt(sum(i * i for i in comparison) / len(comparison)), 5) if usable else None,
        "coverage_cells": areas,
    }
    comparison_board = panel_pair(source_residuals(original, verified), registration_board(context, verified, areas),
                                  "Original point and residual", "Fabric coordinates and visible support", max_side=690)
    return context, result, legend(comparison_board, "Independent calibration inspection", [
        ("Within selected tolerance", TEAL), ("Over tolerance", PINK), ("Held for missing evidence", LILAC),
    ])


def diagnose(store, raw, *, save):
    with store.open() as lib:
        context, result, artwork = review_measure(lib, raw)
    if not save:
        return result, artwork
    from .raster import png
    record = {
        "title": text(raw.get("title"), "checkpoint inspection title", 90, default="Calibration checkpoint review"),
        "tool": "checkpoints", "analysis": result, "proofs": ["map"], "review_state": "unreviewed",
        "parameters": {"capture": context.capture["id"], "capture_role": context.capture["role"],
                       "plane": context.plane["id"], "coordinate_basis": "fabric"},
        "notes": "Independent measured checkpoints. This record retains the original calibration.",
    }
    with store.open(write=True) as lib:
        if lib.count("readings", parent=context.fabric["id"]) >= MAX_RESULTS:
            raise limit("This fabric has reached its limit of retained readings.")
        for item, _ in context.references():
            lib.require_version(lib.get(item["id"], item["kind"]), item["version"])
        stored = lib.add("readings", record, parent=context.fabric["id"], action="calibration checked")
        for item, role in context.references():
            lib.cite(stored["id"], item, role)
        lib.attach(stored["id"], "map", png(artwork), "image/png")
        return lib.public(stored, links=True)

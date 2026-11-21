"""Inspect fabric openings and visible surface texture under their own light roles."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .contracts import WorkbenchError, choice, integer, mapping, real, require_support
from .geometry import box_mask, cut_rectangle
from .inspection import common_box
from .raster import blurred_masked, component_labels, gradients, luma, quantiles, read_fraction, rgba
from .visual import OCHRE, LILAC, PINK, TEAL, blended_overlay, heatmap, histogram_plot, label, legend


@dataclass(frozen=True)
class OpeningRequest:
    selection: list
    contrast: float
    absolute: float
    min_area: int
    max_results: int
    neighborhood: float

    @classmethod
    def parse(cls, raw):
        form = mapping(raw)
        return cls(
            common_box(form),
            real(form.get("contrast"), "minimum opening contrast", 0.015, 0.85, default=0.145),
            real(form.get("absolute"), "minimum opening light", 0.08, 0.98, default=0.44),
            integer(form.get("min_area"), "smallest opening", 2, 900, default=20),
            integer(form.get("max_results"), "retained candidates", 1, 80, default=36),
            real(form.get("neighborhood"), "nearby cloth radius", 3, 40, default=11),
        )


def opening_profile(ctx, settings):
    surface = ctx.surface
    light = luma(surface.linear)
    selection = box_mask(surface.extent, settings.selection)
    visible = ctx.visible & ~surface.bright
    neighborhood, supported = blurred_masked(light, visible, settings.neighborhood)
    good = selection & supported
    require_support(int(selection.sum()), int(good.sum()), "Opening inspection", 110, fraction=0.30)
    residual = light - neighborhood
    plausible = good & (residual >= settings.contrast) & (light >= settings.absolute)
    return selection, good, residual, plausible


def pixel_perimeter(mask):
    a = np.pad(mask, 1, constant_values=False)
    return int(np.count_nonzero(a[:, 1:] != a[:, :-1]) + np.count_nonzero(a[1:] != a[:-1]))


def candidate_reading(index, fragment, bounds, residual, support, scale):
    row, col = bounds
    subsection = fragment[row, col]
    height, width = fragment.shape
    y, x = np.nonzero(subsection)
    xx, yy = x + col.start, y + row.start
    spin = np.column_stack([xx, yy])
    area = int(len(spin))
    y0, y1, x0, x1 = row.start, row.stop, col.start, col.stop
    if area >= 3:
        covariance = np.cov(spin.T)
        spread = np.linalg.eigvalsh(covariance)
        elongation = math.sqrt(max(float(spread[-1]), 0.0) / max(0.0001, float(spread[0])))
    else:
        elongation = None
    border = ndimage.binary_dilation(fragment, iterations=2) & ~fragment
    adjacent = int((border & ~support).sum())
    perimeter = pixel_perimeter(subsection)
    return {
        "index": index,
        "bounds": [x0, y0, x1, y1],
        "box": [x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height],
        "center": [round(float(xx.mean() / width), 6), round(float(yy.mean() / height), 6)],
        "area_px": area, "perimeter_px": perimeter,
        "area_mm2": round(float(area * np.prod(scale)), 5) if scale is not None else None,
        "residual_percentiles": quantiles(residual[fragment], (5, 50, 95)),
        "elongation": round(elongation, 4) if elongation is not None else None,
        "not_fully_bounded": adjacent > 0,
        "adjacent_unknown_px": adjacent,
    }


def classified_openings(active, settings, residual, supported, scale):
    labels, _, areas, cells = component_labels(active)
    proposed = [i for i, n in enumerate(areas) if i and n >= settings.min_area]
    proposed.sort(key=lambda i: (-areas[i], i))
    selection = np.zeros(active.shape, dtype=bool)
    readings = []
    for index, candidate in enumerate(proposed[:settings.max_results], 1):
        fragment = labels == candidate
        definition = candidate_reading(index, fragment, cells[candidate - 1], residual, supported, scale)
        selection |= fragment
        readings.append(definition)
    return selection, readings, len(proposed)


def openings(ctx, raw):
    if ctx.capture["role"] != "backlight":
        raise WorkbenchError("Inspect optical openings on a transmitted-light capture.", "photo")
    request = OpeningRequest.parse(raw)
    region, visible, residual, candidates = opening_profile(ctx, request)
    accepted, pieces, total = classified_openings(candidates, request, residual, visible, ctx.scale)
    classes = np.zeros(visible.shape, dtype=np.uint8)
    classes[region & ~visible] = 1
    classes[candidates & ~accepted] = 2
    classes[accepted] = 3
    canvas = blended_overlay(ctx.surface, classes, {1: LILAC, 2: TEAL, 3: PINK})
    pen = ImageDraw.Draw(canvas)
    for row in pieces:
        x0, y0, x1, y1 = row["bounds"]
        pen.rectangle((x0, y0, x1, y1), outline=PINK, width=2)
        label(pen, (x0, max(0, y0 - 14)), str(row["index"]), fill=PINK)
    report = {
        "box": request.selection,
        "readable_pixels": int(visible.sum()), "unknown_pixels": int((region & ~visible).sum()),
        "possible_count": total, "returned_count": len(pieces),
        "opened_pixel_fraction": read_fraction(accepted, denominator=int(visible.sum())),
        "observations": pieces,
        "note": "A bright opening can be part of the weave or evidence of a defect. Review it in the source.",
    }
    return report, {"map": legend(canvas, "Individual optical opening candidates", [
        ("Not visible", LILAC), ("Small opening signal", TEAL), ("Numbered for review", PINK),
    ])}


@dataclass(frozen=True)
class ReliefRequest:
    selection: list
    fine: float
    coarse: float
    minimum: float
    cells: int

    @classmethod
    def parse(cls, raw):
        form = mapping(raw)
        fine = real(form.get("fine"), "fine detail radius", 0.6, 18, default=1.4)
        coarse = real(form.get("coarse"), "background radius", 2, 70, default=7.0)
        if fine * 1.5 >= coarse:
            raise WorkbenchError("Keep a stronger separation between the two filter radii.", "coarse")
        return cls(
            common_box(form), fine, coarse,
            real(form.get("minimum"), "visible detail threshold", 0.005, 0.32, default=0.035),
            integer(form.get("cells"), "texture cell size", 28, 280, default=66),
        )


def surface_field(ctx, req):
    usable = ctx.readable
    values = luma(ctx.surface.linear)
    fine, f_support = blurred_masked(values, usable, req.fine)
    coarse, c_support = blurred_masked(values, usable, req.coarse)
    differences = fine - coarse
    good = ndimage.binary_erosion(f_support & c_support, iterations=2)
    return differences, good


def relief_tiles(detail, support, domain, step, threshold):
    h, w = detail.shape
    ready = []
    for top in range(0, h, step):
        for left in range(0, w, step):
            bottom, right = min(top + step, h), min(left + step, w)
            bounds = np.s_[top:bottom, left:right]
            requested = int(domain[bounds].sum())
            read = support[bounds]
            accepted = int(read.sum())
            if not requested:
                continue
            row = {"bounds": [left, top, right, bottom], "coverage": round(accepted / requested, 5)}
            if accepted < 36 or accepted / requested < 0.3:
                ready.append(row | {"texture": None, "strong_fraction": None, "status": "held for review"})
                continue
            present = np.abs(detail[bounds][read])
            ready.append(row | {
                "texture": quantiles(present, (15, 50, 85, 97)),
                "strong_fraction": round(float((present >= threshold).mean()), 6),
                "status": "read",
            })
    return ready


def relief(ctx, raw):
    if ctx.capture["role"] != "rake":
        raise WorkbenchError("Choose a raking-light capture for surface relief evidence.", "photo")
    form = ReliefRequest.parse(raw)
    values, valid = surface_field(ctx, form)
    selection = box_mask(ctx.surface.extent, form.selection)
    support = selection & valid
    require_support(int(selection.sum()), int(support.sum()), "Surface relief", 140)
    magnitude = np.abs(values)
    summary = {
        "box": form.selection, "scales_px": [form.fine, form.coarse],
        "detail_percentiles": quantiles(magnitude[support]), "usable_pixels": int(support.sum()),
        "strong_detail_pixels": int(((magnitude >= form.minimum) & support).sum()),
        "threshold": form.minimum, "unknown_pixels": int((selection & ~support).sum()),
        "cells": relief_tiles(values, support, selection, form.cells, form.minimum),
        "note": "Optical relief rather than measured fibre height. Lighting direction affects the result.",
    }
    contour = heatmap(ctx.surface, magnitude, support, (form.minimum / 2, form.minimum * 2.8),
                      unknown=selection & ~valid, hue=OCHRE)
    return summary, {"map": legend(contour, "Surface relief by fine and coarse contrast", [
        ("Stronger optical relief", OCHRE), ("Unsupported", LILAC),
    ])}


def strip_rows(raw, extent):
    request = mapping(raw)
    bounds = common_box(request)
    left, top, right, bottom = cut_rectangle(bounds, extent)
    count = integer(request.get("count"), "sample strips", 3, 25, default=9)
    direction = choice(request.get("direction"), "strip orientation", {"horizontal", "vertical"}, "horizontal")
    if max(right - left, bottom - top) < 60 or (bottom - top if direction == "horizontal" else right - left) < count * 4:
        raise WorkbenchError("Choose a larger region to separate the inspection strips.", "box")
    if direction == "horizontal":
        steps = np.linspace(top, bottom, count + 1).astype(int)
        return bounds, [(left, l, right, r) for l, r in zip(steps, steps[1:])], direction
    steps = np.linspace(left, right, count + 1).astype(int)
    return bounds, [(l, top, r, bottom) for l, r in zip(steps, steps[1:])], direction


def abrasion_strip(arg, detail, energy, support, relative, threshold):
    left, top, right, bottom = arg
    index = np.s_[top:bottom, left:right]
    active = support[index]
    eligible = float(active.mean())
    if eligible < 0.35 or active.sum() < 40:
        return {"bounds": list(arg), "coverage": round(eligible, 5), "status": "not measured", "index": None}
    pixels = detail[index][active]
    measured = energy[index][active]
    extra = measured >= threshold
    normal = float(np.median(pixels))
    estimate = np.clip(float(np.quantile(measured, 0.85)) / max(relative, 1e-7), 0, 25)
    return {
        "bounds": list(arg), "coverage": round(eligible, 5), "status": "measured",
        "index": round(float(estimate), 6), "detail_median": round(normal, 6),
        "higher_than_reference": round(float(extra.mean()), 6),
        "local_energy": quantiles(measured),
    }


def abrasion_profile(ctx, raw):
    if ctx.capture["role"] != "rake":
        raise WorkbenchError("Use raking light for this abrasion review.", "photo")
    data = mapping(raw)
    request = ReliefRequest.parse(data)
    bounds, strips, direction = strip_rows(data, ctx.surface.extent)
    reference = data.get("control", [0.68, 0.20, 0.21, 0.18])
    from .contracts import box
    checked = box(reference, "control cloth")
    detail, eligible = surface_field(ctx, request)
    grad_x, grad_y, safe = gradients(detail, eligible)
    energy = np.hypot(grad_x, grad_y)
    control = safe & box_mask(ctx.surface.extent, checked)
    count = int(control.sum())
    require_support(int(box_mask(ctx.surface.extent, checked).sum()), count, "Unaffected cloth control", 60)
    control_values = energy[control]
    relative = max(0.0025, float(np.quantile(control_values, 0.85)))
    threshold = relative * real(data.get("ratio"), "contrast multiplier", 1, 6, default=1.6)
    entries = [abrasion_strip(arg, detail, energy, safe, relative, threshold) for arg in strips]
    art = rgba(ctx.surface)
    draw = ImageDraw.Draw(art)
    for index, row in enumerate(entries, 1):
        x, y, r, b = row["bounds"]
        color = OCHRE if row["index"] is not None else LILAC
        draw.rectangle((x, y, r - 1, b - 1), outline=(*color, 255), width=2)
        label(draw, (x + 3, y + 3), str(index), fill=color)
    x, y, r, b = cut_rectangle(checked, ctx.surface.extent)
    draw.rectangle((x, y, r - 1, b - 1), outline=(*TEAL, 255), width=2)
    chart = histogram_plot([{"points": [[i, row["index"]] for i, row in enumerate(entries, 1)], "color": OCHRE}],
                           "Contrast across abrasion strips", x_label="Strip number", y_label="Relative to control")
    return {
        "box": bounds, "strip_direction": direction, "control": checked,
        "control_pixels": count, "control_edge_percentiles": quantiles(control_values),
        "threshold": round(threshold, 6), "strips": entries,
        "note": "Visible fibre disturbance in this photo; compare captures only when light setup is comparable.",
    }, {"map": legend(art, "Abrasion inspection strips", [("Control", TEAL), ("Read strip", OCHRE),
                                                          ("Unmeasured strip", LILAC)]), "chart": chart}

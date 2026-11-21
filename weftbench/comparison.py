"""Comply with comparison boundaries by comparing separate calibrated image evidence."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .catalog import View, view
from .contracts import WorkbenchError, box, choice, flag, identifier, integer, mapping, real, require_support
from .geometry import box_mask, cut_rectangle
from .inspection import common_box
from .raster import Raster, blurred_masked, crop_raster, luma, quantiles, rgba, to_lab
from .visual import LILAC, OCHRE, PINK, TEAL, blended_overlay, heatmap, label, legend, panel_pair


@dataclass(frozen=True)
class ComparisonOptions:
    reference: str
    reference_plane: str | None
    region: list
    metric: str
    neutral_area: list | None
    smooth: float
    tolerance: float
    overlay_later: bool

    @classmethod
    def parse(cls, raw):
        form = mapping(raw)
        return cls(
            identifier(form.get("reference"), "reference capture"),
            identifier(form.get("reference_plane"), "reference calibration", optional=True),
            common_box(form),
            choice(form.get("metric"), "comparison metric", {"light", "color"}, "light"),
            box(form.get("neutral"), "stable sample") if form.get("neutral") is not None else None,
            real(form.get("smooth"), "comparison smoothing", 0, 9, default=0.7),
            real(form.get("tolerance"), "comparison threshold", 0.003, 30.0, default=0.085),
            flag(form.get("overlay_later"), "later photo under map", default=True),
        )


def nearest_mask(mask, extent):
    before = np.asarray(mask.shape, dtype=np.float64)
    shapes = np.asarray(extent[::-1], dtype=np.float64)
    y, x = [((np.arange(int(size)) + 0.5) / size) * original - 0.5
            for size, original in zip(shapes, before, strict=True)]
    coords = np.array(np.meshgrid(y, x, indexing="ij"))
    work = ndimage.map_coordinates(mask.astype(np.float32), coords, order=0, mode="nearest")
    return work.astype(bool)


def linked_raster(frame, extent):
    width, height = extent
    orig_width, orig_height = frame.extent
    x = (np.arange(width) + 0.5) * (orig_width / width) - 0.5
    y = (np.arange(height) + 0.5) * (orig_height / height) - 0.5
    grids = np.array(np.meshgrid(y, x, indexing="ij"))
    weights = ndimage.map_coordinates(frame.valid.astype(np.float32), grids, order=1, mode="nearest")
    valid = weights >= 0.997
    data = np.stack([
        ndimage.map_coordinates(frame.linear[:, :, channel], grids, order=1, mode="nearest")
        for channel in range(3)
    ], axis=2)
    return Raster(np.where(valid[:, :, None], data, 0), valid)


def two_views(lib, target, using, *, role_must_agree=True):
    if target.capture["id"] == using.reference:
        raise WorkbenchError("Choose a different reference image.", "reference")
    current = target
    other = view(lib, using.reference, plane_key=using.reference_plane, registered=True)
    if current.plane is None:
        raise WorkbenchError("Calibrate the current image before comparing it.")
    if current.fabric["id"] != other.fabric["id"] or current.capture["face"] != other.capture["face"]:
        raise WorkbenchError("Choose the same fabric and cloth side for both photographs.", "reference")
    if [current.plane["width_mm"], current.plane["height_mm"]] != [other.plane["width_mm"], other.plane["height_mm"]]:
        raise WorkbenchError("Both retained planes must describe the same physical fabric extent.", "reference")
    if role_must_agree and current.capture["role"] != other.capture["role"]:
        raise WorkbenchError("Choose equivalent light roles when comparing image measurements.", "reference")
    if other.surface.extent == current.surface.extent:
        surface_b, good_b = other.surface, other.readable
    else:
        surface_b = linked_raster(other.surface, current.surface.extent)
        good_b = surface_b.readable & nearest_mask(other.readable, current.surface.extent)
    return other, surface_b, good_b


def optical_correction(remembered, current, shared, sample):
    if sample is None:
        return np.ones(3, dtype=np.float32), None
    region = box_mask(current.extent, sample)
    usable = region & shared
    require_support(int(region.sum()), int(usable.sum()), "Stable tonal reference", 70, fraction=0.4)
    before = np.median(remembered.linear[usable], axis=0)
    after = np.median(current.linear[usable], axis=0)
    if np.min(before) < 0.015 or np.min(after) < 0.015:
        raise WorkbenchError("Select a more evenly lit stable region for normalization.", "neutral")
    gains = after / before
    if np.any((gains < 0.4) | (gains > 2.5)):
        raise WorkbenchError("This single stable region cannot explain the exposure change.", "neutral")
    return gains, {
        "box": sample, "used_pixels": int(usable.sum()),
        "before_rgb": np.round(before, 6).tolist(), "after_rgb": np.round(after, 6).tolist(),
        "rgb_multipliers": np.round(gains, 6).tolist(),
    }


def adjusted_views(first, second, shared, gains, radius):
    if radius <= 0:
        return first.linear * gains, second.linear, shared
    a = []
    b = []
    mask = shared.copy()
    for channel in range(3):
        before, can_before = blurred_masked(first.linear[:, :, channel] * gains[channel], shared, radius)
        after, can_after = blurred_masked(second.linear[:, :, channel], shared, radius)
        a.append(before)
        b.append(after)
        mask &= can_before & can_after
    corrected = np.stack(a, axis=2)
    after = np.stack(b, axis=2)
    mask &= (np.max(corrected, axis=2) < 0.99) & (np.max(after, axis=2) < 0.99)
    return corrected, after, mask


def residual_values(before, after, metric, mask):
    if metric == "light":
        return luma(after) - luma(before)
    result = np.zeros(mask.shape, dtype=np.float32)
    result[mask] = np.linalg.norm(to_lab(after[mask]) - to_lab(before[mask]), axis=1)
    return result


def extent_statistics(mask, scale):
    count = int(mask.sum())
    return {"pixels": count, "area_mm2": round(float(count * np.prod(scale)), 5) if scale is not None else None}


def shift_regions(raw, valid, rule, bounds, scale):
    ready = np.abs(raw) >= rule
    classes = {
        "decreased": valid & ready & (raw < 0), "increased": valid & ready & (raw >= 0),
        "within_tolerance": valid & ~ready, "not_supported": bounds & ~valid,
    }
    return classes, {key: extent_statistics(mask, scale) for key, mask in classes.items()}


def explicit_tolerance(method, configured):
    if method == "light" and configured > 1:
        raise WorkbenchError("Use a luminance threshold between 0 and 1.", "tolerance")
    return configured


def side_panel(first, second, region, first_name, second_name):
    bounds = cut_rectangle(region, second.extent)
    reference_view = rgba(crop_raster(first, bounds))
    current_view = rgba(crop_raster(second, bounds))
    return panel_pair(reference_view, current_view, first_name, second_name)


def compare(current, raw, lib):
    settings = ComparisonOptions.parse(raw)
    other, surface_b, readable_b = two_views(lib, current, settings)
    extent = current.surface.extent
    whole = box_mask(extent, settings.region)
    intersected = current.readable & readable_b
    gains, neutral = optical_correction(surface_b, current.surface, intersected, settings.neutral_area)
    adjusted, latest, good = adjusted_views(surface_b, current.surface, intersected, gains, settings.smooth)
    within = good & whole
    require_support(int(whole.sum()), int(within.sum()), "Paired inspection", 85, fraction=0.25)
    limit = explicit_tolerance(settings.metric, settings.tolerance)
    discrepancy = residual_values(adjusted, latest, settings.metric, within)
    classes, counts = shift_regions(discrepancy, within, limit, whole, current.scale)
    canvas = current.surface if settings.overlay_later else surface_b
    composite = np.zeros(whole.shape, dtype=np.uint8)
    composite[classes["not_supported"]] = 1
    composite[classes["decreased"]] = 2
    composite[classes["increased"]] = 3
    titles = [("Unavailable on one or both captures", LILAC)]
    if settings.metric == "light":
        titles.extend([("Lower illumination value", OCHRE), ("Higher illumination value", TEAL)])
    else:
        titles.append(("Greater than selected color tolerance", TEAL))
    panel = blended_overlay(canvas, composite, {1: LILAC, 2: OCHRE, 3: TEAL})
    guide = side_panel(surface_b, current.surface, settings.region, other.capture["title"], current.capture["title"])
    report = {
        "reference": other.capture["id"], "reference_plane": other.plane["id"],
        "box": settings.region, "metric": settings.metric, "normalization": neutral,
        "measured": counts,
        "difference_percentiles": quantiles(discrepancy[within]),
        "tolerance": limit,
        "note": "Difference within a physically aligned image pair, after the stated optional tonal reference.",
    }
    return report, {"map": legend(panel, "Paired image differences", titles), "pair": guide}, other.references()


@dataclass(frozen=True)
class ContactRequest:
    reference: str
    reference_plane: str | None
    region: list
    rows: int
    columns: int
    tile_fraction: float

    @classmethod
    def parse(cls, raw):
        form = mapping(raw)
        return cls(
            identifier(form.get("reference"), "reference capture"),
            identifier(form.get("reference_plane"), "reference calibration", optional=True),
            common_box(form),
            integer(form.get("rows"), "paired area rows", 2, 16, default=5),
            integer(form.get("columns"), "paired area columns", 2, 16, default=7),
            real(form.get("tile_fraction"), "local review coverage", 0.2, 0.99, default=0.55),
        )


def neutral_overlay(surface, first, good_a, good_b, selection, bounds):
    rendering = np.zeros(good_a.shape, dtype=np.uint8)
    rendering[selection & good_a & ~good_b] = 1
    rendering[selection & good_b & ~good_a] = 2
    rendering[selection & ~good_a & ~good_b] = 3
    draw = blended_overlay(surface, rendering, {1: OCHRE, 2: TEAL, 3: LILAC})
    pencil = ImageDraw.Draw(draw)
    pencil.rectangle(bounds, outline=TEAL, width=2)
    return draw


def paired_cell(current, reference, can_a, can_b, top, bottom, left, right, acceptable):
    area = np.s_[top:bottom, left:right]
    a, b = can_a[area], can_b[area]
    both = a & b
    valid_fraction = float(both.mean()) if both.size else 0.0
    summary = {
        "bounds": [left, top, right, bottom],
        "common": int(both.sum()), "current_only": int((a & ~b).sum()),
        "reference_only": int((b & ~a).sum()), "neither": int((~a & ~b).sum()),
        "common_fraction": round(valid_fraction, 4),
    }
    if valid_fraction < acceptable or np.count_nonzero(both) < 24:
        return summary | {"status": "missing shared support", "contrast": None}
    first = luma(current.linear[area][both])
    second = luma(reference.linear[area][both])
    return summary | {
        "status": "shared view",
        "contrast": round(float(np.median(first - second)), 6),
        "reference_luma": quantiles(second, (10, 50, 90)),
        "current_luma": quantiles(first, (10, 50, 90)),
    }


def coverage(ctx, raw, lib):
    options = ContactRequest.parse(raw)
    reference, surface, support = two_views(lib, ctx, options, role_must_agree=False)
    size = ctx.surface.extent
    left, top, right, bottom = cut_rectangle(options.region, size)
    if min((right - left) / options.columns, (bottom - top) / options.rows) < 7:
        raise WorkbenchError("Use fewer rows or columns for this comparison crop.", "box")
    selection = box_mask(size, options.region)
    grid = []
    xs = np.rint(np.linspace(left, right, options.columns + 1)).astype(int)
    ys = np.rint(np.linspace(top, bottom, options.rows + 1)).astype(int)
    for i, (y0, y1) in enumerate(zip(ys, ys[1:]), 1):
        for j, (x0, x1) in enumerate(zip(xs, xs[1:]), 1):
            assessed = paired_cell(ctx.surface, surface, ctx.readable, support,
                                   int(y0), int(y1), int(x0), int(x1), options.tile_fraction)
            grid.append(assessed | {"row": i, "column": j})
    board = neutral_overlay(ctx.surface, surface, ctx.readable, support, selection, (left, top, right, bottom))
    draw = ImageDraw.Draw(board)
    for item in grid:
        x0, y0, x1, y1 = item["bounds"]
        uncertain = item["status"] != "shared view"
        color = LILAC if uncertain else TEAL
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color)
        if min(x1 - x0, y1 - y0) > 36:
            label(draw, (x0 + 2, y0 + 2), f"{item['row']}.{item['column']}", size=9, fill=color)
    return {
        "range": options.region, "reference": reference.capture["id"],
        "reference_plane": reference.plane["id"],
        "rows": options.rows, "columns": options.columns, "grid": grid,
        "shared_pixels": sum(item["common"] for item in grid),
        "note": "Photographic coverage compares the same part of the calibrated fabric on both captures.",
    }, {
        "pair": side_panel(surface, ctx.surface, options.region, reference.capture["title"], ctx.capture["title"]),
        "map": legend(board, "Two-image coverage ledger", [
            ("Current only", OCHRE), ("Reference only", TEAL), ("Absent on both", LILAC)]),
    }, reference.references()

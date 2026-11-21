"""Image support audit, light passage map and normal colour observation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import ImageDraw

from .contracts import WorkbenchError, box, integer, mapping, real, require_support, text
from .geometry import box_mask, cut_rectangle, polygon_mask
from .raster import gradients, luma, quantiles, read_fraction, rgb_sample, to_lab, whitened
from .visual import (
    LILAC, MUTED, OCHRE, PALE, PINK, TEAL, blended_overlay, heatmap, histogram_plot,
    label, legend, neutral_card,
)

AUDIT_CLASSES = {1: LILAC, 2: (171, 146, 202), 3: (109, 130, 188), 4: OCHRE, 5: TEAL}
AUDIT_LABELS = {
    1: "Missing image", 2: "Excluded observation", 3: "Too dark", 4: "Clip-prone highlights",
    5: "Lower edge detail",
}


def common_box(raw):
    return box(raw.get("box"), default=[0, 0, 1, 1])


@dataclass(frozen=True)
class AuditOptions:
    selection: list
    low_threshold: float
    high_threshold: float
    tile: int
    minimum_detail: float

    @classmethod
    def parse(cls, raw):
        form = mapping(raw)
        return cls(
            common_box(form),
            real(form.get("low"), "dark signal threshold", 0.0001, 0.5, default=0.01),
            real(form.get("high"), "high signal threshold", 0.51, 1, default=0.975),
            integer(form.get("tile"), "inspection cell", 16, 300, default=72),
            real(form.get("detail"), "minimum local edge detail", 0.001, 0.4, default=0.012),
        )


def per_tile(grads, can, inside, extent, stride, minimum):
    frame_width, frame_height = extent
    full = inside.shape
    sparse = np.zeros(full, dtype=bool)
    detail = []
    for top in range(0, frame_height, stride):
        for left in range(0, frame_width, stride):
            bottom, right = min(top + stride, frame_height), min(left + stride, frame_width)
            selection = np.s_[top:bottom, left:right]
            window = can[selection]
            region = inside[selection]
            good, total = int(window.sum()), int(region.sum())
            if total <= 0:
                continue
            measured = quantiles(grads[selection][window], (75, 90))
            treated = bool(good >= 32 and good / total >= 0.35)
            category = "ready" if treated else "insufficient"
            if treated and measured[1] < minimum:
                sparse[selection] = window
                category = "low detail"
            detail.append({
                "bounds": [left, top, right, bottom],
                "pixels": total, "readable": good, "edge_p75_p90": measured, "class": category,
            })
    return sparse, detail


def audit(ctx, raw):
    settings = AuditOptions.parse(raw)
    surface = ctx.surface
    inside = box_mask(surface.extent, settings.selection)
    light = luma(surface.linear)
    clipped = np.max(surface.linear, axis=2) >= settings.high_threshold
    available = inside & ctx.visible
    bright = clipped & available
    dark = (light <= settings.low_threshold) & available & ~bright
    readable = available & ~bright & ~dark
    dx, dy, core = gradients(light, readable)
    soft, groups = per_tile(np.hypot(dx, dy), core, inside, surface.extent, settings.tile,
                            settings.minimum_detail)
    classes = np.zeros(inside.shape, dtype=np.uint8)
    classes[inside & ~surface.valid] = 1
    classes[inside & surface.valid & ctx.obscured] = 2
    classes[dark] = 3
    classes[bright] = 4
    classes[soft] = 5
    totals = {AUDIT_LABELS[key]: int((classes == key).sum()) for key in AUDIT_CLASSES}
    report = {
        "box": settings.selection,
        "reviewed_pixels": int(inside.sum()),
        "usable_pixels": int(readable.sum()),
        "categories": totals,
        "light_percentiles": quantiles(light[readable]),
        "edge_detail_percentiles": quantiles(np.hypot(dx[core], dy[core])),
        "grid": groups,
        "note": "Edge detail describes the photo and may be low on deliberately smooth cloth.",
    }
    picture = blended_overlay(surface, classes, AUDIT_CLASSES)
    return report, {"map": legend(picture, "Exposure and usable detail", [
        (label_, AUDIT_CLASSES[key]) for key, label_ in AUDIT_LABELS.items()
    ])}


def corrected_display(ctx, raw):
    form = mapping(raw)
    black = real(form.get("black"), "black level", 0, 0.98, default=0.0)
    white = real(form.get("white"), "white level", 0.02, 1, default=1.0)
    gamma = real(form.get("gamma"), "display gamma", 0.2, 4, default=1.0)
    if black + 0.01 > white:
        raise WorkbenchError("Leave at least a one-percent gap between black and white.", "white")
    grayscale = form.get("channel") == "gray"
    return whitened(ctx.surface, black, white, gamma, grayscale=grayscale)


@dataclass(frozen=True)
class PassageOptions:
    target: list
    lit: list
    blank: list | None
    cell: int

    @classmethod
    def parse(cls, raw):
        form = mapping(raw)
        return cls(
            common_box(form), box(form.get("lit"), "uncovered light reference", default=[0.715, 0.08, 0.18, 0.10]),
            box(form["blank"], "light-blocking reference") if form.get("blank") is not None else None,
            integer(form.get("cell"), "passage grid cell", 25, 400, default=90),
        )


def reference_light(field, valid, region, extent, label_):
    requested = box_mask(extent, region)
    accepted = valid & requested
    require_support(int(requested.sum()), int(accepted.sum()), label_, 35)
    samples = field[accepted]
    percent = quantiles(samples)
    return float(np.median(samples)), {
        "region": region, "accepted": int(accepted.sum()), "percentiles": percent
    }


def passage_grid(values, wanted, good, extent, step):
    width, height = extent
    results = []
    for y in range(0, height, step):
        for x in range(0, width, step):
            rr, bb = min(width, x + step), min(height, y + step)
            region = np.s_[y:bb, x:rr]
            area = int(wanted[region].sum())
            if not area:
                continue
            keep = good[region]
            seen = int(keep.sum())
            enough = seen >= 20 and seen / area >= 0.35
            results.append({
                "box": [x, y, rr, bb], "wanted_px": area, "supported_px": seen,
                "passage_percentiles": quantiles(values[region][keep]) if enough else [],
            })
    return results


def passage(ctx, raw):
    if ctx.capture["role"] != "backlight":
        raise WorkbenchError("Choose a transmitted-light photo to measure passage.", "photo")
    options = PassageOptions.parse(raw)
    surface = ctx.surface
    light = luma(surface.linear)
    requested = box_mask(surface.extent, options.target)
    high, top = reference_light(light, ctx.readable, options.lit, surface.extent,
                                 "Uncovered light reference")
    low, bottom = 0.0, None
    if options.blank:
        low, bottom = reference_light(light, ctx.visible & ~surface.bright, options.blank,
                                       surface.extent, "Light-blocking reference")
    if high - low < 0.055:
        raise WorkbenchError("The light and blocked references are too close to measure passage.")
    expected = (light - low) / (high - low)
    inside = requested & ctx.visible
    reliable = inside & ~surface.bright & (expected >= -0.04) & (expected <= 1.04)
    require_support(int(requested.sum()), int(reliable.sum()), "Passage inspection", 80)
    fraction = np.clip(expected, 0, 1)
    reading = {
        "box": options.target,
        "light_reference": top, "blocked_reference": bottom,
        "measured_pixels": int(reliable.sum()), "unknown_pixels": int((requested & ~reliable).sum()),
        "passage_percentiles": quantiles(fraction[reliable]),
        "grid": passage_grid(fraction, requested, reliable, surface.extent, options.cell),
        "note": "Relative optical passage in this capture; not calibrated laboratory transmittance.",
    }
    if ctx.scale is not None:
        reading["measured_mm2"] = round(float(reliable.sum() * np.prod(ctx.scale)), 4)
    product = heatmap(surface, fraction, reliable, (0, 1), unknown=requested & ~reliable)
    return reading, {"map": legend(product, "Relative optical passage", [
        ("Higher passage", TEAL), ("No reliable reading", LILAC)
    ])}


def region_for_swatch(form, mask, ctx):
    if form.get("mark"):
        chosen = next((item for item in ctx.marks if item["id"] == form["mark"]), None)
        if not chosen or chosen["shape"] != "polygon":
            raise WorkbenchError("Choose a visible outlined region for a color swatch.", "mark")
        mask = polygon_mask(ctx.surface.extent, chosen["points"], chosen.get("holes", []))
        return mask, chosen
    return mask, None


def chroma_distance(linear, ref):
    return np.linalg.norm(to_lab(linear) - np.asarray(ref), axis=1)


def swatch_reading(ctx, raw):
    data = mapping(raw)
    bounds = common_box(data)
    surface = ctx.surface
    region, annotation = region_for_swatch(data, box_mask(surface.extent, bounds), ctx)
    good = region & ctx.readable
    require_support(int(region.sum()), int(good.sum()), "Color sample", 70)
    sample = rgb_sample(surface.linear[good])
    selected = surface.linear[good]
    spread = chroma_distance(selected, sample["lab_d65"])
    groups = int(min(20, max(4, len(selected) / 240)))
    histogram, edges = np.histogram(spread, bins=groups, range=(0, max(0.25, float(np.percentile(spread, 98)))))
    distances = [[float((before + after) * 0.5), int(count)]
                 for before, after, count in zip(edges, edges[1:], histogram)]
    summary = {
        "name": text(data.get("name"), "swatch label", 70, default="Cloth sample"),
        "box": bounds,
        "mark": annotation["id"] if annotation else None,
        "color": sample,
        "calibrated_measures": ctx.scale is not None,
        "pixel_count": int(good.sum()),
        "color_variability": quantiles(spread),
        "variance_distribution": distances,
        "clip_fraction": read_fraction(region & surface.bright, denominator=int(region.sum())),
        "note": "Image color under the recorded illumination. Record a reference card when comparing sessions.",
    }
    all_shades = np.zeros(good.shape, dtype=np.float32)
    all_shades[good] = spread
    mapper = heatmap(surface, all_shades, good, (0, max(0.25, float(np.percentile(spread, 92)))),
                     unknown=region & ~good, hue=OCHRE)
    chart = histogram_plot([{"points": distances, "color": MUTED}], "Color variation in selected area",
                           x_label="Delta E76 from median", y_label="Pixel count")
    card = neutral_card([{"label": summary["name"], "color": summary["color"]}], "Measured color")
    return summary, {
        "map": legend(mapper, "Local color variation", [("Higher variation", OCHRE), ("Unavailable", LILAC)]),
        "chart": chart, "card": card,
    }


def physical_crop_note(ctx, bounds):
    result = cut_rectangle(bounds, ctx.surface.extent)
    extent = [result[2] - result[0], result[3] - result[1]]
    physical = (np.asarray(extent) * ctx.scale).tolist() if ctx.scale is not None else None
    if physical:
        physical = [round(value, 4) for value in physical]
    return {
        "plane": ctx.plane["id"] if ctx.plane else None,
        "box": bounds, "pixel_bounds": list(result), "extent": extent,
        "print_mm": physical,
    }

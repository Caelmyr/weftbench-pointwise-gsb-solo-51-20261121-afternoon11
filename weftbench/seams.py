"""Sewing-line measurements and yarn trace profiles using supported stations."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import ImageDraw
from scipy import signal

from .contracts import WorkbenchError, identifier, integer, mapping, real, require_support
from .geometry import line_points, pixel_points, profile_lattice, trace_stations
from .raster import luma, quantiles, rgba
from .visual import LILAC, OCHRE, PINK, TEAL, histogram_plot, label, legend, line_dashboard

MAX_WIDTH_SAMPLES = 450


def source_trace(ctx, raw, key, kind):
    value = raw.get(key)
    if isinstance(value, list):
        return line_points(value, key), None
    mark_id = identifier(value, key)
    selected = next((row for row in ctx.marks if row["id"] == mark_id), None)
    if selected is None or selected["shape"] != "line" or selected["mark_type"] not in kind:
        raise WorkbenchError("Choose a visible trace of the requested type.", key)
    return selected["points"], selected


def read_path(ctx, form, field="trace"):
    route, reference = source_trace(ctx, form, field, {"seam", "cut", "selvedge"})
    step = real(form.get("step"), "distance between samples in pixels", 2, 90, default=9)
    walked, centers, normals = trace_stations(route, ctx.surface.extent, step, cap=MAX_WIDTH_SAMPLES)
    return route, reference, walked, centers, normals


def cumulative_units(ctx, centers, walked):
    if ctx.scale is None:
        return walked, "px"
    advance = np.diff(centers, axis=0) * ctx.scale
    return np.r_[0, np.cumsum(np.linalg.norm(advance, axis=1))], "mm"


def valid_location(good, xy):
    x, y = map(int, np.rint(xy))
    return 0 <= y < good.shape[0] and 0 <= x < good.shape[1] and bool(good[y, x])


def line_witness(width, height, x, y, color):
    return [float(x / width), float(y / height)], color


def run_segments(path, extents):
    endpoints = np.asarray(pixel_points(path, extents), dtype=np.float64)
    segment_start = endpoints[:-1]
    segment_vector = endpoints[1:] - segment_start
    return segment_start, segment_vector


def intersects_normals(origin, normal, segment_start, segment_delta, max_span):
    # Solve origin + t * normal == start + u * edge independently for each edge segment.
    forward = segment_start - origin[None, :]
    divisor = normal[1] * segment_delta[:, 0] - normal[0] * segment_delta[:, 1]
    estimable = np.abs(divisor) > 1e-8
    safe = np.where(estimable, divisor, 1.0)
    distance = (forward[:, 1] * segment_delta[:, 0] - forward[:, 0] * segment_delta[:, 1]) / safe
    fraction = (forward[:, 1] * normal[0] - forward[:, 0] * normal[1]) / safe
    hits = estimable & (fraction >= 0) & (fraction <= 1) & (np.abs(distance) <= max_span)
    if not hits.any():
        return []
    return sorted([(float(abs(distance[i])), float(distance[i]), int(i)) for i in np.flatnonzero(hits)])


def ray_covered(origin, endpoint, usable):
    distance = float(np.linalg.norm(endpoint - origin))
    checks = np.linspace(origin, endpoint, max(2, int(math.ceil(distance)) + 1))
    xx, yy = np.rint(checks).astype(int).T
    return bool(
        (xx >= 0).all() and (yy >= 0).all() and (xx < usable.shape[1]).all()
        and (yy < usable.shape[0]).all() and usable[yy, xx].all()
    )


@dataclass(frozen=True)
class Allowance:
    limit_px: float
    side: str

    @classmethod
    def parse(cls, raw):
        side = raw.get("side", "nearest")
        if side not in ("positive", "negative", "nearest"):
            raise WorkbenchError("Choose the nearer edge or a single side.", "side")
        return cls(real(raw.get("limit"), "maximum allowance search", 4, 170, default=45), side)


def tested_width(center, normal, side, segments, support, maximum):
    if not valid_location(support, center):
        return None, "seam center unavailable"
    matches = intersects_normals(center, normal, *segments, maximum)
    if side == "positive":
        matches = [row for row in matches if row[1] > 0]
    if side == "negative":
        matches = [row for row in matches if row[1] < 0]
    if not matches:
        return None, "cut edge outside search"
    nearest = matches[0]
    endpoint = center + normal * nearest[1]
    if not ray_covered(center, endpoint, support):
        return endpoint, "width passes through folded or missing fabric"
    return endpoint, "measured"


def display_trace(route, mark, title, kind):
    if mark is not None:
        return mark
    return {
        "id": None, "title": title, "mark_type": kind,
        "shape": "line", "points": route, "holes": [],
    }


def allowance(ctx, raw):
    form = mapping(raw)
    control = Allowance.parse(form)
    route, seam, distances, centers, normals = read_path(ctx, form, "seam")
    trim, edge = source_trace(ctx, form, "cut", {"cut", "selvedge"})
    segments = run_segments(trim, ctx.surface.extent)
    length, unit = cumulative_units(ctx, centers, distances)
    width, height = ctx.surface.extent
    values = []
    rows = []
    for index, (center, normal) in enumerate(zip(centers, normals, strict=True)):
        hit, tag = tested_width(center, normal, control.side, segments, ctx.visible, control.limit_px)
        width_units = None
        if tag == "measured":
            delta = hit - center
            width_units = float(np.linalg.norm(delta * ctx.scale) if ctx.scale is not None else np.linalg.norm(delta))
            values.append(width_units)
        rows.append({
            "station": index + 1, "center": [round(float(center[0] / width), 6), round(float(center[1] / height), 6)],
            "cut_point": [round(float(hit[0] / width), 6), round(float(hit[1] / height), 6)] if hit is not None else None,
            "distance": round(float(length[index]), 6), "width": round(width_units, 6) if width_units is not None else None,
            "status": tag,
        })
    series = [{"points": [[row["distance"], row["width"]] for row in rows], "color": TEAL}]
    entry = {
        "seam": seam["id"] if seam else None, "cut": edge["id"] if edge else None,
        "units": unit, "tested_stations": len(rows), "unmeasured_stations": len(rows) - len(values),
        "width_percentiles": quantiles(values), "stations": rows,
        "note": "Distance to the marked cut edge along each seam normal. Unknown spans remain blank.",
    }
    readings = [display_trace(route, seam, "Sewn route", "seam"), display_trace(trim, edge, "Cut edge", "cut")]
    artwork = line_dashboard(ctx.surface, readings, series, name="Seam allowance")
    return entry, {"map": artwork}


@dataclass(frozen=True)
class StitchOptions:
    flank: int
    minimum_pitch: int
    prominence: float
    turn: str

    @classmethod
    def parse(cls, form):
        style = form.get("polarity", "light")
        if style not in {"dark", "light", "either"}:
            raise WorkbenchError("Choose dark or light stitch contrast.", "polarity")
        return cls(
            integer(form.get("flank"), "seam sample half-width", 2, 38, default=8),
            integer(form.get("minimum_pitch"), "minimum stitch gap in stations", 2, 22, default=3),
            real(form.get("prominence"), "minimum stitch contrast", 0.005, 0.6, default=0.025),
            style,
        )


def central_contrast(profile, mask, offsets):
    core = np.abs(offsets) <= 1
    wing_left = offsets < -1
    wing_right = offsets > 1
    good = mask[:, core].all(axis=1) & (
        mask[:, wing_left].sum(axis=1) >= 1
    ) & (mask[:, wing_right].sum(axis=1) >= 1)
    scores = np.zeros(profile.shape[0], dtype=np.float64)
    for index in np.flatnonzero(good):
        found = mask[index]
        left = np.median(profile[index, wing_left & found])
        right = np.median(profile[index, wing_right & found])
        center = np.mean(profile[index, core])
        scores[index] = center - 0.5 * (left + right)
    return scores, good


def visible_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges > 0), np.flatnonzero(edges < 0), strict=True)


def pitch_marks(data, valid, options):
    usable_score = data if options.turn == "light" else -data if options.turn == "dark" else np.abs(data)
    readings = []
    for start, end in visible_runs(valid):
        if end - start < 4:
            continue
        indexes, report = signal.find_peaks(usable_score[start:end], distance=options.minimum_pitch,
                                            prominence=options.prominence)
        readings.append([(int(start + index), float(level))
                         for index, level in zip(indexes, report["prominences"], strict=True)])
    return readings


def stitches(ctx, raw):
    form = mapping(raw)
    settings = StitchOptions.parse(form)
    fast = {"step": 2.4} | form
    route, seam, walked, centers, normals = read_path(ctx, fast, "seam")
    offsets, _, profiles, mask = profile_lattice(luma(ctx.surface.linear), ctx.readable,
                                                centers, normals, settings.flank)
    contrast, measured = central_contrast(profiles, mask, offsets)
    require_support(len(measured), int(measured.sum()), "Stitch inspection", 15, fraction=0.28)
    values, unit = cumulative_units(ctx, centers, walked)
    events = pitch_marks(contrast, measured, settings)
    pitches = [float(values[b[0]] - values[a[0]]) for run in events for a, b in zip(run, run[1:])]
    marks = [{"station": index + 1, "distance": round(float(values[index]), 5),
              "position": (centers[index] / ctx.surface.extent).tolist(), "prominence": round(level, 6)}
             for run in events for index, level in run]
    series = [[float(values[i]), float(contrast[i]) if measured[i] else None] for i in range(len(contrast))]
    map_ = rgba(ctx.surface)
    draw = ImageDraw.Draw(map_)
    for i, xy in enumerate(centers):
        if measured[i]:
            continue
        draw.point(tuple(xy), fill=(*LILAC, 255))
    for selected in marks:
        x, y = centers[selected["station"] - 1]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(*TEAL, 255), width=2)
    report = {
        "seam": seam["id"] if seam else None, "unit": unit, "rate_percentiles": quantiles(pitches),
        "readable_stations": int(measured.sum()), "unknown_stations": int((~measured).sum()),
        "cand_count": len(marks), "candidate_sites": marks, "signed_contrast": series,
        "note": "Visible optical stitch peaks. Gaps spanning a hidden section are excluded.",
    }
    return report, {
        "map": legend(map_, "Candidate stitch locations", [("Candidate peak", TEAL), ("Unmeasurable trace", LILAC)]),
        "chart": histogram_plot([{"points": series, "color": PINK}], "Stitch line optical profile",
                                 x_label=f"Trace distance ({unit})", y_label="Center contrast"),
    }


def edge_profile(ctx, raw):
    form = mapping(raw)
    _, mark, distances, centers, normals = read_path(ctx, form, "trace")
    flank = integer(form.get("half_width"), "cross-section half-width", 2, 48, default=14)
    offsets, _, values, mask = profile_lattice(luma(ctx.surface.linear), ctx.readable, centers, normals, flank)
    distance, unit = cumulative_units(ctx, centers, distances)
    selected = [i for i in range(len(offsets)) if abs(offsets[i]) in {0, flank}]
    read = []
    series = []
    for color, index in zip((OCHRE, TEAL, PINK), selected, strict=True):
        series.append({
            "color": color,
            "points": [[float(s), float(values[i, index]) if mask[i, index] else None]
                       for i, s in enumerate(distance)],
        })
    for number, (offset, samples, eligible) in enumerate(zip(distance, values, mask, strict=True), 1):
        read.append({
            "sample": number, "distance": round(float(offset), 5),
            "supported_fraction": round(float(eligible.mean()), 4),
            "across_levels": [round(float(v), 6) if ok else None for v, ok in zip(samples, eligible, strict=True)],
        })
    report = {
        "source_mark": mark["id"] if mark else None, "unit": unit,
        "across_offsets_px": offsets.tolist(), "sections": read,
        "visible_fraction": round(float(mask.mean()), 6),
    }
    return report, {"chart": histogram_plot(series, "Center and side trace levels",
                x_label=f"Distance along trace ({unit})", y_label="Linear luminance")}

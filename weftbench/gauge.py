"""Measure calibrated cloth spans and directional stretch from independent landmarks."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import ImageDraw

from .catalog import view
from .contracts import WorkbenchError, identifier, integer, limit, mapping, point, real, sequence, text
from .geometry import pixel_at, pixel_points, trace_stations
from .raster import png, quantiles, rgba
from .visual import LILAC, OCHRE, PINK, TEAL, histogram_plot, label, legend, panel_pair


@dataclass(frozen=True)
class Span:
    name: str
    begin: list
    end: list
    design: float | None

    @classmethod
    def parse(cls, raw, index):
        form = mapping(raw)
        design = form.get("target_mm")
        return cls(
            text(form.get("name"), "span name", 54, default=f"Station {index}"),
            point(form.get("begin"), "first landmark"),
            point(form.get("end"), "second landmark"),
            real(design, "reference length in millimetres", 0.4, 8000) if design is not None else None,
        )


@dataclass(frozen=True)
class GaugeRequest:
    spans: list[Span]
    minimum_coverage: float
    check_centers: bool

    @classmethod
    def parse(cls, raw):
        form = mapping(raw, "span gauge")
        wanted = [Span.parse(item, i) for i, item in enumerate(
            sequence(form.get("spans"), "gauged spans", 1, 36), 1,
        )]
        if len({span.name for span in wanted}) != len(wanted):
            raise WorkbenchError("Label each checked span differently.", "spans")
        return cls(wanted, real(form.get("coverage"), "visible length fraction", 0.35, 1, default=0.9),
                   bool(form.get("check_center", True)))


def point_usable(mask, at):
    height, width = mask.shape
    x, y = pixel_at(at, (width, height))
    return bool(mask[y, x])


def evaluate(ctx, ruler, request):
    a, b = ruler.begin, ruler.end
    width, height = ctx.surface.extent
    p, q = np.array(pixel_points([a, b], ctx.surface.extent))
    span_px = float(np.linalg.norm(q - p))
    if span_px < 3:
        raise WorkbenchError(f"{ruler.name} needs two more widely separated landmarks.", "spans")
    n = min(2200, max(15, int(np.ceil(span_px * 1.3))))
    sampling = np.column_stack([np.linspace(p[0], q[0], n), np.linspace(p[1], q[1], n)])
    xx, yy = np.floor(sampling).astype(int).T
    xx = np.minimum(xx, width - 1)
    yy = np.minimum(yy, height - 1)
    inside = ctx.visible[yy, xx]
    sample = float(inside.mean())
    good = point_usable(ctx.visible, a) and point_usable(ctx.visible, b)
    if request.check_centers:
        good = good and sample >= request.minimum_coverage
    shift = (q - p) * ctx.scale
    span = float(np.linalg.norm(shift))
    ratio = None
    if good and ruler.design is not None:
        ratio = (span / ruler.design - 1) * 100
    readings = {
        "name": ruler.name, "begin": a, "end": b,
        "details": {"visible_fraction": round(sample, 5), "pixels": round(span_px, 5)},
        "classification": "visible span" if good else "held for missing landmark or section",
        "length_mm": round(span, 4) if good else None,
        "components_mm": np.round(shift, 5).tolist() if good else None,
        "delta_percent": round(ratio, 5) if ratio is not None else None,
        "nominal_mm": ruler.design,
    }
    return readings, sampling, inside


def meter_line(draw, coordinates, visibility, shade):
    previous = None
    prev_valid = None
    for at, clear in zip(coordinates, visibility, strict=True):
        if previous is not None and clear and prev_valid:
            draw.line((tuple(previous), tuple(at)), fill=(*shade, 255), width=2)
        elif previous is not None and not clear:
            draw.point(tuple(at), fill=(*LILAC, 255))
        previous, prev_valid = at, clear


def paint_span(painter, result, samples, inside, index):
    status = result["length_mm"] is not None
    shade = TEAL if status else PINK
    meter_line(painter, samples, inside, shade)
    for x, y in samples[[0, -1]]:
        painter.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(*shade, 255))
    x, y = samples[len(samples) // 2]
    size = f'{result["length_mm"]:.1f} mm' if status else "review"
    label(painter, (x + 3, y - 16), f"{index}. {size}", fill=shade, size=11)


def evaluate_spans(ctx, request):
    source = rgba(ctx.surface)
    drawer = ImageDraw.Draw(source)
    reports = []
    for index, row in enumerate(request.spans, 1):
        reading, samples, inside = evaluate(ctx, row, request)
        paint_span(drawer, reading, samples, inside, index)
        reports.append(reading)
    values = [row["length_mm"] for row in reports if row["length_mm"] is not None]
    changes = [row["delta_percent"] for row in reports if row["delta_percent"] is not None]
    percent = [[i, sample["delta_percent"]] for i, sample in enumerate(reports, 1)]
    has_nominal = any(point_[1] is not None for point_ in percent)
    if has_nominal:
        chart = histogram_plot([{"points": percent, "color": OCHRE}], "Percentage from each reference",
                               x_label="Measured span", y_label="Difference (%)")
        source = panel_pair(source, chart, "Numbered fabric spans", "Only visible spans with a nominal length")
    details = {
        "fabric": ctx.fabric["id"], "capture": ctx.capture["id"], "plane": ctx.plane["id"],
        "spans": reports, "tested_spans": len(reports), "supported_spans": len(values),
        "length_percentiles_mm": quantiles(values), "delta_percentiles": quantiles(changes),
        "should_be_visible_fraction": request.minimum_coverage,
        "note": "Straight landmark-to-landmark lengths in the retained image plane.",
    }
    return details, legend(source, "Calibrated landmark spans", [
        ("Visible measured span", TEAL), ("End point for review", PINK), ("Gap in visual support", LILAC),
    ])


def gauge(store, key, raw, *, save):
    form = mapping(raw)
    options = GaugeRequest.parse(form)
    plane = identifier(form.get("plane"), "plane", optional=True)
    with store.open() as lib:
        ctx = view(lib, key, plane_key=plane, registered=True)
    report, art = evaluate_spans(ctx, options)
    if not save:
        return report, art
    with store.open(write=True) as lib:
        if lib.count("readings", parent=ctx.fabric["id"]) >= 120:
            raise limit("This fabric has reached its retained inspection limit.")
        for row, _ in ctx.references():
            lib.require_version(lib.get(row["id"], row["kind"]), row["version"])
        record = lib.add("readings", {
            "title": text(form.get("title"), "span gauge title", 94, default="Fabric landmark gauge"),
            "tool": "gauge", "analysis": report, "proofs": ["map"], "review_state": "unreviewed",
            "parameters": {"capture": key, "capture_role": ctx.capture["role"],
                           "plane": ctx.plane["id"], "coordinate_basis": "fabric"},
        }, parent=ctx.fabric["id"], action="fabric dimensions gauged")
        for linked, role in ctx.references():
            lib.cite(record["id"], linked, role)
        lib.attach(record["id"], "map", png(art), "image/png")
        return lib.public(record, links=True)


def spacing_route(store, capture, raw):
    form = mapping(raw)
    guides = sequence(form.get("points"), "measured route", 2, 24)
    plane = identifier(form.get("plane"), "plane", optional=True)
    with store.open() as lib:
        ctx = view(lib, capture, plane_key=plane, registered=True)
    line = [point(at, "route point") for at in guides]
    interval = real(form.get("step"), "station interval in pixels", 3, 90, default=23)
    distance, sampled, normal = trace_stations(line, ctx.surface.extent, interval, cap=240)
    steps_mm = np.r_[0, np.cumsum(np.linalg.norm(np.diff(sampled, axis=0) * ctx.scale, axis=1))]
    count = integer(form.get("style"), "max displayed sample points", 10, 240, default=160)
    if len(distance) > count:
        raise WorkbenchError("Raise the station interval to fit the available labels.", "step")
    artwork = rgba(ctx.surface)
    draw = ImageDraw.Draw(artwork)
    results = []
    previous = None
    for i, (at, along, perpendicular) in enumerate(zip(sampled, steps_mm, normal, strict=True), 1):
        x, y = at
        visible = point_usable(ctx.visible, (x / artwork.width, y / artwork.height))
        shade = TEAL if visible else LILAC
        if visible and previous is not None:
            draw.line((*previous, x, y), fill=(*shade, 255), width=2)
        u, v = perpendicular * 4
        draw.line((x - u, y - v, x + u, y + v), fill=(*shade, 255), width=2)
        if i % 4 == 1:
            label(draw, (x + 6, y - 15), str(i), fill=shade, size=9)
        results.append({
            "station": i, "point": [round(float(x / artwork.width), 6), round(float(y / artwork.height), 6)],
            "path_mm": round(float(along), 5), "visible": visible,
        })
        previous = at if visible else None
    return results, legend(artwork, "Even stations along drawn fabric route", [("Sample visible", TEAL), ("Held", LILAC)])

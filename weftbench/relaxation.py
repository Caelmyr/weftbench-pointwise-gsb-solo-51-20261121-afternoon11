"""Compare retained before/after cloth dimensions from corresponding physical landmarks."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import ImageDraw

from .catalog import view
from .contracts import WorkbenchError, identifier, limit, mapping, point, real, sequence, text
from .geometry import pixel_at, pixel_points
from .raster import png, quantiles, rgba
from .visual import LILAC, OCHRE, PINK, TEAL, label, legend, panel_pair


@dataclass(frozen=True)
class EvidencePoint:
    name: str
    before: list
    after: list

    @classmethod
    def parse(cls, raw, ordinal):
        item = mapping(raw, "paired physical landmark")
        return cls(
            text(item.get("name"), "landmark name", 45, default=f"Pair {ordinal}"),
            point(item.get("before"), "point before relaxation"),
            point(item.get("after"), "point after relaxation"),
        )


@dataclass(frozen=True)
class RelaxationRequest:
    base: str
    base_plane: str
    latest_plane: str
    stable_pairs: list
    limit: float

    @classmethod
    def parse(cls, raw):
        form = mapping(raw, "relaxation inspection")
        source = identifier(form.get("reference"), "earlier photograph")
        prior = identifier(form.get("reference_plane"), "earlier plane")
        after = identifier(form.get("plane"), "current plane")
        pairs = [EvidencePoint.parse(item, index) for index, item in enumerate(
            sequence(form.get("pairs"), "physical point pairs", 3, 24), 1)]
        if len(set(item.name for item in pairs)) != len(pairs):
            raise WorkbenchError("Name each point pair distinctly.", "pairs")
        if len({tuple(p.before) for p in pairs}) != len(pairs) or len({tuple(p.after) for p in pairs}) != len(pairs):
            raise WorkbenchError("Use distinct points in both photographs.", "pairs")
        return cls(source, prior, after, pairs, real(form.get("tolerance"), "allowable residual in mm", 0.1, 80, default=4))


def position_in_view(position, ctx):
    x, y = pixel_at(position, ctx.surface.extent)
    return bool(ctx.visible[y, x])


def dimensional_pairs(options, before, after):
    readings = []
    old_size = np.array([before.plane["width_mm"], before.plane["height_mm"]])
    new_size = np.array([after.plane["width_mm"], after.plane["height_mm"]])
    for item in options.stable_pairs:
        reference = np.array(item.before) * old_size
        current = np.array(item.after) * new_size
        visible = position_in_view(item.before, before) and position_in_view(item.after, after)
        readings.append({
            "name": item.name, "before": item.before, "after": item.after, "before_mm": reference.tolist(),
            "after_mm": current.tolist(), "usable": visible,
        })
    return readings


def transform_estimate(pairs):
    first = np.array([item["before_mm"] for item in pairs])
    second = np.array([item["after_mm"] for item in pairs])
    centre = first.mean(axis=0)
    change = second.mean(axis=0)
    initial = first - centre
    target = second - change
    linear, _, rank, singular = np.linalg.lstsq(initial, target, rcond=None)
    if rank < 2 or singular[-1] / singular[0] < 0.04:
        raise WorkbenchError("Spread the point pairs across both fabric directions.", "pairs")
    if np.linalg.det(linear) <= 0:
        raise WorkbenchError("The landmarks reverse fabric handedness. Review the photos' orientation.", "pairs")
    shift = change - centre @ linear
    ranges = np.linalg.norm(second - (first @ linear + shift), axis=1)
    return linear, shift, ranges


def scale_reading(linear):
    dx, dy = linear[0], linear[1]
    factor_x, factor_y = np.linalg.norm(dx), np.linalg.norm(dy)
    angle = math.degrees(math.acos(float(np.clip(np.dot(dx, dy) / (factor_x * factor_y), -1, 1))))
    singular = np.linalg.svd(linear, compute_uv=False)
    return {
        "width_change_percent": round(float((factor_x - 1) * 100), 4),
        "length_change_percent": round(float((factor_y - 1) * 100), 4),
        "area_change_percent": round(float((np.linalg.det(linear) - 1) * 100), 4),
        "orthogonality_change_degrees": round(angle - 90, 4),
        "principal_scale_factors": np.round(singular, 6).tolist(),
    }


def fit_and_describe(readings, limit_):
    candidates = [item for item in readings if item["usable"]]
    assumed = len(candidates) == 3
    if len(candidates) < 3:
        return {"status": "needs more visible pairs", "selected": len(candidates), "fit": None}
    linear, offset, errors = transform_estimate(candidates)
    for point_, value in zip(candidates, errors, strict=True):
        point_["residual_mm"] = round(float(value), 5)
        point_["matches_fit"] = bool(value <= limit_)
    reliable = bool(np.max(errors) <= limit_)
    state = "measured" if reliable else "inconsistent visible landmarks"
    return {
        "status": state, "selected": len(candidates),
        "evidence_model": "three noncollinear points fit without redundancy" if assumed else "overdetermined affine inspection",
        "fit": scale_reading(linear) if reliable else None,
        "rms_deviation_mm": round(float(np.sqrt(np.mean(errors**2))), 5),
        "residual_percentiles_mm": quantiles(errors),
        "tolerance_mm": limit_,
        "transform": {"matrix": linear.T.tolist(), "offset": offset.tolist()} if reliable else None,
    }


def dressed(ctx, rows, phase):
    grid = rgba(ctx.surface)
    draw = ImageDraw.Draw(grid)
    for v, row in enumerate(rows, 1):
        x, y = pixel_points([row[phase]], grid.size)[0]
        shade = LILAC if not row["usable"] else PINK if row.get("matches_fit") is False else TEAL
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(*shade, 255), width=2)
        label(draw, (x + 5, y + 5), f"{v} · {row['name']}", size=11, fill=shade)
    return grid


def measure(lib, capture, options):
    if capture == options.base:
        raise WorkbenchError("Choose two separate photographs of the cloth.", "reference")
    current = view(lib, capture, plane_key=options.latest_plane, registered=True)
    previous = view(lib, options.base, plane_key=options.base_plane, registered=True)
    if previous.fabric["id"] != current.fabric["id"] or previous.capture["face"] != current.capture["face"]:
        raise WorkbenchError("Compare the same fabric and side.", "reference")
    inspected = dimensional_pairs(options, previous, current)
    aggregate = fit_and_describe(inspected, options.limit)
    before = dressed(previous, inspected, "before")
    after = dressed(current, inspected, "after")
    comparison = panel_pair(before, after, previous.capture["title"], current.capture["title"])
    report = {
        "before": previous.capture["id"], "after": current.capture["id"],
        "before_plane": previous.plane["id"], "after_plane": current.plane["id"],
        "landmarks": inspected, "measurement": aggregate,
        "note": "Visible planar dimensional changes. A fold or the wrong landmark can prevent a reliable fit.",
    }
    return previous, current, report, legend(comparison, "Relaxation and dimensional inspection", [
        ("Readable landmark", TEAL), ("Missing in one view", LILAC), ("Outlying residual", PINK),
    ])


def audit(store, capture, raw, *, save):
    options = RelaxationRequest.parse(raw)
    with store.open() as lib:
        previous, current, report, panel = measure(lib, capture, options)
    if not save:
        return report, panel
    with store.open(write=True) as lib:
        parent = current.fabric["id"]
        if lib.count("readings", parent=parent) >= 120:
            raise limit("This fabric has reached its retained inspection limit.")
        refs = [(previous.capture, "earlier image"), (previous.plane, "earlier measured plane"),
                (current.capture, "current image"), (current.plane, "current measured plane")]
        for item, _ in refs:
            lib.require_version(lib.get(item["id"], item["kind"]), item["version"])
        item = lib.add("readings", {
            "title": text(raw.get("title"), "relaxation inspection name", 95, default="Cloth dimension change"),
            "tool": "relaxation", "analysis": report, "proofs": ["pair"], "review_state": "unreviewed",
            "parameters": {"capture": current.capture["id"], "plane": current.plane["id"],
                           "coordinate_basis": "fabric", "capture_role": current.capture["role"]},
        }, parent=parent, action="relaxation checkpoint inspection")
        for entry, relation in refs:
            lib.cite(item["id"], entry, relation)
        lib.attach(item["id"], "pair", png(panel), "image/png")
        return lib.public(item, links=True)

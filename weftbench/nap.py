"""Nap (pile lay) direction from two raking-light photographs taken from opposite sides."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import ImageDraw

from .comparison import two_views
from .contracts import WorkbenchError, choice, identifier, integer, mapping, real, require_support
from .geometry import box_mask, cut_rectangle
from .inspection import common_box
from .raster import blurred_masked, gradients, luma, quantiles, rgba
from .visual import LILAC, OCHRE, PINK, TEAL, label, legend, panel_pair

# Compass direction -> unit vector (x right, y down) in the registered fabric view.
COMPASS = {
    "right": (1.0, 0.0), "left": (-1.0, 0.0),
    "down": (0.0, 1.0), "up": (0.0, -1.0),
    "down-right": (math.sqrt(2) / 2, math.sqrt(2) / 2),
    "up-left": (-math.sqrt(2) / 2, -math.sqrt(2) / 2),
    "down-left": (-math.sqrt(2) / 2, math.sqrt(2) / 2),
    "up-right": (math.sqrt(2) / 2, -math.sqrt(2) / 2),
}
OPPOSITE = {
    "right": "left", "left": "right", "up": "down", "down": "up",
    "down-right": "up-left", "up-left": "down-right",
    "down-left": "up-right", "up-right": "down-left",
}
STATUS_DIRECTED = "nap direction read"
STATUS_REVIEW = "axis visible, direction to check"
STATUS_UNSUPPORTED = "folded or clear on one photo only"
STATUS_NO_AXIS = "no nap axis established"
DIRECT_COLOR = TEAL
REVIEW_COLOR = OCHRE
SUPPORT_COLOR = LILAC
NO_AXIS_COLOR = PINK


@dataclass(frozen=True)
class NapRequest:
    reference: str
    reference_plane: str | None
    selection: list
    current_light: str
    reference_light: str
    cells: int
    smooth: float
    weave_radius: float
    polarity: float
    coherence: float
    weave_ratio: float

    @classmethod
    def parse(cls, raw):
        form = mapping(raw)
        current_light = choice(form.get("light"), "light direction of this photo", COMPASS, "right")
        reference_light = choice(
            form.get("reference_light"), "light direction of the other photo", COMPASS,
            OPPOSITE[current_light],
        )
        if OPPOSITE[current_light] != reference_light:
            raise WorkbenchError(
                "Choose opposite raking-light directions for the two photographs.",
                "reference_light",
            )
        smooth = real(form.get("smooth"), "nap smoothing radius", 1.5, 60, default=11.0)
        weave_radius = real(form.get("weave_radius"), "weave texture radius", 0.4, 12, default=1.6)
        if weave_radius * 1.5 >= smooth:
            raise WorkbenchError(
                "Keep the nap smoothing radius clearly larger than the weave texture radius.",
                "smooth",
            )
        return cls(
            identifier(form.get("reference"), "reference raking-light photograph"),
            identifier(form.get("reference_plane"), "reference calibration", optional=True),
            common_box(form),
            current_light,
            reference_light,
            integer(form.get("cells"), "nap grid cell size", 24, 320, default=78),
            smooth,
            weave_radius,
            real(form.get("polarity"), "minimum nap polarity", 0.004, 0.6, default=0.028),
            real(form.get("coherence"), "minimum direction coherence", 0.05, 0.95, default=0.22),
            real(form.get("weave_ratio"), "weave-to-nap signal limit", 0.2, 6.0, default=1.35),
        )


def require_rake(ctx, other):
    for capture in (ctx.capture, other.capture):
        if capture["role"] != "rake":
            raise WorkbenchError(
                "Nap direction needs two raking-light (side light) photographs.", "reference",
            )


def normalized_luma(surface, support):
    """Match one photo's tonal spread to the other so only directional shading remains."""
    values = luma(surface.linear)
    accepted = support & surface.readable
    require_support(int(support.sum()), int(accepted.sum()), "Raking-light tonal match", 120, fraction=0.3)
    lo, hi = np.quantile(values[accepted], (0.05, 0.95))
    if hi - lo < 0.02:
        raise WorkbenchError("The raking-light photo has too little tonal range to read nap lay.")
    return np.clip((values - lo) / (hi - lo), 0.0, 1.2)


def nap_difference(current, other, other_readable, req):
    """Broad-scale signed luma difference; common weave shading cancels between opposite lights."""
    extent = current.surface.extent
    shared = current.readable & other_readable
    region = box_mask(extent, req.selection)
    light_a = normalized_luma(current.surface, shared)
    light_b = normalized_luma(other, shared)
    broad_a, support_a = blurred_masked(light_a, shared, req.smooth)
    broad_b, support_b = blurred_masked(light_b, shared, req.smooth)
    broad_support = support_a & support_b
    differential = broad_a - broad_b
    # Fine, high-frequency response is dominated by the warp/weft texture; broad response is nap lay.
    fine_a, fine_ok = blurred_masked(light_a, shared, req.weave_radius)
    fine_b, _ = blurred_masked(light_b, shared, req.weave_radius)
    fine_detail = np.abs((fine_a - broad_a) - (fine_b - broad_b))
    fine_detail = np.where(fine_ok & broad_support, fine_detail, 0.0)
    gx, gy, gradient_ok = gradients(broad_a, broad_support)
    return {
        "region": region, "shared": shared, "support": broad_support & region,
        "current_support": current.visible & region, "other_support": other_readable & region,
        "differential": differential, "fine_detail": fine_detail,
        "grad_x": gx, "grad_y": gy, "gradient_ok": gradient_ok & region,
    }


def direction_coherence(gx, gy, mask):
    if np.count_nonzero(mask) < 30:
        return 0.0
    dx, dy = gx[mask], gy[mask]
    txx = float(np.mean(dx * dx))
    tyy = float(np.mean(dy * dy))
    txy = float(np.mean(dx * dy))
    power = txx + tyy
    if power < 1e-9:
        return 0.0
    return math.hypot(txx - tyy, 2 * txy) / power


def light_axis_degrees(light):
    dx, dy = COMPASS[light]
    return math.degrees(math.atan2(dy, dx)) % 360.0


def classify_grid(field, req):
    height, width = field["differential"].shape
    left, top, right, bottom = cut_rectangle(req.selection, (width, height))
    if min(right - left, bottom - top) < 48:
        raise WorkbenchError("Choose a larger region for the nap direction grid.", "box")
    differential = field["differential"]
    fine_detail = field["fine_detail"]
    step = req.cells
    rows = []
    index = 0
    for y0 in range(top, bottom, step):
        y1 = min(y0 + step, bottom)
        for x0 in range(left, right, step):
            x1 = min(x0 + step, right)
            window = np.s_[y0:y1, x0:x1]
            requested = int(field["region"][window].sum())
            if not requested:
                continue
            index += 1
            current_cover = field["current_support"][window]
            other_cover = field["other_support"][window]
            shared_cover = field["support"][window]
            current_fraction = float(current_cover.mean())
            other_fraction = float(other_cover.mean())
            shared_fraction = float(shared_cover.mean())
            row = {
                "index": index,
                "bounds": [x0, y0, x1, y1],
                "center": [
                    round(float((x0 + x1) * 0.5 / width), 6),
                    round(float((y0 + y1) * 0.5 / height), 6),
                ],
                "coverage_current": round(current_fraction, 5),
                "coverage_reference": round(other_fraction, 5),
                "coverage_shared": round(shared_fraction, 5),
            }
            # Folded areas or areas sharp on only one photograph never get a definite arrow.
            if current_fraction < 0.35 or other_fraction < 0.35 or shared_fraction < 0.30:
                rows.append(row | {
                    "status": STATUS_UNSUPPORTED, "nap_direction_degrees": None,
                    "polarity": None, "axis_coherence": None, "weave_to_nap": None,
                })
                continue
            values = differential[window][shared_cover]
            polarity = float(np.median(values))
            nap_energy = abs(polarity)
            weave_energy = float(np.median(fine_detail[window][shared_cover]))
            ratio = weave_energy / max(nap_energy, 1e-8)
            coherence = direction_coherence(
                field["grad_x"][window], field["grad_y"][window],
                field["gradient_ok"][window] & shared_cover,
            )
            row["polarity"] = round(polarity, 6)
            row["axis_coherence"] = round(coherence, 5)
            row["weave_to_nap"] = round(ratio, 5)
            if nap_energy < req.polarity or ratio > req.weave_ratio:
                rows.append(row | {
                    "status": STATUS_REVIEW if coherence >= req.coherence else STATUS_NO_AXIS,
                    "nap_direction_degrees": None,
                })
                continue
            if coherence < req.coherence:
                # Fibre axis is not established well enough even when tonal polarity is present.
                rows.append(row | {"status": STATUS_REVIEW, "nap_direction_degrees": None})
                continue
            axis = light_axis_degrees(req.current_light)
            nap_degrees = (axis + (180.0 if polarity < 0 else 0.0)) % 360.0
            rows.append(row | {
                "status": STATUS_DIRECTED,
                "nap_direction_degrees": round(nap_degrees, 4),
            })
    return rows, (left, top, right, bottom)


def draw_arrow(pen, cx, cy, angle_degrees, reach, color):
    angle = math.radians(angle_degrees)
    tip = (cx + math.cos(angle) * reach, cy + math.sin(angle) * reach)
    tail = (cx - math.cos(angle) * reach, cy - math.sin(angle) * reach)
    pen.line((tail, tip), fill=(*color, 255), width=3)
    head = reach * 0.32
    for sweep in (math.radians(150), math.radians(-150)):
        back = (
            tip[0] + math.cos(angle + sweep) * head,
            tip[1] + math.sin(angle + sweep) * head,
        )
        pen.line((tip, back), fill=(*color, 255), width=3)


def draw_axis(pen, cx, cy, angle_degrees, reach, color):
    angle = math.radians(angle_degrees)
    dx, dy = math.cos(angle) * reach, math.sin(angle) * reach
    pen.line(((cx - dx, cy - dy), (cx + dx, cy + dy)), fill=(*color, 255), width=2)


def nap_map(ctx, cells, req):
    art = rgba(ctx.surface)
    pen = ImageDraw.Draw(art)
    status_color = {
        STATUS_DIRECTED: DIRECT_COLOR, STATUS_REVIEW: REVIEW_COLOR,
        STATUS_UNSUPPORTED: SUPPORT_COLOR, STATUS_NO_AXIS: NO_AXIS_COLOR,
    }
    axis = light_axis_degrees(req.current_light)
    for cell in cells:
        x0, y0, x1, y1 = cell["bounds"]
        color = status_color[cell["status"]]
        pen.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(*color, 255), width=1)
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        reach = max(7.0, min(x1 - x0, y1 - y0) * 0.32)
        if cell["status"] == STATUS_DIRECTED:
            draw_arrow(pen, cx, cy, cell["nap_direction_degrees"], reach, color)
        elif cell["status"] == STATUS_REVIEW:
            draw_axis(pen, cx, cy, axis, reach, color)
        label(pen, (x0 + 2, y0 + 1), str(cell["index"]), size=9, fill=color, maximum=46)
    return legend(art, "Nap lay from opposite raking lights", [
        ("Nap direction read", DIRECT_COLOR),
        ("Axis visible, direction to check", REVIEW_COLOR),
        ("No nap axis established", NO_AXIS_COLOR),
        ("Folded or clear on one photo only", SUPPORT_COLOR),
    ])


def nap_direction(ctx, raw, lib):
    form = mapping(raw)
    req = NapRequest.parse(form)
    other, other_surface, other_readable = two_views(lib, ctx, req, role_must_agree=False)
    require_rake(ctx, other)
    field = nap_difference(ctx, other_surface, other_readable, req)
    require_support(
        int(field["region"].sum()), int(field["support"].sum()),
        "Nap direction", 160, fraction=0.25,
    )
    cells, _ = classify_grid(field, req)
    if not cells:
        raise WorkbenchError("The selected region does not cover any readable nap cells.", "box")
    counts = {
        status: sum(1 for cell in cells if cell["status"] == status)
        for status in (STATUS_DIRECTED, STATUS_REVIEW, STATUS_UNSUPPORTED, STATUS_NO_AXIS)
    }
    map_image = nap_map(ctx, cells, req)
    pair_image = panel_pair(
        rgba(other_surface), rgba(ctx.surface),
        f"{other.capture['title']} · light {req.reference_light}",
        f"{ctx.capture['title']} · light {req.current_light}",
    )
    report = {
        "reference": other.capture["id"], "reference_plane": other.plane["id"] if other.plane else None,
        "box": req.selection,
        "current_light_direction": req.current_light,
        "reference_light_direction": req.reference_light,
        "light_axis_degrees": round(light_axis_degrees(req.current_light), 4),
        "grid_cell_px": req.cells, "smoothing_px": req.smooth, "weave_radius_px": req.weave_radius,
        "cells": cells, "cell_counts": counts,
        "directed_fraction": round(counts[STATUS_DIRECTED] / len(cells), 6),
        "shared_support_percentiles": quantiles(
            field["support"][field["region"]].astype(np.float32),
        ),
        "note": (
            "Optical nap lay from two opposite raking lights. A cell brighter under this photo's light "
            "than under the opposite light has its pile tips pointing toward this photo's light, so the "
            "arrow follows the recorded light axis (or the reversed axis for the negative case). "
            "Warp/weft texture is suppressed as common shading and never reported as nap direction. "
            "Cells held for review, and cells folded or clear on one photo only, carry no definite arrow."
        ),
    }
    images = {"map": map_image, "pair": pair_image}
    return report, images, other.references()

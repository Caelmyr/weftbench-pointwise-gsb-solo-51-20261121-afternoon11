"""Fit rectangular cut pieces onto a calibrated cloth plane.

Every rectangle is measured in physical millimetres on the retained fabric plane.
The photographed cloth is the only usable material: transparent areas (cloth the
photo does not show) and fold or ignore observations are excluded, and each
circled defect keeps the operator's safety margin.  Pieces marked grain-locked
are never turned, and no piece is ever shrunk to make a layout work.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import csv
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .catalog import view
from .contracts import (
    MARK_TYPES, WorkbenchError, choice, csv_cell, flag, identifier, limit, mapping,
    real, sequence, text,
)
from .geometry import polygon_mask
from .raster import png
from .visual import (
    DARK, LILAC, OCHRE, PALE, TEAL, blended_overlay, font, label, legend,
)

PIECE_LIMIT = 6
PLAN_LIMIT = 40
MAX_CELLS = 260_000
CELL_STEPS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0,
              8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0)
BEAM = 2
POSITIONS = 3
DEFAULT_DEFECTS = ("stain", "puncture", "slub")
EXCLUSION_TYPES = {"fold", "ignore"}
SCENARIOS = {
    "required": "Mandatory pieces only",
    "optional": "Mandatory and optional pieces",
    "reserve": "Mandatory pieces with clean reserve",
}
ROLE_NAMES = {"required": "Mandatory", "optional": "Optional", "reserve": "Clean reserve"}
ROLE_COLORS = {"required": TEAL, "optional": (119, 179, 236), "reserve": OCHRE}
DEFECT_COLOR = (214, 96, 74)
MARGIN_COLOR = (239, 181, 168)


@dataclass(frozen=True)
class PieceSpec:
    name: str
    role: str
    width_mm: float
    height_mm: float
    locked: bool

    def orientations(self):
        yield self.width_mm, self.height_mm, False
        if not self.locked and abs(self.width_mm - self.height_mm) > 1e-9:
            yield self.height_mm, self.width_mm, True


def _piece(item, role, field_prefix):
    return PieceSpec(
        text(item.get("name"), f"{field_prefix} name", 54),
        role,
        real(item.get("width_mm"), f"{field_prefix} width", 1, 4800),
        real(item.get("height_mm"), f"{field_prefix} height", 1, 4800),
        flag(item.get("locked"), f"{field_prefix} grain lock", default=False),
    )


@dataclass(frozen=True)
class PlanRequest:
    title: str
    clearance_mm: float
    pieces: list
    reserve: PieceSpec | None
    scenario: str
    defect_types: tuple

    @classmethod
    def parse(cls, raw):
        form = mapping(raw, "cut plan")
        rows = sequence(form.get("pieces"), "cut pieces", 1, PIECE_LIMIT)
        pieces = []
        for index, row in enumerate(rows, start=1):
            item = mapping(row, f"cut piece {index}")
            role = choice(item.get("role"), "piece role",
                          {"required": "Mandatory", "optional": "Optional"}, "required")
            spec = _piece(item | {"name": item.get("name") or f"Piece {index}"},
                          role, "piece")
            pieces.append(spec)
        if not any(piece.role == "required" for piece in pieces):
            raise WorkbenchError("Mark at least one piece as mandatory.", "pieces")
        if len({piece.name.casefold() for piece in pieces}) != len(pieces):
            raise WorkbenchError("Give each piece a distinct name.", "pieces")
        reserve = None
        if form.get("reserve"):
            item = mapping(form["reserve"], "clean reserve")
            reserve = _piece(item | {"name": item.get("name") or "Clean reserve"},
                             "reserve", "reserve")
            if any(piece.name.casefold() == reserve.name.casefold() for piece in pieces):
                raise WorkbenchError("Give the clean reserve its own name.", "reserve")
        defects = form.get("defect_types")
        if defects is None:
            defects = list(DEFAULT_DEFECTS)
        if not isinstance(defects, list) or not defects:
            raise WorkbenchError("Choose at least one defect observation category.", "defect_types")
        allowed = {key for key in MARK_TYPES if key not in EXCLUSION_TYPES}
        defects = tuple(dict.fromkeys(
            choice(item, "defect observation category", allowed) for item in defects))
        return cls(
            text(form.get("title"), "cut plan name", 90, default="Cut layout"),
            real(form.get("clearance_mm"), "defect safety margin", 0, 300, default=10),
            pieces, reserve,
            choice(form.get("scenario"), "layout to keep", SCENARIOS, "optional"),
            defects,
        )


def grid_shape(width_mm, height_mm):
    for step in CELL_STEPS:
        cols = max(1, int(math.ceil(width_mm / step)))
        rows = max(1, int(math.ceil(height_mm / step)))
        if cols * rows <= MAX_CELLS:
            return cols, rows
    step = CELL_STEPS[-1]
    return (max(1, int(math.ceil(width_mm / step))),
            max(1, int(math.ceil(height_mm / step))))


def polygon_marks(marks, kinds, extent):
    filled = np.zeros((extent[1], extent[0]), dtype=bool)
    for mark in marks:
        if mark["mark_type"] in kinds and mark["shape"] == "polygon":
            filled |= polygon_mask(extent, mark["points"], mark.get("holes", []))
    return filled


def block_mask(ctx, clearance_mm, defect_types):
    """Pixels that cannot carry a cut piece plus the defect margin footprint."""
    extent = ctx.surface.extent
    sx = ctx.plane["width_mm"] / extent[0]
    sy = ctx.plane["height_mm"] / extent[1]
    defects = polygon_marks(ctx.marks, defect_types, extent)
    excluded = polygon_marks(ctx.marks, EXCLUSION_TYPES, extent)
    margin = np.zeros_like(defects)
    if clearance_mm > 0 and defects.any():
        distance = ndimage.distance_transform_edt(~defects, sampling=(sy, sx))
        margin = distance <= clearance_mm + 1e-9
    blocked = (~ctx.surface.valid) | excluded | defects | margin
    return blocked, defects, excluded, margin


def resample(blocked, cols, rows):
    """A grid cell is blocked when any covered pixel is blocked (conservative)."""
    height, width = blocked.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.int64)
    integral[1:, 1:] = blocked.cumsum(0, dtype=np.int64).cumsum(1, dtype=np.int64)
    xs = np.rint(np.linspace(0, width, cols + 1)).astype(int)
    ys = np.rint(np.linspace(0, height, rows + 1)).astype(int)
    totals = (integral[ys[1:][:, None], xs[1:][None, :]]
              - integral[ys[:-1][:, None], xs[1:][None, :]]
              - integral[ys[1:][:, None], xs[:-1][None, :]]
              + integral[ys[:-1][:, None], xs[:-1][None, :]])
    return totals > 0


def window_map(free, ch, cw):
    rows, cols = free.shape
    if ch > rows or cw > cols:
        return None
    integral = np.zeros((rows + 1, cols + 1), dtype=np.int32)
    integral[1:, 1:] = free.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    window = (integral[ch:, cw:] - integral[:-ch, cw:] - integral[ch:, :-cw]
              + integral[:-ch, :-cw])
    return window == ch * cw


def candidates(free, ch, cw, wanted=POSITIONS):
    """Best snug placements for a fixed window, scored by obstacle contact."""
    valid = window_map(free, ch, cw)
    if valid is None or not valid.any():
        return []
    rows, cols = free.shape
    span_rows, span_cols = rows - ch + 1, cols - cw + 1
    blocked = np.ones((rows + 2, cols + 2), dtype=np.int32)
    blocked[1:-1, 1:-1] = ~free
    integral = np.zeros((rows + 3, cols + 3), dtype=np.int32)
    integral[1:, 1:] = blocked.cumsum(0).cumsum(1)

    def box(top, left, height, width):
        return (integral[top + height:top + height + span_rows, left + width:left + width + span_cols]
                - integral[top:top + span_rows, left + width:left + width + span_cols]
                - integral[top + height:top + height + span_rows, left:left + span_cols]
                + integral[top:top + span_rows, left:left + span_cols])

    # Contact counts blocked cells in the one-cell ring around each window,
    # including the outer border of the whole photographed plane.
    ring = box(0, 0, ch + 2, cw + 2) - box(1, 1, ch, cw)
    score = ring.astype(np.float32) / float(2 * (ch + cw))
    score += 0.02 * (1 - np.arange(score.shape[0], dtype=np.float32)[:, None]
                     / max(1, score.shape[0] - 1))
    score += 0.01 * (1 - np.arange(score.shape[1], dtype=np.float32)[None, :]
                     / max(1, score.shape[1] - 1))
    score[~valid] = -1.0
    flat = score.ravel()
    span_cols = score.shape[1]
    count = int(min(wanted, valid.sum()))
    best = np.argpartition(flat, -count)[-count:]
    best = best[np.argsort(-flat[best])]
    return [(int(index // span_cols), int(index % span_cols), float(flat[index]))
            for index in best if flat[index] >= 0]


def cells_for(width_mm, height_mm, cell_x, cell_y):
    return (max(1, int(math.ceil(width_mm / cell_x - 1e-9))),
            max(1, int(math.ceil(height_mm / cell_y - 1e-9))))


def placement(piece, row, col, w_mm, h_mm, turned, cell_x, cell_y):
    left, top = col * cell_x, row * cell_y
    return {
        "name": piece.name, "role": piece.role, "locked": piece.locked, "turned": turned,
        "requested_mm": [round(piece.width_mm, 2), round(piece.height_mm, 2)],
        "left_mm": round(left, 3), "top_mm": round(top, 3),
        "width_mm": round(w_mm, 3), "height_mm": round(h_mm, 3),
        "right_mm": round(left + w_mm, 3), "bottom_mm": round(top + h_mm, 3),
        "area_mm2": round(w_mm * h_mm, 2),
    }


def missing_reason(piece, plane_mm):
    width_mm, height_mm = plane_mm
    fits_flat = piece.width_mm <= width_mm and piece.height_mm <= height_mm
    fits_turned = (not piece.locked and piece.height_mm <= width_mm
                   and piece.width_mm <= height_mm)
    if not fits_flat and not fits_turned:
        return "larger than the calibrated fabric plane"
    return "no clear area of this size in the searched arrangement"


def search(free, order, cell_x, cell_y, plane_mm):
    """Beam search over one piece order.

    States rank by mandatory pieces placed, then the reserve, then any extras,
    so an optional or reserve piece never costs a mandatory one.
    """
    rows, cols = free.shape
    states = [(free, [], [], 0.0, 0)]
    for piece in order:
        grown = []
        must = {"required": 2, "reserve": 1}.get(piece.role, 0)
        for grid, placed, missing, snug, kept in states:
            options = []
            for w_mm, h_mm, turned in piece.orientations():
                cw, ch = cells_for(w_mm, h_mm, cell_x, cell_y)
                if cw > cols or ch > rows:
                    continue
                for row, col, score in candidates(grid, ch, cw):
                    shape = grid.copy()
                    shape[row:row + ch, col:col + cw] = False
                    options.append((shape, placed + [placement(
                        piece, row, col, w_mm, h_mm, turned, cell_x, cell_y)],
                        missing, snug + 1.0 + score, kept + must))
            if options:
                grown.extend(options)
            else:
                grown.append((grid, placed, missing + [{
                    "name": piece.name, "role": piece.role,
                    "reason": missing_reason(piece, plane_mm),
                }], snug, kept))
        if not grown:
            return None
        grown.sort(key=lambda state: (state[4], len(state[1]), state[3]), reverse=True)
        states = grown[:BEAM]
    return max(states, key=lambda state: (state[4], len(state[1]), state[3]))


def orders(required, optional):
    big = sorted(required, key=lambda p: p.width_mm * p.height_mm, reverse=True)
    small = sorted(required, key=lambda p: p.width_mm * p.height_mm)
    long_side = sorted(required, key=lambda p: max(p.width_mm, p.height_mm), reverse=True)
    short_side = sorted(required, key=lambda p: min(p.width_mm, p.height_mm), reverse=True)
    extra_big = sorted(optional, key=lambda p: p.width_mm * p.height_mm, reverse=True)
    extra_small = list(reversed(extra_big))
    return [big + extra_big, big + extra_small, small + extra_big,
            long_side + extra_big, short_side + extra_big]


def best_layout(free, required, optional, cell_x, cell_y, plane_mm):
    best = None
    for order in orders(required, optional):
        candidate = search(free, order, cell_x, cell_y, plane_mm)
        if candidate is None:
            continue
        if best is None or (candidate[4], len(candidate[1]), candidate[3]) > (
                best[4], len(best[1]), best[3]):
            best = candidate
    return best


def scenario(label, best, usable, must_place):
    placed = [] if best is None else best[1]
    missing = [] if best is None else best[2]
    required_missing = [row["name"] for row in missing if row["role"] in must_place]
    used = round(float(sum(piece["area_mm2"] for piece in placed)), 2)
    return {
        "label": label,
        "status": "fits" if not required_missing else "does_not_fit",
        "placed": placed,
        "missing": missing,
        "shortfall": [row["name"] for row in missing],
        "required_missing": required_missing,
        "used_area_mm2": used,
        "usable_area_mm2": round(usable, 2),
        "used_fraction": round(used / usable, 5) if usable else None,
    }


def area_bound(pieces, usable):
    needed = sum(piece.width_mm * piece.height_mm for piece in pieces)
    return {"needed_area_mm2": round(needed, 2), "usable_area_mm2": round(usable, 2),
            "certainly_impossible": needed > usable}


def comparison_note(request, scenarios, extras, dropped, kept):
    if request.reserve is None:
        if extras:
            return ("No clean reserve was requested. The optional layout adds "
                    + ", ".join(extras) + ".")
        return ("No clean reserve was requested, and no optional piece fits beside "
                "the mandatory pieces.")
    parts = [
        "The optional layout keeps "
        + (", ".join(extras) if extras else "no optional pieces")
        + " and sets aside no clean reserve.",
        "The clean-reserve layout keeps "
        + (", ".join(kept) if kept else "no clean reserve")
        + " and gives up " + (", ".join(dropped) if dropped else "no optional piece") + ".",
    ]
    if not kept:
        parts.append("The requested reserve could not be placed beside the mandatory pieces.")
    missing = scenarios["optional"]["required_missing"]
    if missing:
        parts.append("Even without a reserve, the mandatory piece"
                     + ("s " if len(missing) > 1 else " ")
                     + ", ".join(missing) + " could not be placed.")
    return " ".join(parts)


def build(ctx, request):
    width_mm, height_mm = ctx.plane["width_mm"], ctx.plane["height_mm"]
    cols, rows = grid_shape(width_mm, height_mm)
    cell_x, cell_y = width_mm / cols, height_mm / rows
    blocked, defects, excluded, margin = block_mask(
        ctx, request.clearance_mm, request.defect_types)
    free = ~resample(blocked, cols, rows)
    usable = float(free.sum()) * cell_x * cell_y

    required = [piece for piece in request.pieces if piece.role == "required"]
    optional = [piece for piece in request.pieces if piece.role == "optional"]
    reserve = request.reserve
    plane_mm = (width_mm, height_mm)

    scenarios = {
        "required": scenario(SCENARIOS["required"],
                             best_layout(free, required, [], cell_x, cell_y, plane_mm),
                             usable, {"required"}),
        "optional": scenario(SCENARIOS["optional"],
                             best_layout(free, required, optional, cell_x, cell_y, plane_mm),
                             usable, {"required"}),
    }
    if reserve is not None:
        scenarios["reserve"] = scenario(
            SCENARIOS["reserve"],
            best_layout(free, required + [reserve], optional, cell_x, cell_y, plane_mm),
            usable, {"required", "reserve"})
        scenarios["reserve"]["reserve"] = {
            "name": reserve.name, "width_mm": reserve.width_mm,
            "height_mm": reserve.height_mm, "locked": reserve.locked,
        }
    else:
        scenarios["reserve"] = {
            "label": SCENARIOS["reserve"], "status": "not_requested",
            "placed": [], "missing": [], "shortfall": [], "required_missing": [],
            "used_area_mm2": 0.0, "usable_area_mm2": round(usable, 2),
            "used_fraction": None,
        }
    for name, block in scenarios.items():
        if block["status"] == "not_requested":
            continue
        needed = required + ([reserve] if name == "reserve" and reserve else [])
        block["bound"] = area_bound(needed, usable)

    extra_optional = [row["name"] for row in scenarios["optional"]["placed"]
                      if row["role"] == "optional"]
    reserve_optional = [row["name"] for row in scenarios["reserve"]["placed"]
                        if row["role"] == "optional"]
    dropped = [name for name in extra_optional if name not in reserve_optional]
    extra_mandatory = [row["name"] for row in scenarios["optional"]["placed"]
                       if row["role"] == "required"]
    trimmed_mandatory = [row["name"] for row in scenarios["optional"]["missing"]
                         if row["role"] == "required"]
    kept_reserve = [row["name"] for row in scenarios["reserve"]["placed"]
                    if row["role"] == "reserve"]
    comparison = {
        "optional_extras": extra_optional,
        "reserve_extras": reserve_optional,
        "dropped_for_reserve": dropped,
        "reserve_kept": kept_reserve,
        "optional_shortfall": scenarios["optional"]["shortfall"],
        "reserve_shortfall": scenarios["reserve"]["shortfall"],
        "mandatory_missing": scenarios["optional"]["required_missing"],
        "bound": area_bound(required + ([reserve] if reserve else []) + optional, usable),
        "note": comparison_note(request, scenarios, extra_optional, dropped, kept_reserve),
    }

    notes = [
        "Rectangles are measured in millimetres on the saved fabric plane; every piece keeps "
        "its exact requested size.",
        "Grain-locked pieces are only tried with their stated width across the fabric width.",
        "Unphotographed cloth, folds and observations marked ignore cannot carry a piece.",
    ]
    if request.clearance_mm:
        notes.append("Circled defects of type "
                     + ", ".join(request.defect_types) + f" keep a "
                     f"{request.clearance_mm:g} mm safety margin.")
    if max(cell_x, cell_y) > 1.0:
        notes.append(f"Placement grid is {cell_x:.2f} × {cell_y:.2f} mm; "
                     "reported positions round to it.")
    notes.append("Layouts come from a bounded search of candidate arrangements. A shortfall "
                 "means the search found no arrangement; the area bound only settles clear "
                 "overloads.")

    return {
        "title": request.title,
        "capture": ctx.capture["id"],
        "plane": ctx.plane["id"],
        "coordinate_basis": "fabric",
        "plane_mm": [round(width_mm, 3), round(height_mm, 3)],
        "plane_pixels": list(ctx.surface.extent),
        "grid_mm": [round(cell_x, 4), round(cell_y, 4)],
        "clearance_mm": request.clearance_mm,
        "defect_types": list(request.defect_types),
        "usable_area_mm2": round(usable, 2),
        "usable_fraction": round(usable / (width_mm * height_mm), 5),
        "excluded": {
            "defect_marks": int(sum(1 for mark in ctx.marks
                                    if mark["mark_type"] in request.defect_types
                                    and mark["shape"] == "polygon")),
            "fold_or_ignore_marks": int(sum(1 for mark in ctx.marks
                                            if mark["mark_type"] in EXCLUSION_TYPES)),
            "defect_margin_mm2": round(float(margin.sum()) * cell_x * cell_y, 2),
            "unphotographed_pixels": int((~ctx.surface.valid).sum()),
        },
        "pieces": [{
            "name": piece.name, "role": piece.role, "width_mm": piece.width_mm,
            "height_mm": piece.height_mm, "locked": piece.locked,
        } for piece in request.pieces],
        "reserve": None if reserve is None else {
            "name": reserve.name, "width_mm": reserve.width_mm,
            "height_mm": reserve.height_mm, "locked": reserve.locked,
        },
        "scenarios": scenarios,
        "comparison": comparison,
        "notes": notes,
    }


def cut_sheet(report, key):
    scenario_ = report["scenarios"][key]
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["piece", "role", "left_mm", "top_mm", "width_mm", "height_mm",
                     "right_mm", "bottom_mm", "grain_locked", "turned"])
    for row in scenario_["placed"]:
        writer.writerow([csv_cell(row["name"]), ROLE_NAMES[row["role"]], row["left_mm"],
                         row["top_mm"], row["width_mm"], row["height_mm"], row["right_mm"],
                         row["bottom_mm"], row["locked"], row["turned"]])
    for row in scenario_["missing"]:
        writer.writerow([csv_cell(row["name"]), ROLE_NAMES[row["role"]], "", "", "", "", "",
                         "", "", ""])
    return output.getvalue().encode("utf-8")


def arrow(draw, x0, y0, x1, y1, color, width=2):
    draw.line((x0, y0, x1, y1), fill=(*color, 255), width=width)
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 12:
        return
    angle = math.atan2(y1 - y0, x1 - x0)
    for side in (0.45, -0.45):
        draw.line((x1, y1, x1 - 9 * math.cos(angle - side), y1 - 9 * math.sin(angle - side)),
                  fill=(*color, 255), width=width)


def draw_layout(ctx, request, report, key, *, width=None, scale=1.0):
    """One layout drawn over the cloth, with its margin and a millimetre rule."""
    scenario_ = report["scenarios"][key]
    width_mm, height_mm = report["plane_mm"]
    blocked, defects, excluded, margin = block_mask(
        ctx, request.clearance_mm, request.defect_types)
    classes = np.zeros(blocked.shape, dtype=np.uint8)
    classes[~ctx.surface.valid] = 1
    classes[ctx.surface.valid & excluded] = 2
    classes[ctx.surface.valid & ~excluded & defects] = 3
    classes[ctx.surface.valid & ~excluded & ~defects & margin] = 4
    picture = blended_overlay(ctx.surface, classes, {
        1: DARK, 2: LILAC, 3: DEFECT_COLOR, 4: MARGIN_COLOR,
    }, opacity=0.6).convert("RGBA")
    if width is not None and width != picture.width:
        height = max(1, int(round(picture.height * width / picture.width)))
        picture = picture.resize((width, height), Image.Resampling.LANCZOS)
    mat = max(16, int(round(34 * scale)))
    board = Image.new("RGBA", (picture.width + mat, picture.height + mat), (*DARK, 255))
    board.paste(picture, (0, 0))
    pen = ImageDraw.Draw(board, "RGBA")
    scale_x = picture.width / width_mm
    scale_y = picture.height / height_mm
    font_size = max(9, int(round(12 * scale)))
    for piece in scenario_["placed"]:
        color = ROLE_COLORS[piece["role"]]
        x0, y0 = piece["left_mm"] * scale_x, piece["top_mm"] * scale_y
        x1, y1 = piece["right_mm"] * scale_x, piece["bottom_mm"] * scale_y
        pen.rectangle((x0, y0, x1, y1), fill=(*color, 64), outline=(*color, 255),
                      width=max(1, int(round(3 * scale))))
        if piece["locked"]:
            arrow(pen, (x0 + x1) / 2, y1 - 9, (x0 + x1) / 2, y0 + 9, color)
        text_ = f'{piece["name"]} · {piece["width_mm"]:g}×{piece["height_mm"]:g}'
        if piece["turned"]:
            text_ += " · turned"
        label(pen, (x0 + 4, y0 + 4), text_, size=font_size, fill=color,
              maximum=max(40, int(x1 - x0) - 8))
    for index, row in enumerate(scenario_["missing"]):
        label(pen, (6, 6 + (font_size + 12) * index), f'Not placed: {row["name"]}',
              size=font_size, fill=(255, 255, 255), background=(140, 52, 40),
              maximum=max(80, picture.width - 12))
    step = 100 if max(width_mm, height_mm) > 600 else 50
    for millimetre in range(0, int(width_mm) + 1, step):
        x = millimetre * scale_x
        pen.line((x, picture.height, x, picture.height + max(4, int(7 * scale))),
                 fill=(*PALE, 255), width=1)
        pen.text((min(x + 2, board.width - mat + 4), picture.height + max(5, int(9 * scale))),
                 str(millimetre), fill=(*PALE, 255), font=font(max(8, font_size - 2)))
    for millimetre in range(0, int(height_mm) + 1, step):
        y = millimetre * scale_y
        pen.line((picture.width, y, picture.width + max(4, int(7 * scale)), y),
                 fill=(*PALE, 255), width=1)
        pen.text((picture.width + max(6, int(10 * scale)),
                  min(max(y - 5, 0), picture.height - font_size)),
                 str(millimetre), fill=(*PALE, 255), font=font(max(8, font_size - 2)))
    if scale >= 0.9:
        pen.text((picture.width + max(6, int(10 * scale)),
                  min(picture.height + max(5, int(9 * scale)), board.height - font_size)),
                 "mm", fill=(*PALE, 255), font=font(max(8, font_size - 2)))
    return board.convert("RGB")


def legend_items(request):
    return [
        ("Mandatory piece", ROLE_COLORS["required"]),
        ("Optional piece", ROLE_COLORS["optional"]),
        ("Clean reserve", ROLE_COLORS["reserve"]),
        ("Circled defect", DEFECT_COLOR),
        (f'{request.clearance_mm:g} mm defect margin', MARGIN_COLOR),
        ("Fold, ignore or unphotographed", LILAC),
    ]


def map_image(ctx, request, report, key):
    width_mm, height_mm = report["plane_mm"]
    board = draw_layout(ctx, request, report, key)
    title = (f'{report["title"]} · {SCENARIOS[key]} · '
             f'plane {width_mm:g} × {height_mm:g} mm')
    return legend(board, title, legend_items(request))


def comparison_image(ctx, request, report):
    """All requested layouts side by side, so a choice can be made at a glance."""
    keys = [key for key in SCENARIOS
            if report["scenarios"][key]["status"] != "not_requested"]
    panel = min(420, max(260, 1260 // max(1, len(keys))))
    sheets = [draw_layout(ctx, request, report, key, width=panel, scale=panel / 620)
              for key in keys]
    gap, header = 18, 36
    height = max(sheet.height for sheet in sheets)
    board = Image.new("RGB", (len(sheets) * panel + (len(sheets) + 1) * gap,
                               height + header + 2 * gap), DARK)
    pen = ImageDraw.Draw(board)
    for index, (key, sheet) in enumerate(zip(keys, sheets)):
        left = gap + index * (panel + gap)
        pen.text((left + 2, gap), SCENARIOS[key], fill=PALE, font=font(16, True))
        status = report["scenarios"][key]["status"]
        note = ("All requested pieces placed" if status == "fits"
                else "Not all mandatory pieces fit")
        pen.text((left + 2, gap + 21), note, fill=TEAL if status == "fits" else (239, 160, 145),
                 font=font(12))
        board.paste(sheet, (left, header + gap))
    title = (f'{report["title"]} · layout comparison · plane '
             f'{report["plane_mm"][0]:g} × {report["plane_mm"][1]:g} mm')
    return legend(board, title, legend_items(request))


def analyze(store, capture, raw, *, want_marks=False):
    form = mapping(raw, "cut plan")
    request = PlanRequest.parse(form)
    plane = identifier(form.get("plane"), "calibration", optional=True)
    with store.open() as lib:
        ctx = view(lib, capture, plane_key=plane, registered=True)
        report = build(ctx, request)
        sources = []
        marks = []
        if want_marks:
            sources = ctx.references()
            kinds = set(request.defect_types) | EXCLUSION_TYPES
            marks = [lib.get(mark["id"], "marks") for mark in ctx.marks
                     if mark["mark_type"] in kinds]
    return ctx, request, report, sources, marks


def preview(store, capture, raw):
    _, _, report, _, _ = analyze(store, capture, raw)
    return report


def rendered(store, capture, raw, key):
    if key not in SCENARIOS:
        raise WorkbenchError("Choose a supported cut layout.", status=404)
    ctx, request, report, _, _ = analyze(store, capture, raw)
    if report["scenarios"][key]["status"] == "not_requested":
        raise WorkbenchError("Give a clean reserve size to plan that layout.", "reserve")
    return map_image(ctx, request, report, key)


def comparison(store, capture, raw):
    ctx, request, report, _, _ = analyze(store, capture, raw)
    return comparison_image(ctx, request, report)


def save_plan(store, capture, raw):
    ctx, request, report, sources, marks = analyze(store, capture, raw, want_marks=True)
    key = request.scenario
    if report["scenarios"][key]["status"] == "not_requested":
        raise WorkbenchError("Give a clean reserve size before keeping that layout.", "reserve")
    scenario_ = report["scenarios"][key]
    if scenario_["status"] != "fits":
        raise WorkbenchError(
            "This layout leaves mandatory pieces unplaced: "
            + ", ".join(scenario_["required_missing"])
            + ". Adjust the pieces or the cloth before keeping it.", "scenario")
    picture = png(map_image(ctx, request, report, key))
    sheet = cut_sheet(report, key)
    placed_rows = [{
        "name": row["name"], "role": row["role"],
        "left_mm": row["left_mm"], "top_mm": row["top_mm"],
        "width_mm": row["width_mm"], "height_mm": row["height_mm"],
        "locked": row["locked"], "turned": row["turned"],
    } for row in scenario_["placed"]]
    with store.open(write=True) as lib:
        if lib.count("cut_plans", parent=ctx.fabric["id"]) >= PLAN_LIMIT:
            raise limit("This fabric has reached its retained cut-plan limit.")
        for source, _ in sources:
            lib.require_version(lib.get(source["id"], source["kind"]), source["version"])
        record = lib.add("cut_plans", {
            "title": request.title,
            "capture": ctx.capture["id"],
            "plane": ctx.plane["id"],
            "coordinate_basis": "fabric",
            "kept_scenario": key,
            "kept_label": SCENARIOS[key],
            "clearance_mm": request.clearance_mm,
            "defect_types": list(request.defect_types),
            "plane_mm": report["plane_mm"],
            "pieces": placed_rows,
            "missing": scenario_["missing"],
            "analysis": report,
        }, parent=ctx.fabric["id"], action="cut layout retained")
        for source, role in sources:
            lib.cite(record["id"], source, role)
        for mark in marks:
            lib.cite(record["id"], mark, "defect and exclusion outline")
        lib.attach(record["id"], "map", picture, "image/png")
        lib.attach(record["id"], "csv", sheet, "text/csv")
        return lib.public(record, links=True)


def plan_index(store, parent):
    with store.open() as lib:
        lib.get(parent, "fabrics")
        return [lib.public(row, links=True)
                for row in lib.list("cut_plans", parent=parent, limit=PLAN_LIMIT)]

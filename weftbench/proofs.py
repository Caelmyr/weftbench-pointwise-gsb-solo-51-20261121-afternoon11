"""Printable image evidence, at paper-fit, a calibrated single sheet, or tiled sheets."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
import csv
import math

from PIL import Image, ImageDraw

from .catalog import view
from .contracts import WorkbenchError, box, choice, csv_cell, flag, identifier, identifiers, limit, mapping, real, text
from .geometry import cut_rectangle, pixel_points
from .inspection import physical_crop_note
from .raster import fit_image, opaque, png, rgba
from .visual import DARK, INK, MARK_COLORS, MUTED, OCHRE, PAPER, TEAL, annotated, ellipsize, font

PAPER_MM = {"a4": (210.0, 297.0), "a3": (297.0, 420.0), "letter": (215.9, 279.4)}
PAPER_DPI = 150
MM_PER_INCH = 25.4
OVERLAP_MM = 12.0
GUARD_MM = 3.0
MAX_TILE_PAGES = 10


@dataclass(frozen=True)
class Paper:
    name: str
    width_mm: float
    height_mm: float
    margin_mm: float

    @classmethod
    def parse(cls, form):
        kind = choice(form.get("paper"), "paper size", PAPER_MM, "a4")
        margin = real(form.get("margin"), "empty margin in mm", 8, 30, default=13)
        dims = PAPER_MM[kind]
        if flag(form.get("landscape"), "landscape paper"):
            dims = dims[::-1]
        return cls(kind, *dims, margin)

    def pixels(self, value):
        return int(round(value * PAPER_DPI / MM_PER_INCH))

    @property
    def extent(self):
        return self.pixels(self.width_mm), self.pixels(self.height_mm)

    @property
    def margin(self):
        return self.pixels(self.margin_mm)

    @property
    def header_start(self):
        return self.margin

    @property
    def content_top(self):
        return self.margin + self.pixels(21.0)

    @property
    def bottom_line(self):
        return self.extent[1] - self.margin - self.pixels(8.0)

    def summary(self):
        return {
            "paper": self.name, "paper_mm": [self.width_mm, self.height_mm],
            "margin_mm": self.margin_mm, "print_density_dpi": PAPER_DPI,
            "landscape": self.width_mm > self.height_mm,
        }


def page_heading(page, paper, title, secondary, index=1, total=1):
    draw = ImageDraw.Draw(page)
    margin = paper.margin
    top = paper.header_start
    draw.rectangle((margin, top, page.width - margin, top + paper.pixels(18)), fill=INK)
    caption = ellipsize(title, draw, font(25, True), page.width - 2 * margin - 90)
    draw.text((margin + 15, top + 12), caption, fill=PAPER, font=font(25, True))
    subtitle = ellipsize(secondary, draw, font(13), page.width - 2 * margin - 140)
    draw.text((margin + 15, top + paper.pixels(12.5)), subtitle, fill=(193, 224, 212), font=font(13))
    if total > 1:
        draw.text((page.width - margin - 80, top + paper.pixels(12.5)),
                  f"{index} / {total}", fill=PAPER, font=font(13))
    return draw


def chart_scale(draw, x, y, paper, scale_ratio):
    length_mm = 30
    physical_length = paper.pixels(length_mm * scale_ratio)
    draw.line((x, y, x + physical_length, y), fill=INK, width=2)
    for position in range(4):
        start = int(round(x + physical_length * position / 3))
        draw.line((start, y - 5, start, y + 6), fill=INK, width=2)
    draw.text((x + physical_length + 12, y - 8), f"{length_mm} mm of fabric", fill=MUTED, font=font(13))


def right_scale_bar(draw, paper, bottom, scale_ratio):
    """Place the calibrated scale bar in the right footer, clear of left-aligned notes."""
    label = "30 mm of fabric"
    label_w = int(draw.textlength(label, font=font(12)))
    bar_w = paper.pixels(30 * scale_ratio)
    x = paper.extent[0] - paper.margin - label_w - 10 - bar_w
    y = bottom + 30
    draw.line((x, y, x + bar_w, y), fill=INK, width=2)
    for position in range(4):
        start = int(round(x + bar_w * position / 3))
        draw.line((start, y - 5, start, y + 6), fill=INK, width=2)
    draw.text((x + bar_w + 10, y - 8), label, fill=MUTED, font=font(12))


def adjusted_crop(ctx, region, drawings, *, labels=True):
    bounds = cut_rectangle(region, ctx.surface.extent)
    picture = annotated(ctx.surface, ctx.marks, labels=labels) if drawings else rgba(ctx.surface)
    return opaque(picture.crop(bounds)), bounds


def normal_proof(store, key, raw, *, save=False):
    form = mapping(raw)
    paper = Paper.parse(form)
    name = text(form.get("title"), "proof title", 90, default="Calibrated fabric proof")
    selected = box(form.get("box"), default=[0, 0, 1, 1])
    margin_mm = form.get("print_scale")
    at_scale = None if margin_mm is None else real(margin_mm, "print scale", 0.25, 2.0)
    tiled = flag(form.get("tiled"), "tiled proof", default=False)
    layer = flag(form.get("marks"), "show observations", default=True)
    plane_key = identifier(form.get("plane"), "calibration", optional=True)
    with store.open() as lib:
        current = view(lib, key, plane_key=plane_key, registered=True if at_scale else None)
        crop, bounds = adjusted_crop(current, selected, layer, labels=not (at_scale and tiled))
    note = physical_crop_note(current, selected)
    page = Image.new("RGB", paper.extent, PAPER)
    renderer = page_heading(page, paper, name, current.capture["title"])
    top, bottom = paper.content_top, paper.bottom_line
    capacity = (page.width - 2 * paper.margin, bottom - top)
    placement = None
    if at_scale is None:
        picture = fit_image(crop, *capacity)
        origin = ((page.width - picture.width) // 2, top + (capacity[1] - picture.height) // 2)
        page.paste(picture, origin)
        renderer.rectangle((origin[0], origin[1], origin[0] + picture.width, origin[1] + picture.height),
                           outline=DARK, width=max(1, paper.pixels(0.2)))
        renderer.text((paper.margin, bottom + 10), "Fit to page · figure has no guaranteed printed scale",
                      fill=MUTED, font=font(13))
        right = f"Photo pixels {bounds[0]}, {bounds[1]} to {bounds[2]}, {bounds[3]}"
        renderer.text((page.width - paper.margin - int(renderer.textlength(right, font=font(12))), bottom + 12),
                      right, fill=MUTED, font=font(12))
        pages, plan = [page], None
        placement = [origin[0], origin[1], picture.width, picture.height]
    else:
        sizes = [paper.pixels(part * at_scale) for part in note["print_mm"]]
        fits = all(wanted <= fit for wanted, fit in zip(sizes, capacity))
        if not tiled:
            if not fits:
                raise WorkbenchError(
                    "This crop does not fit one sheet at the selected scale. Choose a smaller crop or "
                    "enable tiled sheets to keep the scale across pages.", "box")
            origin = (paper.margin + (capacity[0] - sizes[0]) // 2,
                      top + (capacity[1] - sizes[1]) // 2)
            pages, plan = [scaled_sheet(page, paper, crop, sizes, origin)], None
            placement = [origin[0], origin[1], sizes[0], sizes[1]]
        else:
            if current.scale is None:
                raise WorkbenchError("Save a physical calibration before printing tiled scale proofs.", "plane")
            pages, plan = tiled_sheets(paper, name, current, crop, bounds, sizes, capacity, layer)
            renderer = None
        footer_draw = ImageDraw.Draw(pages[0])
        if tiled:
            right_scale_bar(footer_draw, paper, bottom, at_scale)
        else:
            chart_scale(footer_draw, paper.margin, bottom + 19, paper, at_scale)
    details = {
        "title": name, "crop": note, "physical_scale": at_scale, "marks": layer,
        "layout": paper.summary(), "placement_pixels": placement,
    }
    if plan is not None:
        details["tiled"] = True
        details["tile_plan"] = plan
    if not save:
        return details, proof_preview(pages, plan is not None)
    pdf = to_pdf(pages)
    with store.open(write=True) as lib:
        if lib.count("proofs", parent=current.fabric["id"]) >= 55:
            raise limit("This fabric has reached its print-proof storage limit.")
        for source, _ in current.references():
            lib.require_version(lib.get(source["id"], source["kind"]), source["version"])
        entry = lib.add("proofs", details, parent=current.fabric["id"], action="fabric proof retained")
        for source, role in current.references():
            lib.cite(entry["id"], source, role)
        lib.attach(entry["id"], "pdf", pdf, "application/pdf")
        lib.attach(entry["id"], "page", png(fit_image(proof_preview(pages, plan is not None), 945, 1150)),
                   "image/png")
        return lib.public(entry, links=True)


def scaled_sheet(page, paper, crop, sizes, origin):
    picture = crop.resize(tuple(max(1, size) for size in sizes), Image.Resampling.LANCZOS)
    page.paste(picture, origin)
    draw = ImageDraw.Draw(page)
    draw.rectangle((origin[0], origin[1], origin[0] + picture.width, origin[1] + picture.height),
                   outline=DARK, width=max(1, paper.pixels(0.2)))
    return page


# ---------------------------------------------------------------- tiled proofs


@dataclass(frozen=True)
class AxisPlan:
    starts: tuple
    step: int
    capacity: int
    overlap: int
    seams: tuple


def merged_guards(intervals):
    ordered = sorted((max(0, lo), hi) for lo, hi in intervals if hi > lo)
    merged = []
    for lo, hi in ordered:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def plan_axis(span, capacity, overlap, guards):
    """Greedy sheet seams; nudge each seam away from mark guard intervals where possible.

    Seams are print positions where one sheet hands over to the next. With ``n``
    sheets there are ``n - 1`` seams. Consecutive seams stay at least
    ``step = capacity - overlap`` apart, so every join keeps the full overlap band
    and sheet windows stay ordered. A seam that cannot be moved off a mark is left
    in place; the mark is then reported as split without a full page.
    """
    if span <= capacity:
        return AxisPlan((0,), span, capacity, overlap, ())
    step = capacity - overlap
    sheet_count = int(math.ceil((span - overlap) / step))
    blocked = merged_guards(guards)
    upper = span - capacity
    seams = []
    previous = 0
    for index in range(sheet_count - 1):
        nominal = min((index + 1) * step, upper)
        nominal = max(nominal, previous + step)
        chosen = nominal
        if any(lo <= nominal < hi for lo, hi in blocked):
            left = [int(math.floor(lo)) - 1 for lo, hi in blocked]
            right = [int(math.ceil(hi)) for lo, hi in blocked]
            candidates = [value for value in left + right if previous + step <= value <= upper]
            if candidates:
                chosen = min(candidates, key=lambda value: abs(value - nominal))
        seams.append(int(chosen))
        previous = chosen
    # Sheet 0 starts at 0 and ends at seam 0. Every later sheet starts at the
    # preceding seam; its full window still reaches ``start + capacity``, which is
    # exactly what creates the overlap band. Pin the final sheet to the end if the
    # nominal grid would otherwise leave the crop tail uncovered.
    starts = [0] + list(seams)
    if starts[-1] + capacity < span:
        starts[-1] = max(0, upper)
    return AxisPlan(tuple(starts), step, capacity, overlap, tuple(seams))


def mark_print_boxes(marks, bounds, surface_extent, size, guard):
    """Print-pixel guard boxes for marks intersecting the selected crop."""
    x0, y0, x1, y1 = bounds
    crop_w, crop_h = max(1, x1 - x0), max(1, y1 - y0)
    print_w, print_h = size
    result = []
    for mark in marks:
        pixels = pixel_points(mark["points"], surface_extent)
        left = min(p[0] for p in pixels)
        top = min(p[1] for p in pixels)
        right = max(p[0] for p in pixels)
        bottom = max(p[1] for p in pixels)
        if right <= x0 or left >= x1 or bottom <= y0 or top >= y1:
            continue
        pl = (max(x0, left) - x0) / crop_w * print_w - guard
        pt = (max(y0, top) - y0) / crop_h * print_h - guard
        pr = (min(x1, right) - x0) / crop_w * print_w + guard
        pb = (min(y1, bottom) - y0) / crop_h * print_h + guard
        result.append({"mark": mark, "box": (max(0.0, pl), max(0.0, pt), min(float(print_w), pr),
                                             min(float(print_h), pb))})
    return result


def interval_for(mark_boxes, axis, end):
    return [(b["box"][axis], b["box"][end]) for b in mark_boxes]


def page_intersects(box_, col, row, xplan, yplan, print_w, print_h):
    x_start = xplan.starts[col]
    x_end = min(print_w, x_start + xplan.capacity)
    y_start = yplan.starts[row]
    y_end = min(print_h, y_start + yplan.capacity)
    lo_x, lo_y, hi_x, hi_y = box_
    return lo_x < x_end and hi_x > x_start and lo_y < y_end and hi_y > y_start


def source_rect(p_start, p_end, print_span, source_span):
    lo = max(0.0, p_start / print_span * source_span)
    hi = min(float(source_span), p_end / print_span * source_span)
    left = int(math.floor(lo + 1e-8))
    right = int(math.ceil(hi - 1e-8))
    return max(0, left), min(source_span, max(left + 1, right))


def dashed(draw, first, last, fill, width, dash=10, gap=7):
    x1, y1 = first
    x2, y2 = last
    length = math.hypot(x2 - x1, y2 - y1)
    if length <= 0:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    travelled = 0
    while travelled < length:
        seg_end = min(travelled + dash, length)
        draw.line((x1 + dx * travelled, y1 + dy * travelled, x1 + dx * seg_end, y1 + dy * seg_end),
                  fill=fill, width=width)
        travelled += dash + gap


def registration_cross(draw, x, y, code, color=OCHRE):
    draw.line((x - 9, y, x + 9, y), fill=color, width=2)
    draw.line((x, y - 9, x, y + 9), fill=color, width=2)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
    draw.text((x + 12, y - 9), code, fill=color, font=font(12, True))


def mark_chip(draw, x, y, text_value, color, paper, maximum=None):
    face = font(12, True)
    text_value = ellipsize(text_value, draw, face, maximum or 300)
    width = draw.textlength(text_value, font=face)
    x = max(paper.margin + 4, min(x, paper.extent[0] - paper.margin - int(width) - 14))
    draw.rounded_rectangle((x, y, x + width + 10, y + 20), radius=4, fill=INK)
    draw.ellipse((x + 4, y + 6, x + 12, y + 14), fill=color)
    draw.text((x + 16, y + 3), text_value, font=face, fill=PAPER)


def tiled_sheets(paper, title, ctx, crop, bounds, sizes, capacity, layer):
    print_w, print_h = sizes
    overlap = min(paper.pixels(OVERLAP_MM), max(1, capacity[0] - 24), max(1, capacity[1] - 24))
    guard = paper.pixels(GUARD_MM)
    boxes = mark_print_boxes(ctx.marks if layer else [], bounds, ctx.surface.extent, sizes, guard)
    xplan = plan_axis(print_w, capacity[0], overlap, interval_for(boxes, 0, 2))
    yplan = plan_axis(print_h, capacity[1], overlap, interval_for(boxes, 1, 3))
    columns, rows = len(xplan.starts), len(yplan.starts)
    if columns * rows > MAX_TILE_PAGES:
        raise limit(f"A tiled proof needs {columns * rows} sheets; keep it to {MAX_TILE_PAGES} pages or fewer. "
                    "Choose a larger paper or a smaller crop.")
    moved_x = [index for index, seam in enumerate(xplan.seams, 1)
               if abs(seam - index * xplan.step) > 1]
    moved_y = [index for index, seam in enumerate(yplan.seams, 1)
               if abs(seam - index * yplan.step) > 1]
    unprotected = unprotected_marks(boxes, xplan, yplan, print_w, print_h)
    pages, numbering, page_info = [], [], []
    page_no = 0
    for row in range(rows):
        for col in range(columns):
            page_no += 1
            page = Image.new("RGB", paper.extent, PAPER)
            page_heading(page, paper, title, ctx.capture["title"], page_no, columns * rows)
            numbering.append(page_no)
            info = render_tile(page, paper, ctx, crop, bounds, sizes, capacity, xplan, yplan, col, row,
                               page_no, columns, rows, boxes)
            page_info.append(info)
            pages.append(page)
    plan = {
        "columns": columns, "rows": rows, "print_pixels": sizes,
        "overlap_px": overlap, "overlap_mm": round(overlap * MM_PER_INCH / PAPER_DPI, 2),
        "axis": [{"starts": list(xplan.starts), "seams": list(xplan.seams), "step": xplan.step},
                 {"starts": list(yplan.starts), "seams": list(yplan.seams), "step": yplan.step}],
        "page_numbers": numbering, "pages": page_info,
        "moved_seams": [f"H{i + 1}" for i in moved_x] + [f"V{i + 1}" for i in moved_y],
        "marks_split_without_full_page": [entry["mark"]["title"] for entry in unprotected],
    }
    return pages, plan


def unprotected_marks(boxes, xplan, yplan, print_w, print_h):
    """Marks that no single sheet contains whole, even after seam nudging."""
    result = []
    for entry in boxes:
        contained = any(
            lo_x >= x_start and hi_x <= min(print_w, x_start + xplan.capacity)
            and lo_y >= y_start and hi_y <= min(print_h, y_start + yplan.capacity)
            for y_start in yplan.starts for x_start in xplan.starts
            for lo_x, lo_y, hi_x, hi_y in [entry["box"]]
        )
        if not contained:
            result.append(entry)
    return result


def render_tile(page, paper, ctx, crop, bounds, sizes, capacity, xplan, yplan, col, row,
                page_no, columns, rows, boxes):
    draw = ImageDraw.Draw(page)
    origin = (paper.margin, paper.content_top)
    print_w, print_h = sizes
    x_start, y_start = xplan.starts[col], yplan.starts[row]
    x_end = min(print_w, x_start + capacity[0])
    y_end = min(print_h, y_start + capacity[1])
    sx0, sx1 = source_rect(x_start, x_end, print_w, crop.width)
    sy0, sy1 = source_rect(y_start, y_end, print_h, crop.height)
    tile = crop.crop((sx0, sy0, sx1, sy1)).resize((x_end - x_start, y_end - y_start),
                                                  Image.Resampling.LANCZOS)
    page.paste(tile, origin)
    draw.rectangle((origin[0], origin[1], origin[0] + tile.width, origin[1] + tile.height),
                   outline=DARK, width=max(1, paper.pixels(0.2)))
    seam_codes = draw_join_guides(draw, paper, origin, tile.size, xplan, yplan, col, row,
                                  page_no, columns, rows)
    visible = [entry for entry in boxes
               if page_no in pages_for_mark(entry, xplan, yplan, print_w, print_h)]
    home_titles = []
    for entry in visible:
        mark = entry["mark"]
        color = MARK_COLORS[mark["mark_type"]]
        lo_x, lo_y, hi_x, hi_y = entry["box"]
        cx = origin[0] + int((max(lo_x, x_start) + min(hi_x, x_end)) / 2 - x_start)
        cy = origin[1] + int((max(lo_y, y_start) + min(hi_y, y_end)) / 2 - y_start)
        home = home_page_for(entry, xplan, yplan, print_w, print_h)
        if page_no == home:
            anchor_x = origin[0] + int(max(lo_x, x_start) - x_start) + 6
            anchor_y = origin[1] + int(max(lo_y, y_start) - y_start) + 4
            mark_chip(draw, anchor_x, anchor_y, mark["title"], color, paper)
            home_titles.append(mark["title"])
        else:
            note = mark["title"] + (f" · continues on p.{home}" if page_no < home
                                    else f" · from p.{home}")
            mark_chip(draw, cx, max(origin[1] + 4, cy - 22), note, color, paper)
    footnotes = tile_footer(draw, paper, ctx, x_start, y_start, x_end, y_end, visible)
    return {"sheet": page_no, "grid": [col + 1, row + 1], "seam_codes": seam_codes,
            "observations": [entry["mark"]["title"] for entry in visible],
            "home_observations": home_titles, "footnotes": footnotes}


def draw_join_guides(draw, paper, origin, tile_size, xplan, yplan, col, row,
                     page_no, columns, rows):
    ox, oy = origin
    tw, th = tile_size
    codes = []
    if col > 0:
        code = f"H{col}"
        x = ox + 2
        dashed(draw, (x, oy), (x, oy + th), OCHRE, 2)
        for fraction in (0.15, 0.5, 0.85):
            registration_cross(draw, x, oy + int(th * fraction), code)
        draw.text((x + 4, oy + 4), f"{code} overlap · align with p.{page_no - 1}", fill=OCHRE,
                  font=font(11, True))
        codes.append(code)
    if col < columns - 1:
        code = f"H{col + 1}"
        x = ox + tw - 2
        dashed(draw, (x, oy), (x, oy + th), OCHRE, 2)
        for fraction in (0.15, 0.5, 0.85):
            registration_cross(draw, x, oy + int(th * fraction), code)
        draw.text((max(ox + 4, x - 190), oy + 4), f"{code} overlap · continues on p.{page_no + 1}",
                  fill=OCHRE, font=font(11, True))
        codes.append(code)
    if row > 0:
        code = f"V{row}"
        y = oy + 2
        dashed(draw, (ox, y), (ox + tw, y), TEAL, 2)
        for fraction in (0.18, 0.5, 0.82):
            registration_cross(draw, ox + int(tw * fraction), y, code, TEAL)
        draw.text((ox + 4, y + 16), f"{code} overlap · align with p.{page_no - columns}", fill=TEAL,
                  font=font(11, True))
        codes.append(code)
    if row < rows - 1:
        code = f"V{row + 1}"
        y = oy + th - 2
        dashed(draw, (ox, y), (ox + tw, y), TEAL, 2)
        for fraction in (0.18, 0.5, 0.82):
            registration_cross(draw, ox + int(tw * fraction), y, code, TEAL)
        draw.text((ox + 4, y - 34), f"{code} overlap · continues on p.{page_no + columns}",
                  fill=TEAL, font=font(11, True))
        codes.append(code)
    return codes


def pages_for_mark(entry, xplan, yplan, print_w, print_h):
    pages = []
    for r, y_start in enumerate(yplan.starts):
        for c, x_start in enumerate(xplan.starts):
            if page_intersects(entry["box"], c, r, xplan, yplan, print_w, print_h):
                pages.append(r * len(xplan.starts) + c + 1)
    return pages


def home_page_for(entry, xplan, yplan, print_w, print_h):
    lo_x, lo_y, hi_x, hi_y = entry["box"]
    anchor_x = int((lo_x + hi_x) / 2)
    anchor_y = int((lo_y + hi_y) / 2)
    home_col = next((c for c, start in enumerate(xplan.starts)
                     if start <= anchor_x < min(print_w, start + xplan.capacity)), len(xplan.starts) - 1)
    home_row = next((r for r, start in enumerate(yplan.starts)
                     if start <= anchor_y < min(print_h, start + yplan.capacity)), len(yplan.starts) - 1)
    return home_row * len(xplan.starts) + home_col + 1


def tile_footer(draw, paper, ctx, x_start, y_start, x_end, y_end, visible):
    scale = ctx.scale
    notes = []
    # Keep left-aligned text inside roughly the left two thirds; the scale bar owns
    # the right footer zone.
    text_width = paper.extent[0] - 2 * paper.margin - 330
    if scale is not None:
        mm = [round(x_start * float(scale[0]), 1), round(y_start * float(scale[1]), 1),
              round(x_end * float(scale[0]), 1), round(y_end * float(scale[1]), 1)]
        coverage = f"Fabric {mm[0]}–{mm[2]} × {mm[1]}–{mm[3]} mm"
        draw.text((paper.margin, paper.bottom_line + 12),
                  ellipsize(coverage, draw, font(12), text_width), fill=MUTED, font=font(12))
        notes.append(coverage)
    if visible:
        line = ellipsize("Observations: " + ", ".join(entry["mark"]["title"] for entry in visible),
                         draw, font(11), text_width)
        draw.text((paper.margin, paper.bottom_line + 30), line, fill=MUTED, font=font(11))
        notes.append(line)
    return notes


def proof_preview(pages, tiled):
    if not tiled or len(pages) == 1:
        return pages[0]
    thumb_w = 430
    gap = 26
    thumbs = [fit_image(page, thumb_w, 560) for page in pages]
    columns = min(len(thumbs), 3)
    rows = int(math.ceil(len(thumbs) / columns))
    width = columns * thumb_w + (columns + 1) * gap
    row_heights = [max(thumbs[r * columns:(r + 1) * columns], key=lambda image: image.height).height
                   for r in range(rows)]
    height = sum(row_heights) + (rows + 1) * gap + rows * 22
    board = Image.new("RGB", (width, height), (232, 236, 226))
    draw = ImageDraw.Draw(board)
    for index, thumb in enumerate(thumbs):
        r, c = divmod(index, columns)
        x = gap + c * (thumb_w + gap)
        y = gap + sum(row_heights[:r]) + r * (gap + 22) + (22 if r else 0)
        draw.text((x, y - 18), f"Sheet {index + 1} of {len(thumbs)} · same order as the PDF",
                  fill=INK, font=font(14, True))
        board.paste(thumb, (x, y))
    return board


def to_pdf(pages):
    if not pages:
        raise WorkbenchError("No evidence page could be prepared.")
    if len(pages) > 10:
        raise limit("Keep this document at ten pages or fewer.")
    out = BytesIO()
    pages[0].save(out, "PDF", save_all=True, append_images=pages[1:], resolution=PAPER_DPI,
                  quality=86, subsampling=0)
    return out.getvalue()


def overview_cells(paper, columns, rows):
    spacing = paper.pixels(5)
    width, height = paper.extent
    fixed = (
        paper.margin, paper.content_top,
        width - paper.margin, paper.bottom_line,
    )
    usable_width = (fixed[2] - fixed[0] - (columns - 1) * spacing) // columns
    usable_height = (fixed[3] - fixed[1] - (rows - 1) * spacing) // rows
    for index in range(columns * rows):
        col, row = index % columns, index // columns
        left = fixed[0] + col * (usable_width + spacing)
        top = fixed[1] + row * (usable_height + spacing)
        yield left, top, usable_width, usable_height


def board_picture(db, item):
    choices = ("working",) if item["kind"] == "captures" else ("map", "pair", "chart", "card", "page")
    for role in choices:
        attached = db.asset(item["id"], role, optional=True)
        if attached:
            with Image.open(attached.path) as image:
                return opaque(image), attached
    raise WorkbenchError("One selected record does not have retained image evidence.", "entries")


def card_for(page, table, position, image, source):
    x, y, w, h = position
    table.rounded_rectangle((x, y, x + w - 1, y + h - 1), radius=11,
                            fill=(236, 232, 218), outline=(191, 199, 184), width=2)
    adapted = fit_image(image, w - 22, h - 84)
    page.paste(adapted, (x + (w - adapted.width) // 2, y + (h - 74 - adapted.height) // 2))
    title = ellipsize(source["title"], table, font(16, True), w - 29)
    table.text((x + 13, y + h - 65), title, fill=INK, font=font(16, True))
    tag = f'{source["kind"].capitalize()} · recorded version {source["version"]}'
    table.text((x + 13, y + h - 39), tag, fill=MUTED, font=font(13))


def contact_pages(entries, paper, title, layout):
    cols, rows = layout
    count = cols * rows
    groups = [entries[start:start + count] for start in range(0, len(entries), count)]
    result = []
    for offset, group in enumerate(groups, 1):
        canvas = Image.new("RGB", paper.extent, PAPER)
        text_ = page_heading(canvas, paper, title, "Retained fabric image evidence", offset, len(groups))
        for frame, cell in zip(group, overview_cells(paper, cols, rows)):
            photo, document, _ = frame
            card_for(canvas, text_, cell, photo, document)
        text_.text((paper.margin, paper.bottom_line + 14), "Local image workbench · visual inspection record",
                   fill=MUTED, font=font(13))
        result.append(canvas)
    return result


def evidence_rows(entries):
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["record type", "title", "id", "recorded version", "image role", "sha256"])
    for _, item, asset in entries:
        writer.writerow([
            item["kind"], csv_cell(item["title"]), item["id"], item["version"], asset.role, asset.sha256,
        ])
    return output.getvalue().encode("utf-8")


def gather_entries(lib, parent, wanted):
    allowed = {"captures", "readings", "swatches", "proofs", "details"}
    result = []
    for key in wanted:
        one = lib.get(key, parent=parent)
        if one["kind"] not in allowed:
            raise WorkbenchError("Select retained photos, readings, swatches, print proofs or detail sheets.", "entries")
        photo, origin = board_picture(lib, one)
        result.append((photo, one, origin))
    return result


def contact(store, parent, raw, *, save=False):
    form = mapping(raw)
    selected = identifiers(form.get("entries"), "selected evidence", 20)
    layouts = {"large": (2, 2), "grid": (2, 3), "strip": (1, 3)}
    matrix = layouts[choice(form.get("layout"), "contact layout", layouts, "large")]
    title = text(form.get("title"), "contact sheet name", 90, default="Fabric evidence overview")
    paper = Paper.parse(form)
    with store.open() as lib:
        lib.get(parent, "fabrics")
        entries = gather_entries(lib, parent, selected)
    pages = contact_pages(entries, paper, title, matrix)
    if not save:
        return pages[0]
    payload = to_pdf(pages)
    table = evidence_rows(entries)
    with store.open(write=True) as lib:
        if lib.count("proofs", parent=parent) >= 55:
            raise limit("This fabric has reached its print-proof storage limit.")
        for _, item, _ in entries:
            lib.require_version(lib.get(item["id"]), item["version"])
        record = lib.add("proofs", {
            "title": title, "layout": paper.summary(), "page_count": len(pages),
            "entries": selected, "physical_scale": None, "contact_sheet": True,
        }, parent=parent, action="contact sheet saved")
        for _, item, _ in entries:
            lib.cite(record["id"], item, "evidence panel")
        lib.attach(record["id"], "pdf", payload, "application/pdf")
        lib.attach(record["id"], "page", png(fit_image(pages[0], 945, 1150)), "image/png")
        lib.attach(record["id"], "csv", table, "text/csv")
        return lib.public(record, links=True)

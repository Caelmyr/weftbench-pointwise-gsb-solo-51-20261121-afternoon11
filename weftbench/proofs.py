"""Printable image evidence, at paper-fit or a calibrated single-sheet scale."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO, StringIO
import csv
import math

from PIL import Image, ImageDraw

from .catalog import view
from .contracts import WorkbenchError, box, choice, csv_cell, flag, identifier, identifiers, limit, mapping, real, text
from .geometry import cut_rectangle
from .inspection import physical_crop_note
from .raster import fit_image, opaque, png, rgba
from .visual import DARK, INK, MUTED, OCHRE, PAPER, TEAL, annotated, ellipsize, font

PAPER_MM = {"a4": (210.0, 297.0), "a3": (297.0, 420.0), "letter": (215.9, 279.4)}
PAPER_DPI = 150
MM_PER_INCH = 25.4


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


def adjusted_crop(ctx, region, drawings):
    bounds = cut_rectangle(region, ctx.surface.extent)
    picture = annotated(ctx.surface, ctx.marks) if drawings else rgba(ctx.surface)
    return opaque(picture.crop(bounds)), bounds


def normal_proof(store, key, raw, *, save=False):
    form = mapping(raw)
    paper = Paper.parse(form)
    name = text(form.get("title"), "proof title", 90, default="Calibrated fabric proof")
    selected = box(form.get("box"), default=[0, 0, 1, 1])
    margin_mm = form.get("print_scale")
    at_scale = None if margin_mm is None else real(margin_mm, "print scale", 0.25, 2.0)
    layer = flag(form.get("marks"), "show observations", default=True)
    plane_key = identifier(form.get("plane"), "calibration", optional=True)
    with store.open() as lib:
        current = view(lib, key, plane_key=plane_key, registered=True if at_scale else None)
        crop, bounds = adjusted_crop(current, selected, layer)
    note = physical_crop_note(current, selected)
    page = Image.new("RGB", paper.extent, PAPER)
    renderer = page_heading(page, paper, name, current.capture["title"])
    top, bottom = paper.content_top, paper.bottom_line
    capacity = (page.width - 2 * paper.margin, bottom - top)
    if at_scale is not None:
        sizes = [paper.pixels(part * at_scale) for part in note["print_mm"]]
        if any(wanted > fit for wanted, fit in zip(sizes, capacity, strict=True)):
            raise WorkbenchError("This crop does not fit one sheet at the selected scale. Choose a smaller crop.", "box")
        picture = crop.resize(tuple(max(1, size) for size in sizes), Image.Resampling.LANCZOS)
    else:
        picture = fit_image(crop, *capacity)
    origin = ((page.width - picture.width) // 2, top + (capacity[1] - picture.height) // 2)
    page.paste(picture, origin)
    renderer.rectangle((origin[0], origin[1], origin[0] + picture.width, origin[1] + picture.height),
                       outline=DARK, width=max(1, paper.pixels(0.2)))
    if at_scale is not None:
        chart_scale(renderer, paper.margin, bottom + 19, paper, at_scale)
    else:
        renderer.text((paper.margin, bottom + 10), "Fit to page · figure has no guaranteed printed scale",
                      fill=MUTED, font=font(13))
    right = f"Photo pixels {bounds[0]}, {bounds[1]} to {bounds[2]}, {bounds[3]}"
    renderer.text((page.width - paper.margin - int(renderer.textlength(right, font=font(12))), bottom + 12),
                  right, fill=MUTED, font=font(12))
    details = {
        "title": name, "crop": note, "physical_scale": at_scale, "marks": layer,
        "layout": paper.summary(), "placement_pixels": [*origin, picture.width, picture.height],
    }
    if not save:
        return details, page
    pdf = to_pdf([page])
    with store.open(write=True) as lib:
        if lib.count("proofs", parent=current.fabric["id"]) >= 55:
            raise limit("This fabric has reached its print-proof storage limit.")
        for source, _ in current.references():
            lib.require_version(lib.get(source["id"], source["kind"]), source["version"])
        entry = lib.add("proofs", details, parent=current.fabric["id"], action="fabric proof retained")
        for source, role in current.references():
            lib.cite(entry["id"], source, role)
        lib.attach(entry["id"], "pdf", pdf, "application/pdf")
        lib.attach(entry["id"], "page", png(fit_image(page, 945, 1150)), "image/png")
        return lib.public(entry, links=True)


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

"""Retain native-resolution cloth details with independent context and coverage evidence."""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import csv

import numpy as np
from PIL import Image, ImageDraw

from .catalog import view
from .contracts import (
    WorkbenchError, box, csv_cell, flag, identifier, limit, mapping, sequence, text,
)
from .geometry import cut_rectangle
from .raster import crop_raster, fit_image, png, rgba
from .visual import DARK, LILAC, OCHRE, PALE, TEAL, annotated, ellipsize, font, label, legend

MAX_DETAILS = 48
MAX_CROP_PIXELS = 2_000_000
MAX_TOTAL_PIXELS = 5_200_000


@dataclass(frozen=True)
class DetailRegion:
    title: str
    selection: list[float]

    @classmethod
    def parse(cls, raw, order):
        row = mapping(raw, "detail crop")
        return cls(text(row.get("title"), "detail label", 55, default=f"Detail {order}"),
                   box(row.get("box"), "detail crop"))


@dataclass(frozen=True)
class DetailRequest:
    title: str
    regions: list[DetailRegion]
    plane: str | None
    registered: bool
    with_marks: bool

    @classmethod
    def parse(cls, raw):
        form = mapping(raw, "native image detail")
        labels = [DetailRegion.parse(row, number) for number, row in enumerate(
            sequence(form.get("regions"), "detail crops", 1, 6), start=1)]
        if len({region.title.casefold() for region in labels}) != len(labels):
            raise WorkbenchError("Give each detail a distinct name.", "regions")
        calibrated = flag(form.get("registered"), "use calibrated image", default=True)
        plane = identifier(form.get("plane"), "calibration", optional=True)
        if plane and not calibrated:
            raise WorkbenchError("Switch to calibrated coordinates to use the selected plane.", "plane")
        return cls(
            text(form.get("title"), "detail sheet name", 90, default="Fabric details"),
            labels, plane, calibrated, flag(form.get("marks"), "show observation outlines", default=True),
        )


@dataclass(frozen=True)
class ImageCrop:
    receipt: dict
    picture: Image.Image
    coverage: Image.Image


def crop_bounds(options, extent):
    selected = [cut_rectangle(row.selection, extent) for row in options.regions]
    shapes = [(right - left, bottom - top) for left, top, right, bottom in selected]
    if any(min(size) < 5 for size in shapes):
        raise WorkbenchError("A detail needs at least five source pixels on both sides.", "regions")
    sizes = [width * height for width, height in shapes]
    if any(size > MAX_CROP_PIXELS for size in sizes) or sum(sizes) > MAX_TOTAL_PIXELS:
        raise limit("Keep each detail below two million pixels and the group below 5.2 million pixels.")
    return selected


def physical_crop(bounds, scale):
    if scale is None:
        return None
    left, top, right, bottom = bounds
    return {
        "origin_mm": [round(float(left * scale[0]), 5), round(float(top * scale[1]), 5)],
        "extent_mm": [round(float((right - left) * scale[0]), 5),
                      round(float((bottom - top) * scale[1]), 5)],
        "millimetres_per_pixel": [round(float(part), 8) for part in scale],
    }


def coverage_image(sensor, visible, readable):
    height, width = sensor.shape
    diagnostic = np.empty((height, width, 3), dtype=np.uint8)
    diagnostic[:] = DARK
    diagnostic[sensor] = LILAC
    diagnostic[visible] = OCHRE
    diagnostic[readable] = TEAL
    return Image.fromarray(diagnostic)


def get_crop(context, input_, bounds, number):
    left, top, right, bottom = bounds
    footprint = np.s_[top:bottom, left:right]
    source = crop_raster(context.surface, bounds)
    visible = context.visible[footprint]
    readable = context.readable[footprint]
    if not source.valid.any():
        raise WorkbenchError(f"{input_.title} does not cover any source image.", "regions")
    pixels = int(source.valid.size)
    fractions = {
        "source": round(float(source.valid.mean()), 6),
        "visible": round(float(visible.mean()), 6),
        "readable": round(float(readable.mean()), 6),
    }
    receipt = {
        "title": input_.title,
        "normalized_box": input_.selection,
        "views": {"detail": f"crop-{number:02d}", "coverage": f"support-{number:02d}"},
        "source_bounds_px": list(bounds),
        "size_px": list(source.extent),
        "physical": physical_crop(bounds, context.scale),
        "pixels": pixels,
        "coverage": fractions,
    }
    # The native crop keeps source opacity. Coverage shows the analysis exclusions separately.
    return ImageCrop(receipt, rgba(source), coverage_image(source.valid, visible, readable))


def source_overview(context, crops, show_marks):
    base = annotated(context.surface, context.marks) if show_marks else rgba(context.surface)
    scaled = fit_image(base, 1000, 735)
    pen = ImageDraw.Draw(scaled)
    native_w, native_h = base.size
    for number, crop in enumerate(crops, 1):
        x0, y0, x1, y1 = crop.receipt["source_bounds_px"]
        x0, x1 = (round(coord * scaled.width / native_w) for coord in (x0, x1))
        y0, y1 = (round(coord * scaled.height / native_h) for coord in (y0, y1))
        pen.rectangle((x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)), outline=(*OCHRE, 255), width=2)
        label(pen, (max(0, min(x0, scaled.width - 22)), max(0, min(y0, scaled.height - 22))),
              str(number), size=11, fill=OCHRE)
    return legend(scaled, "Source context · numbered crop locations", [("Retained detail", OCHRE)])


def overview_sheet(context, title, crops, show_marks):
    overview = source_overview(context, crops, show_marks)
    w = max(overview.width, 600)
    card_h = 152
    canvas = Image.new("RGB", (w, 45 + overview.height + len(crops) * card_h), DARK)
    canvas.paste(overview, ((w - overview.width) // 2, 45))
    pen = ImageDraw.Draw(canvas)
    heading = ellipsize(title, pen, font(22, True), w - 30)
    pen.text((14, 10), heading, fill=PALE, font=font(22, True))
    for index, crop in enumerate(crops):
        top = overview.height + 48 + index * card_h
        for slot, picture in enumerate((crop.picture, crop.coverage)):
            thumbnail = fit_image(picture, 140, 118)
            left = 14 + slot * 152
            canvas.paste(thumbnail, (left, top + 21), thumbnail if thumbnail.mode == "RGBA" else None)
            legend_text = "Native crop" if slot == 0 else "Analysis coverage"
            pen.text((left, top + 3), legend_text, font=font(11), fill=PALE)
        data = crop.receipt
        x = 325
        heading = ellipsize(f"{index + 1}. {data['title']}", pen, font(14, True), w - x - 14)
        pen.text((x, top + 22), heading, font=font(14, True), fill=PALE)
        width, height = data["size_px"]
        pen.text((x, top + 51), f"{width} × {height} original working pixels", font=font(11), fill=PALE)
        if data["physical"]:
            width, height = data["physical"]["extent_mm"]
            pen.text((x, top + 72), f"{width:.2f} × {height:.2f} mm", font=font(11), fill=PALE)
        coverage = data["coverage"]["readable"] * 100
        pen.text((x, top + 95), f"{coverage:.1f}% available for optical reading", font=font(11), fill=TEAL)
    return legend(canvas, "Fabric detail evidence", [
        ("Readable", TEAL), ("Dim or clipped", OCHRE), ("Observation exclusion", LILAC), ("Missing image", DARK),
    ])


def detail_csv(crops):
    output = StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([
        "detail", "crop", "coverage map", "left px", "top px", "right px", "bottom px",
        "width mm", "height mm", "image fraction", "visible fraction", "readable fraction",
    ])
    for crop in crops:
        row = crop.receipt
        physical = row["physical"]["extent_mm"] if row["physical"] else ["", ""]
        writer.writerow([
            csv_cell(row["title"]), row["views"]["detail"], row["views"]["coverage"], *row["source_bounds_px"],
            *physical, row["coverage"]["source"], row["coverage"]["visible"], row["coverage"]["readable"],
        ])
    return output.getvalue().encode("utf-8")


def collect(store, capture, raw, *, save=False):
    request = DetailRequest.parse(raw)
    with store.open() as lib:
        current = view(lib, capture, plane_key=request.plane, registered=None if request.registered else False)
        bounds = crop_bounds(request, current.surface.extent)
        crops = [get_crop(current, region, boundary, index) for index, (region, boundary) in enumerate(
            zip(request.regions, bounds, strict=True), 1)]
        sources = current.references()
        for mark in current.marks:
            if request.with_marks or mark["mark_type"] in {"ignore", "fold"}:
                sources.append((lib.get(mark["id"], "marks"), "observation in image view"))
    sheet = overview_sheet(current, request.title, crops, request.with_marks)
    report = {
        "title": request.title,
        "capture": current.capture["id"],
        "plane": current.plane["id"] if current.plane else None,
        "coordinate_basis": current.basis,
        "source_extent_px": list(current.surface.extent),
        "show_observations": request.with_marks,
        "regions": [crop.receipt for crop in crops],
    }
    if not save:
        return report, sheet
    attachments = [("map", png(sheet), "image/png"), ("csv", detail_csv(crops), "text/csv")]
    for crop in crops:
        attachments.extend([
            (crop.receipt["views"]["detail"], png(crop.picture), "image/png"),
            (crop.receipt["views"]["coverage"], png(crop.coverage), "image/png"),
        ])
    with store.open(write=True) as lib:
        parent = current.fabric["id"]
        if lib.count("details", parent=parent) >= MAX_DETAILS:
            raise limit("This fabric already has the maximum number of retained detail sheets.")
        for source, _ in sources:
            lib.require_version(lib.get(source["id"]), source["version"])
        record = lib.add("details", report, parent=parent, action="native image details retained")
        for source, role in sources:
            lib.cite(record["id"], source, role)
        for role, payload, mime in attachments:
            lib.attach(record["id"], role, payload, mime)
        return lib.public(record, links=True)


def crop_integrity(lib, record, region, checks):
    width, height = region["size_px"]
    details = []
    for name, role in region["views"].items():
        check = checks.get(role, "missing")
        retained = lib.asset(record["id"], role, optional=True) if check == "verified" else None
        dimensions = None
        if retained and retained.path.is_file():
            try:
                with Image.open(retained.path) as decoded:
                    dimensions = list(decoded.size)
            except (OSError, ValueError):
                dimensions = None
        details.append({
            "name": name,
            "role": role,
            "asset_check": check,
            "size_px": dimensions,
            "matches_saved_size": dimensions == [width, height],
        })
    return {"title": region["title"], "expected_size_px": [width, height], "views": details}


def audit_record(store, key):
    with store.open() as lib:
        record = lib.get(key, "details")
        listed = lib.integrity(key)
        checks = {row["role"]: row["check"] for row in listed}
        regions = [crop_integrity(lib, record, row, checks) for row in record["regions"]]
        sources = []
        for citation in lib.citations(key):
            latest = lib.get(citation["source"])
            original = lib.revision(latest["id"], citation["version"])
            sources.append({
                "id": latest["id"], "title": original["title"], "kind": latest["kind"],
                "saved_version": original["version"], "current_version": latest["version"],
                "revised_since_capture": latest["version"] != original["version"],
            })
    bad_views = [
        item["role"] for region in regions for item in region["views"]
        if item["asset_check"] != "verified" or not item["matches_saved_size"]
    ]
    failed_assets = [row["role"] for row in listed if row["check"] != "verified"]
    if "map" not in checks or "csv" not in checks:
        failed_assets.extend(role for role in ("map", "csv") if role not in checks)
    return {
        "detail_record": key,
        "coordinate_basis": record["coordinate_basis"],
        "verified": not (bad_views or failed_assets),
        "failed_views": bad_views,
        "failed_assets": failed_assets,
        "revised_source_count": sum(row["revised_since_capture"] for row in sources),
        "regions": regions,
        "retained_assets": listed,
        "sources": sources,
    }

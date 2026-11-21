"""Review a proposed source photo without writing an intake record or changing its pixels."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from . import catalog
from .contracts import ROLES, WorkbenchError, choice
from .raster import decode_image, fit_image, from_image, luma, quantiles
from .visual import LILAC, OCHRE, TEAL, histogram_plot, label

CONTACT_SIZE = 48
RECENT_LIMIT = 48
NEAR_LIMIT = 5


@dataclass(frozen=True)
class ThumbnailSignature:
    aspect: float
    values: np.ndarray
    support: np.ndarray

    @classmethod
    def from_photo(cls, picture):
        resized = fit_image(picture, 240, 200)
        width, height = resized.size
        photo = resized.convert("RGBA").resize((CONTACT_SIZE, CONTACT_SIZE), Image.Resampling.BOX)
        image = from_image(photo)
        return cls(width / height, image.linear, image.valid)


def compare_shapes(source, stored):
    if abs(math.log(source.aspect / stored.aspect)) > 0.035:
        return None
    all_pixels = source.support | stored.support
    shared = source.support & stored.support
    if int(shared.sum()) < 160:
        return None
    overlap = int(shared.sum()) / int(all_pixels.sum())
    if overlap < 0.88:
        return None
    first, second = source.values[shared], stored.values[shared]
    errors = np.mean(np.abs(first - second), axis=1)
    mean = float(errors.mean())
    p90 = float(np.percentile(errors, 90))
    if mean > 0.045 or p90 > 0.105:
        return None
    return {"mean_linear_difference": round(mean, 5), "larger_difference_p90": round(p90, 5),
            "support_overlap": round(overlap, 5)}


def visual_neighbors(lib, fabric, image, role, exact):
    current = ThumbnailSignature.from_photo(image)
    matches = []
    inspected = 0
    for other in lib.list("captures", parent=fabric, limit=RECENT_LIMIT):
        if other["id"] == exact or other["role"] != role:
            continue
        artifact = lib.asset(other["id"], "thumbnail", optional=True)
        if artifact is None:
            continue
        inspected += 1
        with Image.open(artifact.path) as bitmap:
            other_sig = ThumbnailSignature.from_photo(bitmap)
        estimate = compare_shapes(current, other_sig)
        if estimate:
            matches.append({"id": other["id"], "title": other["title"],
                            "captured_day": other["captured_day"], "match": estimate})
    matches.sort(key=lambda row: (row["match"]["mean_linear_difference"], row["id"]))
    return matches[:NEAR_LIMIT], inspected


def read_exposure(image):
    raster = from_image(image)
    pixels = int(raster.valid.sum())
    if not pixels:
        raise WorkbenchError("This source has no opaque photo pixels for intake review.", "image")
    value = luma(raster.linear)
    distribution = value[raster.valid]
    rows, edges = np.histogram(distribution, bins=32, range=(0, 1))
    chart = [[float((a + b) / 2), int(n)] for a, b, n in zip(edges[:-1], edges[1:], rows, strict=True)]
    return {
        "source_pixels": pixels,
        "unavailable_fraction": round(1 - float(raster.valid.mean()), 6),
        "deep_shadow_fraction": round(float(raster.dark.sum() / pixels), 6),
        "bright_clip_fraction": round(float(raster.bright.sum() / pixels), 6),
        "linear_luminance_percentiles": quantiles(distribution, q=(5, 50, 95)),
        "histogram": chart,
    }


def local_focus(image):
    # Measure the contrast of adjacent pixels only where the underlying sample is all visible.
    thumbnail = fit_image(image, 420, 420)
    raster = from_image(thumbnail)
    enough = ndimage.binary_erosion(raster.readable, iterations=1)
    values = luma(raster.linear)
    blurred = ndimage.gaussian_filter(values, sigma=1.05)
    residual = np.abs(values - blurred)
    width, height = thumbnail.size
    edges_x = np.linspace(0, width, 4, dtype=int)
    edges_y = np.linspace(0, height, 4, dtype=int)
    cells = []
    for y0, y1 in zip(edges_y, edges_y[1:]):
        for x0, x1 in zip(edges_x, edges_x[1:]):
            area = np.s_[y0:y1, x0:x1]
            mask = enough[area]
            samples = int(mask.sum())
            detail = float(np.median(residual[area][mask])) if samples >= 35 else None
            cells.append({
                "box_normalized": [x0 / width, y0 / height, x1 / width, y1 / height],
                "readable_fraction": round(float(mask.mean()), 5),
                "detail": round(detail, 7) if detail is not None else None,
            })
    return cells


def render_exposure(image, exposure, focus):
    chart = histogram_plot([{"points": exposure["histogram"], "color": TEAL}],
                           "Source luminance before intake", x_label="Linear luminance", y_label="Pixels",
                           limits=(0, max(amount for _, amount in exposure["histogram"])))
    preview = fit_image(image, chart.width, 400)
    board = Image.new("RGB", (chart.width,max(300, preview.height) + chart.height), (36, 47, 44))
    board.paste(preview, ((chart.width - preview.width) // 2,0), preview)
    draw = ImageDraw.Draw(board)
    margin = (chart.width - preview.width) // 2
    for number, cell in enumerate(focus, 1):
        x0,y0,x1,y1 = cell["box_normalized"]
        color = LILAC if cell["detail"] is None else OCHRE
        p0 = (round(margin + x0 * preview.width), round(y0 * preview.height))
        p1 = (round(margin + x1 * preview.width)-1, round(y1 * preview.height)-1)
        draw.rectangle((*p0,*p1),outline=color,width=1)
        if p1[0] - p0[0] >= 30:
            label(draw, (p0[0] + 3, p0[1] + 3), f"{number}", size=10, fill=color)
    board.paste(chart, (0,board.height - chart.height))
    return board


def review(store, fabric_key, raw, file):
    source, filename = catalog.upload_source(file)
    role = choice(raw.get("role"), "light role", ROLES, "face")
    image = decode_image(source)
    digest = sha256(source).hexdigest()
    with store.open() as lib:
        lib.get(fabric_key, "fabrics")
        exact_key = lib.includes_blob("captures", fabric_key, digest)
        exact = lib.get(exact_key, "captures") if exact_key else None
        neighbors, checked = visual_neighbors(lib, fabric_key, image.working, role, exact_key)
    exposure = read_exposure(image.working)
    focus = local_focus(image.working)
    result = {
        "filename": filename,
        "source_sha256": digest,
        "inspection": image.receipt(),
        "byte_count": len(source),
        "exact_source": {"id": exact["id"], "title": exact["title"]} if exact else None,
        "visual_matches": neighbors,
        "checked_recent_same_role": checked,
        "exposure": exposure,
        "detail_grid": focus,
        "scope": "Similar thumbnails may be separate captures. Only matching original bytes count as a duplicate.",
    }
    return result, (image.working, exposure, focus)

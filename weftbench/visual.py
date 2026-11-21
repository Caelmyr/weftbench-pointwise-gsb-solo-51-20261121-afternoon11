"""Deterministic image overlays, charts and printable inspection panels."""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import clip_segment, pixel_points
from .raster import fit_image, opaque, rgba

INK = (24, 34, 37)
PAPER = (247, 243, 233)
MUTED = (83, 101, 104)
TEAL = (67, 171, 157)
OCHRE = (232, 156, 82)
PINK = (235, 130, 114)
LILAC = (155, 161, 215)
PALE = (236, 236, 222)
DARK = (43, 53, 58)
MARK_COLORS = {
    "ignore": LILAC, "fold": (203, 158, 228), "stain": OCHRE, "puncture": PINK,
    "slub": (210, 113, 167), "seam": (88, 202, 201), "cut": (242, 199, 83),
    "selvedge": (119, 179, 236), "area": TEAL, "proof": (184, 234, 200),
}


@lru_cache(maxsize=30)
def font(size, strong=False):
    family = "DejaVuSans-Bold.ttf" if strong else "DejaVuSans.ttf"
    for path in (family, "/usr/share/fonts/truetype/dejavu/" + family):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=max(size, 8))


def ellipsize(value, drawer, face, width):
    if drawer.textlength(value, font=face) <= width:
        return value
    suffix = "…"
    length = max(0.0, width - drawer.textlength(suffix, font=face))
    low, high = 0, len(value)
    while low < high:
        mid = (low + high + 1) // 2
        if drawer.textlength(value[:mid], font=face) <= length:
            low = mid
        else:
            high = mid - 1
    return value[:low] + suffix


def label(drawer, xy, value, *, size=13, fill=PALE, background=INK, maximum=290):
    face = font(size, True)
    value = ellipsize(str(value), drawer, face, maximum)
    x, y = [int(v) for v in xy]
    width = drawer.textlength(value, font=face)
    drawer.rounded_rectangle((x, y, x + width + 10, y + size + 9), radius=4, fill=background)
    drawer.text((x + 5, y + 2), value, font=face, fill=fill)


def legend(canvas, title, labels):
    artwork = opaque(canvas)
    # Keep the image data at the original positions beneath a collapsible-width legend.
    width = max(artwork.width, 360)
    measure = ImageDraw.Draw(Image.new("RGB", (2, 2)))
    badge_font = font(11)
    line = []
    stack = []
    left = 15
    for name, shade in labels:
        step = int(measure.textlength(name, font=badge_font)) + 31
        if line and left + step > width - 10:
            stack.append(line)
            line = []
            left = 15
        line.append((left, name, shade))
        left += step
    if line:
        stack.append(line)
    top = 35 + 24 * len(stack)
    target = Image.new("RGB", (width, artwork.height + top), INK)
    target.paste(artwork, (0, top))
    draw = ImageDraw.Draw(target)
    title = ellipsize(title, draw, font(15, True), width - 31)
    draw.text((15, 9), title, font=font(15, True), fill=PALE)
    for i, items in enumerate(stack):
        y = 34 + i * 24
        for x, name, shade in items:
            draw.rounded_rectangle((x, y, x + 13, y + 13), fill=shade, radius=2)
            name = ellipsize(name, draw, badge_font, width - x - 27)
            draw.text((x + 19, y - 1), name, fill=PALE, font=badge_font)
    return target


def blended_overlay(surface, classes, palette, *, opacity=0.48):
    rgb = np.asarray(opaque(rgba(surface))).astype(np.float32)
    for key, color in palette.items():
        mask = classes == key
        rgb[mask] = rgb[mask] * (1 - opacity) + np.array(color) * opacity
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def heatmap(surface, values, visible, bounds, *, unknown=None, hue=TEAL):
    lo, hi = bounds
    if hi <= lo:
        hi = lo + 1e-5
    strength = np.clip((values - lo) / (hi - lo), 0, 1)
    original = np.asarray(opaque(rgba(surface)), dtype=np.float32)
    alpha = np.where(visible, 0.07 + 0.63 * strength, 0.0)
    painted = original * (1 - alpha[..., None]) + np.asarray(hue) * alpha[..., None]
    if unknown is not None and np.any(unknown):
        y, x = np.ogrid[:painted.shape[0], :painted.shape[1]]
        faded = np.where((x + y) % 11 < 3, 0.56, 0.22)
        mask = unknown & (faded > 0)
        painted[mask] = painted[mask] * (1 - faded[mask, None]) + np.asarray(LILAC) * faded[mask, None]
    return Image.fromarray(np.rint(painted).astype(np.uint8), "RGB")


def outline(draw, points, extents, shade, *, width=2, loop=False):
    pixel = pixel_points(points, extents)
    links = pixel[1:] + ([pixel[0]] if loop else [])
    bounds = (0, 0, extents[0] - 1, extents[1] - 1)
    for one, two in zip(pixel, links):
        clipped = clip_segment(one, two, bounds)
        if clipped is not None:
            draw.line(clipped, fill=shade, width=width)


def annotated(surface, marks, *, selected=None):
    board = opaque(rgba(surface))
    painter = ImageDraw.Draw(board)
    width, height = board.size
    for mark in marks:
        if mark.get("hidden"):
            continue
        active = selected is not None and selected == mark["id"]
        color = MARK_COLORS[mark["mark_type"]]
        thickness = 3 if active else max(1, width // 650)
        outline(painter, mark["points"], board.size, color, width=thickness, loop=mark["shape"] == "polygon")
        for hole in mark.get("holes", []):
            outline(painter, hole, board.size, color, width=thickness, loop=True)
        x, y = min(pixel_points(mark["points"], board.size), key=lambda p: (p[1], p[0]))
        x, y = max(0, min(width - 80, x + 4)), max(0, min(height - 21, y + 5))
        label(painter, (x, y), mark["title"], size=max(9, min(14, width // 50)), fill=color,
              maximum=max(50, min(width - int(x) - 12, 235)))
    return board


def panel_pair(first, second, left_title, right_title, *, max_side=530):
    left = fit_image(opaque(first), max_side, max_side)
    right = fit_image(opaque(second), max_side, max_side)
    gap = 20
    total_width = max(250, left.width) + max(250, right.width) + gap * 3
    board = Image.new("RGB", (total_width, max(left.height, right.height) + 67), INK)
    draw = ImageDraw.Draw(board)
    for x, picture, title in (
        (gap, left, left_title), (2 * gap + max(250, left.width), right, right_title),
    ):
        board.paste(picture, (x, 47))
        label(draw, (x, 14), title, maximum=max(244, picture.width - 10))
    return board


def histogram_plot(series, title, *, x_label="", y_label="", limits=None):
    width, height = 740, 333
    board = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(board)
    left, top, right, bottom = 65, 64, width - 24, height - 51
    draw.text((21, 19), title, fill=INK, font=font(20, True))
    available = [(x, y) for item in series for x, y in item["points"]
                 if y is not None and math.isfinite(y) and math.isfinite(x)]
    if not available:
        draw.text((left, top + 22), "No readable samples", fill=MUTED, font=font(15))
        return board
    xx, yy = zip(*available)
    x0, x1 = min(xx), max(xx)
    y0, y1 = limits if limits is not None else (min(yy), max(yy))
    x1 = max(x0 + 1e-6, x1)
    y1 = max(y0 + 1e-6, y1)
    def plot_point(x, y):
        return left + (right - left) * ((x - x0) / (x1 - x0)), bottom - (bottom - top) * ((y - y0) / (y1 - y0))
    for n in range(5):
        f = n / 4
        x = left + int(f * (right - left))
        y = bottom - int(f * (bottom - top))
        draw.line((left, y, right, y), fill=(209, 213, 205))
        draw.line((x, top, x, bottom), fill=(225, 228, 219))
        draw.text((left - 58, y - 7), f"{y0 + f * (y1 - y0):.3g}", fill=MUTED, font=font(11))
        draw.text((x - 7, bottom + 8), f"{x0 + f * (x1 - x0):.3g}", fill=MUTED, font=font(11))
    for item in series:
        previous = None
        for x, y in item["points"]:
            if y is None or not math.isfinite(y):
                previous = None
                continue
            point = plot_point(x, y)
            if previous is not None:
                draw.line([previous, point], fill=item["color"], width=2)
            previous = point
    draw.text((left, bottom + 28), x_label, fill=MUTED, font=font(11))
    draw.text((left, top - 17), y_label, fill=MUTED, font=font(11))
    return board


def line_dashboard(surface, selected, samples, *, box=None, name="Physical-width inspection"):
    image = annotated(surface, selected)
    chart = histogram_plot(samples, name, x_label="Distance along trace", y_label="Width")
    if box:
        draw = ImageDraw.Draw(image)
        draw.rectangle(box, outline=OCHRE, width=2)
    return panel_pair(image, chart, "Source and trace positions", name, max_side=610)


def neutral_card(swatches, title):
    width = max(375, min(1060, 130 * len(swatches)))
    columns = max(1, width // 132)
    rows = math.ceil(max(1, len(swatches)) / columns)
    card = Image.new("RGB", (width, 61 + 105 * rows), PAPER)
    paint = ImageDraw.Draw(card)
    paint.text((17, 15), title, fill=INK, font=font(19, True))
    for index, sample in enumerate(swatches):
        col, row = index % columns, index // columns
        x, y = col * 130 + 10, row * 105 + 55
        rgb = tuple(sample["color"]["srgb_bytes"]) if sample.get("color") else DARK
        paint.rounded_rectangle((x, y, x + 114, y + 55), radius=7, fill=rgb, outline=(177, 183, 171))
        title = ellipsize(sample["label"], paint, font(11), 114)
        paint.text((x, y + 62), title, fill=INK, font=font(11, True))
        if sample.get("lab_distance") is not None:
            paint.text((x, y + 78), f"ΔE76 {sample['lab_distance']:.2f}", fill=MUTED, font=font(10))
    return card

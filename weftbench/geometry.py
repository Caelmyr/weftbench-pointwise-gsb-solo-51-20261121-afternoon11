"""Inspection geometry shared by image calibration, annotations and measurements."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .contracts import MAX_EDGE, MAX_PIXELS, WorkbenchError, points, real
from .raster import Raster

EPS = 1e-10
UNIT_CORNERS = [[0, 0], [1, 0], [1, 1], [0, 1]]


def point_pixels(position, extent):
    width, height = extent
    return position[0] * width, position[1] * height


def pixel_at(position, extent):
    width, height = extent
    x, y = point_pixels(position, extent)
    return min(int(x), width - 1), min(int(y), height - 1)


def pixel_points(positions, extent):
    return [point_pixels(item, extent) for item in positions]


def area(points_):
    result = 0.0
    for before, after in zip(points_, points_[1:] + points_[:1]):
        result += before[0] * after[1] - before[1] * after[0]
    return result / 2


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def between(a, b, point):
    return (min(a[0], b[0]) - EPS <= point[0] <= max(a[0], b[0]) + EPS
            and min(a[1], b[1]) - EPS <= point[1] <= max(a[1], b[1]) + EPS
            and abs(cross(a, b, point)) < EPS)


def intersects(a, b, c, d):
    turn1, turn2 = cross(a, b, c), cross(a, b, d)
    turn3, turn4 = cross(c, d, a), cross(c, d, b)
    if turn1 * turn2 < -EPS and turn3 * turn4 < -EPS:
        return True
    return any((between(a, b, c), between(a, b, d), between(c, d, a), between(c, d, b)))


def inside(point, polygon):
    result = False
    x, y = point
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if between(a, b, point):
            return True
        if (a[1] > y) != (b[1] > y):
            cut = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < cut:
                result = not result
    return result


def polygon(raw, field="outline", *, maximum=80):
    vertices = points(raw, field, 3, maximum)
    if abs(area(vertices)) < 0.000005:
        raise WorkbenchError("Draw a wider outline or choose a smaller crop before marking it.", field)
    links = list(zip(vertices, vertices[1:] + vertices[:1]))
    for i, (a, b) in enumerate(links):
        for j in range(i + 2, len(links)):
            if i == 0 and j == len(links) - 1:
                continue
            if intersects(a, b, *links[j]):
                raise WorkbenchError("Keep the outline from crossing itself.", field)
    return vertices


def line_points(raw, field="trace"):
    values = points(raw, field, 2, 100)
    if any(math.dist(a, b) < 0.00015 for a, b in zip(values, values[1:])):
        raise WorkbenchError("Spread neighboring trace vertices farther apart.", field)
    return values


def cut_rectangle(selection, extent):
    x, y, width, height = selection
    w, h = extent
    # Coordinates denote continuous image edges, not inclusively numbered pixels.
    left, top = int(math.floor(x * w + 1e-8)), int(math.floor(y * h + 1e-8))
    right, bottom = int(math.ceil((x + width) * w - 1e-8)), int(math.ceil((y + height) * h - 1e-8))
    return max(0, left), max(0, top), min(w, right), min(h, bottom)


def box_mask(extent, selection):
    w, h = extent
    if selection is None:
        return np.ones((h, w), dtype=bool)
    left, top, right, bottom = cut_rectangle(selection, extent)
    result = np.zeros((h, w), dtype=bool)
    result[top:bottom, left:right] = True
    return result


def polygon_mask(extent, outline, holes=()):
    brush = Image.new("1", extent, 0)
    painter = ImageDraw.Draw(brush)
    painter.polygon(pixel_points(outline, extent), fill=1)
    for hole in holes:
        painter.polygon(pixel_points(hole, extent), fill=0)
    return np.array(brush, dtype=bool)


def masks_for(extent, marks, kinds):
    w, h = extent
    fill = np.zeros((h, w), dtype=bool)
    for mark in marks:
        if mark["mark_type"] in kinds and mark["shape"] == "polygon" and not mark.get("hidden"):
            fill |= polygon_mask(extent, mark["points"], mark.get("holes", []))
    return fill


def hole_polygons(outline, raw):
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > 12:
        raise WorkbenchError("Keep an outline to at most twelve internal holes.", "holes")
    result = []
    outer_edges = list(zip(outline, outline[1:] + outline[:1]))
    for item in raw:
        inner = polygon(item, "hole", maximum=40)
        links = list(zip(inner, inner[1:] + inner[:1]))
        if not all(inside(node, outline) for node in inner):
            raise WorkbenchError("Keep all hole vertices inside their outer outline.", "holes")
        if any(intersects(a, b, c, d) for a, b in links for c, d in outer_edges):
            raise WorkbenchError("A hole cannot touch or cross the outer outline.", "holes")
        for old in result:
            if inside(inner[0], old) or inside(old[0], inner):
                raise WorkbenchError("Avoid overlapping or nested hole outlines.", "holes")
            if any(intersects(a, b, c, d) for a, b in links for c, d in zip(old, old[1:] + old[:1])):
                raise WorkbenchError("Avoid crossing hole outlines.", "holes")
        result.append(inner)
    return result


def approved_quad(raw):
    corners = points(raw, "four calibration corners", 4, 4)
    turns = [cross(corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]) for i in range(4)]
    if min(turns) < 0.000025 or area(corners) < 0.003:
        raise WorkbenchError(
            "Mark a convex outline clockwise from the top left, with the fabric front facing you.",
            "corners",
        )
    return corners


def normalize_for_fit(points_):
    center = np.mean(points_, axis=0)
    diff = np.asarray(points_, dtype=np.float64) - center
    rms = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
    if rms <= EPS:
        raise WorkbenchError("The calibration points are too close together.", "corners")
    s = math.sqrt(2) / rms
    matrix = np.array([[s, 0, -s * center[0]], [0, s, -s * center[1]], [0, 0, 1]])
    return [list(x) for x in diff * s], matrix


def homography(before, after):
    source, left = normalize_for_fit(before)
    target, right = normalize_for_fit(after)
    rows = []
    for (x, y), (u, v) in zip(source, target, strict=True):
        rows.extend([
            [x, y, 1, 0, 0, 0, -x * u, -y * u, -u],
            [0, 0, 0, x, y, 1, -x * v, -y * v, -v],
        ])
    _, values, vectors = np.linalg.svd(rows)
    if values[-1] / values[0] < 0.001:
        raise WorkbenchError("That set of corners cannot support a stable fabric plane.", "corners")
    estimate = np.linalg.inv(right) @ vectors[-1].reshape(3, 3) @ left
    if abs(np.linalg.det(estimate)) < EPS:
        raise WorkbenchError("The calibration mapping collapsed.", "corners")
    return estimate / np.linalg.norm(estimate)


def transform(matrix, points_):
    xy = np.asarray(points_, dtype=np.float64)
    extended = np.column_stack([xy, np.ones(len(xy))])
    values = (np.asarray(matrix) @ extended.T).T
    if np.any(np.abs(values[:, 2]) < EPS):
        raise WorkbenchError("A requested point is too near the calibration horizon.")
    return values[:, :2] / values[:, [2]]


def plane_shape(width_mm, height_mm, pitch):
    width = max(24, int(round(width_mm / pitch)))
    height = max(24, int(round(height_mm / pitch)))
    if max(width, height) > MAX_EDGE or width * height > MAX_PIXELS:
        raise WorkbenchError("Use a coarser plane detail: this grid is too large.", "pitch")
    return (width, height)


@dataclass(frozen=True)
class Calibration:
    extent: tuple[int, int]
    corners: list[list[float]]
    width_mm: float
    height_mm: float
    matrix: np.ndarray

    @classmethod
    def parse(cls, form, width_mm, height_mm):
        corners = approved_quad(form.get("corners"))
        pitch = real(form.get("pitch"), "millimetres per pixel", 0.06, 8.0, default=0.66)
        extent = plane_shape(width_mm, height_mm, pitch)
        matrix = homography(UNIT_CORNERS, corners)
        return cls(extent, corners, width_mm, height_mm, matrix)

    @classmethod
    def recorded(cls, item):
        return cls(tuple(item["extent"]), item["corners"], item["width_mm"], item["height_mm"],
                   np.asarray(item["matrix"], dtype=np.float64))

    @property
    def scale(self):
        return np.array([self.width_mm / self.extent[0], self.height_mm / self.extent[1]])

    def receipt(self, source_extent):
        inverse = np.linalg.inv(self.matrix)
        error = transform(inverse, transform(self.matrix, UNIT_CORNERS)) - np.asarray(UNIT_CORNERS)
        return {
            "extent": list(self.extent), "corners": self.corners, "matrix": self.matrix.tolist(),
            "width_mm": self.width_mm, "height_mm": self.height_mm,
            "millimetres_per_pixel": self.scale.tolist(),
            "source_extent": list(source_extent),
            "round_trip_mm": round(float(np.max(np.linalg.norm(error * [self.width_mm, self.height_mm],
                                                               axis=1))), 6),
        }


def sampled(surface, source_x, source_y):
    w, h = surface.extent
    limits = (source_x >= 0) & (source_y >= 0) & (source_x <= w - 1) & (source_y <= h - 1)
    coords = np.array([source_y, source_x])
    passband = ndimage.map_coordinates(surface.valid.astype(np.float32), coords, order=1, mode="nearest")
    good = limits & (passband >= 0.997)
    image = np.zeros((*good.shape, 3), dtype=np.float32)
    for channel in range(3):
        sampled_values = ndimage.map_coordinates(surface.linear[:, :, channel], coords, order=1, mode="nearest")
        image[:, :, channel] = np.where(good, sampled_values, 0)
    return Raster(image, good)


def rectify(surface, calibration):
    width, height = calibration.extent
    source_width, source_height = surface.extent
    matrix = calibration.matrix
    a = np.linspace(0.5 / width, 1 - 0.5 / width, width)[None, :]
    b = np.linspace(0.5 / height, 1 - 0.5 / height, height)[:, None]
    divisor = matrix[2, 0] * a + matrix[2, 1] * b + matrix[2, 2]
    if np.any(np.abs(divisor) < EPS):
        raise WorkbenchError("The calibration plane crosses a projective horizon.", "corners")
    x = source_width * (matrix[0, 0] * a + matrix[0, 1] * b + matrix[0, 2]) / divisor - 0.5
    y = source_height * (matrix[1, 0] * a + matrix[1, 1] * b + matrix[1, 2]) / divisor - 0.5
    return sampled(surface, x, y)


def move_geometry(mark, matrix):
    pts = transform(matrix, mark["points"]).tolist()
    holes = [transform(matrix, ring).tolist() for ring in mark.get("holes", [])]
    return mark | {"points": pts, "holes": holes}


def physical_span(points_, extent, scale=None):
    values = np.array(pixel_points(points_, extent))
    delta = np.diff(values, axis=0)
    if scale is not None:
        delta *= scale
    return float(np.linalg.norm(delta, axis=1).sum())


def trace_stations(vertices, extent, step, *, cap=1500):
    line = np.asarray(pixel_points(vertices, extent), dtype=np.float64)
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    if distances[-1] < 2:
        raise WorkbenchError("Draw a longer stitch or yarn route.", "trace")
    count = int(math.ceil(distances[-1] / step)) + 1
    if count > cap:
        raise WorkbenchError("Use a larger station interval for this trace.", "station interval")
    stations = np.linspace(0, distances[-1], count)
    centers = np.column_stack([np.interp(stations, distances, line[:, i]) for i in range(2)])
    tangent = np.gradient(centers, axis=0)
    norms = np.linalg.norm(tangent, axis=1)
    if np.any(norms < 0.1):
        raise WorkbenchError("Remove the self-reversing section of the trace.", "trace")
    tangent /= norms[:, None]
    return stations, centers, np.column_stack([-tangent[:, 1], tangent[:, 0]])


def profile_lattice(values, mask, centers, normals, half_span):
    offsets = np.arange(-half_span, half_span + 1)
    positions = centers[:, None, :] + normals[:, None, :] * offsets[None, :, None]
    coord = [positions[..., 1], positions[..., 0]]
    measured = ndimage.map_coordinates(mask.astype(np.float32), coord, order=1, mode="constant") >= 0.99
    signals = ndimage.map_coordinates(values, coord, order=1, mode="constant", cval=0)
    return offsets, positions, signals, measured


def clip_segment(first, last, bounds):
    left, top, right, bottom = bounds
    x, y = first
    dx, dy = last[0] - x, last[1] - y
    before, after = 0.0, 1.0
    for p, q in ((-dx, x - left), (dx, right - x), (-dy, y - top), (dy, bottom - y)):
        if abs(p) <= EPS and q < 0:
            return None
        if abs(p) <= EPS:
            continue
        value = q / p
        if p < 0:
            before = max(before, value)
        else:
            after = min(after, value)
        if before > after:
            return None
    return (x + before * dx, y + before * dy), (x + after * dx, y + after * dy)

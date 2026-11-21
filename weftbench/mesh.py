"""Reference-guided netting inspection for transmitted-light mesh cloth.

The tool learns the regular mesh lattice (its two slanted repeat vectors and the
normal opening size) from an intact reference patch drawn by the operator, then
walks that lattice into the inspection region. Connected bright components are
matched against expected lattice sites so routine openings stay quiet while new
holes, enlarged openings and merged tears are reported separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .contracts import WorkbenchError, box, integer, mapping, real, require_support
from .geometry import box_mask, cut_rectangle
from .inspection import common_box
from .raster import component_labels, luma, rgba
from .visual import LILAC, OCHRE, PINK, TEAL, label, legend

# Site/component classification.
REGULAR = "regular"
ENLARGED = "enlarged opening"
NEW = "new opening"
MERGED = "merged opening"
REVIEW = "review"
MAX_LISTED_REGULAR = 220
MIN_ANCHORS = 8
MIN_REFERENCE_SITES = 9


@dataclass(frozen=True)
class MeshRequest:
    target: list
    reference: list
    contrast: float
    growth: float
    search_fraction: float
    max_results: int

    @classmethod
    def parse(cls, raw):
        form = mapping(raw)
        reference = box(form.get("reference"), "intact mesh reference")
        target = common_box(form)
        rx, ry, rw, rh = reference
        tx, ty, tw, th = target
        overlap = not (rx + rw <= tx or tx + tw <= rx or ry + rh <= ty or ty + th <= ry)
        if overlap:
            raise WorkbenchError(
                "Keep the intact reference patch outside the inspection region.", "reference",
            )
        return cls(
            target, reference,
            real(form.get("contrast"), "opening contrast", 0.02, 0.8, default=0.12),
            real(form.get("growth"), "enlarged opening ratio", 1.05, 4.0, default=1.35),
            real(form.get("search"), "site match radius fraction", 0.18, 0.6, default=0.34),
            integer(form.get("max_results"), "retained defects", 1, 120, default=60),
        )


def opening_mask(ctx, request):
    """Bright connected components in the inspection region under backlight."""
    surface = ctx.surface
    light = luma(surface.linear)
    region = box_mask(surface.extent, request.target)
    usable = region & ctx.visible & ~surface.bright
    require_support(int(region.sum()), int(usable.sum()), "Mesh inspection region", 160, fraction=0.30)
    background = ndimage.gaussian_filter(np.where(usable, light, np.nan), 3.0, mode="nearest")
    with np.errstate(invalid="ignore"):
        local = np.nanmedian(background[usable])
    background = np.where(np.isfinite(background), background, local)
    bright = (light >= background + request.contrast) & region & ctx.visible
    # Drop single-pixel speckle before labelling.
    bright = ndimage.binary_opening(bright, iterations=1)
    return light, region, bright


def lattice_basis(centers, shortest, longest):
    """Recover the two shortest non-parallel repeat vectors by vector voting.

    Bright opening centroids inside the intact reference patch repeat along the
    two mesh directions. A histogram of pairwise displacement vectors votes for
    the lattice basis, which stays reliable on small or slightly warped patches
    where a windowed FFT would lock onto a harmonic instead of the fundamental.
    """
    points = np.asarray(centers, dtype=np.float64)
    if len(points) < MIN_REFERENCE_SITES:
        raise WorkbenchError(
            "The reference patch needs at least nine clear mesh openings.", "reference",
        )
    deltas = (points[:, None, :] - points[None, :, :]).reshape(-1, 2)
    lengths = np.linalg.norm(deltas, axis=1)
    plausible = deltas[(lengths >= shortest * 0.85) & (lengths <= longest * 1.15)]
    if len(plausible) < 6:
        raise WorkbenchError("No repeating mesh openings were found in the reference patch.", "reference")
    # Canonical half-plane orientation so opposite vectors vote together.
    oriented = plausible.copy()
    flip = (oriented[:, 0] < -1e-9) | ((np.abs(oriented[:, 0]) <= 1e-9) & (oriented[:, 1] < 0))
    oriented[flip] *= -1
    bins = 64
    angles = (np.degrees(np.arctan2(oriented[:, 1], oriented[:, 0])) % 180.0)
    angle_index = np.clip((angles / 180.0 * bins).astype(int), 0, bins - 1)
    length_index = np.clip(
        ((np.linalg.norm(oriented, axis=1) - shortest) / max(1e-6, longest - shortest) * bins).astype(int),
        0, bins - 1,
    )
    ballot = np.zeros((bins, bins), dtype=np.int32)
    np.add.at(ballot, (angle_index, length_index), 1)
    smoothed = ndimage.gaussian_filter(ballot.astype(np.float32), 1.1)
    peaks = ndimage.maximum_filter(smoothed, size=5)
    cells = np.argwhere((smoothed == peaks) & (ballot > 0))
    if len(cells) == 0:
        raise WorkbenchError("No repeating mesh openings were found in the reference patch.", "reference")

    def vector_for(angle_cell, length_cell):
        members = oriented[(angle_index == angle_cell) & (length_index == length_cell)]
        return np.median(members, axis=0)

    # Collapse the length dimension per direction, then pick the shortest well
    # populated directions first: lattice basis vectors are the fundamental
    # repeats, so longer multiples must not win the vote.
    directions = sorted(cells, key=lambda cell: (cell[1], -float(smoothed[tuple(cell)])))
    chosen = []
    for angle_cell, length_cell in directions:
        candidate = vector_for(angle_cell, length_cell)
        if any(
            abs(float(np.dot(candidate, other))) / (
                np.linalg.norm(candidate) * np.linalg.norm(other)
            ) >= 0.8
            for other in chosen
        ):
            continue
        chosen.append(candidate)
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        raise WorkbenchError(
            "The reference patch must show openings in both mesh directions (try a wider patch).",
            "reference",
        )
    first, second = chosen
    if second[1] < 0:
        second = -second
    spacing = (float(np.linalg.norm(first)), float(np.linalg.norm(second)))
    if not (shortest * 0.8 <= min(spacing) and max(spacing) <= longest * 1.2):
        raise WorkbenchError("The detected mesh repeat is unreliable; redraw the intact reference.", "reference")
    return first, second


class Lattice:
    def __init__(self, basis_a, basis_b, origin):
        self.a = basis_a
        self.b = basis_b
        self.origin = np.asarray(origin, dtype=np.float64)
        self.metric = np.column_stack([self.a, self.b])
        self.inverse = np.linalg.inv(self.metric)

    def position(self, i, j):
        return self.origin + i * self.a + j * self.b

    def indices_near(self, point, radius):
        """All lattice index pairs whose site lies within radius pixels."""
        coords = self.inverse @ (np.asarray(point, dtype=np.float64) - self.origin)
        ci, cj = int(round(float(coords[0]))), int(round(float(coords[1])))
        reach = max(1, int(math.ceil(radius / min(np.linalg.norm(self.a), np.linalg.norm(self.b)))) + 1)
        for i in range(ci - reach, ci + reach + 1):
            for j in range(cj - reach, cj + reach + 1):
                if np.linalg.norm(self.position(i, j) - point) <= radius:
                    yield i, j


def reference_learning(ctx, request):
    surface = ctx.surface
    width, height = surface.extent
    x0, y0, x1, y1 = cut_rectangle(request.reference, surface.extent)
    if min(x1 - x0, y1 - y0) < 46:
        raise WorkbenchError("Draw a larger intact reference patch.", "reference")
    light = luma(surface.linear)
    patch_mask = np.zeros((height, width), dtype=bool)
    patch_mask[y0:y1, x0:x1] = True
    patch_mask &= ctx.visible & ~surface.bright
    require_support((y1 - y0) * (x1 - x0), int(patch_mask.sum()), "Intact mesh reference", 500, fraction=0.45)
    local = ndimage.gaussian_filter(np.where(patch_mask, light, np.nan), 1.4, mode="nearest")
    threshold = float(np.nanpercentile(local[patch_mask], 62))
    anchors = patch_mask & (light >= threshold)
    labels, count, areas, cells = component_labels(anchors)
    anchor_centers = []
    anchor_areas = []
    for candidate in range(1, count + 1):
        if areas[candidate] < 3:
            continue
        row, col = cells[candidate - 1]
        yy, xx = np.nonzero(labels[row, col] == candidate)
        anchor_centers.append(np.array([xx.mean() + col.start, yy.mean() + row.start]))
        anchor_areas.append(int(areas[candidate]))
    if len(anchor_centers) < MIN_REFERENCE_SITES:
        raise WorkbenchError(
            "The reference patch needs at least nine clear mesh openings.", "reference",
        )
    anchor_centers = np.asarray(anchor_centers)
    anchor_areas = np.asarray(anchor_areas)
    median_area = float(np.median(anchor_areas))
    basis_a, basis_b = lattice_basis(anchor_centers, 4.0, 64.0)
    # Place the origin on the anchor nearest the patch centre.
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    seed = anchor_centers[np.argmin(np.linalg.norm(anchor_centers - centre, axis=1))]
    lattice = Lattice(basis_a, basis_b, seed)
    matched = fit_origin(lattice, anchor_centers)
    if len(matched) < MIN_ANCHORS:
        raise WorkbenchError("Could not align the mesh lattice with the reference openings.", "reference")
    origin = np.median(np.stack([
        point - i * basis_a - j * basis_b for point, (i, j) in matched
    ]), axis=0)
    lattice = Lattice(basis_a, basis_b, origin)
    matched = fit_origin(lattice, anchor_centers)
    if matched:
        matched_areas = [
            int(anchor_areas[int(np.argmin(np.linalg.norm(anchor_centers - point, axis=1)))])
            for point, _ in matched
        ]
        fit_area = float(np.median(matched_areas))
    else:
        fit_area = median_area
    return lattice, fit_area, anchor_centers


def fit_origin(lattice, centers, *, fraction=0.30):
    """Nearest lattice index per anchor when it sits inside the match radius."""
    radius = fraction * min(np.linalg.norm(lattice.a), np.linalg.norm(lattice.b))
    matched = []
    for point in centers:
        coords = lattice.inverse @ (point - lattice.origin)
        i, j = int(round(float(coords[0]))), int(round(float(coords[1])))
        if np.linalg.norm(lattice.position(i, j) - point) <= radius:
            matched.append((point, (i, j)))
    return matched


def inspection_sites(lattice, region):
    height, width = region.shape
    rows, cols = np.nonzero(region)
    if not len(rows):
        return []
    corners = np.array([
        [cols.min(), rows.min()], [cols.max(), rows.min()],
        [cols.min(), rows.max()], [cols.max(), rows.max()],
    ], dtype=np.float64)
    coords = np.stack([lattice.inverse @ (corner - lattice.origin) for corner in corners])
    i_lo, i_hi = int(np.floor(coords[:, 0].min())) - 1, int(np.ceil(coords[:, 0].max())) + 1
    j_lo, j_hi = int(np.floor(coords[:, 1].min())) - 1, int(np.ceil(coords[:, 1].max())) + 1
    if (i_hi - i_lo) * (j_hi - j_lo) > 9000:
        raise WorkbenchError(
            "The inspection region covers too many mesh openings; draw a tighter region.", "box",
        )
    sites = []
    for i in range(i_lo, i_hi + 1):
        for j in range(j_lo, j_hi + 1):
            x, y = lattice.position(i, j)
            ix, iy = int(round(float(x))), int(round(float(y)))
            if 0 <= ix < width and 0 <= iy < height and region[iy, ix]:
                sites.append((i, j, ix, iy))
    return sites


def drift_field(lattice, component_centers, block=150):
    """Spatially varying lattice offset so the grid follows cloth that is not flat.

    Bright component centroids across the whole frame vote, per block, for the
    median displacement between nominal lattice sites and observed openings.
    Returns per-block offsets measured from the original lattice, plus a global
    median used where a block lacks evidence. A single missing or merged hole
    cannot drag the grid because every value is a median over several openings.
    """
    if not len(component_centers):
        return np.zeros(2), {}
    points = np.asarray(component_centers, dtype=np.float64)
    offsets = []
    for point in points:
        coords = lattice.inverse @ (point - lattice.origin)
        i, j = int(round(float(coords[0]))), int(round(float(coords[1])))
        offset = point - lattice.position(i, j)
        if np.linalg.norm(offset) <= 0.5 * min(np.linalg.norm(lattice.a), np.linalg.norm(lattice.b)):
            offsets.append((point, offset))
    if not offsets:
        return np.zeros(2), {}
    global_shift = np.median(np.stack([offset for _, offset in offsets]), axis=0)
    field = {}
    for point, offset in offsets:
        key = (int(point[0] // block), int(point[1] // block))
        field.setdefault(key, []).append(offset)
    shifts = {key: np.median(np.stack(values), axis=0) for key, values in field.items() if len(values) >= 3}
    return global_shift, shifts


def shift_at(shifts, fallback, x, y, block=150):
    key = (int(x // block), int(y // block))
    if key in shifts:
        return shifts[key]
    # Nearest populated block when this block lacks evidence.
    if shifts:
        nearest = min(shifts, key=lambda cell: (cell[0] - key[0]) ** 2 + (cell[1] - key[1]) ** 2)
        return shifts[nearest]
    return fallback


def component_pixel(label_id, labels):
    rows, cols = np.nonzero(labels == label_id)
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1, len(rows)


def classify(ctx, request, lattice, normal_area, components, shifts, fallback):
    surface = ctx.surface
    width, height = surface.extent
    labels, total, areas, cells = components
    search_radius = request.search_fraction * min(np.linalg.norm(lattice.a), np.linalg.norm(lattice.b))
    comp_info = {}
    centroids = {}
    for label_id in range(1, total + 1):
        x0, y0, x1, y1, area = component_pixel(label_id, labels)
        touches = x0 == 0 or y0 == 0 or x1 >= width or y1 >= height
        comp_info[label_id] = {"bounds": [x0, y0, x1, y1], "area": area, "touches_edge": touches}
        rows, cols = np.nonzero(labels == label_id)
        centroids[label_id] = (float(cols.mean()), float(rows.mean()))

    nominal_sites = inspection_sites(lattice, box_mask(surface.extent, request.target))
    visible_lookup = ctx.visible
    site_visibility = {}
    site_position = {}
    for i, j, ix, iy in nominal_sites:
        dx, dy = shift_at(shifts, fallback, ix, iy)
        px = int(round(ix + dx))
        py = int(round(iy + dy))
        px = min(max(px, 0), width - 1)
        py = min(max(py, 0), height - 1)
        site_position[(i, j)] = (px, py)
        site_visibility[(i, j)] = bool(visible_lookup[py, px])

    # Distance field to hidden/occluded pixels, so an opening pressed against a
    # clamp or fold edge is held for review instead of being called enlarged.
    target_mask = box_mask(surface.extent, request.target)
    hidden = target_mask & ~ctx.visible
    near_hidden = ndimage.binary_dilation(hidden, iterations=10) if hidden.any() else np.zeros_like(hidden)

    # Attach lattice sites to each component in two steps:
    #  * a site whose position lies inside the component is covered by it (this
    #    is how one connected tear keeps every lattice position it occupies);
    #  * otherwise a site within centroid match-radius belongs to the component.
    # Centroid-only matching stops a large new hole from stealing the routine
    # opening it merely sits next to.
    component_sites = {}
    for label_id in range(1, total + 1):
        cx, cy = centroids[label_id]
        attached = set()
        for (i, j), (ix, iy) in site_position.items():
            if not site_visibility[(i, j)]:
                continue
            if 0 <= iy < labels.shape[0] and 0 <= ix < labels.shape[1] and labels[iy, ix] == label_id:
                attached.add((i, j))
                continue
            if math.hypot(cx - ix, cy - iy) <= search_radius:
                attached.add((i, j))
        if attached:
            component_sites[label_id] = attached

    defects = []
    review = []
    regular_sites = []
    occupied_sites = set()
    used_components = set()

    # Components that explain one or more expected lattice sites.
    for label_id, sites in component_sites.items():
        info = comp_info[label_id]
        occupied_sites.update(sites)
        used_components.add(label_id)
        ratio = info["area"] / max(1.0, normal_area * len(sites))
        obscured = sum(1 for site in sites if not site_visibility[site])
        entry = {
            "component": label_id,
            "bounds": info["bounds"],
            "box": [round(info["bounds"][0] / width, 6), round(info["bounds"][1] / height, 6),
                    round((info["bounds"][2] - info["bounds"][0]) / width, 6),
                    round((info["bounds"][3] - info["bounds"][1]) / height, 6)],
            "center": [round(((info["bounds"][0] + info["bounds"][2]) / 2) / width, 6),
                       round(((info["bounds"][1] + info["bounds"][3]) / 2) / height, 6)],
            "area_px": info["area"], "area_to_reference": round(float(ratio), 4),
            "lattice_sites": [list(site) for site in sorted(sites)],
            "obscured_site_count": obscured,
        }
        if ctx.scale is not None:
            entry["area_mm2"] = round(float(info["area"] * np.prod(ctx.scale)), 5)
        bx0, by0, bx1, by1 = info["bounds"]
        beside_occlusion = bool(near_hidden[by0:by1, bx0:bx1].any())
        if len(sites) >= 2 and ratio >= 0.85:
            if beside_occlusion:
                entry["kind"] = REVIEW
                entry["reason"] = (
                    f"Connected opening spans {len(sites)} mesh positions but touches a hidden area; "
                    "confirm the full extent by eye."
                )
                review.append(entry)
            else:
                entry["kind"] = MERGED
                entry["reason"] = f"One connected opening covers {len(sites)} regular mesh positions."
                defects.append(entry)
        elif ratio >= request.growth:
            if beside_occlusion and len(sites) == 1:
                entry["kind"] = REVIEW
                entry["reason"] = "Enlarged signal sits beside a hidden or clamped area; confirm by eye."
                review.append(entry)
            else:
                entry["kind"] = ENLARGED
                entry["reason"] = f"Opening is {ratio:.2f}x the reference opening area."
                defects.append(entry)
        elif obscured or info["touches_edge"]:
            entry["kind"] = REVIEW
            entry["reason"] = "Partly hidden or cut by the frame; opening size cannot be confirmed."
            review.append(entry)
        else:
            entry["kind"] = REGULAR
            regular_sites.append(entry)

    # Bright components without an expected lattice site: real new holes are at
    # least as large as a routine opening fragment; sub-opening speckle is quiet.
    min_new_area = normal_area * 0.6
    for label_id in range(1, total + 1):
        if label_id in used_components:
            continue
        info = comp_info[label_id]
        if not info["touches_edge"] and info["area"] < min_new_area:
            continue
        near = set()
        cx = (info["bounds"][0] + info["bounds"][2]) / 2
        cy = (info["bounds"][1] + info["bounds"][3]) / 2
        for i, j in lattice.indices_near((cx, cy), search_radius):
            if (i, j) in site_visibility:
                near.add((i, j))
        entry = {
            "component": label_id, "bounds": info["bounds"],
            "box": [round(info["bounds"][0] / width, 6), round(info["bounds"][1] / height, 6),
                    round((info["bounds"][2] - info["bounds"][0]) / width, 6),
                    round((info["bounds"][3] - info["bounds"][1]) / height, 6)],
            "center": [round(cx / width, 6), round(cy / height, 6)],
            "area_px": info["area"], "area_to_reference": round(info["area"] / max(1.0, normal_area), 4),
            "lattice_sites": [list(site) for site in sorted(near)],
        }
        if ctx.scale is not None:
            entry["area_mm2"] = round(float(info["area"] * np.prod(ctx.scale)), 5)
        if info["touches_edge"]:
            entry["kind"] = REVIEW
            entry["reason"] = "Bright opening crosses the image edge; hold for review."
            review.append(entry)
        else:
            entry["kind"] = NEW
            entry["reason"] = "Bright opening where the reference lattice has no mesh position."
            defects.append(entry)

    # Expected sites with no detectable opening: review when visibility is doubtful.
    for (i, j), (ix, iy) in site_position.items():
        if (i, j) in occupied_sites:
            continue
        if not site_visibility[(i, j)]:
            review.append({
                "kind": REVIEW, "lattice_sites": [[i, j]],
                "center": [round(ix / width, 6), round(iy / height, 6)],
                "reason": "Expected mesh position is hidden or unreadable.",
            })

    defects.sort(key=lambda row: (-row["area_px"], row["component"]))
    for index, row in enumerate(defects[: request.max_results], 1):
        row["index"] = index
    return defects[: request.max_results], review, regular_sites, len(site_position), site_position


def render_map(ctx, request, lattice, defects, review, regular_sites, site_position):
    canvas = rgba(ctx.surface)
    pen = ImageDraw.Draw(canvas)
    width, height = canvas.size

    def rectangle(selection, color, width_px=3):
        x0, y0, x1, y1 = cut_rectangle(selection, (width, height))
        pen.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(*color, 255), width=width_px)

    rectangle(request.reference, TEAL, 3)
    rx, ry, _, _ = cut_rectangle(request.reference, (width, height))
    label(pen, (rx + 4, max(0, ry - 22)), "Intact mesh reference", fill=TEAL)
    rectangle(request.target, OCHRE, 2)

    # Quiet markers for regular openings that were matched.
    for row in regular_sites:
        x0, y0, x1, y1 = row["bounds"]
        pen.ellipse((x0 - 1, y0 - 1, x1, y1), outline=(*TEAL, 180), width=1)
    # Held-for-review markers.
    for row in review:
        if "bounds" in row:
            x0, y0, x1, y1 = row["bounds"]
            pen.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(*LILAC, 255), width=2)
        else:
            ix = int(round(row["center"][0] * width))
            iy = int(round(row["center"][1] * height))
            pen.ellipse((ix - 6, iy - 6, ix + 6, iy + 6), outline=(*LILAC, 255), width=2)
    # Reported defects: one outline per connected component, numbered once.
    for row in defects:
        x0, y0, x1, y1 = row["bounds"]
        pen.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(*PINK, 255), width=3)
        label(pen, (x0, max(0, y0 - 16)), str(row["index"]), fill=PINK)
        for i, j in row["lattice_sites"]:
            x, y = site_position.get((i, j), lattice.position(i, j))
            pen.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(*PINK, 220))

    legend_rows = [
        ("Intact reference patch", TEAL), ("Inspection region", OCHRE),
        ("Regular mesh opening", TEAL), ("New / enlarged / merged defect", PINK),
        ("Hold for review", LILAC),
    ]
    return legend(canvas, "Reference-guided mesh inspection", legend_rows)


def mesh_audit(ctx, raw):
    if ctx.capture["role"] != "backlight":
        raise WorkbenchError("Mesh inspection needs a transmitted-light photograph.", "photo")
    request = MeshRequest.parse(raw)
    lattice, normal_area, anchors = reference_learning(ctx, request)
    light, region, bright = opening_mask(ctx, request)
    components = component_labels(bright)

    # Build a spatially varying drift field from every detected opening, so the
    # reference lattice follows cloth that was not pulled perfectly flat.
    labels, total, _, cells = components
    centers = []
    for label_id in range(1, total + 1):
        row, col = cells[label_id - 1]
        yy, xx = np.nonzero(labels[row, col] == label_id)
        centers.append((xx.mean() + col.start, yy.mean() + row.start))
    global_shift, shifts = drift_field(lattice, centers)
    defects, review, regular_sites, site_count, site_position = classify(
        ctx, request, lattice, normal_area, components, shifts, global_shift,
    )
    artwork = render_map(ctx, request, lattice, defects, review, regular_sites, site_position)
    counts = {kind: sum(1 for item in defects if item["kind"] == kind)
              for kind in (NEW, ENLARGED, MERGED)}
    report = {
        "reference_box": request.reference, "box": request.target,
        "lattice_repeat_px": [round(float(np.linalg.norm(lattice.a)), 4),
                              round(float(np.linalg.norm(lattice.b)), 4)],
        "lattice_angle_degrees": round(
            float(math.degrees(math.atan2(
                np.cross(lattice.a, lattice.b), np.dot(lattice.a, lattice.b),
            ))) % 180, 4,
        ),
        "reference_opening_area_px": round(normal_area, 3),
        "reference_opening_count": len(anchors),
        "expected_sites": site_count,
        "regular_openings_matched": len(regular_sites),
        "defect_counts": counts,
        "review_count": len(review),
        "defects": defects,
        "held_for_review": review,
        "regular_examples": [
            {key: row[key] for key in ("component", "bounds", "area_px", "area_to_reference", "lattice_sites")}
            for row in regular_sites[:MAX_LISTED_REGULAR]
        ],
        "readable_pixels": int((region & ctx.visible).sum()),
        "unknown_pixels": int((region & ~ctx.visible).sum()),
        "note": "Routine mesh openings are matched to the reference lattice; a connected tear spanning "
                "several positions is reported once with the positions it occupies.",
    }
    return report, {"map": artwork}

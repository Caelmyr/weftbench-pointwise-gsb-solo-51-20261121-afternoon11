"""Procedural woven samples for exercising image measurements locally."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .catalog import Calibration, area_readout, install_capture, line_readout, readable_stats, view
from .contracts import conflict
from .geometry import rectify
from .raster import Raster, open_work, png, rgba, to_linear
from .visual import font

PhotoShapes = tuple[int, int]

SEAM_ROUTE = [[0.17, 0.67], [0.33, 0.66], [0.45, 0.666], [0.59, 0.674], [0.76, 0.665], [0.85, 0.659]]
CUT_ROUTE = [[0.17, 0.70], [0.33, 0.691], [0.45, 0.700], [0.59, 0.712], [0.76, 0.701], [0.85, 0.689]]
CORNERS = [[0.025, 0.024], [0.975, 0.024], [0.975, 0.976], [0.025, 0.976]]


@dataclass(frozen=True)
class WeavePlan:
    width: int
    height: int
    warp_pitch: float
    weft_pitch: float
    skew: float
    curl: float
    front_a: tuple
    front_b: tuple
    resin: float
    band_px: float
    graphic: str

    @property
    def size(self):
        return self.width, self.height


@dataclass(frozen=True)
class CameraTake:
    title: str
    role: str
    visit: str
    gain: float
    shift: float
    patina: float
    torn: float
    shade: float
    seed: int


JUNIPER = WeavePlan(
    750, 645, 9.0, 6.7, 0.09, 1.6,
    (0.35, 0.57, 0.48), (0.053, 0.29, 0.31), 0.18, 44.0, "bands",
)
BLUESTONE = WeavePlan(
    700, 560, 7.0, 8.0, -0.055, 2.1,
    (0.47, 0.39, 0.64), (0.11, 0.16, 0.28), 0.25, 59.5, "pane",
)
MESH = WeavePlan(
    720, 600, 15.0, 13.5, 0.12, 1.4,
    (0.32, 0.40, 0.45), (0.10, 0.15, 0.19), 0.10, 48.0, "bands",
)
SERIES = {
    "Juniper twill": [
        CameraTake("Juniper · incoming face", "face", "woven roll on bench", 1.0, 0.0, 0.15, 1.0, 0.17, 31),
        CameraTake("Juniper · transmitted light", "backlight", "woven roll on bench", 1.0, 0.0, 0.15, 1.2, 0.08, 32),
        CameraTake("Juniper · raking beam right", "rake", "woven roll on bench", 1.0, 0.0, 0.17, 1.0, 0.17, 33),
        CameraTake("Juniper · finish review face", "face", "finish review", 1.05, 0.0, 0.32, 1.5, 0.11, 34),
        CameraTake("Juniper · finish review rake", "rake", "finish review", 1.07, 0.0, 0.44, 1.5, 0.17, 35),
    ],
    "Bluestone check": [
        CameraTake("Bluestone · prior face", "face", "reference swatch", 0.92, 0.0, 0.0, 0.8, 0.13, 43),
        CameraTake("Bluestone · open backlight", "backlight", "reference swatch", 1.0, 0.0, 0.1, 1.0, 0.09, 44),
        CameraTake("Bluestone · oblique beam", "rake", "reference swatch", 1.03, 0.0, 0.18, 0.8, 0.17, 45),
        CameraTake("Bluestone · current face", "face", "second inspection", 1.03, 1.2, 0.38, 1.1, 0.11, 46),
        CameraTake("Bluestone · neutral reference", "reference", "second inspection", 1.0, 0.0, 0.0, 0.0, 0.09, 47),
    ],
    "Halcyon netting": [
        CameraTake("Halcyon · mesh face", "face", "netting roll on bench", 1.0, 0.0, 0.10, 0.6, 0.10, 61),
        CameraTake("Halcyon · transmitted light", "backlight", "netting roll on bench", 1.0, 0.0, 0.10, 0.9, 0.06, 62),
        CameraTake("Halcyon · raking beam", "rake", "netting roll on bench", 1.0, 0.0, 0.14, 0.6, 0.14, 63),
    ],
}


@dataclass(frozen=True)
class WeaveFields:
    x: np.ndarray
    y: np.ndarray
    gates_x: np.ndarray
    gates_y: np.ndarray
    lift: np.ndarray
    base: np.ndarray
    holes: np.ndarray
    ordinary_voids: np.ndarray
    shading: np.ndarray


def windows(size):
    width, height = size
    y, x = np.ogrid[:height, :width]
    return x.astype(np.float32), y.astype(np.float32)


def thread_curve(coord, spacing, shift):
    phase = 2 * np.pi * (coord + shift) / spacing
    return np.abs(np.cos(phase * 0.5)) ** 1.65


def carrier_layout(plan, xy):
    x, y = xy
    if plan.graphic == "bands":
        widths = np.mod(x + 0.07 * y, plan.width * 0.325)
        mask = (widths > plan.width * 0.20) | ((widths > plan.width * 0.125) & (widths < plan.width * 0.14))
        return ndimage.gaussian_filter(mask.astype(np.float32), 1)
    stripes_x = (np.mod(x, plan.width * 0.31) < plan.width * 0.13).astype(np.float32)
    stripes_y = (np.mod(y + 8 * np.sin(x / 70), plan.height * 0.38) < plan.height * 0.115).astype(np.float32)
    return 0.23 + 0.33 * stripes_x + 0.41 * stripes_y


def gap_map(plan, x, y, size):
    gaps = np.zeros((plan.height, plan.width), dtype=bool)
    for dx, dy, rx, ry, a in [(0.42, 0.40, 8, 16, 0.28), (0.66, 0.77, 11, 8, -0.34),
                               (0.57, 0.48, 4, 13, 0.17), (0.28, 0.84, 5, 12, 0.61)]:
        u, v = x - dx * plan.width, y - dy * plan.height
        spin_x = np.cos(a) * u - np.sin(a) * v
        spin_y = np.sin(a) * u + np.cos(a) * v
        radius = (spin_x / max(1, rx * size)) ** 2 + (spin_y / max(1, ry * size)) ** 2
        gaps |= radius <= 1 if size else False
    return gaps


def fabric_signals(plan, take):
    x, y = windows(plan.size)
    curve = plan.curl * np.sin(y / 59) + plan.skew * y
    length = plan.curl * np.sin(x / 111) + take.shift
    horizontal = thread_curve(x, plan.warp_pitch, curve)
    vertical = thread_curve(y, plan.weft_pitch, length)
    a = np.floor((x + curve) / plan.warp_pitch)
    b = np.floor((y + length) / plan.weft_pitch)
    pick = (np.mod(a + b * (1 if plan.graphic == "bands" else 2), 4) < 2).astype(np.float32)
    height = np.where(pick >= 0.5, horizontal * 0.59 + vertical * 0.18, vertical * 0.52 + horizontal * 0.16)
    base = carrier_layout(plan, (x, y))
    clear = np.maximum(0.0, 0.56 - (0.32 * horizontal + 0.36 * vertical))
    gaps = gap_map(plan, x, y, take.torn)
    shading = 1 + 0.032 * np.sin(2 * np.pi * y / plan.band_px) - take.shade * (0.5 - x / plan.width)
    return WeaveFields(x, y, horizontal, vertical, height, base, gaps, clear, shading)


def colors_under_face(plan, fields, take, rng):
    a = np.array(plan.front_a, dtype=np.float32)
    b = np.array(plan.front_b, dtype=np.float32)
    colors = a * (1 - fields.base[..., None]) + b * fields.base[..., None]
    shift = 0.77 + fields.lift * (0.54 + plan.resin) + 0.10 * fields.gates_x
    fiber = colors * shift[..., None] * fields.shading[..., None]
    fibre = np.where(fields.holes[:, :, None], 0.047, fiber)
    cloud = ndimage.gaussian_filter(rng.normal(0, 0.012, fields.holes.shape), 0.75)
    lobe = np.exp(-((fields.x - plan.width * 0.61) / 85) ** 2 - ((fields.y - plan.height * 0.46) / 73) ** 2)
    dust = cloud + take.patina * lobe * 0.07
    return fibre * take.gain + dust[..., None]


def colors_under_transmission(plan, fields, take):
    passing = 0.045 + 0.21 * fields.ordinary_voids + 0.17 * fields.ordinary_voids**2
    # Broad woven apertures and isolated puncture witnesses can be visually distinct in this demo.
    repeated = (fields.gates_x < 0.23) & (fields.gates_y < 0.26)
    passing += repeated * 0.11
    passing = np.where(fields.holes, 0.71 + 0.07 * fields.ordinary_voids, passing)
    passing = passing * fields.shading * take.gain
    data = passing[..., None] * np.array([0.98, 0.88, 0.71])[None, None, :]
    h, w = passing.shape
    clear_x, clear_y = int(w * 0.725), int(h * 0.067)
    data[clear_y:clear_y + int(h * 0.11), clear_x:clear_x + int(w * 0.17)] = [0.76, 0.68, 0.53]
    data[clear_y:clear_y + int(h * 0.11), int(w * 0.53):int(w * 0.65)] = [0.009, 0.009, 0.009]
    return data


def cotton_fibres(fields, take, plan, rng):
    shape = fields.holes.shape
    impulses = np.zeros(shape, dtype=np.float32)
    amount = int(165 + 870 * take.patina)
    xx = rng.integers(0, plan.width, amount)
    yy = rng.integers(0, plan.height, amount)
    brightness = rng.uniform(0.13, 0.72, amount)
    impulses[yy, xx] = brightness
    pull = ndimage.gaussian_filter(impulses, (0.6, 1.85))
    extra = np.exp(-((fields.x - plan.width * 0.59) / 115)**2 - ((fields.y - plan.height * 0.43) / 88)**2)
    return pull * 3.8 + extra * take.patina * 0.08


def colors_under_rake(plan, fields, take, rng):
    _, bump = np.gradient(fields.lift)
    color = np.array([0.23, 0.31, 0.274]) * (0.82 + fields.lift[..., None] * 0.65)
    shadow = 1 + 0.72 * np.tanh(bump * 6)
    relief = cotton_fibres(fields, take, plan, rng)
    natural = color * shadow[..., None] * fields.shading[..., None]
    return np.clip((natural + relief[..., None]) * take.gain, 0, 1)


def cloth_edge_mask(plan, take, fields):
    width, height = plan.size
    x, y = fields.x, fields.y
    main = (x >= 8 + 2.1 * np.sin(y / 76)) & (x < width - 5 - 1.8 * np.cos(y / 73))
    main &= (y >= 7 + 2.0 * np.sin(x / 83)) & (y < height - 7 - 2.4 * np.cos(x / 85))
    if "finish" in take.visit or "second" in take.visit:
        missing = ((x - width * 0.90) / 33) ** 2 + ((y - height * 0.31) / 56) ** 2 < 1
        main &= ~missing
    return main


def insert_stitches(image, taking, plan):
    if taking.role == "backlight":
        return image
    art = image.copy()
    brush = ImageDraw.Draw(art)
    outline = [(x * art.width, y * art.height) for x, y in SEAM_ROUTE]
    trim = [(x * art.width, y * art.height) for x, y in CUT_ROUTE]
    for index, (a, b) in enumerate(zip(outline, outline[1:]), 1):
        pitch = max(1, int(math.dist(a, b) / 7.5))
        for k in range(pitch):
            p = k / pitch
            x, y = (a[0] * (1 - p) + b[0] * p, a[1] * (1 - p) + b[1] * p)
            color = (239, 214, 173, 255) if k % 3 else (213, 192, 159, 255)
            brush.line((x - 2, y, x + 2, y + 1), fill=color, width=1 + index % 2)
    brush.line(trim, fill=(38, 45, 54, 255), width=2, joint="curve")
    return art


def digital_reference(plan, take):
    picture = Image.new("RGB", plan.size, (57, 56, 53))
    draw = ImageDraw.Draw(picture)
    tones = [(46, 46, 48), (87, 89, 90), (123, 125, 127), (165, 168, 167),
             (222, 224, 222), (112, 156, 164), (150, 121, 158), (183, 131, 103)]
    for index, swatch in enumerate(tones):
        x = 75 + (index % 4) * 136
        y = 115 + (index // 4) * 185
        draw.rounded_rectangle((x, y, x + 110, y + 135), radius=5, fill=swatch)
        draw.text((x + 7, y + 146), f"Lab reference {index + 1}", fill=(224, 219, 202), font=font(13))
    draw.text((75, 31), "Synthetic neutral reference", fill=(237, 235, 217), font=font(25, True))
    return picture


@lru_cache(maxsize=10)
def textile_picture(plan, take):
    rng = np.random.default_rng(take.seed)
    if take.role == "reference":
        return png(digital_reference(plan, take))
    if plan is MESH and take.role == "backlight":
        return mesh_picture(plan, take)
    fields = fabric_signals(plan, take)
    if take.role == "backlight":
        data = colors_under_transmission(plan, fields, take)
    elif take.role == "rake":
        data = colors_under_rake(plan, fields, take, rng)
    else:
        data = colors_under_face(plan, fields, take, rng)
    valid = cloth_edge_mask(plan, take, fields)
    raster = Raster(np.asarray(data, dtype=np.float32).clip(0, 1), valid)
    return png(insert_stitches(rgba(raster), take, plan))


def demo_gallery_photo():
    return textile_picture(JUNIPER, SERIES["Juniper twill"][0])


# Fixed landmarks for the procedural Halcyon netting sample.
MESH_BASIS_A = np.array([15.0, 2.6])
MESH_BASIS_B = np.array([-2.2, 13.4])
MESH_ORIGIN = np.array([46.0, 40.0])
MESH_CLAMP = (150.0, 470.0, 78.0, 46.0)
MESH_ENLARGED = (18, 6)
MESH_MERGED = [(24, 20), (25, 20), (24, 21), (25, 21)]
MESH_NEW_CELL = (16, 26)


def mesh_site_centers(plan):
    """Slanted, gently warped lattice positions for the netting sample."""
    width, height = plan.size
    y_index = np.arange(height)
    x_index = np.arange(width)
    warp_x = 1.6 * np.sin(y_index / 95.0)
    warp_y = 1.2 * np.cos(x_index / 110.0)
    centers = {}
    for i in range(-60, 61):
        for j in range(-60, 61):
            cx, cy = (MESH_ORIGIN + i * MESH_BASIS_A + j * MESH_BASIS_B).tolist()
            cx += float(warp_x[int(np.clip(round(cy), 0, height - 1))])
            cy += float(warp_y[int(np.clip(round(cx), 0, width - 1))])
            if 6 <= cx < width - 6 and 6 <= cy < height - 6:
                centers[(i, j)] = (cx, cy)
    return centers


def mesh_picture(plan, take):
    """Transmitted-light netting with routine holes and known defect witnesses."""
    rng = np.random.default_rng(take.seed)
    width, height = plan.size
    yy, xx = np.mgrid[:height, :width]
    centers = mesh_site_centers(plan)
    data = np.full((height, width, 3), 0.05, np.float32)
    data += rng.normal(0, 0.006, (height, width, 1)).astype(np.float32)
    for cx, cy in centers.values():
        glow = 0.62 * np.exp(-(((xx - cx) ** 2) / 9.5 + ((yy - cy) ** 2) / 9.5))
        data[..., 0] += glow * 0.96
        data[..., 1] += glow * 0.86
        data[..., 2] += glow * 0.70

    # Clamp occlusion across the lower-left section.
    clamp_x, clamp_y, clamp_rx, clamp_ry = MESH_CLAMP
    clamp = ((xx - clamp_x) / clamp_rx) ** 2 + ((yy - clamp_y) / clamp_ry) ** 2 < 1
    data[clamp] = [0.012, 0.011, 0.010]

    # One routine opening deliberately enlarged.
    ex, ey = centers[MESH_ENLARGED]
    swell = np.exp(-(((xx - ex) ** 2) / 26 + ((yy - ey) ** 2) / 26))
    data[..., 0] += 0.55 * 0.96 * swell
    data[..., 1] += 0.50 * 0.86 * swell
    data[..., 2] += 0.40 * 0.70 * swell

    # A single connected tear spanning four adjacent lattice positions.
    merged_x = float(np.mean([centers[site][0] for site in MESH_MERGED]))
    merged_y = float(np.mean([centers[site][1] for site in MESH_MERGED]))
    tear = 0.8 * np.exp(-(((xx - merged_x) ** 2) / 150 + ((yy - merged_y) ** 2) / 120))
    data[..., 0] += tear * 0.96
    data[..., 1] += tear * 0.86
    data[..., 2] += tear * 0.70

    # A new opening inside a lattice cell where no routine mesh position exists.
    nx_, ny_ = centers[MESH_NEW_CELL]
    nx_ += 7.0
    ny_ += 7.5
    new_hole = 0.8 * np.exp(-(((xx - nx_) ** 2) / 20 + ((yy - ny_) ** 2) / 20))
    data[..., 0] += new_hole * 0.96
    data[..., 1] += new_hole * 0.86
    data[..., 2] += new_hole * 0.70

    valid = ((xx >= 10 + 1.5 * np.sin(yy / 60)) & (xx < width - 8)
             & (yy >= 9) & (yy < height - 7) & ~clamp)
    raster = Raster(np.clip(data, 0, 1).astype(np.float32), valid)
    return png(rgba(raster))


def install_training_plane(lib, item, fabric):
    source = open_work(lib.asset(item["id"], "working").path)
    plan = Calibration.parse({"corners": CORNERS, "pitch": 0.71}, fabric["width_mm"], fabric["length_mm"])
    surface = rectify(source, plan)
    entry = lib.add("planes", plan.receipt(source.extent) | {
        "capture": item["id"], "title": "Bench reference corners",
        "coverage": readable_stats(surface), "physical_version": fabric["version"],
    }, parent=fabric["id"], action="training physical plane recorded")
    lib.attach(entry["id"], "plane", png(rgba(surface)), "image/png")
    lib.cite(entry["id"], item, "calibrated photo")
    lib.cite(entry["id"], fabric, "physical dimensions")
    return entry


def install_training_mark(lib, ctx, title, kind, vertices, *, across=True, observed="new"):
    shape = "line" if kind in {"seam", "cut", "selvedge"} else "polygon"
    measure = line_readout(ctx, vertices) if shape == "line" else area_readout(ctx, vertices, [])
    item = lib.add("marks", {
        "title": title, "memo": "Procedural training evidence", "shape": shape, "mark_type": kind,
        "points": vertices, "holes": [], "basis": ctx.basis if across else "capture",
        "capture": ctx.capture["id"], "plane": ctx.plane["id"] if ctx.plane and across else None,
        "fabric_extent_mm": [ctx.plane["width_mm"], ctx.plane["height_mm"]] if ctx.plane and across else None,
        "state": observed, "hidden": False, "reading": measure,
    }, parent=ctx.fabric["id"], action="training observation retained")
    lib.cite(item["id"], ctx.capture, "recorded on training photo")
    return item


def selector_band():
    return [[0.21, 0.27], [0.41, 0.27], [0.43, 0.46], [0.20, 0.46]]


def seed_fabric(lib, title, settings, dimensions, index):
    is_mesh = settings is MESH
    material = lib.add("fabrics", {
        "title": title, "roll_code": f"TRAIN-{109 + index}", "lot": "Procedural training collection",
        "weave": "open" if is_mesh else ("twill" if index == 1 else "plain"),
        "width_mm": dimensions[0], "length_mm": dimensions[1],
        "memo": "Synthetic woven images for inspecting the tools and visible evidence.", "state": "active",
        "training": True,
    }, action="training fabric created")
    files = []
    retained_planes = {}
    for j, take in enumerate(SERIES[title]):
        document = install_capture(lib, material, {
            "title": take.title, "role": take.role, "face": "face", "session": take.visit,
            "captured_day": "2026-10-09" if j <= 2 else "2026-10-15",
            "memo": "Generated woven image, not a real mill observation.",
        }, textile_picture(settings, take), f"training-{index}-{take.seed}.png", training=True)
        if take.role != "reference":
            retained_planes[document["id"]] = install_training_plane(lib, document, material)
        files.append(document)
    if is_mesh:
        backlight = next(row for row in files if row["title"].endswith("transmitted light"))
        plane = retained_planes[backlight["id"]]
        mesh_ctx = view(lib, backlight["id"], plane_key=plane["id"])
        clamp_x, clamp_y, clamp_rx, clamp_ry = MESH_CLAMP
        width, height = settings.size
        install_training_mark(
            lib, mesh_ctx, "Support clamp occlusion", "ignore",
            [[(clamp_x - clamp_rx) / width, (clamp_y - clamp_ry) / height],
             [(clamp_x + clamp_rx) / width, (clamp_y - clamp_ry) / height],
             [(clamp_x + clamp_rx) / width, (clamp_y + clamp_ry) / height],
             [(clamp_x - clamp_rx) / width, (clamp_y + clamp_ry) / height]], observed="accepted",
        )
        return {"fabric": material["id"], "captures": [row["id"] for row in files]}
    source = files[0]
    built = retained_planes[source["id"]]
    current = view(lib, source["id"], plane_key=built["id"])
    install_training_mark(lib, current, "Reference color panel", "proof", selector_band(), observed="accepted")
    install_training_mark(lib, current, "Stitched test route", "seam", SEAM_ROUTE, observed="accepted")
    install_training_mark(lib, current, "Cut edge of sewn strip", "cut", CUT_ROUTE, observed="accepted")
    install_training_mark(lib, current, "Suspect opening", "puncture",
                          [[0.40, 0.36], [0.44, 0.36], [0.45, 0.425], [0.396, 0.431]], observed="check")
    install_training_mark(lib, current, "Support clamp shadow", "ignore",
                          [[0.04, 0.14], [0.16, 0.15], [0.14, 0.233], [0.08, 0.22]])
    return {"fabric": material["id"], "captures": [row["id"] for row in files]}


def seed_training(store):
    with store.open(write=True) as lib:
        if any(item.get("training") for item in lib.list("fabrics", limit=100)):
            raise conflict("Training fabrics are already in the library.")
        result = [
            seed_fabric(lib, "Juniper twill", JUNIPER, (444, 385), 1),
            seed_fabric(lib, "Bluestone check", BLUESTONE, (397, 324), 2),
            seed_fabric(lib, "Halcyon netting", MESH, (420, 350), 3),
        ]
    return {"training": result, "count": sum(len(row["captures"]) for row in result)}

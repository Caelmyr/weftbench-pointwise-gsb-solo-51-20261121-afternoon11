"""End-to-end check for the opposite-raking-light nap direction tool.

Builds two synthetic calibrated raking-light photos of a napped cloth whose nap
lay is known, then runs weftbench.nap.nap_direction through the real reading
pipeline and checks arrows / review / support behaviour.
"""
from __future__ import annotations

import math
import sys

sys.path.insert(0, "/opt/pointwise-env/site-packages")

import numpy as np
from PIL import ImageDraw

sys.path.insert(0, "/workspace")

from weftbench.catalog import Calibration, install_capture, view  # noqa: E402
from weftbench.geometry import rectify  # noqa: E402
from weftbench.nap import (  # noqa: E402
    COMPASS, STATUS_DIRECTED, STATUS_NO_AXIS, STATUS_REVIEW, STATUS_UNSUPPORTED,
)
from weftbench.raster import Raster, open_work, png, rgba  # noqa: E402
from weftbench.readings import analyze  # noqa: E402
from weftbench.store import Store  # noqa: E402
from weftbench.visual import font  # noqa: E402

WIDTH, HEIGHT = 720, 600
CORNERS = [[0.02, 0.02], [0.98, 0.02], [0.98, 0.98], [0.02, 0.98]]
# Two opposite nap lays along the x axis. Region A tips point right; region B tips point left.
SPLIT_X = int(WIDTH * 0.52)


def nap_field():
    """Unit nap-tip vector field: right in the left band, left in the right band."""
    tip = np.zeros((HEIGHT, WIDTH, 2), dtype=np.float32)
    tip[:, :SPLIT_X, 0] = 1.0
    tip[:, SPLIT_X:, 0] = -1.0
    return tip


def weave_base(rng):
    """Strong warp/weft texture that is identical for both raking photos."""
    x = np.arange(WIDTH, dtype=np.float32)[None, :]
    y = np.arange(HEIGHT, dtype=np.float32)[:, None]
    warp = 0.06 * np.sin(2 * math.pi * x / 7.0)
    weft = 0.06 * np.sin(2 * math.pi * y / 9.0)
    noise = rng.normal(0, 0.012, (HEIGHT, WIDTH)).astype(np.float32)
    base = 0.42 + warp + weft + noise
    return base.astype(np.float32)


def rake_photo(tip, base, light_dir, rng, *, fold=False, right_only_clear=False):
    dx, dy = COMPASS[light_dir]
    light = np.array([dx, dy], dtype=np.float32)
    # Tips pointing away from the light read brighter: shading = +k * (tip . light).
    shading = 0.16 * (tip[..., 0] * light[0] + tip[..., 1] * light[1])
    broad = rng.normal(0, 0.006, (HEIGHT, WIDTH)).astype(np.float32)
    data = np.clip(base + shading + broad, 0.02, 0.98)
    valid = np.ones((HEIGHT, WIDTH), dtype=bool)
    if fold:
        # A fold band is unreadable (occluded) on both takes.
        fy0, fy1 = int(HEIGHT * 0.62), int(HEIGHT * 0.72)
        valid[fy0:fy1, :] = False
        data[fy0:fy1, :] = 0.0
    if right_only_clear:
        # A patch readable on this take but dark/missing on the other take.
        px0, px1 = int(WIDTH * 0.70), int(WIDTH * 0.88)
        py0, py1 = int(HEIGHT * 0.10), int(HEIGHT * 0.30)
        data[py0:py1, px0:px1] = 0.5 + 0.05 * rng.normal(size=(py1 - py0, px1 - px0))
    rgb = np.repeat(data[..., None], 3, axis=2).astype(np.float32)
    return Raster(rgb, valid)


def title_panel(raster, title):
    art = rgba(raster)
    pen = ImageDraw.Draw(art)
    pen.rectangle((6, 6, 360, 34), fill=(24, 34, 37))
    pen.text((12, 10), title, fill=(236, 236, 222), font=font(16, True))
    return art


def install(lib, fabric, raster, title, light):
    document = install_capture(
        lib, fabric,
        {"title": title, "role": "rake", "face": "face", "session": "nap check",
         "captured_day": "2026-11-21", "memo": f"Synthetic raking photo, light {light}"},
        png(title_panel(raster, title)), f"{light}.png", training=True,
    )
    source = open_work(lib.asset(document["id"], "working").path)
    plan = Calibration.parse({"corners": CORNERS, "pitch": 0.6}, fabric["width_mm"], fabric["length_mm"])
    rectified = rectify(source, plan)
    plane = lib.add("planes", plan.receipt(source.extent) | {
        "capture": document["id"], "title": f"Plane {light}",
        "coverage": {"visible": 1.0, "low_signal": 0.0, "clipped": 0.0},
        "physical_version": fabric["version"],
    }, parent=fabric["id"], action="nap plane")
    lib.attach(plane["id"], "plane", png(rgba(rectified)), "image/png")
    lib.cite(plane["id"], document, "calibrated photo")
    lib.cite(plane["id"], fabric, "physical dimensions")
    return document, plane


def main():
    store = Store("/workspace/verify_nap/data")
    rng = np.random.default_rng(20261121)
    tip = nap_field()
    base = weave_base(rng)
    photo_a = rake_photo(tip, base, "right", rng, fold=True)
    # Opposite light; same fold, plus a patch clear here only (dark on the first take).
    rng_b = np.random.default_rng(20261122)
    photo_b = rake_photo(tip, base, "left", rng_b, fold=True, right_only_clear=True)
    # Make the matching patch unreadable on the first take.
    px0, px1 = int(WIDTH * 0.70), int(WIDTH * 0.88)
    py0, py1 = int(HEIGHT * 0.10), int(HEIGHT * 0.30)
    photo_a.valid[py0:py1, px0:px1] = False
    photo_a.linear[py0:py1, px0:px1] = 0.0

    with store.open(write=True) as lib:
        fabric = lib.add("fabrics", {
            "title": "Nap verification cloth", "roll_code": "NAP-1", "lot": "synthetic",
            "weave": "plain", "width_mm": 420.0, "length_mm": 350.0,
            "memo": "Synthetic nap pair", "state": "active",
        }, action="fabric")
        capture_a, plane_a = install(lib, fabric, photo_a, "Rake from right", "right")
        capture_b, plane_b = install(lib, fabric, photo_b, "Rake from left", "left")
        fabric_id = fabric["id"]
        capture_a_id, capture_b_id = capture_a["id"], capture_b["id"]

    settings = {
        "reference": capture_b_id, "light": "right", "reference_light": "left",
        "box": [0.04, 0.04, 0.92, 0.92], "cells": 70,
    }
    form = {"tool": "nap", "title": "Nap lay check", "settings": settings,
            "plane": None, "registered": True}
    with store.open() as lib:
        ctx, record, pictures, references = analyze(lib, capture_a_id, form)
        report = record["analysis"]
        pictures["map"].save("/workspace/verify_nap/out/nap-map.png")
        pictures["pair"].save("/workspace/verify_nap/out/nap-pair.png")

    failures = []

    def check(name, condition, detail=""):
        print(f"{'PASS' if condition else 'FAIL'}: {name} {detail}")
        if not condition:
            failures.append(name)

    check("proof roles", sorted(pictures) == ["map", "pair"], str(sorted(pictures)))
    check("both captures cited",
          {r[0]["id"] for r in references if r[0]["kind"] == "captures"} == {capture_a_id, capture_b_id})
    check("recorded opposite lights",
          report["current_light_direction"] == "right" and report["reference_light_direction"] == "left")

    cells = report["cells"]
    by_status = {}
    for cell in cells:
        by_status.setdefault(cell["status"], []).append(cell)
    print("counts:", report["cell_counts"])
    check("has directed cells", report["cell_counts"][STATUS_DIRECTED] > 0)
    check("has unsupported fold/one-photo cells", report["cell_counts"][STATUS_UNSUPPORTED] > 0)
    check("no definite arrow where unsupported",
          all(c["nap_direction_degrees"] is None for c in by_status.get(STATUS_UNSUPPORTED, [])))
    check("review/no-axis cells carry no arrow",
          all(c["nap_direction_degrees"] is None
              for c in by_status.get(STATUS_REVIEW, []) + by_status.get(STATUS_NO_AXIS, [])))

    # Directed cells on the left band must point right (~0 deg), right band must point left (~180 deg).
    bad = []
    for cell in by_status.get(STATUS_DIRECTED, []):
        cx = cell["center"][0]
        ang = cell["nap_direction_degrees"]
        expected = 0.0 if cx < 0.52 else 180.0
        diff = abs((ang - expected + 180) % 360 - 180)
        if diff > 35:
            bad.append((cell["index"], round(cx, 3), ang))
    check("directed arrows follow known nap lay", not bad, str(bad[:8]))

    # Reject non-opposite light directions.
    bad_form = {"tool": "nap", "settings": {**settings, "reference_light": "right"}}
    try:
        with store.open() as lib:
            analyze(lib, capture_a_id, bad_form)
        check("opposite light enforced", False)
    except Exception as exc:  # noqa: BLE001
        check("opposite light enforced", "opposite" in str(exc), str(exc))

    # Reject a face-role capture paired as the raking reference.
    with store.open(write=True) as lib:
        face_doc = install_capture(
            lib, lib.get(fabric_id, "fabrics"),
            {"title": "Face photo", "role": "face", "face": "face"},
            png(rgba(photo_b)), "face.png", training=True,
        )
        face_source = open_work(lib.asset(face_doc["id"], "working").path)
        face_plan = Calibration.parse(
            {"corners": CORNERS, "pitch": 0.6}, fabric["width_mm"], fabric["length_mm"],
        )
        face_plane = lib.add("planes", face_plan.receipt(face_source.extent) | {
            "capture": face_doc["id"], "title": "Plane face",
            "coverage": {"visible": 1.0, "low_signal": 0.0, "clipped": 0.0},
            "physical_version": lib.get(fabric_id, "fabrics")["version"],
        }, parent=fabric_id, action="face plane")
        lib.attach(face_plane["id"], "plane", png(rgba(rectify(face_source, face_plan))), "image/png")
        face_id = face_doc["id"]
    face_form = {"tool": "nap", "settings": {**settings, "reference": face_id}}
    try:
        with store.open() as lib:
            analyze(lib, capture_a_id, face_form)
        check("rake role enforced", False)
    except Exception as exc:  # noqa: BLE001
        check("rake role enforced", "raking" in str(exc).lower(), str(exc))

    # Full save path: images retained with the reading.
    from weftbench.readings import save_reading

    saved = save_reading(store, capture_a_id, form)
    roles = sorted(asset["role"] for asset in saved["assets"])
    check("saved reading retains images", roles == ["map", "pair"], str(roles))

    print("\nRESULT:", "ALL PASS" if not failures else f"{len(failures)} FAILURES -> {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

"""Bind inspection notes to the exact fabric observations reviewed on a visit."""
from __future__ import annotations

from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

from .catalog import detached, version
from .contracts import WorkbenchError, choice, date, identifiers, integer, limit, mapping, text
from .visual import INK, MUTED, OCHRE, PAPER, TEAL, ellipsize, font
from .raster import fit_image, opaque, png

CITABLE = {
    "marks": "Observation", "captures": "Source photo", "readings": "Inspection",
    "swatches": "Approved color", "planes": "Fabric plane", "details": "Detail sheet",
}
VISIT_KINDS = {
    "incoming": "Incoming quality check", "process": "In-process recheck",
    "finish": "Finished fabric release", "return": "Return review",
}
OUTCOMES = {"check": "Further inspection required", "ready": "Reviewed", "hold": "Keep under review"}
VISIT_LIMIT = 100


def reference_entry(lib, fabric, cited):
    target = lib.get(cited)
    if target["parent"] != fabric or target["kind"] not in CITABLE:
        raise WorkbenchError("Select observations and image evidence from this fabric.", "observations")
    return target


def card_note(form, before=None):
    prior = before or {}
    return {
        "title": text(form.get("title", prior.get("title")), "visit title", 94),
        "observed_day": date(form.get("observed_day", prior.get("observed_day")), "observed day"),
        "visit_type": choice(form.get("visit_type", prior.get("visit_type")), "visit type", VISIT_KINDS, "incoming"),
        "outcome": choice(form.get("outcome", prior.get("outcome")), "decision", OUTCOMES, "check"),
        "memo": text(form.get("memo", prior.get("memo")), "review note", 1400, default="", multiline=True),
    }


def save_visit(store, parent, raw):
    form = mapping(raw)
    keys = identifiers(form.get("observations", []), "evidence records", 22, minimum=0)
    values = card_note(form)
    with store.open(write=True) as lib:
        owner = lib.get(parent, "fabrics")
        if lib.count("visits", parent=parent) >= VISIT_LIMIT:
            raise limit("This fabric has reached its visit limit.")
        references = [reference_entry(lib, parent, key) for key in keys]
        entry = lib.add("visits", values | {
            "evidence": [{"id": item["id"], "version": item["version"],
                          "kind": item["kind"], "title": item["title"]} for item in references],
            "fabric_at_visit": owner["version"],
        }, parent=parent, action="inspection visit saved")
        lib.cite(entry["id"], owner, "fabric at visit")
        for item in references:
            lib.cite(entry["id"], item, "visited evidence")
        return lib.public(entry, links=True)


def update_visit(store, parent, key, raw):
    form = mapping(raw)
    with store.open(write=True) as lib:
        before = lib.get(key, "visits", parent=parent)
        details = card_note(form, before)
        entry = lib.revise(before, version(form.get("version")), detached(before) | details,
                           action="visit note and decision revised")
        return lib.public(entry, links=True)


def joined_visit(lib, visit):
    joined = []
    for entry in visit["evidence"]:
        live = lib.get(entry["id"], optional=True)
        if live is None:
            joined.append(entry | {"status": "not available", "latest_version": None})
            continue
        held = lib.revision(entry["id"], entry["version"])
        joined.append(entry | {
            "status": "still current" if held["version"] == live["version"] else "newer revision available",
            "latest_version": live["version"],
            "pinned": held,
        })
    return visit | {"evidence": joined}


def timeline(store, parent, limit_=None):
    count = integer(limit_, "visit count", 1, VISIT_LIMIT, default=45)
    with store.open() as lib:
        lib.get(parent, "fabrics")
        selected = sorted(lib.list("visits", parent=parent, limit=VISIT_LIMIT),
                          key=lambda row: (row["observed_day"], row["created"]), reverse=True)[:count]
        readings = Counter(ref["kind"] for entry in selected for ref in entry["evidence"])
        return {
            "visits": [joined_visit(lib, entry) for entry in selected],
            "decisions": dict(Counter(entry["outcome"] for entry in selected)), "evidence_types": dict(readings),
        }


def thumbnail_for(lib, owner):
    preference = {
        "readings": ("map", "pair", "chart", "card"),
        "captures": ("thumbnail", "working"),
        "swatches": ("card",),
        "planes": ("plane",),
    }
    for label_ in preference.get(owner["kind"], ()):
        attached = lib.asset(owner["id"], label_, optional=True)
        if attached is None:
            continue
        with Image.open(attached.path) as image:
            return fit_image(opaque(image), 220, 125)
    return None


def proof_summary(lib, visit):
    picked = visit["evidence"][:12]
    width = 980
    height = 165 + 170 * int(np.ceil(max(1, len(picked)) / 3))
    card = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(card)
    draw.rectangle((0, 0, width, 97), fill=INK)
    title = ellipsize(visit["title"], draw, font(25, True), width - 55)
    draw.text((26, 19), title, fill=PAPER, font=font(25, True))
    draw.text((27, 65), f'{visit["observed_day"]} · {VISIT_KINDS[visit["visit_type"]]} · {OUTCOMES[visit["outcome"]]}',
              fill=(182, 218, 204), font=font(15))
    for i, pick in enumerate(picked):
        y = 132 + 170 * (i // 3)
        x = 29 + 318 * (i % 3)
        pinned = lib.revision(pick["id"], pick["version"])
        photo = thumbnail_for(lib, pinned)
        if photo:
            card.paste(photo, (x, y))
        else:
            draw.rounded_rectangle((x, y, x + 215, y + 98), radius=5, fill=(219, 227, 215))
            draw.text((x + 11, y + 38), CITABLE[pinned["kind"]], fill=MUTED, font=font(13))
        name = ellipsize(pick["title"], draw, font(14, True), 275)
        draw.text((x, y + 129), name, fill=INK, font=font(14, True))
        draw.text((x, y + 149), f"Recorded version {pick['version']}", fill=MUTED, font=font(12))
    if not picked:
        draw.text((30, 151), "No images or observations were pinned on this visit.", fill=MUTED, font=font(15))
    if len(visit["evidence"]) > 12:
        draw.text((29, height - 24), "One page shows the first twelve pinned records.", fill=OCHRE, font=font(12))
    return card


def visit_sheet(store, key, version_=None):
    with store.open() as lib:
        head = lib.get(key, "visits")
        visit = head if version_ is None else lib.revision(key, version_)
        return proof_summary(lib, visit)

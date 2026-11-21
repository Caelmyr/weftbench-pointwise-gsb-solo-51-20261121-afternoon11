"""HTTP smoke test for the nap tool against the verification library."""
import json
import sys

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

from weftbench.web import create_app  # noqa: E402

app = create_app("/workspace/verify_nap/data")
client = app.test_client()

options = client.get("/api/options").get_json()
assert "nap" in options["tools"], options["tools"]
print("PASS: nap listed in /api/options ->", options["tools"]["nap"])

# Find the two rake captures and a fabric.
fabrics = client.get("/api/fabrics").get_json()["fabrics"]
fabric_id = fabrics[0]["id"]
detail = client.get(f"/api/fabrics/{fabric_id}").get_json()
rakes = [c for c in detail["captures"] if c["role"] == "rake"]
assert len(rakes) >= 2, detail["captures"]
current, reference = rakes[0], rakes[1]
print("rake captures:", [c["title"] for c in rakes[:2]])

payload = {
    "tool": "nap",
    "settings": {
        "reference": reference["id"], "light": "right", "reference_light": "left",
        "box": [0.04, 0.04, 0.92, 0.92], "cells": 70,
    },
    "registered": True,
}

preview = client.post(f"/api/captures/{current['id']}/reading/preview", json=payload)
assert preview.status_code == 200, preview.get_data(as_text=True)
body = preview.get_json()
assert body["tool"] == "nap" and "map" in body["proofs"] and "pair" in body["proofs"]
print("PASS: preview 200, proofs:", body["proofs"], "| counts:", body["analysis"]["cell_counts"])

map_png = client.post(f"/api/captures/{current['id']}/reading/preview/map.png", json=payload)
pair_png = client.post(f"/api/captures/{current['id']}/reading/preview/pair.png", json=payload)
assert map_png.status_code == 200 and map_png.headers["Content-Type"] == "image/png"
assert pair_png.status_code == 200 and pair_png.headers["Content-Type"] == "image/png"
print("PASS: preview renders map.png (%d bytes) and pair.png (%d bytes)" % (
    len(map_png.data), len(pair_png.data)))

saved = client.post(f"/api/captures/{current['id']}/reading", json={**payload, "title": "HTTP nap reading"})
assert saved.status_code == 201, saved.get_data(as_text=True)
record = saved.get_json()
roles = sorted(a["role"] for a in record["assets"])
assert roles == ["map", "pair"], roles
print("PASS: saved reading 201 with assets:", roles)

# Invalid (non-opposite) lights must be a 400 with a clear field.
bad = client.post(f"/api/captures/{current['id']}/reading/preview",
                  json={**payload, "settings": {**payload["settings"], "reference_light": "right"}})
assert bad.status_code == 400 and bad.get_json()["error"]["field"] == "reference_light"
print("PASS: non-opposite lights rejected:", bad.get_json()["error"]["message"])

print("\nHTTP SMOKE: ALL PASS")

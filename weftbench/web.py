"""Run the WeftBench image library on one local Flask server."""
from __future__ import annotations

from functools import wraps
from io import BytesIO
from pathlib import Path
from threading import BoundedSemaphore

from flask import Blueprint, Flask, Response, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from . import calibration_review, catalog, details, findings, gauge, inspection, intake_review, packets, proofs, readings, relaxation, visits
from .contracts import (
    FACES, MARK_TYPES, MAX_PACKET_BYTES, ROLES, STATES, WEAVES,
    WorkbenchError, asset_label, choice, identifier, integer, mapping, point, slug,
)
from .demo import demo_gallery_photo, seed_training
from .raster import png, probe, rgba
from .store import Store
from .visual import annotated

ROOT = Path(__file__).resolve().parents[1]


def data():
    return mapping(request.get_json(), "request")


def truth(selector, default=True):
    value = request.args.get(selector)
    if value is None:
        return default
    return choice(value, selector, {"true", "false"}) == "true"


def key(raw, field="record"):
    return identifier(raw, field)


def bitmap(image):
    result = send_file(BytesIO(png(image)), mimetype="image/png", max_age=0, conditional=False)
    result.headers["X-Content-Type-Options"] = "nosniff"
    return result


def file_result(owner, attachment, download):
    extensions = {
        "image/jpeg": ".jpg", "image/png": ".png", "image/tiff": ".tif", "image/webp": ".webp",
        "application/pdf": ".pdf", "application/zip": ".zip", "text/csv": ".csv", "application/json": ".json",
    }
    canonical = slug(owner.get("title"), default="weftbench")
    hint = canonical + "-" + attachment.role + extensions.get(attachment.mime, "")
    response = send_file(attachment.path, mimetype=attachment.mime, download_name=hint, conditional=True,
                         as_attachment=download, etag=attachment.sha256, max_age=0)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def protect_heavy(parallel):
    active = BoundedSemaphore(parallel)

    def wrapper(action):
        @wraps(action)
        def hold(*args, **kwargs):
            if not active.acquire(timeout=0.2):
                raise WorkbenchError("The image workbench is busy. Wait for the current rendering.", status=429)
            try:
                return action(*args, **kwargs)
            finally:
                active.release()
        return hold

    return wrapper


def catalog_routes(api, store, heavy):
    @api.get("/fabrics")
    def fabrics():
        return {"fabrics": catalog.list_fabrics(store, archived=truth("archived", False))}

    @api.post("/fabrics")
    def create_fabric():
        return catalog.save_fabric(store, data()), 201

    @api.get("/fabrics/<fabric>")
    def fabric_record(fabric):
        return catalog.fabric_detail(store, key(fabric, "fabric"))

    @api.put("/fabrics/<fabric>")
    def revise_fabric(fabric):
        return catalog.save_fabric(store, data(), key(fabric, "fabric"))

    @api.post("/fabrics/<fabric>/captures")
    @heavy
    def add_capture(fabric):
        return catalog.intake(store, key(fabric, "fabric"), request.form.to_dict(), request.files.get("image")), 201

    @api.post("/fabrics/<fabric>/captures/preflight")
    @heavy
    def check_capture(fabric):
        result, _ = intake_review.review(store, key(fabric, "fabric"), request.form.to_dict(), request.files.get("image"))
        return result

    @api.post("/fabrics/<fabric>/captures/preflight.png")
    @heavy
    def render_capture_check(fabric):
        _, evidence = intake_review.review(store, key(fabric, "fabric"), request.form.to_dict(), request.files.get("image"))
        return bitmap(intake_review.render_exposure(*evidence))

    @api.put("/captures/<capture>")
    def revise_capture(capture):
        return catalog.edit_capture(store, key(capture, "capture"), data())

    @api.post("/captures/<capture>/plane/preview")
    @heavy
    def preview_plane(capture):
        values, _ = catalog.approve_plane(store, key(capture, "capture"), data(), preview=True)
        return values

    @api.post("/captures/<capture>/plane/preview.png")
    @heavy
    def preview_plane_render(capture):
        _, image = catalog.approve_plane(store, key(capture, "capture"), data(), preview=True)
        return bitmap(image)

    @api.post("/captures/<capture>/plane")
    @heavy
    def retain_plane(capture):
        return catalog.approve_plane(store, key(capture, "capture"), data()), 201

    @api.post("/fabrics/<fabric>/observations")
    def add_annotation(fabric):
        return catalog.save_mark(store, key(fabric, "fabric"), data()), 201

    @api.put("/fabrics/<fabric>/observations/<mark>")
    def revise_annotation(fabric, mark):
        return catalog.save_mark(store, key(fabric, "fabric"), data(), key(mark, "observation"))


def image_routes(api, store, heavy):
    @api.get("/captures/<capture>/display.png")
    @heavy
    def display(capture):
        plane = identifier(request.args.get("plane"), "calibration", optional=True)
        registered = truth("registered", True)
        marked = truth("marks", True)
        with store.open() as lib:
            selected = catalog.view(lib, key(capture, "capture"), plane_key=plane,
                                    registered=None if registered else False)
        if marked:
            return bitmap(annotated(selected.surface, selected.marks))
        return bitmap(rgba(selected.surface))

    @api.post("/captures/<capture>/tone.png")
    @heavy
    def tone(capture):
        form = data()
        with store.open() as lib:
            current = readings.parse_context(lib, key(capture, "capture"), form)
        return bitmap(inspection.corrected_display(current, form.get("settings", {})))

    @api.post("/captures/<capture>/probe")
    def sample(capture):
        form = data()
        spot = point(form.get("point"), "probe position")
        radius = integer(form.get("radius"), "probe radius", 0, 21, default=3)
        with store.open() as lib:
            current = readings.parse_context(lib, key(capture, "capture"), form)
        reading = probe(current.surface, spot, radius, current.visible)
        if current.scale is not None:
            reading["millimetres"] = [round(float((position + 0.5) * scale), 5)
                                      for position, scale in zip(reading["working_pixel"], current.scale)]
        reading["coordinate_basis"] = current.basis
        return reading

    @api.post("/captures/<capture>/reading/preview")
    @heavy
    def preview(capture):
        return readings.preview(store, key(capture, "capture"), data())

    @api.post("/captures/<capture>/reading/preview/<role>.png")
    @heavy
    def render_reading(capture, role):
        return bitmap(readings.rendered_preview(store, key(capture, "capture"), data(), role))

    @api.post("/captures/<capture>/reading")
    @heavy
    def save_reading(capture):
        return readings.save_reading(store, key(capture, "capture"), data()), 201

    studies = {"findings": findings.saved_findings, "gauge": gauge.gauge, "relaxation": relaxation.audit}

    def image_study(study, capture, *, save):
        method = studies.get(study)
        if method is None:
            raise WorkbenchError("Choose a supported image study.", "study", status=404)
        return method(store, key(capture, "capture"), data(), save=save)

    @api.post("/captures/<capture>/study/<study>/preview")
    @heavy
    def study_preview(capture, study):
        report, _ = image_study(study, capture, save=False)
        return {"analysis": report}

    @api.post("/captures/<capture>/study/<study>/preview.png")
    @heavy
    def study_preview_render(capture, study):
        _, image = image_study(study, capture, save=False)
        return bitmap(image)

    @api.post("/captures/<capture>/study/<study>")
    @heavy
    def save_study(capture, study):
        return image_study(study, capture, save=True), 201

    @api.post("/captures/<capture>/details/preview.png")
    @heavy
    def preview_details(capture):
        _, image = details.collect(store, key(capture, "capture"), data())
        return bitmap(image)

    @api.post("/captures/<capture>/details")
    @heavy
    def save_details(capture):
        return details.collect(store, key(capture, "capture"), data(), save=True), 201

    @api.get("/details/<record>/audit")
    @heavy
    def audit_details(record):
        return details.audit_record(store, key(record, "detail sheet"))

    @api.put("/readings/<reading>/decision")
    def revise_reading(reading):
        return readings.review_state(store, key(reading, "reading"), data())

    @api.post("/planes/checkpoints/preview.png")
    @heavy
    def check_preview():
        _, image = calibration_review.diagnose(store, data(), save=False)
        return bitmap(image)

    @api.post("/planes/checkpoints")
    @heavy
    def save_checkpoints():
        return calibration_review.diagnose(store, data(), save=True), 201

    @api.post("/readings/<reading>/swatch")
    def swatch(reading):
        return readings.approve_swatch(store, key(reading, "reading"), data()), 201

    @api.put("/swatches/<swatch>/retire")
    def retire_swatch(swatch):
        return readings.retire_swatch(store, key(swatch, "swatch"), data())

    @api.get("/fabrics/<fabric>/swatches")
    def swatches(fabric):
        return {"swatches": readings.swatch_book(store, key(fabric, "fabric"), retired=truth("retired", False))}

    @api.get("/swatches/pair")
    def swatch_comparison():
        comparison, _ = readings.compare_swatches(store, key(request.args.get("first"), "first swatch"),
                                                  key(request.args.get("second"), "second swatch"))
        return comparison

    @api.get("/swatches/pair.png")
    def swatch_comparison_image():
        _, image = readings.compare_swatches(store, key(request.args.get("first"), "first swatch"),
                                             key(request.args.get("second"), "second swatch"))
        return bitmap(image)


def archive_routes(api, store, heavy):
    @api.get("/fabrics/<fabric>/timeline")
    def see_timeline(fabric):
        return visits.timeline(store, key(fabric, "fabric"), request.args.get("limit"))

    @api.post("/fabrics/<fabric>/visits")
    def visit(fabric):
        return visits.save_visit(store, key(fabric, "fabric"), data()), 201

    @api.put("/fabrics/<fabric>/visits/<entry>")
    def amend_visit(fabric, entry):
        return visits.update_visit(store, key(fabric, "fabric"), key(entry, "visit"), data())

    @api.get("/visits/<entry>/proof.png")
    @heavy
    def visit_proof(entry):
        number = request.args.get("version")
        selected = catalog.version(number) if number is not None else None
        return bitmap(visits.visit_sheet(store, key(entry, "visit"), selected))

    @api.post("/captures/<capture>/proof/preview.png")
    @heavy
    def preview_proof(capture):
        _, image = proofs.normal_proof(store, key(capture, "capture"), data(), save=False)
        return bitmap(image)

    @api.post("/captures/<capture>/proof")
    @heavy
    def proof(capture):
        return proofs.normal_proof(store, key(capture, "capture"), data(), save=True), 201

    @api.post("/fabrics/<fabric>/contact/preview.png")
    @heavy
    def contact_preview(fabric):
        return bitmap(proofs.contact(store, key(fabric, "fabric"), data(), save=False))

    @api.post("/fabrics/<fabric>/contact")
    @heavy
    def contact_sheet(fabric):
        return proofs.contact(store, key(fabric, "fabric"), data(), save=True), 201

    @api.post("/fabrics/<fabric>/pack")
    @heavy
    def package(fabric):
        return packets.save_bundle(store, key(fabric, "fabric"), data()), 201

    @api.post("/received")
    @heavy
    def receive():
        return packets.receive_bundle(store, request.files.get("archive")), 201

    @api.get("/packages")
    def all_packages():
        return packets.pack_index(store)

    @api.get("/records/<record>")
    def read_record(record):
        desired = request.args.get("version")
        selected = catalog.version(desired) if desired is not None else None
        return catalog.review_case(store, key(record), version_=selected)

    @api.get("/records/<record>/integrity")
    def integrity(record):
        with store.open() as lib:
            return {"assets": lib.integrity(key(record))}

    @api.get("/assets/<record>/<role>")
    def serve_attachment(record, role):
        with store.open() as lib:
            found = lib.get(key(record))
            file = lib.asset(found["id"], asset_label(role))
        return file_result(found, file, truth("download", file.mime in {
            "application/zip", "text/csv", "application/json"}))


def create_app(directory):
    app = Flask("weftbench", static_folder=str(ROOT / "static"), static_url_path="/ui")
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_PACKET_BYTES + 4096, MAX_FORM_MEMORY_SIZE=384 * 1024,
        MAX_FORM_PARTS=32, JSON_SORT_KEYS=False,
    )
    store = Store(directory)
    app.extensions["weftbench_store"] = store
    api = Blueprint("api", __name__, url_prefix="/api")
    heavy = protect_heavy(2)
    catalog_routes(api, store, heavy)
    image_routes(api, store, heavy)
    archive_routes(api, store, heavy)

    @api.get("/health")
    def health():
        with store.open() as lib:
            lib.conn.execute("SELECT 1")
        return {"status": "ready"}

    @api.get("/options")
    def options():
        return {
            "roles": ROLES, "faces": FACES, "weaves": WEAVES,
            "mark_types": MARK_TYPES, "observation_states": STATES,
            "tools": readings.inventory(), "visit_types": visits.VISIT_KINDS,
            "visit_outcomes": visits.OUTCOMES,
        }

    @api.post("/training")
    @heavy
    def training():
        return seed_training(store), 201

    @api.get("/training-example.png")
    def training_example():
        return send_file(BytesIO(demo_gallery_photo()), mimetype="image/png", max_age=3600)

    @app.errorhandler(WorkbenchError)
    def known(error):
        return jsonify(error.public()), error.status

    @app.errorhandler(BadRequest)
    def invalid(_error):
        return jsonify({"error": {"message": "The request could not be read as a valid form or JSON."}}), 400

    @app.errorhandler(RequestEntityTooLarge)
    def oversize(_error):
        return jsonify({"error": {"message": "The uploaded data exceeds the local intake limit."}}), 413

    @app.after_request
    def response_headers(response):
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    app.register_blueprint(api)
    return app

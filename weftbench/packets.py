"""Transport retained local fabric proofs with bounded binary validation."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import PurePosixPath
import re
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from PIL import Image, UnidentifiedImageError

from .contracts import (
    MAX_ATTACHMENT_BYTES, MAX_PACKET_BYTES, MAX_PACKET_MEMBERS, WorkbenchError,
    identifier, identifiers, integer, limit, mapping, sequence, text,
)
from .proofs import Paper, contact_pages, evidence_rows, gather_entries, to_pdf
from .raster import fit_image, jpeg, png
from .store import SHA_RE, dump

SCHEMA = "weftbench-evidence-v1"
RECEIPT = "weftbench.json"
NUMBERED_PATH = re.compile(r"^evidence/[0-9]{3}\.jpg$")
PACKET_LIMIT = 38
ENTRY_KINDS = {"captures", "readings", "swatches", "proofs", "details", "cut_plans"}


@dataclass(frozen=True)
class PacketFiles:
    manifest: dict
    contents: dict[str, bytes]
    first_page: bytes | None = None

    @property
    def title(self):
        return self.manifest["title"]

    @property
    def digest(self):
        return sha256(self.contents[RECEIPT]).hexdigest()

    def zip(self):
        out = BytesIO()
        with ZipFile(out, "w", compression=ZIP_DEFLATED, compresslevel=4) as bundle:
            for name, body in self.contents.items():
                bundle.writestr(name, body)
        archive = out.getvalue()
        if len(archive) > MAX_PACKET_BYTES:
            raise limit("The evidence pack exceeds the transferable size; select fewer images.")
        return archive


def image_entry(origin, adjusted, asset_role, key, payload):
    return {
        "filename": key,
        "title": origin["title"],
        "record": origin["id"], "type": origin["kind"], "version": origin["version"],
        "image": asset_role, "width": adjusted.width, "height": adjusted.height,
        "image_sha256": sha256(payload).hexdigest(),
    }


def entry_assets(rows, files):
    material = []
    for index, (picture, source, asset) in enumerate(rows, start=1):
        name = f"evidence/{index:03d}.jpg"
        usable = fit_image(picture, 1360, 1100)
        payload = jpeg(usable, quality=89)
        files[name] = payload
        material.append(image_entry(source, usable, asset.role, name, payload))
    return material


def create_files(title, sources, paper):
    pages = contact_pages(sources, paper, title, (2, 2))
    documents = {"contact.pdf": to_pdf(pages), "index.csv": evidence_rows(sources)}
    declarations = entry_assets(sources, documents)
    checks = {
        name: {"sha256": sha256(contents).hexdigest(), "bytes": len(contents)}
        for name, contents in documents.items()
    }
    manifest = {"format": SCHEMA, "title": title, "assets": checks, "entries": declarations}
    documents[RECEIPT] = dump(manifest).encode("utf-8")
    return PacketFiles(manifest, documents, png(fit_image(pages[0], 945, 1150)))


def valid_member(path):
    if path in {RECEIPT, "contact.pdf", "index.csv"}:
        return True
    pure = PurePosixPath(path)
    return not pure.is_absolute() and ".." not in pure.parts and NUMBERED_PATH.fullmatch(path) is not None


def registered_files(bundle):
    items = bundle.infolist()
    if not 4 <= len(items) <= MAX_PACKET_MEMBERS:
        raise WorkbenchError("The evidence archive has an unsupported number of entries.", "archive")
    sizes = {}
    for one in items:
        if one.is_dir() or not valid_member(one.filename) or one.filename in sizes:
            raise WorkbenchError("The bundle has unsupported or repeated file names.", "archive")
        if one.flag_bits & 1:
            raise WorkbenchError("Read the evidence ZIP without password protection.", "archive")
        if one.file_size < 1 or one.file_size > MAX_ATTACHMENT_BYTES:
            raise limit("A transferable attachment is too large or empty.")
        sizes[one.filename] = one.file_size
    if sum(sizes.values()) > MAX_PACKET_BYTES * 2:
        raise limit("The expanded evidence archive is too large.")
    required = {RECEIPT, "contact.pdf", "index.csv"}
    if not required <= set(sizes):
        raise WorkbenchError("The bundle is missing its index, PDF or evidence manifest.", "archive")
    return sizes


def safe_binary(bundle, path, sizes):
    value = bundle.read(path)
    if len(value) != sizes[path]:
        raise WorkbenchError("An attachment length does not agree with the ZIP directory.", "archive")
    return value


def require_sha(value, field):
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise WorkbenchError("The evidence manifest contains an invalid digest.", field)
    return value


def clean_manifest(data, sizes):
    try:
        supplied = json.loads(data)
    except (json.JSONDecodeError, UnicodeError, RecursionError):
        raise WorkbenchError("The evidence manifest is not valid JSON.", "archive") from None
    parsed = mapping(supplied, "manifest")
    if parsed.get("format") != SCHEMA:
        raise WorkbenchError("This archive is not a supported local WeftBench evidence bundle.", "archive")
    title = text(parsed.get("title"), "bundle title", 96)
    checks = mapping(parsed.get("assets"), "attachment declarations")
    if set(checks) != set(sizes) - {RECEIPT}:
        raise WorkbenchError("Archive files do not agree with the attachment manifest.", "archive")
    for key, value in checks.items():
        record = mapping(value, "attachment")
        require_sha(record.get("sha256"), "archive")
        count = integer(record.get("bytes"), "attachment length", 1, MAX_ATTACHMENT_BYTES)
        if count != sizes[key]:
            raise WorkbenchError("An attachment has a different byte length than declared.", "archive")
    entries = []
    pictures = set()
    for item in sequence(parsed.get("entries"), "image entries", 1, 20):
        declared = mapping(item, "image declaration")
        path = declared.get("filename")
        if not isinstance(path, str) or not NUMBERED_PATH.fullmatch(path) or path in pictures or path not in checks:
            raise WorkbenchError("Every image declaration must match exactly one numbered attachment.", "archive")
        pictures.add(path)
        role = declared.get("type")
        if not isinstance(role, str) or role not in ENTRY_KINDS:
            raise WorkbenchError("The manifest identifies an unsupported record type.", "archive")
        width = integer(declared.get("width"), "image width", 1, 1360)
        height = integer(declared.get("height"), "image height", 1, 1100)
        image_sha = require_sha(declared.get("image_sha256"), "image digest")
        if image_sha != checks[path]["sha256"]:
            raise WorkbenchError("An image has mismatching declaration and asset digests.", "archive")
        entries.append({
            "filename": path, "type": role, "width": width, "height": height,
            "title": text(declared.get("title"), "source title", 95),
            "record": identifier(declared.get("record"), "retained source"),
            "version": integer(declared.get("version"), "recorded revision", 1, 99999999),
            "image": text(declared.get("image"), "image role", 56),
            "image_sha256": image_sha,
        })
    if pictures != set(checks) - {"contact.pdf", "index.csv"}:
        raise WorkbenchError("At least one included image has no source declaration.", "archive")
    return {"format": SCHEMA, "title": title, "assets": checks, "entries": entries}


def image_matches(raw, entry):
    try:
        with Image.open(BytesIO(raw)) as image:
            if image.format != "JPEG" or image.size != (entry["width"], entry["height"]):
                raise WorkbenchError("A numbered preview is not the declared JPEG size.", "archive")
            image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise WorkbenchError("A numbered preview is unreadable.", "archive") from None


def decode_bundle(payload):
    if not payload or len(payload) > MAX_PACKET_BYTES:
        raise limit("Select a non-empty local ZIP below the transferable size limit.")
    try:
        with ZipFile(BytesIO(payload)) as bundle:
            sizes = registered_files(bundle)
            contents = {RECEIPT: safe_binary(bundle, RECEIPT, sizes)}
            manifest = clean_manifest(contents[RECEIPT], sizes)
            for filename, expected in manifest["assets"].items():
                value = safe_binary(bundle, filename, sizes)
                if sha256(value).hexdigest() != expected["sha256"]:
                    raise WorkbenchError("A transferred attachment failed the manifest checksum.", "archive")
                contents[filename] = value
            for entry in manifest["entries"]:
                image_matches(contents[entry["filename"]], entry)
            if not contents["contact.pdf"].startswith(b"%PDF-"):
                raise WorkbenchError("The contact proof is not a PDF file.", "archive")
    except (BadZipFile, RuntimeError, NotImplementedError, EOFError):
        raise WorkbenchError("The supplied ZIP could not be read completely.", "archive") from None
    return PacketFiles(manifest, contents)


def attach_bundle(lib, record, files, archive):
    owner = record["id"]
    lib.attach(owner, "zip", archive, "application/zip")
    lib.attach(owner, "manifest", files.contents[RECEIPT], "application/json")
    lib.attach(owner, "pdf", files.contents["contact.pdf"], "application/pdf")
    lib.attach(owner, "csv", files.contents["index.csv"], "text/csv")
    if files.first_page:
        lib.attach(owner, "page", files.first_page, "image/png")
    for index, item in enumerate(files.manifest["entries"], 1):
        lib.attach(owner, f"image-{index:03d}", files.contents[item["filename"]], "image/jpeg")


def save_bundle(store, parent, raw):
    values = mapping(raw)
    title = text(values.get("title"), "bundle title", 90, default="Portable fabric inspection")
    keys = identifiers(values.get("entries"), "selected evidence", 16)
    paper = Paper.parse(values)
    with store.open() as lib:
        fabric = lib.get(parent, "fabrics")
        inputs = gather_entries(lib, parent, keys)
    files = create_files(title, inputs, paper)
    archive = files.zip()
    with store.open(write=True) as lib:
        lib.require_version(lib.get(parent, "fabrics"), fabric["version"])
        if lib.count("packets", parent=parent) >= PACKET_LIMIT:
            raise limit("This fabric has reached its portable evidence limit.")
        for _, source, _ in inputs:
            lib.require_version(lib.get(source["id"]), source["version"])
        record = lib.add("packets", {
            "title": title, "format": SCHEMA, "status": "outgoing",
            "checksum": files.digest, "entries": files.manifest["entries"],
        }, parent=parent, action="evidence bundle retained")
        lib.cite(record["id"], fabric, "fabric exported")
        for _, source, _ in inputs:
            lib.cite(record["id"], source, "portable image evidence")
        attach_bundle(lib, record, files, archive)
        return lib.public(record, links=True)


def receive_bundle(store, file):
    if file is None:
        raise WorkbenchError("Choose a local evidence ZIP.", "archive")
    data = file.read(MAX_PACKET_BYTES + 1)
    verified = decode_bundle(data)
    with store.open(write=True) as lib:
        seen = lib.list("received", limit=PACKET_LIMIT * 2)
        for record in seen:
            if record["checksum"] == verified.digest:
                return lib.public(record) | {"duplicate": True}
        if len(seen) >= PACKET_LIMIT * 2:
            raise limit("The received evidence library has reached its capacity.")
        record = lib.add("received", {
            "title": verified.title, "format": SCHEMA,
            "status": "verified", "checksum": verified.digest,
            "entries": verified.manifest["entries"],
        }, action="received bundle verified and retained")
        attach_bundle(lib, record, verified, data)
        return lib.public(record)


def pack_index(store):
    with store.open() as lib:
        return {
            kind: [lib.public(value) for value in lib.list(kind, limit=PACKET_LIMIT * 2)]
            for kind in ("received", "packets")
        }

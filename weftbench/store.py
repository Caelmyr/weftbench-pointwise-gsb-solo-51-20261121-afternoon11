"""Versioned local library with independently addressable image bytes."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import secrets
import sqlite3
import tempfile
from typing import Iterable

from .contracts import LOCKED_FIELDS, WorkbenchError, asset_label, conflict, missing, stamp

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    key TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    parent TEXT REFERENCES records(key),
    version INTEGER NOT NULL CHECK(version > 0),
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS by_parent ON records(kind,parent,updated);
CREATE TABLE IF NOT EXISTS revisions (
    key TEXT NOT NULL REFERENCES records(key),
    version INTEGER NOT NULL,
    body TEXT NOT NULL,
    happened TEXT NOT NULL,
    action TEXT NOT NULL,
    PRIMARY KEY (key,version)
);
CREATE TABLE IF NOT EXISTS assets (
    owner TEXT NOT NULL REFERENCES records(key),
    role TEXT NOT NULL,
    digest TEXT NOT NULL,
    mime TEXT NOT NULL,
    length INTEGER NOT NULL,
    PRIMARY KEY(owner,role)
);
CREATE TABLE IF NOT EXISTS links (
    owner TEXT NOT NULL REFERENCES records(key),
    source TEXT NOT NULL REFERENCES records(key),
    role TEXT NOT NULL,
    version INTEGER NOT NULL,
    PRIMARY KEY(owner,source,role),
    FOREIGN KEY(source,version) REFERENCES revisions(key,version)
);
CREATE INDEX IF NOT EXISTS by_source ON links(source,owner);
"""

SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def dump(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False,
                      ensure_ascii=False)


def validated_payload(payload):
    if not isinstance(payload, dict) or LOCKED_FIELDS.intersection(payload):
        raise ValueError("Invalid record payload")
    return dump(payload)


def decode(row):
    if row is None:
        return None
    return json.loads(row["body"]) | {
        "id": row["key"],
        "kind": row["kind"],
        "parent": row["parent"],
        "version": row["version"],
        "created": row["created"],
        "updated": row["updated"],
    }


@dataclass(frozen=True)
class StoredAsset:
    owner: str
    role: str
    sha256: str
    mime: str
    length: int
    path: Path

    def key(self):
        return (self.owner, self.role)

    def public(self):
        return {
            "role": self.role,
            "sha256": self.sha256,
            "mime": self.mime,
            "bytes": self.length,
        }


class BlobStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def address(self, digest):
        if not isinstance(digest, str) or not SHA_RE.fullmatch(digest):
            raise ValueError("Invalid stored object digest")
        return self.root / digest[:2] / digest[2:]

    def put(self, payload):
        digest = sha256(payload).hexdigest()
        address = self.address(digest)
        if address.is_file():
            if address.stat().st_size != len(payload):
                raise WorkbenchError("An existing local attachment has an invalid length.", status=500)
            return digest
        address.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=address.parent, delete=False) as temp:
            temp.write(payload)
            staged = Path(temp.name)
        try:
            staged.replace(address)
        finally:
            staged.unlink(missing_ok=True)
        return digest

    def available(self, digest):
        address = self.address(digest)
        if not address.is_file():
            raise missing("local attachment")
        return address

    def audit(self, digest, size):
        address = self.address(digest)
        if not address.is_file():
            return "missing"
        if address.stat().st_size != size:
            return "different length"
        with address.open("rb") as content:
            actual = sha256()
            while chunk := content.read(1024 * 1024):
                actual.update(chunk)
        return "verified" if actual.hexdigest() == digest else "different content"


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database = self.directory / "weftbench.sqlite3"
        self.blobs = BlobStore(self.directory / "objects")
        conn = self.connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
        finally:
            conn.close()

    def connect(self):
        conn = sqlite3.connect(self.database, timeout=12, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=12000")
        return conn

    @contextmanager
    def open(self, *, write=False):
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            value = Library(connection, self.blobs)
            yield value
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()


class Library:
    def __init__(self, connection, blobs):
        self.conn = connection
        self.blobs = blobs

    def get(self, key, kind=None, *, parent=None, optional=False):
        row = self.conn.execute("SELECT * FROM records WHERE key=?", (key,)).fetchone()
        if row is None or (kind and row["kind"] != kind) or (parent and row["parent"] != parent):
            if optional:
                return None
            raise missing(kind or "record")
        return decode(row)

    def list(self, kind, *, parent=None, limit=120, oldest=False):
        where = ["kind = ?"]
        arguments = [kind]
        if parent is not None:
            where.append("parent = ?")
            arguments.append(parent)
        arguments.append(limit)
        order = "ASC" if oldest else "DESC"
        command = ("SELECT * FROM records WHERE " + " AND ".join(where)
                   + f" ORDER BY updated {order},rowid {order} LIMIT ?")
        return [decode(row) for row in self.conn.execute(command, arguments)]

    def count(self, kind, *, parent=None):
        if parent is None:
            sql, arguments = "SELECT count(*) FROM records WHERE kind=?", (kind,)
        else:
            sql = "SELECT count(*) FROM records WHERE kind=? AND parent=?"
            arguments = (kind, parent)
        return int(self.conn.execute(sql, arguments).fetchone()[0])

    def scoped_totals(self, parent, kinds: Iterable[str]):
        rows = self.conn.execute(
            "SELECT kind,count(*) AS total FROM records WHERE parent=? GROUP BY kind", (parent,)
        )
        totals = dict.fromkeys(kinds, 0)
        for row in rows:
            if row["kind"] in totals:
                totals[row["kind"]] = row["total"]
        return totals

    def add(self, kind, payload, *, parent=None, action="created"):
        current = stamp()
        key = secrets.token_hex(12)
        raw = validated_payload(payload)
        self.conn.execute(
            "INSERT INTO records(key,kind,parent,version,created,updated,body) VALUES(?,?,?,1,?,?,?)",
            (key, kind, parent, current, current, raw),
        )
        self.conn.execute(
            "INSERT INTO revisions(key,version,body,happened,action) VALUES(?,1,?,?,?)",
            (key, raw, current, action),
        )
        return self.get(key, kind)

    def require_version(self, record, version):
        if version != record["version"]:
            raise conflict("This record changed in another operation. Reload its current version.")

    def revise(self, record, version, payload, *, action="edited"):
        self.require_version(record, version)
        current = stamp()
        new_version = version + 1
        raw = validated_payload(payload)
        changed = self.conn.execute(
            "UPDATE records SET version=?,updated=?,body=? WHERE key=? AND version=?",
            (new_version, current, raw, record["id"], version),
        )
        if changed.rowcount != 1:
            raise conflict("This record changed during the operation.")
        self.conn.execute(
            "INSERT INTO revisions(key,version,body,happened,action) VALUES(?,?,?,?,?)",
            (record["id"], new_version, raw, current, action),
        )
        return self.get(record["id"])

    def revision(self, key, version):
        current = self.get(key)
        row = self.conn.execute(
            "SELECT * FROM revisions WHERE key=? AND version=?", (key, version)
        ).fetchone()
        if row is None:
            raise missing("record revision")
        return json.loads(row["body"]) | {
            "id": current["id"], "kind": current["kind"], "parent": current["parent"],
            "created": current["created"], "version": version, "updated": row["happened"],
        }

    def history(self, key):
        self.get(key)
        revisions = self.conn.execute(
            "SELECT version,body,happened,action FROM revisions WHERE key=? ORDER BY version DESC",
            (key,),
        )
        return [{
            "version": row["version"], "record": json.loads(row["body"]),
            "date": row["happened"], "action": row["action"],
        } for row in revisions]

    def attach(self, owner, role, payload, mime):
        role = asset_label(role)
        digest = self.blobs.put(payload)
        self.conn.execute(
            "INSERT INTO assets(owner,role,digest,mime,length) VALUES(?,?,?,?,?)",
            (owner, role, digest, mime, len(payload)),
        )
        return self.asset(owner, role)

    def load_asset(self, row):
        return StoredAsset(
            owner=row["owner"], role=row["role"], sha256=row["digest"],
            mime=row["mime"], length=row["length"], path=self.blobs.address(row["digest"]),
        )

    def asset(self, owner, role, *, optional=False):
        role = asset_label(role)
        row = self.conn.execute(
            "SELECT * FROM assets WHERE owner=? AND role=?", (owner, role),
        ).fetchone()
        if row is None:
            if optional:
                return None
            raise missing("attachment")
        asset = self.load_asset(row)
        self.blobs.available(asset.sha256)
        return asset

    def assets(self, owner):
        rows = self.conn.execute("SELECT * FROM assets WHERE owner=? ORDER BY role", (owner,))
        return [self.load_asset(row) for row in rows]

    def integrity(self, owner):
        self.get(owner)
        return [
            asset.public() | {"check": self.blobs.audit(asset.sha256, asset.length)}
            for asset in self.assets(owner)
        ]

    def cite(self, owner, source, role):
        self.conn.execute(
            "INSERT INTO links(owner,source,role,version) VALUES(?,?,?,?)",
            (owner, source["id"], role, source["version"]),
        )

    def citations(self, owner):
        rows = self.conn.execute(
            "SELECT source,role,version FROM links WHERE owner=? ORDER BY role,source", (owner,),
        )
        return [dict(row) for row in rows]

    def cited_by(self, source, limit=36):
        rows = self.conn.execute(
            """SELECT l.owner AS id, l.role, l.version, r.kind, r.body
                 FROM links l JOIN records r ON l.owner=r.key
               WHERE l.source=? ORDER BY r.updated DESC LIMIT ?""",
            (source, limit),
        )
        return [{
            "id": row["id"], "role": row["role"], "source_version": row["version"],
            "kind": row["kind"], "title": json.loads(row["body"]).get("title", ""),
        } for row in rows]

    def includes_blob(self, kind, parent, digest):
        sql = """SELECT r.key FROM records r JOIN assets a ON a.owner=r.key
                 WHERE r.kind=? AND r.parent=? AND a.role='original' AND a.digest=? LIMIT 1"""
        row = self.conn.execute(sql, (kind, parent, digest)).fetchone()
        return None if row is None else row["key"]

    def public(self, item, *, attachments=True, links=False):
        value = item.copy()
        if attachments:
            value["assets"] = [entry.public() for entry in self.assets(item["id"])]
        if links:
            value["sources"] = self.citations(item["id"])
        return value

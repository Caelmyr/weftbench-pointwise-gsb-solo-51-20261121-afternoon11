"""Validate user inputs shared by the image and evidence tools."""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

MAX_FILE_BYTES = 24 * 1024 * 1024
MAX_PIXELS = 5200000
MAX_EDGE = 2700
MAX_FABRICS = 96
MAX_CAPTURES = 120
MAX_MARKS = 180
MAX_RESULTS = 120
MAX_PACKET_BYTES = 36 * 1024 * 1024
MAX_ATTACHMENT_BYTES = 18 * 1024 * 1024
MAX_PACKET_MEMBERS = 52

ROLES = {
    "face": "Frontal reflected light",
    "backlight": "Transmitted light",
    "rake": "Raking light",
    "reference": "Neutral proof reference",
}
FACES = {"face": "Front", "reverse": "Reverse"}
WEAVES = {"plain": "Plain weave", "twill": "Twill", "sateen": "Sateen", "open": "Open weave", "unassigned": "Unassigned"}
MARK_TYPES = {
    "ignore": "Exclude from analysis",
    "fold": "Fold or occlusion",
    "stain": "Colour anomaly",
    "puncture": "Hole under review",
    "slub": "Yarn irregularity",
    "seam": "Sewn line",
    "cut": "Cut edge",
    "selvedge": "Selvedge",
    "area": "Inspection area",
    "proof": "Colour proof region",
}
LINE_TYPES = {"seam", "cut", "selvedge"}
MASK_TYPES = {"ignore", "fold"}
STATES = {"new": "Unreviewed", "check": "Keep for review", "accepted": "Accepted", "closed": "Closed"}
ACTIONS = {"active": "Active", "archived": "Archived"}
LOCKED_FIELDS = {"id", "parent", "kind", "created", "updated", "version"}
ID_RE = re.compile(r"^[0-9a-f]{24}$")
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ASSET_RE = re.compile(r"^[a-z][a-z0-9.-]{0,55}$")


class WorkbenchError(Exception):
    def __init__(self, message: str, field: str | None = None, *, status: int = 400):
        super().__init__(message)
        self.message = message
        self.field = field
        self.status = status

    def public(self) -> dict[str, Any]:
        item = {"message": self.message}
        if self.field:
            item["field"] = self.field
        return {"error": item}


def missing(kind: str) -> WorkbenchError:
    return WorkbenchError(f"The requested {kind} was not found.", status=404)


def conflict(message: str) -> WorkbenchError:
    return WorkbenchError(message, status=409)


def limit(message: str) -> WorkbenchError:
    return WorkbenchError(message, status=413)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def mapping(value: Any, field: str = "request") -> dict:
    if not isinstance(value, dict):
        raise WorkbenchError(f"{field.capitalize()} must be an object.", field)
    return value


def sequence(value: Any, field: str, lower: int, upper: int) -> list:
    if not isinstance(value, list) or not lower <= len(value) <= upper:
        raise WorkbenchError(f"Supply {lower} to {upper} values for {field}.", field)
    return value


def text(value: Any, field: str, maximum: int, default: str | None = None,
         *, multiline: bool = False) -> str:
    if value is None:
        value = default
    if not isinstance(value, str):
        raise WorkbenchError(f"Enter {field}.", field)
    value = value.strip()
    unsafe = any(ord(char) < 32 and not (multiline and char in "\r\n\t") for char in value)
    if (not value and default is None) or len(value) > maximum or unsafe:
        raise WorkbenchError(f"Enter {field} with no more than {maximum} characters.", field)
    return value


def choice(value: Any, field: str, available, default: str | None = None) -> str:
    result = default if value is None else value
    if not isinstance(result, str) or result not in available:
        raise WorkbenchError(f"Choose a supported {field}.", field)
    return result


def real(value: Any, field: str, lo: float, hi: float,
         *, default: float | None = None) -> float:
    if value is None:
        value = default
    if isinstance(value, bool) or value is None:
        raise WorkbenchError(f"Enter a number for {field}.", field)
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        raise WorkbenchError(f"Enter a number for {field}.", field) from None
    if not math.isfinite(result) or not lo <= result <= hi:
        raise WorkbenchError(f"{field.capitalize()} must be between {lo:g} and {hi:g}.", field)
    return result


def integer(value: Any, field: str, lo: int, hi: int, *, default: int | None = None) -> int:
    result = real(value, field, lo, hi, default=default)
    if int(result) != result:
        raise WorkbenchError(f"Enter an integer for {field}.", field)
    return int(result)


def flag(value: Any, field: str, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise WorkbenchError(f"{field.capitalize()} must be true or false.", field)


def identifier(value: Any, field: str = "record", *, optional: bool = False) -> str | None:
    if optional and not value:
        return None
    if not isinstance(value, str) or ID_RE.fullmatch(value) is None:
        raise WorkbenchError(f"Pick a valid {field}.", field)
    return value


def identifiers(value: Any, field: str, maximum: int, *, minimum: int = 1) -> list[str]:
    keys = [identifier(item, field) for item in sequence(value, field, minimum, maximum)]
    if len(set(keys)) != len(keys):
        raise WorkbenchError(f"Choose distinct items for {field}.", field)
    return keys


def asset_label(value: Any) -> str:
    if not isinstance(value, str) or ASSET_RE.fullmatch(value) is None:
        raise missing("asset")
    return value


def date(value: Any, field: str, *, optional: bool = False) -> str | None:
    if optional and not value:
        return None
    if not isinstance(value, str) or DAY_RE.fullmatch(value) is None:
        raise WorkbenchError(f"Enter a calendar date for {field}.", field)
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        raise WorkbenchError(f"Enter a calendar date for {field}.", field) from None


def point(value: Any, field: str = "point", *, lower: float = 0, upper: float = 1) -> list[float]:
    items = sequence(value, field, 2, 2)
    return [real(n, field, lower, upper) for n in items]


def points(value: Any, field: str, lower: int, upper: int) -> list[list[float]]:
    result = [point(item, field) for item in sequence(value, field, lower, upper)]
    if len({tuple(item) for item in result}) != len(result):
        raise WorkbenchError(f"Separate the points of {field}.", field)
    return result


def box(value: Any, field: str = "image area", *, default=None) -> list[float]:
    if value is None:
        value = default
    x, y, w, h = [real(item, field, 0, 1) for item in sequence(value, field, 4, 4)]
    if w <= 0.001 or h <= 0.001 or x + w > 1.0000001 or y + h > 1.0000001:
        raise WorkbenchError(f"Keep {field} inside the image and give it a visible area.", field)
    return [x, y, w, h]


def require_support(total: int, good: int, label: str, minimum: int,
                    *, fraction: float = 0.18) -> None:
    if good < minimum or (total and good / total < fraction):
        raise WorkbenchError(f"{label} needs a larger clearly visible area.")


def csv_cell(value):
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def slug(value, default="weftbench", length=55):
    if not isinstance(value, str):
        return default
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-_")
    return safe[:length] or default

"""Original capture intake and numerical views of oriented pixels."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
import warnings

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from scipy import ndimage

from .contracts import MAX_EDGE, MAX_FILE_BYTES, MAX_PIXELS, WorkbenchError, limit

Image.MAX_IMAGE_PIXELS = MAX_PIXELS * 3
SUPPORTED = {"JPEG": "image/jpeg", "PNG": "image/png", "TIFF": "image/tiff", "WEBP": "image/webp"}
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


@dataclass(frozen=True)
class Ingested:
    working: Image.Image
    original_type: str
    source_size: tuple[int, int]
    orientation: int
    profile: str
    frame_count: int

    @property
    def working_size(self):
        return self.working.size

    def receipt(self):
        return {
            "original_size": list(self.source_size),
            "extent": list(self.working_size),
            "orientation": self.orientation,
            "color_profile": self.profile,
            "original_type": self.original_type,
            "frames_in_source": self.frame_count,
        }


@dataclass(frozen=True)
class Raster:
    linear: np.ndarray
    valid: np.ndarray

    @property
    def extent(self):
        return (self.linear.shape[1], self.linear.shape[0])

    @property
    def bright(self):
        return self.valid & (np.max(self.linear, axis=2) >= 0.975)

    @property
    def dark(self):
        return self.valid & (luma(self.linear) <= 0.0045)

    @property
    def readable(self):
        return self.valid & ~self.bright & ~self.dark


def working_size(width, height):
    factor = min(1.0, MAX_EDGE / max(width, height), math.sqrt(MAX_PIXELS / (width * height)))
    return max(1, int(width * factor)), max(1, int(height * factor))


def alpha_pixels(image):
    if image.mode == "P" and "transparency" in image.info:
        return image.convert("RGBA").getchannel("A")
    if "A" in image.getbands():
        return image.getchannel("A")
    return Image.new("L", image.size, 255)


def rgb_profile(image):
    colorspace = image.info.get("icc_profile")
    if not colorspace:
        return image.convert("RGB"), "sRGB assumed"
    if len(colorspace) > 1024 * 1024:
        return image.convert("RGB"), "sRGB assumed; embedded profile too large"
    try:
        profile = ImageCms.ImageCmsProfile(BytesIO(colorspace))
        destination = ImageCms.createProfile("sRGB")
        data = image if image.mode in {"L", "RGB", "CMYK"} else image.convert("RGB")
        changed = ImageCms.profileToProfile(data, profile, destination, outputMode="RGB",
                                             renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)
        return changed, "embedded profile converted to sRGB"
    except (OSError, ValueError, ImageCms.PyCMSError):
        return image.convert("RGB"), "sRGB assumed; unsupported embedded profile"


def decode_image(payload, *, max_bytes=MAX_FILE_BYTES):
    if not payload:
        raise WorkbenchError("The selected image is empty.", "image")
    if len(payload) > max_bytes:
        raise limit("Keep each source photo below the upload limit.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as original:
                mime = SUPPORTED.get(original.format)
                if not mime:
                    raise WorkbenchError("Choose a PNG, JPEG, TIFF or WebP photograph.", "image")
                if min(original.size) < 24:
                    raise WorkbenchError("Choose an image at least 24 pixels wide and high.", "image")
                extent = original.size
                page_count = int(getattr(original, "n_frames", 1))
                if page_count > 1:
                    raise WorkbenchError("Export a single photo from a multi-page image first.", "image")
                orientation = int(original.getexif().get(274, 1) or 1)
                image = ImageOps.exif_transpose(original)
                alpha = alpha_pixels(image)
                rgb, profile = rgb_profile(image)
                rgb.putalpha(alpha)
                reduced = working_size(*rgb.size)
                if reduced != rgb.size:
                    rgb = rgb.resize(reduced, Image.Resampling.LANCZOS)
                return Ingested(rgb, mime, extent, orientation, profile, page_count)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise limit("The source image has too many decoded pixels.") from None
    except (OSError, UnidentifiedImageError, ValueError, OverflowError):
        raise WorkbenchError("This photo could not be decoded.", "image") from None


def open_work(path):
    with Image.open(path) as picture:
        return from_image(picture)


def from_image(image):
    data = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    clear = data[:, :, 3] >= 0.995
    linear = to_linear(data[:, :, :3])
    return Raster(np.where(clear[:, :, None], linear, 0), clear)


def to_linear(encoded):
    return np.where(encoded <= 0.04045, encoded / 12.92,
                    ((encoded + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * linear ** (1 / 2.4) - 0.055)


def luma(linear):
    return np.einsum("...k,k->...", linear, LUMA, optimize=False)


def to_lab(linear):
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz = np.asarray(linear) @ np.array([
        [0.4124564, 0.2126729, 0.0193339],
        [0.3575761, 0.7151522, 0.1191920],
        [0.1804375, 0.0721750, 0.9503041],
    ], dtype=np.float32)
    colors = xyz / white
    transform = np.where(colors > 0.008856, np.cbrt(colors), 7.787 * colors + 16 / 116)
    x, y, z = np.moveaxis(transform, -1, 0)
    return np.stack((116 * y - 16, 500 * (x - y), 200 * (y - z)), axis=-1)


def encoded_image(values, supported=None, *, linear=True):
    rgb = to_srgb(values) if linear else np.clip(values, 0, 1)
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    data = np.rint(255 * rgb).astype(np.uint8)
    if supported is None:
        return Image.fromarray(data, "RGB")
    opacity = np.where(supported, 255, 0).astype(np.uint8)
    return Image.fromarray(np.dstack((data, opacity)), "RGBA")


def rgba(surface):
    return encoded_image(surface.linear, surface.valid)


def png(image):
    out = BytesIO()
    image.save(out, format="PNG", compress_level=4)
    return out.getvalue()


def jpeg(image, quality=92):
    out = BytesIO()
    opaque(image).save(out, format="JPEG", quality=quality, optimize=True, subsampling=0)
    return out.getvalue()


def opaque(image, background=(31, 34, 40)):
    if image.mode == "RGB":
        return image.copy()
    top = image.convert("RGBA")
    fill = Image.new("RGBA", top.size, (*background, 255))
    return Image.alpha_composite(fill, top).convert("RGB")


def fit_image(image, width, height):
    copy = image.copy()
    copy.thumbnail((max(1, int(width)), max(1, int(height))), Image.Resampling.LANCZOS)
    return copy


def crop_raster(surface, bounds):
    x0, y0, x1, y1 = bounds
    return Raster(surface.linear[y0:y1, x0:x1], surface.valid[y0:y1, x0:x1])


def blurred_masked(values, valid, width, *, coverage=0.68):
    if width <= 0.0001:
        return np.where(valid, values, 0), valid.copy()
    basis = valid.astype(np.float32)
    weighted = ndimage.gaussian_filter(values * basis, width, mode="nearest")
    weight = ndimage.gaussian_filter(basis, width, mode="nearest")
    result = weighted / np.maximum(weight, 1e-8)
    return result, (weight >= coverage) & valid


def gradients(values, valid, *, trim=1):
    safe = ndimage.binary_erosion(valid, iterations=trim) if trim else valid
    dy = ndimage.sobel(values, axis=0, mode="nearest") / 8
    dx = ndimage.sobel(values, axis=1, mode="nearest") / 8
    return np.where(safe, dx, 0), np.where(safe, dy, 0), safe


def stable_scale(values, lower=5, upper=95):
    if not len(values):
        return None
    lo, hi = np.percentile(values, [lower, upper])
    return float(lo), float(hi)


def quantiles(values, q=(5, 25, 50, 75, 95), *, digits=6):
    values = np.asarray(values)
    if values.size == 0:
        return []
    return [round(float(v), digits) for v in np.percentile(values, q)]


def rgb_sample(linear_values):
    if len(linear_values) == 0:
        return None
    average = np.median(linear_values, axis=0)
    encoded = to_srgb(average)
    return {
        "linear_rgb": [round(float(v), 6) for v in average],
        "srgb_bytes": np.rint(255 * encoded).astype(int).tolist(),
        "lab_d65": np.round(to_lab(average), 4).tolist(),
    }


def whitened(surface, black, white, gamma, *, grayscale=False):
    samples = luma(surface.linear) if grayscale else surface.linear
    mapped = np.clip((to_srgb(samples) - black) / (white - black), 0, 1)
    return encoded_image(mapped ** (1 / gamma), surface.valid, linear=False)


def probe(surface, at, radius, available=None):
    width, height = surface.extent
    x, y = min(width - 1, int(at[0] * width)), min(height - 1, int(at[1] * height))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    bounds = np.s_[y0:y1, x0:x1]
    mask = surface.valid[bounds]
    if available is not None:
        mask = mask & available[bounds]
    raw = surface.linear[bounds][mask]
    return {
        "working_pixel": [x, y], "bounds": [x0, y0, x1, y1],
        "readable": int((surface.readable[bounds] & mask).sum()), "visible": int(mask.sum()),
        "sampled": int(mask.size), "color": rgb_sample(raw),
        "luminance": quantiles(luma(raw)),
    }


def component_labels(active, *, diagonal=True):
    structure = np.ones((3, 3), dtype=bool) if diagonal else ndimage.generate_binary_structure(2, 1)
    labels, count = ndimage.label(active, structure=structure)
    return labels, count, np.bincount(labels.ravel()), ndimage.find_objects(labels)


def read_fraction(mask, *, denominator=None):
    whole = mask.size if denominator is None else denominator
    return round(float(mask.sum()) / max(1, whole), 6)


def radial_energy(dx, dy, valid):
    return np.hypot(dx[valid], dy[valid]) if np.any(valid) else np.array([], dtype=np.float32)

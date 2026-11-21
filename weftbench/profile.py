"""Segmented real-scale yarn density curves across the cloth width.

Each station is solved independently: a station whose image support is held by
clamps, out of focus, or too weak to separate a repeat is returned blank.
Autocorrelation supplies the repeat period while a half-lag sub-peak and the
neighbouring stations decide whether a strong spectral peak is a stripe
harmonic (reject the doubled count) or a genuinely tighter woven section.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
from PIL import ImageDraw
from scipy import signal

from .contracts import WorkbenchError, integer, mapping, real, require_support
from .geometry import cut_rectangle
from .inspection import common_box
from .raster import blurred_masked, luma, quantiles, rgba
from .visual import LILAC, OCHRE, PINK, TEAL, histogram_plot, label, legend

HARMONIC_RATIO = 0.62
CONTINUITY = 0.15


@dataclass(frozen=True)
class ProfileRequest:
    selection: list
    stations: int
    min_period: float
    max_period: float
    smooth: float

    @classmethod
    def parse(cls, raw):
        body = mapping(raw)
        smallest = real(body.get("min_period"), "smallest pitch in pixels", 2.2, 24, default=3.0)
        largest = real(body.get("max_period"), "largest pitch in pixels", 5, 150, default=38)
        if largest <= smallest * 1.25:
            raise WorkbenchError("Choose a wider pitch interval.", "max_period")
        return cls(
            common_box(body),
            integer(body.get("stations"), "number of width stations", 4, 24, default=10),
            smallest, largest,
            real(body.get("background"), "shading radius in pixels", 4, 95, default=25),
        )


def widest_run(valid):
    bounded = np.r_[False, valid, False].astype(np.int8)
    edges = np.diff(bounded)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, 0
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])


def autocorrelation_peaks(series, lo, hi):
    windowed = series * np.hanning(len(series))
    corr = np.correlate(windowed, windowed, mode="full")[len(series) - 1:]
    corr = corr / max(corr[0], 1e-12)
    band = (np.arange(len(corr)) >= lo) & (np.arange(len(corr)) <= hi)
    indices, props = signal.find_peaks(corr[band], height=0.14, prominence=0.06)
    if not len(indices):
        return np.array([]), np.array([]), corr
    lag_index = np.where(band)[0][indices]
    heights = props["peak_heights"]
    refined = []
    for k in lag_index:
        if 0 < k < len(corr) - 1:
            y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
            denom = y0 - 2 * y1 + y2
            refined.append(k + 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else float(k))
        else:
            refined.append(float(k))
    order = np.argsort(-heights)
    return np.array(refined)[order], heights[order], corr


def station_evidence(profile, usable, pmin, pmax):
    """Local periodic evidence for one station, or (None, blank reason)."""
    if usable.size == 0 or usable.mean() < 0.55:
        return None, "held for low visible coverage"
    start, stop = widest_run(usable)
    if stop - start < max(4 * pmin, 28):
        return None, "held for a short visible run"
    n = len(profile)
    lo, hi = max(2, int(np.floor(pmin))), min(n - 3, int(np.ceil(pmax)))
    if hi <= lo + 1:
        return None, "held for an unusable pitch band"
    lags, heights, corr = autocorrelation_peaks(profile[start:stop], lo, hi)
    if not len(lags):
        return None, "held without a repeat peak"
    freq, power = signal.periodogram(profile, window="hann")
    inside = (freq >= 1.0 / pmax) & (freq <= 1.0 / pmin)
    fb, pb = freq[inside], power[inside]
    spectral = pb.max() / max(np.median(pb), 1e-12)
    if spectral < 6.0 or heights[0] < 0.24:
        return None, "held for weak or unclear periodicity"
    # A genuine repeat (a) re-appears near its second or third lag and
    # (b) concentrates the in-band spectrum; chance peaks in unreadable,
    # blurred, or unresolved imagery fail at least one of these checks.
    if not _repeats(lags, heights) or _concentration(pb) < 0.60:
        return None, "held because the repeat peaks do not separate"
    fft_pitch = 1.0 / fb[np.argmax(pb)]
    # The periodogram may lock onto a half-pitch harmonic; accept either the
    # matching lag or half of an autocorrelation fundamental lag.
    if min(min(abs(lags[:3] - fft_pitch) / fft_pitch),
           min(abs(lags[:3] / 2 - fft_pitch) / fft_pitch)) > 0.22:
        return None, "held because the repeat peaks do not separate"
    return {"lags": lags, "heights": heights, "correlation": corr}, "ok"


def _repeats(lags, heights):
    top, top_h = lags[0], heights[0]
    for multiple in (2.0, 3.0):
        near = np.abs(lags - top * multiple) <= max(1.6, 0.14 * top * multiple)
        if np.any(near) and heights[near].max() >= max(0.14, 0.3 * top_h):
            return True
    return False


def _concentration(power):
    strongest = np.argsort(-power)[:3]
    return float(power[strongest].sum() / max(power.sum(), 1e-12))


def half_lag_pair(evidence):
    lags, heights = evidence["lags"], evidence["heights"]
    top, top_h = lags[0], heights[0]
    half = top / 2.0
    if half < 2.0:
        return None
    corr = evidence["correlation"]

    def value_at(lag):
        if not 0 <= lag <= len(corr) - 2:
            return 0.0
        k = int(np.floor(lag))
        frac = lag - k
        return float(corr[k] * (1 - frac) + corr[k + 1] * frac)

    # A half-period that lands between integer lags has no discrete peak, so
    # read its strength by interpolation and require a corroborating peak near
    # the 1.5x lag (a harmonic lattice) before treating it as a finer repeat.
    near = np.abs(lags - half) <= max(1.2, 0.12 * half)
    half_h = heights[near].max() if np.any(near) else value_at(half)
    lattice = np.abs(lags - top * 1.5) <= max(1.5, 0.13 * top * 1.5)
    if not np.any(near) and not np.any(lattice):
        return None
    if half_h < 0.45 * top_h:
        return None
    return float(half), float(half_h), top, top_h


def resolve_axis(profiles, usable, pmin, pmax):
    count = len(profiles)
    evidence, reasons = [], []
    for series, good in zip(profiles, usable, strict=True):
        found, why = station_evidence(series, good, pmin, pmax)
        evidence.append(found)
        reasons.append(why)
    # First-pass dominant repeats give the continuity reference without any
    # smoothing of the final values; real tight stations keep their own read.
    dominant = [ev["lags"][0] if ev else None for ev in evidence]
    results = []
    for i, ev in enumerate(evidence):
        if ev is None:
            results.append({"pitch_px": None, "strength": None, "status": reasons[i]})
            continue
        pair = half_lag_pair(ev)
        if pair is None:
            results.append({"pitch_px": float(ev["lags"][0]), "strength": float(ev["heights"][0]),
                            "status": "visible repeat"})
            continue
        half, half_h, top, top_h = pair
        neighbours = [dominant[j] for j in (i - 1, i + 1)
                      if 0 <= j < count and dominant[j] is not None]
        if half_h / top_h >= HARMONIC_RATIO:
            pitch, strength, why = half, half_h, "fine repeat supported inside the station"
        elif neighbours and all(abs(top - x) / x <= CONTINUITY for x in neighbours):
            pitch, strength, why = top, top_h, "half-pitch stripe peak rejected by continuity"
        elif neighbours and any(abs(half - x) / x <= CONTINUITY for x in neighbours):
            pitch, strength, why = half, half_h, "tighter weave confirmed against neighbours"
        else:
            results.append({"pitch_px": None, "strength": None,
                            "status": "held while half-pitch doubling is unresolved"})
            continue
        results.append({"pitch_px": float(pitch), "strength": float(strength), "status": why})
    return results


def station_traces(value, good, axis, edges):
    """Per-station median trace along the measured axis.

    Each width station keeps its own image band, so the weft trace (along the
    rows) and the warp trace (along the columns) describe the same cloth strip.
    """
    traces, supports = [], []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        band_value = value[:, left:right]
        band_good = good[:, left:right]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median = np.nanmedian(np.where(band_good, band_value, np.nan), axis=axis)
        support = band_good.mean(axis=axis)
        ready = support >= 0.55
        if np.any(ready):
            filled = np.interp(np.arange(len(median)), np.flatnonzero(ready), median[ready])
        else:
            filled = np.nan_to_num(median)
        slow = np.convolve(filled, np.ones(9) / 9, mode="same")
        traces.append(filled - slow)
        supports.append(ready)
    return traces, supports


def density_axis(ctx, value, good, edges, axis, options):
    traces, supports = station_traces(value, good, axis, edges)
    rows = resolve_axis(traces, supports, options.min_period, options.max_period)
    scale = ctx.scale[1 - axis] if ctx.scale is not None else None
    curve = []
    for i, row in enumerate(rows):
        pitch_px = row["pitch_px"]
        pitch_mm = pitch_px * scale if (pitch_px is not None and scale is not None) else None
        curve.append({
            "station": i + 1,
            "count_per_100_mm": round(100 / pitch_mm, 3) if pitch_mm else None,
            "pitch_px": round(pitch_px, 4) if pitch_px is not None else None,
            "pitch_mm": round(pitch_mm, 5) if pitch_mm is not None else None,
            "strength": round(row["strength"], 4) if row["strength"] is not None else None,
            "status": row["status"],
        })
    return curve


def paint_map(ctx, box, edges, curves):
    frame = rgba(ctx.surface)
    draw = ImageDraw.Draw(frame)
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(*OCHRE, 255), width=3)
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        blank = any(axis[i]["count_per_100_mm"] is None for axis in curves.values())
        shade = LILAC if blank else TEAL
        draw.rectangle((int(left), y0, int(right - 1), y1 - 1), outline=(*shade, 255))
        center = (int(left) + int(right)) // 2
        label(draw, (max(x0 + 2, center - 12), y0 + 4), str(i + 1), fill=shade,
              maximum=max(18, int(right - left) - 6))
        if blank:
            draw.line((int(left) + 3, y1 - 14, int(right) - 3, y0 + 14), fill=(*PINK, 255), width=2)
    return legend(frame, "Width stations for segmented yarn density", [
        ("Station with a reading", TEAL), ("Station left blank", LILAC), ("Selected region", OCHRE),
    ])


def density_profile(ctx, raw):
    options = ProfileRequest.parse(raw)
    if ctx.scale is None:
        raise WorkbenchError("Save a physical calibration before reading real-scale yarn density.", "plane")
    surface = ctx.surface
    box = cut_rectangle(options.selection, surface.extent)
    x0, y0, x1, y1 = box
    if min(x1 - x0, y1 - y0) < 48 or (x1 - x0) * (y1 - y0) > 2500000:
        raise WorkbenchError("Select at least 48 pixels on each side and at most 2.5 million pixels.", "box")
    if (x1 - x0) / options.stations < 24:
        raise WorkbenchError("Use fewer stations so each station is at least 24 pixels wide.", "stations")
    usable = ctx.readable[y0:y1, x0:x1]
    require_support(int(usable.size), int(usable.sum()), "Density profile", 900, fraction=0.34)
    value = luma(surface.linear[y0:y1, x0:x1])
    shade, good = blurred_masked(value, usable, options.smooth)
    detail = value - shade
    edges = np.rint(np.linspace(0, x1 - x0, options.stations + 1)).astype(int)
    curves = {
        "warp": density_axis(ctx, detail, good, edges, 0, options),
        "weft": density_axis(ctx, detail, good, edges, 1, options),
    }
    report = {
        "box": options.selection, "stations": options.stations,
        "source_dimensions": list(surface.extent),
        "station_bounds": [[int(x0 + left), y0, int(x0 + right), y1]
                           for left, right in zip(edges[:-1], edges[1:], strict=True)],
        "band": [options.min_period, options.max_period],
        "support_pixels": int(good.sum()),
        "coverage_percentiles": quantiles(good.mean(axis=0)),
        **curves,
        "note": "Blank stations mark held, obscured, or unresolvable evidence. Stripe harmonics never "
                "double a neighbour's count; a locally supported finer repeat is retained.",
    }
    map_art = paint_map(ctx, box, edges, curves)
    chart = histogram_plot(
        [{"points": [[row["station"], row["count_per_100_mm"]] for row in curves["warp"]], "color": TEAL},
         {"points": [[row["station"], row["count_per_100_mm"]] for row in curves["weft"]], "color": OCHRE}],
        "Yarn density per width station", x_label="Station across the cloth width",
        y_label="Threads per 100 mm",
    )
    return report, {"map": map_art, "chart": chart}

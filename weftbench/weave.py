"""Optical thread pitch, cross-width skew, and periodic shading."""
from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
from PIL import ImageDraw
from scipy import ndimage, signal

from .contracts import WorkbenchError, choice, integer, mapping, real, require_support
from .geometry import box_mask, cut_rectangle
from .inspection import common_box
from .raster import blurred_masked, gradients, luma, quantiles, rgba
from .visual import INK, LILAC, OCHRE, PINK, TEAL, histogram_plot, label, legend

AXES = {"cross-roll": 0, "along-roll": 1}


@dataclass(frozen=True)
class YarnRequest:
    selection: list
    min_period: float
    max_period: float
    angle: float
    smooth: float

    @classmethod
    def parse(cls, raw):
        body = mapping(raw)
        smallest = real(body.get("min_period"), "smallest pitch in pixels", 2.2, 24, default=3.0)
        largest = real(body.get("max_period"), "largest pitch in pixels", 5, 150, default=38)
        if largest <= smallest * 1.25:
            raise WorkbenchError("Choose a wider pitch interval.", "max_period")
        return cls(
            common_box(body), smallest, largest,
            real(body.get("angle"), "allowed skew", 3, 40, default=17),
            real(body.get("background"), "shading radius in pixels", 4, 95, default=25),
        )


def power_map(profile, mask):
    height, width = profile.shape
    weight = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32) * mask
    average = float(np.sum(profile * weight) / max(1e-8, weight.sum()))
    energy = np.abs(np.fft.fftshift(np.fft.fft2((profile - average) * weight))) ** 2
    u, v = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(width)), np.fft.fftshift(np.fft.fftfreq(height)))
    return energy, u, v


def summarize_axis(energy, u, v, options, axis, scale):
    angle = np.degrees(np.arctan2(v, u))
    normal_angle = 0 if axis == 0 else 90
    difference = (angle - normal_angle + 90) % 180 - 90
    frequency = np.hypot(u, v)
    range_ = (frequency >= 1 / options.max_period) & (frequency <= 1 / options.min_period)
    possible = range_ & (np.abs(difference) <= options.angle)
    visible = np.where(possible, energy, 0)
    maxima = (visible == ndimage.maximum_filter(visible, size=(5, 5))) & (visible > 0)
    candidates = np.argwhere(maxima & ((v > 0) | ((v == 0) & (u > 0))))
    candidates = sorted(candidates, key=lambda pair: -float(visible[tuple(pair)]))
    credible = candidates[:5]
    noise = float(np.median(visible[possible])) if np.any(possible) else 0
    peaks = []
    for y, x in credible:
        level = float(visible[y, x])
        if level <= max(1e-8, noise * 5):
            continue
        raw_period = float(1 / frequency[y, x])
        angle_rad = float(np.arctan2(v[y, x], u[y, x]))
        width_mm = None
        if scale is not None:
            width_mm = raw_period * float(np.linalg.norm(np.array([np.cos(angle_rad), np.sin(angle_rad)]) * scale))
        peaks.append({
            "pitch_px": round(raw_period, 4),
            "pitch_mm": round(width_mm, 5) if width_mm is not None else None,
            "crossing_count_per_100_mm": round(100 / width_mm, 3) if width_mm else None,
            "thread_direction_degrees": round((math.degrees(angle_rad) + 90) % 180, 4),
            "strength_over_noise": round(level / max(noise, 1e-8), 2),
        })
    lo, hi = 1 / options.max_period, 1 / options.min_period
    edges = np.linspace(lo, hi, 50)
    group = np.clip(np.searchsorted(edges, frequency, side="right") - 1, 0, len(edges) - 2)
    n = np.bincount(group[possible].ravel(), minlength=len(edges) - 1)
    sums = np.bincount(group[possible].ravel(), weights=energy[possible], minlength=len(edges) - 1)
    values = sums / np.maximum(1, n)
    normalized = values / max(1e-8, float(values.max()))
    result = [
        [round(2 / (start + finish), 5), round(float(strength), 6) if count else None]
        for start, finish, strength, count in zip(edges, edges[1:], normalized, n)
    ]
    return peaks, sorted(result, key=lambda piece: piece[0])


def yarn_density(ctx, raw):
    options = YarnRequest.parse(raw)
    surface = ctx.surface
    x0, y0, x1, y1 = cut_rectangle(options.selection, surface.extent)
    if min(x1 - x0, y1 - y0) < 40 or (x1 - x0) * (y1 - y0) > 2500000:
        raise WorkbenchError("Select 40 or more pixels on each side and at most 2.5 million pixels.", "box")
    usable = ctx.readable[y0:y1, x0:x1]
    require_support(int(usable.size), int(usable.sum()), "Thread pitch", 700, fraction=0.42)
    value = luma(surface.linear[y0:y1, x0:x1])
    shade, good = blurred_masked(value, usable, options.smooth)
    detail = value - shade
    spectrum, u, v = power_map(detail, good)
    warp, warp_plot = summarize_axis(spectrum, u, v, options, 0, ctx.scale)
    weft, weft_plot = summarize_axis(spectrum, u, v, options, 1, ctx.scale)
    summary = {
        "box": options.selection, "support_pixels": int(good.sum()),
        "warp": {"candidate_peaks": warp, "profile": warp_plot},
        "weft": {"candidate_peaks": weft, "profile": weft_plot},
        "band": [options.min_period, options.max_period],
        "note": "A strong repeated pitch is a visible image pattern, not automatically the selected weave count.",
    }
    frame = rgba(surface)
    draw = ImageDraw.Draw(frame)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(*OCHRE, 255), width=3)
    label(draw, (x0 + 8, y0 + 8), "Window read for yarn pitch")
    chart = histogram_plot([
        {"points": warp_plot, "color": TEAL}, {"points": weft_plot, "color": OCHRE},
    ], "Warp and weft pitch evidence", x_label="Optical repeat in pixels", y_label="Relative power",
        limits=(0, 1))
    return summary, {"map": legend(frame, "Thread pitch window", [("Selected textile", OCHRE)]), "chart": chart}


def summarize_direction(x, y, use):
    if np.count_nonzero(use) < 45:
        return None
    first, second = x[use], y[use]
    tensor_x = float(np.mean(first * first))
    tensor_y = float(np.mean(second * second))
    crossed = float(np.mean(first * second))
    power = tensor_x + tensor_y
    if power < 1e-6:
        return None
    coherence = math.hypot(tensor_x - tensor_y, 2 * crossed) / power
    normal = math.degrees(0.5 * math.atan2(2 * crossed, tensor_x - tensor_y)) % 180
    return {"dominant_direction": (normal + 90) % 180, "coherence": coherence, "mean_energy": power}


def local_skew(ctx, raw):
    settings = mapping(raw)
    selection = common_box(settings)
    cells = integer(settings.get("cells"), "number of width stations", 3, 18, default=9)
    minimum = real(settings.get("minimum_coherence"), "minimum direction coherence", 0.05, 0.9, default=0.22)
    box_ = cut_rectangle(selection, ctx.surface.extent)
    x0, y0, x1, y1 = box_
    if min((x1 - x0) / cells, y1 - y0) < 12:
        raise WorkbenchError("Choose greater width for these stations or reduce their count.")
    light = luma(ctx.surface.linear)
    filtered, allowed = blurred_masked(light, ctx.readable, 1.15)
    gx, gy, eligible = gradients(filtered, allowed)
    marks = rgba(ctx.surface)
    draw = ImageDraw.Draw(marks)
    records = []
    edges = np.rint(np.linspace(x0, x1, cells + 1)).astype(int)
    for index, (left, right) in enumerate(zip(edges, edges[1:]), 1):
        patch = np.s_[y0:y1, left:right]
        support = eligible[patch]
        reading = summarize_direction(gx[patch], gy[patch], support)
        enough = bool(reading and reading["coherence"] >= minimum and support.mean() >= 0.36)
        angle = round(reading["dominant_direction"], 4) if enough else None
        cx, cy = (right - 1 + left) * 0.5, (y0 + y1 - 1) * 0.5
        shade = TEAL if enough else LILAC
        draw.rectangle((int(left), y0, int(right - 1), y1 - 1), outline=(*shade, 255))
        if enough:
            reach = min(31, (right - left) / 2, (y1 - y0) / 2)
            shift_x = math.cos(math.radians(angle)) * reach
            shift_y = math.sin(math.radians(angle)) * reach
            draw.line((cx - shift_x, cy - shift_y, cx + shift_x, cy + shift_y),
                      fill=(*shade, 255), width=3)
        label(draw, (int(left) + 2, y0 + 4), str(index), fill=shade, maximum=max(16, right - left - 11))
        records.append({
            "station": index, "x": round(float(cx / ctx.surface.extent[0]), 6),
            "x_mm": round(float(cx * ctx.scale[0]), 4) if ctx.scale is not None else None,
            "bounds": [int(left), y0, int(right), y1],
            "dominant_direction_degrees": angle,
            "coherence": round(reading["coherence"], 5) if reading else None,
            "status": "visible direction" if enough else "insufficient contrast or coverage",
        })
    return {
        "box": selection, "stations": records,
        "reading_kind": "dominant thread-axis hints; no thread is extrapolated between cells",
    }, {"map": legend(marks, "Dominant weave direction across the selected width",
                      [("Readable direction", TEAL), ("Left for review", LILAC)])}


def brightness_trace(values, good, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(np.where(good, values, np.nan), axis=axis)
    support = good.mean(axis=axis)
    ready = (support >= 0.36) & np.isfinite(median)
    return median, ready, support


def widest_run(valid):
    bounded = np.r_[False, valid, False].astype(np.int8)
    edges = np.diff(bounded)
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return 0, 0
    index = int(np.argmax(ends - starts))
    return int(starts[index]), int(ends[index])


def banding(ctx, raw):
    data = mapping(raw)
    selection = common_box(data)
    direction = choice(data.get("axis"), "band axis", AXES, "along-roll")
    axis = (1 if direction == "along-roll" else 0)
    trend_width = real(data.get("radius"), "average shading radius", 3, 100, default=17)
    x0, y0, x1, y1 = cut_rectangle(selection, ctx.surface.extent)
    if min(x1 - x0, y1 - y0) < 28:
        raise WorkbenchError("Choose a larger image region for the banding profile.")
    shape = np.s_[y0:y1, x0:x1]
    value = luma(ctx.surface.linear)[shape]
    medians, good, support = brightness_trace(value, ctx.readable[shape], axis)
    if int(good.sum()) < 25:
        raise WorkbenchError("Not enough supported optical row or column samples.")
    x = np.arange(len(medians))
    base = np.interp(x, x[good], medians[good])
    slow = ndimage.gaussian_filter1d(base, trend_width, mode="nearest")
    signal_ = medians - slow
    start, stop = widest_run(good)
    peaks = []
    if stop - start >= 36:
        run = signal_[start:stop]
        freq, energy = signal.periodogram(run, window="hann")
        floor = max(1e-11, float(np.median(energy)))
        important, _ = signal.find_peaks(energy, prominence=floor * 4)
        for index in sorted(important, key=lambda n: -energy[n])[:7]:
            if freq[index] <= 0.0:
                continue
            period = float(1 / freq[index])
            if not 3 <= period <= (stop - start) / 2:
                continue
            scale = ctx.scale[(0 if axis == 0 else 1)] if ctx.scale is not None else None
            peaks.append({
                "period_px": round(period, 4),
                "period_mm": round(float(period * scale), 4) if scale is not None else None,
                "power_over_floor": round(float(energy[index]) / floor, 4),
            })
    if len(x) > 950:
        points_ = sorted(set(np.linspace(0, len(x) - 1, 950).astype(int)))
    else:
        points_ = list(range(len(x)))
    report = {
        "box": selection, "axis": direction, "ready_samples": int(good.sum()),
        "coverage_percentiles": quantiles(support), "continuous_read_window_px": [start, stop],
        "period_candidates": peaks,
        "profile": [[int(i), round(float(signal_[i]), 7) if good[i] else None] for i in points_],
        "note": "Periodic shading is reported only in the longest continuously visible run.",
    }
    chart = histogram_plot([{"points": report["profile"], "color": TEAL}], "Residual shading bands",
                           x_label="Distance in pixels", y_label="Relative light")
    picture = rgba(ctx.surface)
    draw = ImageDraw.Draw(picture)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(*OCHRE, 255), width=2)
    if axis == 1:
        draw.line((x0, y0 + start, x0, y0 + stop), fill=(*PINK, 255), width=4)
    else:
        draw.line((x0 + start, y0, x0 + stop, y0), fill=(*PINK, 255), width=4)
    return report, {"map": legend(picture, "Banding window", [("Selected region", OCHRE), ("Continuous run", PINK)]),
                    "chart": chart}

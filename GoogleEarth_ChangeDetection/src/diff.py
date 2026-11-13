"""Pixel-difference, SSIM, change-mask and annotation utilities."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from .config import DiffSettings


class DiffError(RuntimeError):
    """Raised when a comparison cannot be computed."""


@dataclass
class ChangeRegion:
    index: int
    x: int
    y: int
    width: int
    height: int
    area: int
    mean_abs_diff: float = 0.0
    mean_ssim: float = 1.0

    @property
    def label(self) -> str:
        return f"R{self.index}"


@dataclass
class AnalysisResult:
    reference: np.ndarray
    target: np.ndarray
    diff_gray: np.ndarray
    ssim_map: np.ndarray
    change_mask: np.ndarray
    heatmap: np.ndarray
    ssim_heatmap: np.ndarray
    overlay: np.ndarray
    annotated: np.ndarray
    regions: list[ChangeRegion] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def analyze_pair(
    reference_rgb: np.ndarray,
    target_rgb: np.ndarray,
    valid_mask: np.ndarray,
    settings: DiffSettings,
    *,
    target_resized: bool = False,
) -> AnalysisResult:
    if reference_rgb.shape != target_rgb.shape:
        raise DiffError(
            f"Reference and target shapes differ: {reference_rgb.shape} vs {target_rgb.shape}."
        )
    if reference_rgb.ndim != 3 or reference_rgb.shape[2] != 3:
        raise DiffError("Expected RGB images.")
    height, width = reference_rgb.shape[:2]
    if height < 3 or width < 3:
        raise DiffError("Images are too small for SSIM analysis.")
    if valid_mask.shape[:2] != (height, width):
        raise DiffError("Valid mask dimensions do not match the images.")

    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2GRAY)

    valid = valid_mask.astype(bool).copy()
    border_excluded = _exclude_borders(valid, settings.exclude_border_px)
    valid_for_diff = valid & border_excluded
    if np.count_nonzero(valid_for_diff) < 100:
        raise DiffError("Insufficient valid overlap for a meaningful comparison.")

    brightness_delta = float(
        reference_gray[valid_for_diff].mean() - target_gray[valid_for_diff].mean()
    )
    contrast_reference = float(reference_gray[valid_for_diff].std())
    contrast_target = float(target_gray[valid_for_diff].std())
    target_for_diff = target_gray
    brightness_normalized = False
    if settings.normalize_brightness:
        target_for_diff = _match_brightness(reference_gray, target_gray, valid_for_diff)
        brightness_normalized = True

    diff_gray = cv2.absdiff(reference_gray, target_for_diff)
    color_diff = np.abs(
        reference_rgb.astype(np.int16) - target_rgb.astype(np.int16)
    ).mean(axis=2)

    window = _ssim_window(settings.ssim_win_size, height, width)
    try:
        _, ssim_map = structural_similarity(
            reference_gray,
            target_for_diff,
            full=True,
            data_range=255,
            win_size=window,
            gaussian_weights=False,
        )
    except (ValueError, TypeError) as exc:
        raise DiffError(f"SSIM computation failed: {exc}") from exc
    ssim_map = np.nan_to_num(ssim_map.astype(np.float64), nan=0.0, posinf=1.0, neginf=-1.0)

    if settings.combine_mode == "and":
        condition = (diff_gray >= settings.pixel_threshold) & (
            ssim_map < settings.ssim_threshold
        )
    else:
        condition = (diff_gray >= settings.pixel_threshold) | (
            ssim_map < settings.ssim_threshold
        )
    raw_mask = (condition & valid_for_diff).astype(np.uint8) * 255
    filtered_mask, regions = _filter_regions(
        raw_mask, diff_gray, ssim_map, valid_for_diff, settings
    )

    valid_pixels = int(np.count_nonzero(valid_for_diff))
    total_pixels = int(height * width)
    changed_pixels = int(np.count_nonzero(filtered_mask))
    changed_pct = 100.0 * changed_pixels / valid_pixels if valid_pixels else 0.0
    ssim_below = int(np.count_nonzero((ssim_map < settings.ssim_threshold) & valid_for_diff))
    largest = max((region.area for region in regions), default=0)
    excluded_pixels = int(np.count_nonzero(valid) - np.count_nonzero(valid_for_diff))

    stats: dict[str, Any] = {
        "width": width,
        "height": height,
        "total_pixels": total_pixels,
        "valid_pixels": valid_pixels,
        "valid_pct": round(100.0 * valid_pixels / total_pixels, 3),
        "border_excluded_pct": round(
            100.0 * excluded_pixels / max(1, np.count_nonzero(valid)), 3
        ),
        "changed_pixels": changed_pixels,
        "changed_pct_of_valid": round(changed_pct, 3),
        "changed_pct_of_image": round(
            100.0 * changed_pixels / total_pixels if total_pixels else 0.0, 3
        ),
        "mean_abs_diff": round(float(diff_gray[valid_for_diff].mean()), 3),
        "median_abs_diff": round(float(np.median(diff_gray[valid_for_diff])), 3),
        "p95_abs_diff": round(float(np.percentile(diff_gray[valid_for_diff], 95)), 3),
        "max_abs_diff": int(diff_gray[valid_for_diff].max()),
        "mean_color_diff": round(float(color_diff[valid_for_diff].mean()), 3),
        "ssim_mean": round(float(ssim_map[valid_for_diff].mean()), 4),
        "ssim_p05": round(float(np.percentile(ssim_map[valid_for_diff], 5)), 4),
        "ssim_below_threshold_pct": round(
            100.0 * ssim_below / valid_pixels if valid_pixels else 0.0, 3
        ),
        "brightness_delta": round(brightness_delta, 3),
        "contrast_reference": round(contrast_reference, 3),
        "contrast_target": round(contrast_target, 3),
        "brightness_normalized": brightness_normalized,
        "region_count": len(regions),
        "largest_region_pixels": int(largest),
        "largest_region_pct_of_valid": round(
            100.0 * largest / valid_pixels if valid_pixels else 0.0, 3
        ),
        "pixel_threshold": settings.pixel_threshold,
        "ssim_threshold": settings.ssim_threshold,
        "combine_mode": settings.combine_mode,
        "exclude_border_px": settings.exclude_border_px,
        "ssim_window": window,
        "target_resized": target_resized,
    }
    stats["border_excluded_pct"] = float(stats["border_excluded_pct"])
    stats["image_flags"] = _image_flags(stats)

    heatmap = _render_heatmap(
        diff_gray, valid_for_diff, settings.pixel_threshold, settings.heatmap_percentile
    )
    ssim_heatmap = _render_ssim_heatmap(ssim_map, valid_for_diff)
    overlay, annotated = _render_change_views(
        target_rgb, filtered_mask, regions, settings.max_annotated_regions, stats
    )

    return AnalysisResult(
        reference=reference_rgb,
        target=target_rgb,
        diff_gray=diff_gray,
        ssim_map=ssim_map,
        change_mask=filtered_mask,
        heatmap=heatmap,
        ssim_heatmap=ssim_heatmap,
        overlay=overlay,
        annotated=annotated,
        regions=regions,
        stats=stats,
    )


def _exclude_borders(valid: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return np.ones_like(valid, dtype=bool)
    height, width = valid.shape[:2]
    inner = np.zeros_like(valid, dtype=bool)
    inner[pixels : max(pixels, height - pixels), pixels : max(pixels, width - pixels)] = True
    return inner


def _match_brightness(
    reference_gray: np.ndarray, target_gray: np.ndarray, valid: np.ndarray
) -> np.ndarray:
    reference_values = reference_gray[valid].astype(np.float32)
    target_values = target_gray[valid].astype(np.float32)
    ref_mean, ref_std = float(reference_values.mean()), float(reference_values.std())
    tgt_mean, tgt_std = float(target_values.mean()), float(target_values.std())
    if tgt_std < 1e-6 or ref_std < 1e-6:
        return target_gray
    corrected = (target_gray.astype(np.float32) - tgt_mean) * (ref_std / tgt_std) + ref_mean
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _ssim_window(configured: int, height: int, width: int) -> int:
    maximum = min(configured, height, width)
    if maximum % 2 == 0:
        maximum -= 1
    return max(3, maximum)


def _filter_regions(
    mask: np.ndarray,
    diff_gray: np.ndarray,
    ssim_map: np.ndarray,
    valid: np.ndarray,
    settings: DiffSettings,
) -> tuple[np.ndarray, list[ChangeRegion]]:
    kernel_size = settings.morphology_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = cleaned & (valid.astype(np.uint8) * 255)

    count, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        (cleaned > 0).astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(mask), []

    allowed_labels: list[int] = []
    regions: list[ChangeRegion] = []
    for label in range(1, count):
        x = int(component_stats[label, cv2.CC_STAT_LEFT])
        y = int(component_stats[label, cv2.CC_STAT_TOP])
        region_width = int(component_stats[label, cv2.CC_STAT_WIDTH])
        region_height = int(component_stats[label, cv2.CC_STAT_HEIGHT])
        area = int(component_stats[label, cv2.CC_STAT_AREA])
        if area < settings.min_region_area:
            continue
        allowed_labels.append(label)
        region_mask = labels[y : y + region_height, x : x + region_width] == label
        diff_crop = diff_gray[y : y + region_height, x : x + region_width]
        ssim_crop = ssim_map[y : y + region_height, x : x + region_width]
        regions.append(
            ChangeRegion(
                index=0,
                x=x,
                y=y,
                width=region_width,
                height=region_height,
                area=area,
                mean_abs_diff=float(diff_crop[region_mask].mean()),
                mean_ssim=float(ssim_crop[region_mask].mean()),
            )
        )
    regions.sort(key=lambda region: region.area, reverse=True)
    for index, region in enumerate(regions, start=1):
        region.index = index
    if not allowed_labels:
        return np.zeros_like(mask), []
    filtered = (np.isin(labels, allowed_labels) & valid).astype(np.uint8) * 255
    return filtered, regions


def _render_heatmap(
    diff_gray: np.ndarray, valid: np.ndarray, threshold: int, percentile: float
) -> np.ndarray:
    values = diff_gray[valid]
    if values.size == 0:
        return np.zeros((*diff_gray.shape, 3), dtype=np.uint8)
    vmax = max(float(np.percentile(values, percentile)), float(threshold), 1.0)
    normalized = np.clip(diff_gray.astype(np.float32) / vmax, 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def _render_ssim_heatmap(ssim_map: np.ndarray, valid: np.ndarray) -> np.ndarray:
    dissimilarity = np.clip(1.0 - ssim_map, 0.0, 1.0)
    colored = cv2.applyColorMap((dissimilarity * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def _render_change_views(
    target_rgb: np.ndarray,
    mask: np.ndarray,
    regions: Sequence[ChangeRegion],
    max_annotated_regions: int,
    stats: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    overlay = target_rgb.copy()
    changed = mask > 0
    if np.any(changed):
        overlay[changed] = (
            overlay[changed].astype(np.float32) * 0.5
            + np.array([230, 35, 35], dtype=np.float32) * 0.5
        ).astype(np.uint8)

    annotated = target_rgb.copy()
    height, width = annotated.shape[:2]
    font_scale = max(0.4, min(1.0, width / 1600.0))
    thickness = max(1, round(width / 900.0))
    for region in regions[:max_annotated_regions]:
        padding = max(2, round(width / 500.0))
        top_left = (max(0, region.x - padding), max(0, region.y - padding))
        bottom_right = (
            min(width - 1, region.x + region.width + padding),
            min(height - 1, region.y + region.height + padding),
        )
        cv2.rectangle(annotated, top_left, bottom_right, (255, 40, 40), thickness + 1)
        label = region.label
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = max(0, top_left[0])
        text_y = max(text_size[1] + 3, top_left[1])
        cv2.rectangle(
            annotated,
            (text_x, text_y - text_size[1] - 3),
            (text_x + text_size[0] + 4, text_y + 2),
            (120, 0, 0),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (text_x + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    summary = (
        f"{stats['region_count']} candidate region(s) | "
        f"{stats['changed_pct_of_valid']:.2f}% of valid pixels | "
        f"SSIM {stats['ssim_mean']:.3f}"
    )
    _draw_text_box(annotated, summary, (8, height - 8), font_scale, bottom=True)
    return overlay, annotated


def _draw_text_box(
    image: np.ndarray, text: str, origin: tuple[int, int], font_scale: float, bottom: bool = False
) -> None:
    thickness = 1
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = origin
    box_top = y - text_size[1] - 8 if bottom else y
    box_bottom = y if bottom else y + text_size[1] + 8
    cv2.rectangle(
        image,
        (x, max(0, box_top)),
        (min(image.shape[1] - 1, x + text_size[0] + 10), min(image.shape[0] - 1, box_bottom)),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y - 4 if bottom else y + text_size[1] + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _image_flags(stats: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if stats["valid_pct"] < 70.0:
        flags.append("low_valid_overlap")
    if abs(stats["brightness_delta"]) > 25.0:
        flags.append("large_brightness_shift")
    if stats["changed_pct_of_valid"] > 40.0:
        flags.append("very_high_change_fraction_possible_misalignment")
    if min(stats["contrast_reference"], stats["contrast_target"]) < 8.0:
        flags.append("low_contrast")
    if stats["brightness_normalized"]:
        flags.append("brightness_normalization_applied")
    if stats["region_count"] == 0 and stats["mean_abs_diff"] > stats["pixel_threshold"]:
        flags.append("differences_below_region_area_threshold")
    return flags


def render_comparison(
    result: AnalysisResult,
    reference_label: str,
    target_label: str,
    footer_lines: Sequence[str] = (),
    max_panel_width: int = 1200,
) -> np.ndarray:
    reference = _fit_rgb(result.reference, max_panel_width)
    target = _fit_rgb(result.target, max_panel_width)
    heatmap = _fit_rgb(result.heatmap, max_panel_width)
    annotated = _fit_rgb(result.annotated, max_panel_width)
    panel_width, panel_height = reference.shape[1], reference.shape[0]

    margin = 16
    gap = 12
    header_height = 30
    title_height = 44
    canvas_width = margin * 2 + panel_width * 2 + gap
    wrap_width = max(40, int((canvas_width - margin * 2) / 7.2))
    wrapped_footer: list[str] = []
    for line in footer_lines:
        wrapped_footer.extend(textwrap.wrap(str(line), width=wrap_width) or [""])
    footer_height = max(56, 20 * len(wrapped_footer) + 20)
    canvas_height = (
        title_height
        + margin
        + header_height * 2
        + panel_height * 2
        + gap
        + footer_height
        + margin
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), (28, 30, 34))
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(20, bold=True)
    label_font = _load_font(15, bold=True)
    footer_font = _load_font(13)

    draw.text(
        (margin, 10), "GOOGLE EARTH SCREENSHOT COMPARISON", fill=(240, 240, 240), font=title_font
    )
    draw.text(
        (margin, title_height - 10),
        f"{reference_label}  vs  {target_label}",
        fill=(170, 200, 240),
        font=label_font,
    )

    panels = [
        (reference, reference_label, (0, 0)),
        (target, target_label, (1, 0)),
        (heatmap, "Absolute grayscale difference (heatmap)", (0, 1)),
        (annotated, "Candidate change regions (image differences only)", (1, 1)),
    ]
    for panel, label, (column, row) in panels:
        x = margin + column * (panel_width + gap)
        y = title_height + margin + header_height + row * (panel_height + header_height + gap)
        draw.text((x + 4, y - header_height + 6), label, fill=(220, 220, 220), font=label_font)
        canvas.paste(Image.fromarray(panel, mode="RGB"), (x, y))

    footer_y = (
        title_height
        + margin
        + header_height
        + panel_height * 2
        + header_height
        + gap
        + 8
    )
    draw.line(
        (margin, footer_y - 6, canvas_width - margin, footer_y - 6), fill=(70, 70, 74), width=1
    )
    for index, line in enumerate(wrapped_footer):
        draw.text(
            (margin, footer_y + 4 + index * 20),
            line,
            fill=(190, 190, 190),
            font=footer_font,
        )
    return np.asarray(canvas, dtype=np.uint8)


def _fit_rgb(image: np.ndarray, max_width: int) -> np.ndarray:
    if image.shape[1] <= max_width:
        return image
    scale = max_width / float(image.shape[1])
    size = (max_width, max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(["arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"])
    candidates.extend(["arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"])
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

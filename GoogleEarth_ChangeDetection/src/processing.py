"""Batch processing of validated tile pairs into comparison artefacts."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .alignment import AlignmentResult, align_pair
from .config import AppConfig
from .dataset import DatasetManager, PairInfo
from .diff import AnalysisResult, analyze_pair, render_comparison
from .logging_setup import get_logger
from .utils import (
    DataFileError,
    atomic_write_json,
    display_path,
    load_image_rgb,
    read_json,
    save_image,
    utc_now,
    verify_written_image,
    write_csv,
)

LOGGER = get_logger(__name__)

STATS_COLUMNS = [
    "tile_id",
    "status",
    "year_a",
    "year_b",
    "alignment_method",
    "alignment_status",
    "alignment_reliable",
    "alignment_notes",
    "width",
    "height",
    "valid_pct",
    "changed_pixels",
    "changed_pct_of_valid",
    "mean_abs_diff",
    "p95_abs_diff",
    "ssim_mean",
    "ssim_below_threshold_pct",
    "brightness_delta",
    "region_count",
    "largest_region_pct_of_valid",
    "flags",
    "error",
]

ProgressCallback = Callable[[int, int, str, str], None]


class ProcessingError(RuntimeError):
    """Raised when the batch could not run safely."""


def artefact_paths(config: AppConfig, tile_id: str) -> dict[str, Path]:
    year_a, year_b = config.years
    comparisons = config.paths.comparisons
    changes = config.paths.change_maps
    return {
        "reference": config.paths.aligned / str(year_a) / f"{tile_id}.png",
        "target": config.paths.aligned / str(year_b) / f"{tile_id}.png",
        "comparison": comparisons / f"{tile_id}_comparison.png",
        "diff": changes / f"{tile_id}_diff.png",
        "heatmap": changes / f"{tile_id}_diff_heatmap.png",
        "ssim_map": changes / f"{tile_id}_ssim.png",
        "ssim_heatmap": changes / f"{tile_id}_ssim_heatmap.png",
        "mask": changes / f"{tile_id}_change_mask.png",
        "valid_mask": changes / f"{tile_id}_valid_mask.png",
        "overlay": changes / f"{tile_id}_change_overlay.png",
        "annotated": changes / f"{tile_id}_annotated.png",
    }


def _existing_outputs(config: AppConfig, tile_id: str, save_aligned: bool) -> bool:
    paths = artefact_paths(config, tile_id)
    required = [
        paths["comparison"],
        paths["diff"],
        paths["heatmap"],
        paths["ssim_map"],
        paths["ssim_heatmap"],
        paths["mask"],
        paths["valid_mask"],
        paths["overlay"],
        paths["annotated"],
    ]
    if save_aligned:
        required.extend([paths["reference"], paths["target"]])
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def _relative_map(config: AppConfig, tile_id: str, save_aligned: bool) -> dict[str, str]:
    paths = artefact_paths(config, tile_id)
    names = [
        "comparison",
        "diff",
        "heatmap",
        "ssim_map",
        "ssim_heatmap",
        "mask",
        "valid_mask",
        "overlay",
        "annotated",
    ]
    if save_aligned:
        names.extend(["reference", "target"])
    return {name: display_path(paths[name], config.root) for name in names}


def process_tile(
    config: AppConfig,
    pair: PairInfo,
    *,
    save_aligned: bool = True,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Process a single processable pair and return its result index entry."""
    entry: dict[str, Any] = {
        "tile_id": pair.tile_id,
        "status": "failed",
        "years": {"reference": pair.year_a, "target": pair.year_b},
        "processed_utc": utc_now(),
        "error": "",
        "paths": {},
        "alignment": {},
        "stats": {},
        "regions": [],
        "flags": [],
    }
    if not pair.processable:
        entry["status"] = "not_processable"
        entry["error"] = pair.notes or pair.status
        return entry
    if pair.record_a is None or pair.record_b is None:
        entry["error"] = "Pair records are incomplete."
        return entry

    reference_path = pair.record_a.absolute_path(config)
    target_path = pair.record_b.absolute_path(config)
    if progress:
        progress(0, 1, pair.tile_id, "load")

    try:
        reference = load_image_rgb(reference_path)
        target = load_image_rgb(target_path)
    except (OSError, ValueError) as exc:
        entry["error"] = f"Could not load screenshots: {exc}"
        LOGGER.error("%s: %s", pair.tile_id, entry["error"])
        return entry

    if progress:
        progress(0, 2, pair.tile_id, "align")
    try:
        alignment: AlignmentResult = align_pair(
            reference,
            target,
            config.processing.alignment,
            resize_mismatched=config.processing.resize_mismatched,
        )
    except Exception as exc:  # noqa: BLE001 - keep the batch running and record failure
        entry["error"] = f"Alignment failed: {exc}"
        LOGGER.exception("%s: alignment failed", pair.tile_id)
        return entry

    if progress:
        progress(0, 3, pair.tile_id, "compare")
    try:
        analysis: AnalysisResult = analyze_pair(
            alignment.reference,
            alignment.aligned_target,
            alignment.valid_mask,
            config.processing.diff,
            target_resized=alignment.resized,
        )
    except Exception as exc:  # noqa: BLE001
        entry["error"] = f"Difference analysis failed: {exc}"
        LOGGER.exception("%s: difference analysis failed", pair.tile_id)
        return entry

    flags = list(analysis.stats.get("image_flags", []))
    if (
        config.processing.alignment.enabled
        and config.processing.alignment.method != "none"
        and not alignment.reliable
    ):
        flags.append("alignment_unreliable")
    if alignment.resized:
        flags.append("target_resized")
    if pair.record_a.metadata_status != "complete" or pair.record_b.metadata_status != "complete":
        flags.append("metadata_incomplete")

    dates = [
        pair.record_a.imagery_date or "unknown",
        pair.record_b.imagery_date or "unknown",
    ]
    footer_lines = [
        f"Imagery dates (user-entered): {pair.year_a}: {dates[0]} | {pair.year_b}: {dates[1]}",
        (
            f"Thresholds: pixel diff >= {config.processing.diff.pixel_threshold} | "
            f"SSIM < {config.processing.diff.ssim_threshold:.2f} | "
            f"combine={config.processing.diff.combine_mode} | "
            f"min region {config.processing.diff.min_region_area}px"
        ),
        (
            f"Alignment: {alignment.method} ({'reliable' if alignment.reliable else 'UNRELIABLE'}) | "
            f"overlap {alignment.overlap:.0%} | "
            f"{pair.record_a.width}x{pair.record_a.height} vs "
            f"{pair.record_b.width}x{pair.record_b.height}"
        ),
        "Measured image differences only - not confirmed geographic change.",
    ]
    if alignment.notes:
        footer_lines.append("Alignment notes: " + " ".join(alignment.notes))
    comparison = render_comparison(
        analysis,
        f"{pair.year_a} reference",
        f"{pair.year_b} target",
        footer_lines,
    )

    if progress:
        progress(0, 4, pair.tile_id, "write")
    paths = artefact_paths(config, pair.tile_id)
    try:
        if save_aligned:
            save_image(paths["reference"], analysis.reference)
            save_image(paths["target"], analysis.target)
            verify_written_image(
                paths["reference"], analysis.reference.shape[1], analysis.reference.shape[0]
            )
            verify_written_image(
                paths["target"], analysis.target.shape[1], analysis.target.shape[0]
            )
        save_image(paths["comparison"], comparison)
        save_image(paths["diff"], analysis.diff_gray)
        save_image(paths["heatmap"], analysis.heatmap)
        save_image(paths["ssim_map"], np.clip(analysis.ssim_map * 255, 0, 255).astype(np.uint8))
        save_image(paths["ssim_heatmap"], analysis.ssim_heatmap)
        save_image(paths["mask"], analysis.change_mask)
        save_image(paths["valid_mask"], alignment.valid_mask)
        save_image(paths["overlay"], analysis.overlay)
        save_image(paths["annotated"], analysis.annotated)
        verify_written_image(paths["comparison"], comparison.shape[1], comparison.shape[0])
        for key in (
            "diff",
            "heatmap",
            "ssim_map",
            "ssim_heatmap",
            "mask",
            "valid_mask",
            "overlay",
            "annotated",
        ):
            verify_written_image(
                paths[key], analysis.change_mask.shape[1], analysis.change_mask.shape[0]
            )
    except (OSError, ValueError, DataFileError) as exc:
        entry["error"] = f"Could not write verified outputs: {exc}"
        LOGGER.error("%s: %s", pair.tile_id, entry["error"])
        return entry

    entry.update(
        status="processed",
        paths=_relative_map(config, pair.tile_id, save_aligned),
        alignment={
            **alignment.metrics(),
            "notes_list": list(alignment.notes),
        },
        stats=analysis.stats,
        regions=[asdict(region) for region in analysis.regions],
        flags=list(dict.fromkeys(flags)),
    )
    if progress:
        progress(1, 1, pair.tile_id, "done")
    LOGGER.info(
        "%s processed: %.2f%% changed pixels of valid overlap, SSIM %.3f, alignment %s",
        pair.tile_id,
        analysis.stats["changed_pct_of_valid"],
        analysis.stats["ssim_mean"],
        alignment.status,
    )
    return entry


def _flatten_entry(entry: dict[str, Any]) -> dict[str, Any]:
    alignment = entry.get("alignment", {})
    stats = entry.get("stats", {})
    return {
        "tile_id": entry.get("tile_id", ""),
        "status": entry.get("status", ""),
        "year_a": entry.get("years", {}).get("reference", ""),
        "year_b": entry.get("years", {}).get("target", ""),
        "alignment_method": alignment.get("alignment_method", ""),
        "alignment_status": alignment.get("alignment_status", ""),
        "alignment_reliable": alignment.get("alignment_reliable", ""),
        "alignment_notes": alignment.get("alignment_notes", ""),
        "width": stats.get("width", ""),
        "height": stats.get("height", ""),
        "valid_pct": stats.get("valid_pct", ""),
        "changed_pixels": stats.get("changed_pixels", ""),
        "changed_pct_of_valid": stats.get("changed_pct_of_valid", ""),
        "mean_abs_diff": stats.get("mean_abs_diff", ""),
        "p95_abs_diff": stats.get("p95_abs_diff", ""),
        "ssim_mean": stats.get("ssim_mean", ""),
        "ssim_below_threshold_pct": stats.get("ssim_below_threshold_pct", ""),
        "brightness_delta": stats.get("brightness_delta", ""),
        "region_count": stats.get("region_count", ""),
        "largest_region_pct_of_valid": stats.get("largest_region_pct_of_valid", ""),
        "flags": entry.get("flags", []),
        "error": entry.get("error", ""),
    }


def run_processing(
    config: AppConfig,
    *,
    tile_ids: Sequence[str] | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Validate the dataset, then process all (or selected) processable pairs."""
    manager = DatasetManager(config)
    scan = manager.scan(save=True)
    selected = set(tile_ids) if tile_ids else set(config.tile_ids)
    unknown = selected - set(config.tile_ids)
    if unknown:
        raise ProcessingError(f"Unknown tile id(s): {', '.join(sorted(unknown))}")

    previous = read_json(config.paths.results_index, default={})
    previous_tiles = previous.get("tiles", {}) if isinstance(previous, dict) else {}
    if not isinstance(previous_tiles, dict):
        previous_tiles = {}

    entries: dict[str, Any] = {
        tile_id: entry
        for tile_id, entry in previous_tiles.items()
        if tile_id not in selected or isinstance(entry, dict)
    }
    pairs = list(scan.pairs.values())
    work = [pair for pair in pairs if pair.tile_id in selected]
    total = len(work)
    counts = {"processed": 0, "skipped": 0, "failed": 0, "not_processable": 0}

    for index, pair in enumerate(work, start=1):
        if progress:
            progress(index - 1, total, pair.tile_id, "start")
        if not pair.processable:
            entries[pair.tile_id] = {
                "tile_id": pair.tile_id,
                "status": "not_processable",
                "years": {"reference": pair.year_a, "target": pair.year_b},
                "processed_utc": "",
                "error": pair.status + (f": {pair.notes}" if pair.notes else ""),
                "paths": {},
                "alignment": {},
                "stats": {},
                "regions": [],
                "flags": ["pair_not_processable"],
            }
            counts["not_processable"] += 1
            continue

        skip_existing = config.processing.skip_existing and not force
        if skip_existing and pair.tile_id in previous_tiles:
            old = previous_tiles[pair.tile_id]
            if (
                isinstance(old, dict)
                and old.get("status") == "processed"
                and _existing_outputs(config, pair.tile_id, config.processing.save_aligned)
            ):
                entries[pair.tile_id] = old
                counts["skipped"] += 1
                LOGGER.info("%s: reusing existing outputs (use --force to recompute)", pair.tile_id)
                continue

        entry = process_tile(
            config,
            pair,
            save_aligned=config.processing.save_aligned,
            progress=progress,
        )
        entries[pair.tile_id] = entry
        if entry["status"] == "processed":
            counts["processed"] += 1
        elif entry["status"] == "not_processable":
            counts["not_processable"] += 1
        else:
            counts["failed"] += 1

        payload = _build_index(config, entries, scan.summary, counts)
        atomic_write_json(config.paths.results_index, payload)
        write_csv(
            config.paths.tile_stats,
            [_flatten_entry(entries[tile_id]) for tile_id in sorted(entries)],
            columns=STATS_COLUMNS,
        )

    payload = _build_index(config, entries, scan.summary, counts)
    atomic_write_json(config.paths.results_index, payload)
    write_csv(
        config.paths.tile_stats,
        [_flatten_entry(entries[tile_id]) for tile_id in sorted(entries)],
        columns=STATS_COLUMNS,
    )
    processed_entries = [
        entry for entry in entries.values() if entry.get("status") == "processed"
    ]
    payload["summary"]["processed"] = len(processed_entries)
    payload["summary"]["alignment_unreliable"] = sum(
        1
        for entry in processed_entries
        if entry.get("alignment", {}).get("alignment_reliable") is False
    )
    payload["summary"]["flagged"] = sum(1 for entry in processed_entries if entry.get("flags"))
    payload["summary"]["regions_total"] = sum(
        len(entry.get("regions", [])) for entry in processed_entries
    )
    atomic_write_json(config.paths.results_index, payload)
    LOGGER.info(
        "Processing finished: %d processed, %d skipped, %d failed, %d not processable",
        counts["processed"],
        counts["skipped"],
        counts["failed"],
        counts["not_processable"],
    )
    return payload


def _build_index(
    config: AppConfig,
    entries: dict[str, Any],
    dataset_summary: dict[str, Any],
    counts: dict[str, int],
) -> dict[str, Any]:
    ordered = {tile_id: entries[tile_id] for tile_id in sorted(entries)}
    processed = sum(1 for entry in ordered.values() if entry.get("status") == "processed")
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "project": config.project_name,
        "years": {"reference": config.years[0], "target": config.years[1]},
        "thresholds": {
            "pixel_threshold": config.processing.diff.pixel_threshold,
            "ssim_threshold": config.processing.diff.ssim_threshold,
            "combine_mode": config.processing.diff.combine_mode,
            "min_region_area": config.processing.diff.min_region_area,
        },
        "summary": {
            **dataset_summary,
            "processed": processed,
            "run_counts": counts,
            "total_indexed": len(ordered),
        },
        "tiles": ordered,
    }

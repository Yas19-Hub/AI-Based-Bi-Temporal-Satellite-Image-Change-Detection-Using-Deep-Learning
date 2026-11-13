"""Copy only tile pairs whose measured difference exceeds a chosen threshold."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Sequence

from .config import AppConfig
from .dataset import DatasetManager
from .logging_setup import get_logger
from .utils import atomic_write_json, atomic_write_text, hash_file, read_json, utc_now, write_csv

LOGGER = get_logger(__name__)

DEFAULT_OUTPUT_DIR = "changes_only"
DEFAULT_THRESHOLD = 10.0
DEFAULT_MIN_VALID_OVERLAP = 50.0

EXPORT_COLUMNS = [
    "tile_id",
    "year_a",
    "year_b",
    "changed_pct_of_valid",
    "ssim_mean",
    "mean_abs_diff",
    "valid_pct",
    "alignment_method",
    "alignment_reliable",
    "source_year_a",
    "source_year_b",
    "copied_year_a",
    "copied_year_b",
    "metadata_status_year_a",
    "metadata_status_year_b",
]
SKIPPED_COLUMNS = [
    "tile_id",
    "status",
    "reason",
    "changed_pct_of_valid",
    "alignment_reliable",
    "notes",
]


class ExportError(RuntimeError):
    """Raised when changed pairs cannot be exported safely."""


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _resolve_output(config: AppConfig, output_dir: Path | None) -> Path:
    directory = Path(output_dir) if output_dir is not None else Path(DEFAULT_OUTPUT_DIR)
    return directory if directory.is_absolute() else (config.root / directory).resolve()


def _copy_verified(source: Path, destination: Path, force: bool) -> tuple[str, str]:
    """Copy source and verify both files share the same SHA-256 digest."""
    source_hash = hash_file(source)
    if destination.exists():
        destination_hash = hash_file(destination)
        if destination_hash == source_hash:
            return "existing", source_hash
        if not force:
            raise ExportError(
                f"{destination} already exists with different content. "
                "Use --force to replace the exported copy."
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temp)
        if hash_file(temp) != source_hash:
            raise ExportError(f"Copy verification failed for {source} -> {destination}.")
        os.replace(temp, destination)
        if hash_file(destination) != source_hash:
            raise ExportError(f"Destination verification failed for {destination}.")
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass
    return "copied", source_hash


def _safe_relative_file(output: Path, relative: str) -> Path | None:
    candidate = (output / relative).resolve()
    try:
        candidate.relative_to(output.resolve())
    except ValueError:
        return None
    return candidate


def _remove_stale(previous: dict[str, Any], current: set[str], output: Path) -> int:
    removed = 0
    for relative in previous.get("files", []):
        if not isinstance(relative, str) or relative in current:
            continue
        target = _safe_relative_file(output, relative)
        if target is None or not target.is_file():
            continue
        try:
            target.unlink()
            removed += 1
        except OSError as exc:
            LOGGER.warning("Could not remove stale exported file %s: %s", target, exc)
    for year_dir in sorted(output.iterdir()) if output.exists() else []:
        if year_dir.is_dir() and not any(year_dir.iterdir()):
            try:
                year_dir.rmdir()
            except OSError:
                pass
    return removed


def export_changed_pairs(
    config: AppConfig,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_valid_overlap: float = DEFAULT_MIN_VALID_OVERLAP,
    output_dir: Path | None = None,
    include_unreliable: bool = False,
    force: bool = False,
    keep_stale: bool = False,
) -> dict[str, Any]:
    """Copy 2019/2024 originals only for tiles above the measured-change threshold.

    Originals are never moved or deleted. The output folder is managed through
    `.export_manifest.json` so re-exporting with a stricter threshold removes
    previously exported copies that no longer qualify.
    """
    if not 0.0 <= threshold <= 100.0:
        raise ExportError("Threshold must be between 0 and 100 percent.")
    if not 0.0 <= min_valid_overlap <= 100.0:
        raise ExportError("Minimum valid overlap must be between 0 and 100 percent.")

    index = read_json(config.paths.results_index, default={})
    tiles = index.get("tiles", {}) if isinstance(index, dict) else {}
    processed_tiles = (
        [entry for entry in tiles.values() if isinstance(entry, dict) and entry.get("status") == "processed"]
        if isinstance(tiles, dict)
        else []
    )
    if not processed_tiles:
        raise ExportError(
            "No processed tile pairs found. Add paired screenshots, run `validate`, "
            "then run `process` before exporting changed pairs."
        )
    scan = DatasetManager(config).scan(save=True)
    output = _resolve_output(config, output_dir)
    output.mkdir(parents=True, exist_ok=True)
    previous = read_json(output / ".export_manifest.json", default={})
    previous = previous if isinstance(previous, dict) else {}
    current_files: set[str] = set()

    exported: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    year_a, year_b = config.years

    for tile_id in config.tile_ids:
        pair = scan.pairs.get(tile_id)
        entry = index["tiles"].get(tile_id, {}) if isinstance(index["tiles"], dict) else {}
        if not isinstance(entry, dict):
            entry = {}
        stats = entry.get("stats", {}) if isinstance(entry.get("stats"), dict) else {}
        alignment = entry.get("alignment", {}) if isinstance(entry.get("alignment"), dict) else {}
        changed_pct = _as_float(stats.get("changed_pct_of_valid"))
        valid_pct = _as_float(stats.get("valid_pct"))
        method = str(alignment.get("alignment_method", ""))
        reliable = alignment.get("alignment_reliable")
        # Alignment explicitly disabled by the user is a deliberate per-capture mode;
        # identity fallback after failed alignment is not equivalently trustworthy.
        alignment_ok = reliable is True or method == "none"
        if include_unreliable:
            alignment_ok = True

        reason = ""
        if pair is None:
            reason = "pair_not_found"
        elif entry.get("status") != "processed":
            reason = f"not_processed:{entry.get('status', 'unknown')}"
        elif not alignment_ok:
            reason = "alignment_unreliable"
        elif changed_pct <= threshold:
            reason = f"below_threshold:{changed_pct:.3f}<={threshold:.3f}"
        elif valid_pct < min_valid_overlap:
            reason = f"insufficient_overlap:{valid_pct:.1f}<{min_valid_overlap:.1f}"
        elif pair.record_a is None or pair.record_b is None:
            reason = "missing_source_record"

        if reason:
            skipped.append(
                {
                    "tile_id": tile_id,
                    "status": entry.get("status", "unknown"),
                    "reason": reason,
                    "changed_pct_of_valid": stats.get("changed_pct_of_valid", ""),
                    "alignment_reliable": reliable if reliable is not None else "",
                    "notes": entry.get("error", "") or (pair.notes if pair else ""),
                }
            )
            continue

        source_a = pair.record_a.absolute_path(config)
        source_b = pair.record_b.absolute_path(config)
        if not source_a.is_file() or not source_b.is_file():
            skipped.append(
                {
                    "tile_id": tile_id,
                    "status": entry.get("status", "unknown"),
                    "reason": "source_file_missing",
                    "changed_pct_of_valid": stats.get("changed_pct_of_valid", ""),
                    "alignment_reliable": reliable if reliable is not None else "",
                    "notes": f"{source_a} / {source_b}",
                }
            )
            continue

        relative_a = Path(str(year_a)) / source_a.name
        relative_b = Path(str(year_b)) / source_b.name
        destination_a = output / relative_a
        destination_b = output / relative_b
        try:
            state_a, _ = _copy_verified(source_a, destination_a, force=force)
            state_b, _ = _copy_verified(source_b, destination_b, force=force)
        except (ExportError, OSError) as exc:
            failures.append(
                {
                    "tile_id": tile_id,
                    "error": str(exc),
                    "source_year_a": str(source_a),
                    "source_year_b": str(source_b),
                }
            )
            LOGGER.error("%s: %s", tile_id, exc)
            continue

        current_files.update({relative_a.as_posix(), relative_b.as_posix()})
        exported.append(
            {
                "tile_id": tile_id,
                "year_a": year_a,
                "year_b": year_b,
                "changed_pct_of_valid": stats.get("changed_pct_of_valid", ""),
                "ssim_mean": stats.get("ssim_mean", ""),
                "mean_abs_diff": stats.get("mean_abs_diff", ""),
                "valid_pct": stats.get("valid_pct", ""),
                "alignment_method": method,
                "alignment_reliable": reliable if reliable is not None else "",
                "source_year_a": str(source_a),
                "source_year_b": str(source_b),
                "copied_year_a": str(destination_a),
                "copied_year_b": str(destination_b),
                "metadata_status_year_a": pair.record_a.metadata_status,
                "metadata_status_year_b": pair.record_b.metadata_status,
                "copy_state_year_a": state_a,
                "copy_state_year_b": state_b,
            }
        )

    write_csv(output / "changes.csv", exported, columns=EXPORT_COLUMNS)
    write_csv(output / "skipped.csv", skipped, columns=SKIPPED_COLUMNS)
    removed = 0
    if not keep_stale and not failures:
        removed = _remove_stale(previous, current_files, output)
    elif not keep_stale and failures:
        LOGGER.warning(
            "Skipped cleanup of stale exported files because %d copy failure(s) occurred.",
            len(failures),
        )
    atomic_write_json(
        output / ".export_manifest.json",
        {
            "generated_utc": utc_now(),
            "threshold_pct": threshold,
            "min_valid_overlap_pct": min_valid_overlap,
            "include_unreliable": include_unreliable,
            "files": sorted(current_files),
        },
    )
    atomic_write_text(
        output / "README.txt",
        (
            "Changed screenshot pairs\n"
            "========================\n\n"
            f"Generated: {utc_now()}\n"
            f"Tile pairs copied: {len(exported)}\n"
            f"Threshold: measured changed pixels > {threshold:.2f}% of valid overlap\n"
            f"Minimum valid overlap: {min_valid_overlap:.2f}%\n"
            f"Unreliable alignment included: {'yes' if include_unreliable else 'no'}\n\n"
            "Only the original 2019 and 2024 screenshots for each qualifying tile are copied.\n"
            "Originals remain untouched in raw_screenshots/. Missing dates/coordinates are not\n"
            "inferred; see metadata/tile_manifest.csv and manual_overrides.csv.\n\n"
            "These are measured image differences, not confirmed geographic change. Review the\n"
            "comparison outputs and source imagery before drawing conclusions.\n"
        ),
    )
    result = {
        "output": str(output),
        "threshold": threshold,
        "min_valid_overlap": min_valid_overlap,
        "exported": len(exported),
        "skipped": len(skipped),
        "failed": len(failures),
        "removed_stale": removed,
        "files": sorted(current_files),
        "changes_csv": str(output / "changes.csv"),
        "skipped_csv": str(output / "skipped.csv"),
    }
    LOGGER.info(
        "Exported %d changed tile pair(s) above %.2f%% to %s",
        result["exported"],
        threshold,
        output,
    )
    return result

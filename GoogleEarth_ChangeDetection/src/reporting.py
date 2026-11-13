"""Human-readable validation/processing reports and CSV exports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import AppConfig
from .dataset import DatasetManager, ScanResult
from .logging_setup import get_logger
from .utils import atomic_write_text, read_json, utc_now, write_csv

LOGGER = get_logger(__name__)

ISSUE_COLUMNS = [
    "tile_id",
    "year",
    "filename",
    "relative_path",
    "validation_status",
    "validation_notes",
    "metadata_status",
    "metadata_notes",
    "pair_status",
    "pair_notes",
]

METADATA_COLUMNS = [
    "tile_id",
    "year",
    "filename",
    "metadata_status",
    "imagery_date",
    "image_capture_date",
    "latitude",
    "longitude",
    "metadata_notes",
]

FLAG_COLUMNS = ["tile_id", "flag"]

PROCESSING_COLUMNS = [
    "tile_id",
    "status",
    "alignment_method",
    "alignment_status",
    "alignment_reliable",
    "valid_pct",
    "changed_pct_of_valid",
    "mean_abs_diff",
    "p95_abs_diff",
    "ssim_mean",
    "ssim_below_threshold_pct",
    "region_count",
    "largest_region_pct_of_valid",
    "flags",
    "error",
]


def generate_reports(
    config: AppConfig,
    scan: ScanResult | None = None,
    index: dict[str, Any] | None = None,
) -> dict[str, Path]:
    if scan is None:
        scan = DatasetManager(config).scan(save=True)
    if index is None:
        loaded = read_json(config.paths.results_index, default={})
        index = loaded if isinstance(loaded, dict) else {}

    directory = config.paths.reports
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "validation_markdown": directory / "validation_report.md",
        "processing_markdown": directory / "processing_report.md",
        "issues": directory / "dataset_issues.csv",
        "missing": directory / "missing_files.csv",
        "unrecognized": directory / "unrecognized_files.csv",
        "metadata_gaps": directory / "metadata_gaps.csv",
        "processing_csv": directory / "processing_results.csv",
        "quality_flags": directory / "quality_flags.csv",
        "change_summary": directory / "change_summary.csv",
    }

    issue_rows = _issue_rows(scan)
    write_csv(paths["issues"], issue_rows, columns=ISSUE_COLUMNS)
    write_csv(
        paths["missing"],
        [row for row in issue_rows if row["validation_status"] == "missing_file"],
        columns=ISSUE_COLUMNS,
    )
    write_csv(
        paths["unrecognized"],
        scan.unrecognized,
        columns=["relative_path", "reason", "detail"],
    )
    metadata_rows = [
        {
            "tile_id": row.tile_id,
            "year": row.year,
            "filename": row.filename,
            "metadata_status": row.metadata_status,
            "imagery_date": row.imagery_date,
            "image_capture_date": row.image_capture_date,
            "latitude": row.latitude,
            "longitude": row.longitude,
            "metadata_notes": row.metadata_notes,
        }
        for row in scan.rows
        if row.file_exists and row.metadata_status != "complete"
    ]
    write_csv(paths["metadata_gaps"], metadata_rows, columns=METADATA_COLUMNS)

    processing_rows = _processing_rows(index)
    write_csv(paths["processing_csv"], processing_rows, columns=PROCESSING_COLUMNS)
    flag_rows = [
        {"tile_id": row["tile_id"], "flag": flag}
        for row in processing_rows
        for flag in str(row.get("flags", "")).split(";")
        if flag.strip()
    ]
    write_csv(paths["quality_flags"], flag_rows, columns=FLAG_COLUMNS)
    change_rows = sorted(
        [
            row
            for row in processing_rows
            if row.get("status") == "processed"
        ],
        key=lambda row: _as_float(row.get("changed_pct_of_valid")),
        reverse=True,
    )
    change_columns = [
        "tile_id",
        "changed_pct_of_valid",
        "ssim_mean",
        "mean_abs_diff",
        "region_count",
        "largest_region_pct_of_valid",
        "alignment_reliable",
        "flags",
    ]
    write_csv(paths["change_summary"], change_rows, columns=change_columns)

    atomic_write_text(paths["validation_markdown"], _validation_markdown(config, scan))
    atomic_write_text(paths["processing_markdown"], _processing_markdown(config, scan, index))
    LOGGER.info("Reports written to %s", directory)
    return paths


def _issue_rows(scan: ScanResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in scan.rows:
        if row.validation_status == "valid" and row.metadata_status == "complete":
            continue
        rows.append(
            {
                "tile_id": row.tile_id,
                "year": row.year,
                "filename": row.filename,
                "relative_path": row.relative_path,
                "validation_status": row.validation_status,
                "validation_notes": row.validation_notes,
                "metadata_status": row.metadata_status,
                "metadata_notes": row.metadata_notes,
                "pair_status": row.pair_status,
                "pair_notes": row.pair_notes,
            }
        )
    return rows


def _processing_rows(index: dict[str, Any]) -> list[dict[str, Any]]:
    tiles = index.get("tiles", {}) if isinstance(index, dict) else {}
    if not isinstance(tiles, dict):
        return []
    rows: list[dict[str, Any]] = []
    for tile_id in sorted(tiles):
        entry = tiles[tile_id]
        if not isinstance(entry, dict):
            continue
        alignment = entry.get("alignment", {})
        stats = entry.get("stats", {})
        rows.append(
            {
                "tile_id": tile_id,
                "status": entry.get("status", ""),
                "alignment_method": alignment.get("alignment_method", ""),
                "alignment_status": alignment.get("alignment_status", ""),
                "alignment_reliable": alignment.get("alignment_reliable", ""),
                "valid_pct": stats.get("valid_pct", ""),
                "changed_pct_of_valid": stats.get("changed_pct_of_valid", ""),
                "mean_abs_diff": stats.get("mean_abs_diff", ""),
                "p95_abs_diff": stats.get("p95_abs_diff", ""),
                "ssim_mean": stats.get("ssim_mean", ""),
                "ssim_below_threshold_pct": stats.get("ssim_below_threshold_pct", ""),
                "region_count": stats.get("region_count", ""),
                "largest_region_pct_of_valid": stats.get("largest_region_pct_of_valid", ""),
                "flags": ";".join(str(flag) for flag in entry.get("flags", [])),
                "error": entry.get("error", ""),
            }
        )
    return rows


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _escape(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]], limit: int | None = None) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    count = 0
    for row in rows:
        if limit is not None and count >= limit:
            lines.append(f"| ... | {' | '.join([''] * (len(headers) - 1))} |")
            break
        lines.append("| " + " | ".join(_escape(value) for value in row) + " |")
        count += 1
    return "\n".join(lines)


def _validation_markdown(config: AppConfig, scan: ScanResult) -> str:
    summary = scan.summary
    status_counts = Counter(row.validation_status for row in scan.rows)
    metadata_counts = Counter(row.metadata_status for row in scan.rows if row.file_exists)
    pair_counts = Counter(pair.status for pair in scan.pairs.values())
    lines = [
        "# Dataset validation report",
        "",
        f"Generated: {utc_now()}",
        f"Project: {config.project_name}",
        f"Years: {config.years[0]} (reference) / {config.years[1]} (target)",
        f"Expected tiles: {summary['expected_tiles']} | "
        f"expected screenshots: {summary['expected_screenshots']}",
        "",
        "## Completion",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Valid screenshot files | {summary['valid_files']} / {summary['expected_screenshots']} "
        f"({summary['file_completion_pct']:.2f}%) |",
        f"| Complete tile pairs | {summary['complete_pairs']} / {summary['expected_tiles']} "
        f"({summary['completion_pct']:.2f}%) |",
        f"| Processable tile pairs | {summary['processable_pairs']} |",
        f"| Missing files | {summary['missing_files']} |",
        f"| Invalid files | {summary['invalid_files']} |",
        f"| Duplicate valid files | {summary['duplicate_files']} |",
        f"| Unrecognized files | {summary['unrecognized_files']} |",
        "",
        "## Validation statuses",
        "",
        _table(["Status", "Count"], sorted(status_counts.items())),
        "",
        "## Metadata statuses (existing files)",
        "",
        _table(["Status", "Count"], sorted(metadata_counts.items())),
        "",
        "## Pair statuses",
        "",
        _table(["Status", "Count"], sorted(pair_counts.items())),
        "",
        "## Missing tile/year slots",
        "",
    ]
    missing = [row for row in scan.rows if row.validation_status == "missing_file"]
    if missing:
        lines.append(_table(["Tile", "Year", "Expected filename"], [
            (row.tile_id, row.year, row.expected_filename) for row in missing
        ]))
    else:
        lines.append("No missing screenshot slots.")
    lines.extend(["", "## Invalid or duplicate files", ""])
    problems = [
        row
        for row in scan.rows
        if row.validation_status not in {"valid", "missing_file"}
    ]
    if problems:
        lines.append(
            _table(
                ["Tile", "Year", "File", "Status", "Notes"],
                [
                    (row.tile_id, row.year, row.filename, row.validation_status, row.validation_notes)
                    for row in problems
                ],
                limit=200,
            )
        )
    else:
        lines.append("No invalid or duplicate files found.")

    lines.extend(["", "## Unrecognized files", ""])
    if scan.unrecognized:
        lines.append(
            _table(
                ["Path", "Reason", "Detail"],
                [
                    (item["relative_path"], item["reason"], item["detail"])
                    for item in scan.unrecognized
                ],
                limit=200,
            )
        )
    else:
        lines.append("No unrecognized files found.")

    lines.extend(["", "## Metadata gaps", ""])
    gaps = [row for row in scan.rows if row.file_exists and row.metadata_status != "complete"]
    if gaps:
        lines.append(
            _table(
                ["Tile", "Year", "Status", "Imagery date", "Notes"],
                [
                    (row.tile_id, row.year, row.metadata_status, row.imagery_date, row.metadata_notes)
                    for row in gaps
                ],
                limit=200,
            )
        )
    else:
        lines.append("All existing files have complete user-entered metadata.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Dates and coordinates are never inferred from filenames, folder names or file "
            "timestamps. Unknown values stay blank.",
            "- Imagery dates can only be transcribed by the user from Google Earth Pro's "
            "historical imagery slider/status display.",
            "- The manifest is authoritative; fixes should be made in the source files or "
            "manual_overrides.csv and then re-validated.",
            "",
        ]
    )
    return "\n".join(lines)


def _processing_markdown(config: AppConfig, scan: ScanResult, index: dict[str, Any]) -> str:
    summary = index.get("summary", {}) if isinstance(index, dict) else {}
    tiles = index.get("tiles", {}) if isinstance(index, dict) else {}
    entries = [entry for entry in tiles.values() if isinstance(entry, dict)] if isinstance(tiles, dict) else []
    status_counts = Counter(entry.get("status", "unknown") for entry in entries)
    flag_counts = Counter(
        flag for entry in entries for flag in entry.get("flags", []) if isinstance(flag, str)
    )
    processed = [entry for entry in entries if entry.get("status") == "processed"]
    top_changed = sorted(
        processed,
        key=lambda entry: _as_float(entry.get("stats", {}).get("changed_pct_of_valid")),
        reverse=True,
    )[:20]
    unreliable = [
        entry
        for entry in processed
        if entry.get("alignment", {}).get("alignment_reliable") is False
    ]
    failures = [
        entry
        for entry in entries
        if entry.get("status") in {"failed", "not_processable"}
    ]
    processed_at = index.get("generated_utc", "not yet run") if isinstance(index, dict) else "not yet run"

    lines = [
        "# Processing report",
        "",
        f"Generated: {utc_now()}",
        f"Processing results timestamp: {processed_at}",
        f"Project: {config.project_name}",
        "",
        "> These outputs measure pixel/structural differences between two screenshots. "
        "They do not confirm geographic, land-use or radiometric change. Review every flagged "
        "tile against the source imagery.",
        "",
        "## Summary",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Dataset complete pairs | {scan.summary['complete_pairs']} / {scan.summary['expected_tiles']} |",
        f"| Processable pairs | {scan.summary['processable_pairs']} |",
        f"| Processed results in index | {summary.get('processed', len(processed))} |",
        f"| Alignment unreliable | {summary.get('alignment_unreliable', len(unreliable))} |",
        f"| Tiles with quality flags | {summary.get('flagged', sum(1 for e in processed if e.get('flags')))} |",
        "",
        "## Result statuses",
        "",
        _table(["Status", "Count"], sorted(status_counts.items())) if status_counts else "No results index found.",
        "",
        "## Quality flags",
        "",
        _table(["Flag", "Count"], sorted(flag_counts.items())) if flag_counts else "No quality flags.",
        "",
        "## Highest measured difference (top 20)",
        "",
    ]
    if top_changed:
        lines.append(
            _table(
                [
                    "Tile",
                    "Changed % of valid",
                    "SSIM mean",
                    "Mean abs diff",
                    "Regions",
                    "Alignment reliable",
                    "Flags",
                ],
                [
                    (
                        entry.get("tile_id", ""),
                        entry.get("stats", {}).get("changed_pct_of_valid", ""),
                        entry.get("stats", {}).get("ssim_mean", ""),
                        entry.get("stats", {}).get("mean_abs_diff", ""),
                        entry.get("stats", {}).get("region_count", ""),
                        entry.get("alignment", {}).get("alignment_reliable", ""),
                        ", ".join(entry.get("flags", [])),
                    )
                    for entry in top_changed
                ],
            )
        )
    else:
        lines.append("No processed tiles yet. Run `python -m src.cli process`.")

    lines.extend(["", "## Unreliable alignment", ""])
    if unreliable:
        lines.append(
            _table(
                ["Tile", "Method", "Inliers", "Overlap", "Notes"],
                [
                    (
                        entry.get("tile_id", ""),
                        entry.get("alignment", {}).get("alignment_method", ""),
                        entry.get("alignment", {}).get("alignment_inliers", ""),
                        entry.get("alignment", {}).get("alignment_overlap", ""),
                        entry.get("alignment", {}).get("alignment_notes", ""),
                    )
                    for entry in unreliable
                ],
            )
        )
    else:
        lines.append("No processed tiles are flagged with unreliable alignment.")

    lines.extend(["", "## Failed or not processable", ""])
    if failures:
        lines.append(
            _table(
                ["Tile", "Status", "Reason"],
                [
                    (entry.get("tile_id", ""), entry.get("status", ""), entry.get("error", ""))
                    for entry in failures
                ],
                limit=200,
            )
        )
    else:
        lines.append("No failed or non-processable tiles in the current index.")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Alignment is optional and can fail when screenshots lack shared texture, have "
            "different zoom/rotation, or contain large seasonal changes.",
            "- Candidate regions are threshold/morphology dependent and are not classification "
            "results.",
            "- Metadata gaps do not stop processing, but dates/coordinates must be verified "
            "manually before using results in analysis.",
            "",
        ]
    )
    return "\n".join(lines)

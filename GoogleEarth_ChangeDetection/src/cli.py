"""Command-line interface for validation, processing, reports and capture."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .capture_assistant import (
    CaptureError,
    build_capture_plan,
    capture_countdown,
    capture_instructions,
    save_screenshot_clipboard,
    save_screenshot_file,
)
from .config import AppConfig, ConfigError, load_config
from .dataset import DatasetError, DatasetManager, init_project, set_review_status, upsert_override
from .export_changes import DEFAULT_THRESHOLD, ExportError, export_changed_pairs
from .logging_setup import get_logger, setup_logging
from .processing import ProcessingError, run_processing
from .reporting import generate_reports
from .utils import DataFileError, normalize_tile_id, read_json

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description=(
            "Validate paired Google Earth Pro screenshots, measure image differences, "
            "and generate reports. Dates must be entered manually; nothing is inferred."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", type=Path, help="Path to config.yaml (default: project root).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="Create folders/templates and the initial manifest.")

    validate = subparsers.add_parser("validate", help="Scan, validate and pair all screenshots.")
    validate.add_argument(
        "--no-reports", action="store_true", help="Do not regenerate the validation report."
    )

    process = subparsers.add_parser("process", help="Align and compare processable tile pairs.")
    process.add_argument(
        "--tile",
        action="append",
        metavar="T###",
        help="Process only this tile; repeat for multiple tiles.",
    )
    process.add_argument(
        "--force", action="store_true", help="Recompute outputs instead of reusing existing ones."
    )

    subparsers.add_parser("report", help="Regenerate validation, processing and CSV reports.")

    export = subparsers.add_parser(
        "export-changes",
        help="Copy only 2019/2024 pairs whose measured difference exceeds a threshold.",
    )
    export.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Changed percent of valid overlap required (default {DEFAULT_THRESHOLD}, must be exceeded).",
    )
    export.add_argument(
        "--min-valid-overlap",
        type=float,
        default=50.0,
        help="Minimum valid overlap percent for an exported pair (default 50).",
    )
    export.add_argument(
        "--output",
        type=Path,
        help="Output folder, relative to the project root (default changes_only).",
    )
    export.add_argument(
        "--include-unreliable",
        action="store_true",
        help="Include pairs whose alignment was flagged unreliable.",
    )
    export.add_argument(
        "--force",
        action="store_true",
        help="Replace exported copies that differ from the originals.",
    )
    export.add_argument(
        "--keep-stale",
        action="store_true",
        help="Keep previously exported copies that no longer pass the current threshold.",
    )

    status = subparsers.add_parser("status", help="Print dataset/processing completion status.")
    status.add_argument("--no-scan", action="store_true", help="Use existing files without rescanning.")

    capture_plan = subparsers.add_parser(
        "capture-plan", help="Show missing tile/year slots and manual capture steps."
    )
    capture_plan.add_argument("--year", type=int, help="Only show one configured year.")
    capture_plan.add_argument(
        "--missing-only", action="store_true", help="Hide slots that already have valid files."
    )

    capture = subparsers.add_parser(
        "capture",
        help="Countdown screen capture after you manually position Google Earth Pro.",
    )
    capture.add_argument("--tile", required=True, help="Tile id, for example T001.")
    capture.add_argument("--year", required=True, type=int, help="Configured imagery year.")
    capture.add_argument("--countdown", type=int, default=5, help="Seconds before capture (default 5).")
    capture.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing original. Off by default to protect captures.",
    )

    save_screenshot = subparsers.add_parser(
        "save-screenshot",
        help="Save an existing screenshot file or clipboard image into raw_screenshots/YYYY.",
    )
    save_screenshot.add_argument("--tile", required=True, help="Tile id, for example T001.")
    save_screenshot.add_argument("--year", required=True, type=int, help="Configured imagery year.")
    source_group = save_screenshot.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", type=Path, help="Existing screenshot file to copy.")
    source_group.add_argument(
        "--clipboard",
        action="store_true",
        help="Save the image currently on the Windows clipboard.",
    )
    save_screenshot.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing original. Off by default to protect captures.",
    )

    metadata = subparsers.add_parser(
        "metadata",
        help="Record user-observed dates/coordinates in manual_overrides.csv.",
    )
    metadata.add_argument("--tile", required=True)
    metadata.add_argument("--year", type=int, help="Year the metadata applies to; blank applies to both.")
    metadata.add_argument("--imagery-date", help="Date observed in Google Earth Pro (YYYY[-MM[-DD]]).")
    metadata.add_argument("--capture-date", help="When the screenshot was taken (ISO date/time).")
    metadata.add_argument("--latitude", help="WGS84 latitude, if known.")
    metadata.add_argument("--longitude", help="WGS84 longitude, if known.")
    metadata.add_argument("--notes", help="Free-text note.")

    review = subparsers.add_parser("review", help="Set a tile review status.")
    review.add_argument("--tile", required=True)
    review.add_argument("--status", required=True, choices=["pending", "reviewed", "unreliable"])
    review.add_argument("--reviewer", default="", help="Reviewer name.")
    review.add_argument("--notes", help="Review notes.")

    dashboard = subparsers.add_parser("dashboard", help="Launch the Streamlit dashboard.")
    dashboard.add_argument(
        "--port", type=int, default=8501, help="Streamlit port (default 8501)."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.verbose:
            config = _with_debug(config)
        setup_logging(config, console=True)
        return _dispatch(args, config)
    except (
        ConfigError,
        DataFileError,
        DatasetError,
        ProcessingError,
        ExportError,
        CaptureError,
        ValueError,
        OSError,
    ) as exc:
        LOGGER.error("%s", exc)
        return 1


def _with_debug(config: AppConfig):
    from dataclasses import replace

    return replace(config, logging=replace(config.logging, level="DEBUG"))


def _dispatch(args: argparse.Namespace, config: AppConfig) -> int:
    if args.command == "init":
        result = init_project(config)
        summary = result["summary"]
        print(f"Initialized project at {config.root}")
        for path in result["created"]:
            print(f"  created/updated: {path}")
        print(
            f"Expected screenshots: {summary['expected_screenshots']} | "
            f"currently valid: {summary['valid_files']} | "
            f"processable pairs: {summary['processable_pairs']}"
        )
        print(f"Put screenshots in: {config.paths.raw_year_dir(config.years[0])} and "
              f"{config.paths.raw_year_dir(config.years[1])}")
        return 0

    if args.command == "validate":
        manager = DatasetManager(config)
        scan = manager.scan(save=True)
        print(
            f"Valid: {scan.summary['valid_files']}/{scan.summary['expected_screenshots']} | "
            f"complete pairs: {scan.summary['complete_pairs']}/{scan.summary['expected_tiles']} | "
            f"processable: {scan.summary['processable_pairs']} | "
            f"missing: {scan.summary['missing_files']} | invalid: {scan.summary['invalid_files']} | "
            f"duplicates: {scan.summary['duplicate_files']} | unrecognized: "
            f"{scan.summary['unrecognized_files']}"
        )
        if not args.no_reports:
            paths = generate_reports(config, scan=scan)
            print(f"Validation report: {paths['validation_markdown']}")
        if scan.unrecognized:
            print("Unrecognized files:")
            for item in scan.unrecognized[:20]:
                print(f"  {item['relative_path']}: {item['reason']} - {item['detail']}")
            if len(scan.unrecognized) > 20:
                print(f"  ... and {len(scan.unrecognized) - 20} more (see report).")
        return 0

    if args.command == "process":
        tile_ids = [
            normalize_tile_id(tile)
            for tile in (args.tile or [])
            if str(tile).strip()
        ]
        index = run_processing(config, tile_ids=tile_ids or None, force=args.force)
        counts = index["summary"].get("run_counts", {})
        print(
            f"Processed: {counts.get('processed', 0)} | skipped: {counts.get('skipped', 0)} | "
            f"failed: {counts.get('failed', 0)} | not processable: "
            f"{counts.get('not_processable', 0)}"
        )
        print(f"Results index: {config.paths.results_index}")
        print(
            "Reminder: outputs are measured image differences, not confirmed geographic change."
        )
        return 0

    if args.command == "report":
        manager = DatasetManager(config)
        scan = manager.scan(save=True)
        paths = generate_reports(config, scan=scan)
        for name, path in paths.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "export-changes":
        result = export_changed_pairs(
            config,
            threshold=args.threshold,
            min_valid_overlap=args.min_valid_overlap,
            output_dir=args.output,
            include_unreliable=args.include_unreliable,
            force=args.force,
            keep_stale=args.keep_stale,
        )
        print(
            f"Exported {result['exported']} changed pair(s) above {result['threshold']:.2f}% | "
            f"skipped: {result['skipped']} | failed: {result['failed']} | "
            f"removed stale: {result['removed_stale']}"
        )
        print(f"Folder: {result['output']}")
        print(f"List: {result['changes_csv']}")
        print(
            "Only matching 2019/2024 originals were copied; originals in raw_screenshots/ "
            "were not moved or deleted."
        )
        return 0

    if args.command == "status":
        manager = DatasetManager(config)
        scan = manager.scan(save=not args.no_scan)
        summary = scan.summary
        index = read_json(config.paths.results_index, default={})
        tiles = index.get("tiles", {}) if isinstance(index, dict) else {}
        processed = (
            sum(1 for entry in tiles.values() if isinstance(entry, dict) and entry.get("status") == "processed")
            if isinstance(tiles, dict)
            else 0
        )
        print(f"Project: {config.project_name}")
        print(f"Years: {config.years[0]} (reference) / {config.years[1]} (target)")
        print(
            f"Screenshots: {summary['valid_files']}/{summary['expected_screenshots']} valid "
            f"({summary['file_completion_pct']:.1f}%)"
        )
        print(
            f"Tile pairs: {summary['complete_pairs']}/{summary['expected_tiles']} complete "
            f"({summary['completion_pct']:.1f}%), {summary['processable_pairs']} processable"
        )
        print(f"Processed results: {processed}")
        print(f"Missing files: {summary['missing_files']} | invalid: {summary['invalid_files']} | "
              f"duplicates: {summary['duplicate_files']} | unrecognized: "
              f"{summary['unrecognized_files']}")
        if summary["processable_pairs"] > processed:
            print(f"Next: python -m src.cli process  (run {summary['processable_pairs'] - processed} more)")
        elif processed:
            print("Next: python -m src.cli report && python -m src.cli export-changes")
        else:
            print("Next: add paired screenshots, then run validate and process")
        return 0

    if args.command == "capture-plan":
        slots = build_capture_plan(config, year=args.year)
        missing_by_year: dict[int, list[str]] = {}
        for slot in slots:
            if not slot.exists:
                missing_by_year.setdefault(slot.year, []).append(slot.tile_id)
        if args.missing_only:
            slots = [slot for slot in slots if not slot.exists]
        for slot in slots:
            marker = "OK" if slot.exists else "MISSING"
            print(
                f"{slot.tile_id} {slot.year} {marker:7s} {slot.expected_filename} "
                f"{slot.validation_status}"
            )
        for year in ((args.year,) if args.year is not None else config.years):
            print()
            print(
                capture_instructions(
                    config,
                    year,
                    list(dict.fromkeys(missing_by_year.get(year, [])))[:100],
                )
            )
        return 0

    if args.command == "capture":
        destination = capture_countdown(
            config,
            args.tile,
            args.year,
            countdown=args.countdown,
            overwrite=args.overwrite,
        )
        print(f"Saved screenshot: {destination}")
        print(
            "Record the imagery date you observed with the metadata command; it is not inferred."
        )
        return 0

    if args.command == "save-screenshot":
        if args.clipboard:
            destination = save_screenshot_clipboard(
                config,
                args.tile,
                args.year,
                overwrite=args.overwrite,
            )
        else:
            destination = save_screenshot_file(
                config,
                args.tile,
                args.year,
                args.source,
                overwrite=args.overwrite,
            )
        print(f"Saved {args.year} screenshot: {destination}")
        print("Run `python -m src.cli validate` to update the manifest.")
        return 0

    if args.command == "metadata":
        year = args.year
        if year is not None and year not in config.years:
            raise ValueError(f"Year {year} is not one of {config.years}.")
        upsert_override(
            config,
            args.tile,
            year,
            imagery_date=args.imagery_date,
            image_capture_date=args.capture_date,
            latitude=args.latitude,
            longitude=args.longitude,
            notes=args.notes,
        )
        manager = DatasetManager(config)
        scan = manager.scan(save=True)
        print(f"Updated manual_overrides.csv for {normalize_tile_id(args.tile)} "
              f"({year if year else 'all years'}).")
        print(
            f"Valid files: {scan.summary['valid_files']}/{scan.summary['expected_screenshots']}"
        )
        return 0

    if args.command == "review":
        set_review_status(
            config,
            args.tile,
            args.status,
            reviewer=args.reviewer,
            notes=args.notes,
        )
        print(f"Review status for {normalize_tile_id(args.tile)} set to {args.status}.")
        return 0

    if args.command == "dashboard":
        app_path = config.root / "app.py"
        if not app_path.exists():
            raise FileNotFoundError(f"Dashboard not found: {app_path}")
        command = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.port",
            str(args.port),
        ]
        print(f"Launching dashboard: {' '.join(command)}")
        return subprocess.call(command, cwd=str(config.root))

    parser_error = f"Unhandled command: {args.command}"
    LOGGER.error(parser_error)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

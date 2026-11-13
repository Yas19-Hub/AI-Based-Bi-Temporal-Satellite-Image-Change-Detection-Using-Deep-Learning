"""Streamlit dashboard for reviewing paired screenshot comparisons."""

from __future__ import annotations
import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.alignment import AlignmentError, align_pair
from src.config import AppConfig, load_config
from src.dataset import DatasetManager, PairInfo, ScanResult
from src.diff import DiffError, analyze_pair, render_comparison
from src.export_changes import ExportError, export_changed_pairs
from src.logging_setup import setup_logging
from src.reporting import generate_reports
from src.utils import DataFileError, display_path, load_image_rgb, read_json

st.set_page_config(
    page_title="Google Earth Change Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False, max_entries=6)
def _align_cached(
    reference_path: str,
    target_path: str,
    reference_mtime: int,
    target_mtime: int,
    alignment_settings: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], bool]:
    reference = load_image_rgb(Path(reference_path))
    target = load_image_rgb(Path(target_path))
    result = align_pair(reference, target, alignment_settings, resize_mismatched=True)
    return (
        result.reference,
        result.aligned_target,
        result.valid_mask,
        result.metrics(),
        result.resized,
    )


def _get_config() -> AppConfig:
    config = load_config()
    setup_logging(config, console=False)
    return config


def _get_scan(config: AppConfig) -> ScanResult:
    if "scan" not in st.session_state:
        with st.spinner("Scanning screenshots and validating files..."):
            st.session_state["scan"] = DatasetManager(config).scan(save=True)
    return st.session_state["scan"]


def _load_index(config: AppConfig) -> dict[str, Any]:
    try:
        payload = read_json(config.paths.results_index, default={})
    except DataFileError as exc:
        st.warning(f"Could not read the processing index: {exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _png_bytes(array: np.ndarray) -> bytes:
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L" if array.ndim == 2 else "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def _refresh() -> None:
    st.session_state.pop("scan", None)
    st.cache_data.clear()
    st.rerun()


def _download_file(path: Path, label: str, mime: str, key: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        st.download_button(
            label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            key=key,
        )
    else:
        st.caption(f"Not generated yet: {path.name}")


def _pair_frame(scan: ScanResult) -> pd.DataFrame:
    rows = []
    for pair in scan.pairs.values():
        rows.append(
            {
                "tile_id": pair.tile_id,
                "status": pair.status,
                "processable": pair.processable,
                "missing_years": ", ".join(str(year) for year in pair.missing_years),
                "invalid_years": ", ".join(str(year) for year in pair.invalid_years),
                "duplicate_years": ", ".join(str(year) for year in pair.duplicate_years),
                "notes": pair.notes,
            }
        )
    return pd.DataFrame(rows)


def _row_frame(scan: ScanResult) -> pd.DataFrame:
    rows = []
    for row in scan.rows:
        rows.append(
            {
                "tile_id": row.tile_id,
                "year": row.year,
                "filename": row.filename,
                "expected_filename": row.expected_filename,
                "validation_status": row.validation_status,
                "metadata_status": row.metadata_status,
                "imagery_date": row.imagery_date,
                "image_capture_date": row.image_capture_date,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "width": row.width,
                "height": row.height,
                "pair_status": row.pair_status,
                "validation_notes": row.validation_notes,
                "metadata_notes": row.metadata_notes,
                "relative_path": row.relative_path,
            }
        )
    return pd.DataFrame(rows)


def _flag_frame(index: dict[str, Any]) -> pd.DataFrame:
    tiles = index.get("tiles", {})
    rows = []
    if isinstance(tiles, dict):
        for tile_id, entry in sorted(tiles.items()):
            if not isinstance(entry, dict):
                continue
            rows.append(
                {
                    "tile_id": tile_id,
                    "status": entry.get("status", ""),
                    "alignment_reliable": entry.get("alignment", {}).get(
                        "alignment_reliable", ""
                    ),
                    "changed_pct_of_valid": entry.get("stats", {}).get(
                        "changed_pct_of_valid", ""
                    ),
                    "ssim_mean": entry.get("stats", {}).get("ssim_mean", ""),
                    "flags": ", ".join(str(flag) for flag in entry.get("flags", [])),
                    "error": entry.get("error", ""),
                }
            )
    return pd.DataFrame(rows)


def _overview(config: AppConfig, scan: ScanResult, index: dict[str, Any]) -> None:
    st.subheader("Overview")
    st.caption(
        "This dashboard reports measured differences between user-captured Google Earth Pro "
        "screenshots. It does not confirm real-world geographic change."
    )
    summary = scan.summary
    processed = sum(
        1
        for entry in index.get("tiles", {}).values()
        if isinstance(entry, dict) and entry.get("status") == "processed"
    )
    unreliable = sum(
        1
        for entry in index.get("tiles", {}).values()
        if isinstance(entry, dict)
        and entry.get("status") == "processed"
        and entry.get("alignment", {}).get("alignment_reliable") is False
    )
    columns = st.columns(5)
    columns[0].metric(
        "Screenshots", f"{summary['valid_files']}/{summary['expected_screenshots']}",
        f"{summary['file_completion_pct']:.1f}%",
    )
    columns[1].metric(
        "Complete pairs", f"{summary['complete_pairs']}/{summary['expected_tiles']}",
        f"{summary['completion_pct']:.1f}%",
    )
    columns[2].metric("Processable pairs", summary["processable_pairs"])
    columns[3].metric("Processed", processed)
    columns[4].metric("Alignment flagged", unreliable)

    st.progress(min(1.0, summary["file_completion_pct"] / 100.0), text="Screenshot completion")
    st.progress(min(1.0, summary["completion_pct"] / 100.0), text="Complete pair completion")
    if summary["processable_pairs"]:
        st.progress(
            min(1.0, processed / summary["processable_pairs"]),
            text=f"Processing progress ({processed}/{summary['processable_pairs']})",
        )

    left, right = st.columns(2)
    with left:
        st.markdown("**File validation status**")
        status_counts = pd.Series(summary["validation_statuses"], dtype="int64").sort_index()
        if not status_counts.empty:
            st.bar_chart(status_counts.rename("files"))
        else:
            st.info("No manifest rows yet.")
    with right:
        st.markdown("**Pair status**")
        pair_counts = pd.Series(summary["pair_statuses"], dtype="int64").sort_index()
        if not pair_counts.empty:
            st.bar_chart(pair_counts.rename("tiles"))
        else:
            st.info("No pairs yet.")

    st.markdown("**Dataset issues at a glance**")
    issue_cols = st.columns(5)
    issue_cols[0].metric("Missing files", summary["missing_files"])
    issue_cols[1].metric("Invalid files", summary["invalid_files"])
    issue_cols[2].metric("Duplicate files", summary["duplicate_files"])
    issue_cols[3].metric("Unrecognized files", summary["unrecognized_files"])
    issue_cols[4].metric("Tiles with flags", index.get("summary", {}).get("flagged", 0))

    with st.expander("Configuration"):
        st.write(
            {
                "project": config.project_name,
                "root": str(config.root),
                "years": config.years,
                "tile_count": len(config.tile_ids),
                "expected_dimensions": config.expected_dimensions,
                "alignment_method": config.processing.alignment.method,
                "pixel_threshold": config.processing.diff.pixel_threshold,
                "ssim_threshold": config.processing.diff.ssim_threshold,
                "results_index": display_path(config.paths.results_index, config.root),
            }
        )
        st.caption(
            "First year is the reference. Changing configured years invalidates old pairs; "
            "re-run validation and processing after editing config.yaml."
        )
    if isinstance(index, dict) and index.get("generated_utc"):
        st.caption(f"Processing index generated: {index['generated_utc']}")


def _select_tile(scan: ScanResult, index: dict[str, Any]) -> tuple[str, PairInfo] | None:
    pairs = scan.pairs
    tiles = index.get("tiles", {}) if isinstance(index, dict) else {}
    filter_choice = st.sidebar.selectbox(
        "Tile filter",
        [
            "All tiles",
            "Processable",
            "Not processed",
            "Processed",
            "Flagged",
            "Needs attention",
        ],
    )
    options: list[str] = []
    for tile_id in config_tile_ids(scan):
        pair = pairs[tile_id]
        entry = tiles.get(tile_id, {}) if isinstance(tiles, dict) else {}
        processed = isinstance(entry, dict) and entry.get("status") == "processed"
        flagged = isinstance(entry, dict) and bool(entry.get("flags"))
        if filter_choice == "Processable" and not pair.processable:
            continue
        if filter_choice == "Not processed" and (not pair.processable or processed):
            continue
        if filter_choice == "Processed" and not processed:
            continue
        if filter_choice == "Flagged" and not flagged:
            continue
        if filter_choice == "Needs attention" and pair.processable and not flagged:
            continue
        options.append(tile_id)
    if not options:
        st.info("No tiles match this filter.")
        return None
    labels = {}
    for tile_id in options:
        pair = pairs[tile_id]
        entry = tiles.get(tile_id, {}) if isinstance(tiles, dict) else {}
        suffix = ""
        if isinstance(entry, dict) and entry.get("status") == "processed":
            suffix = " | processed"
        elif pair.processable:
            suffix = " | ready"
        labels[tile_id] = f"{tile_id} — {pair.status}{suffix}"
    selected = st.sidebar.selectbox("Tile", options, format_func=lambda value: labels[value])
    return selected, pairs[selected]


def config_tile_ids(scan: ScanResult) -> list[str]:
    return sorted(scan.pairs)


def _load_tile_images(
    config: AppConfig,
    pair: PairInfo,
    entry: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], bool, str]:
    paths = entry.get("paths", {}) if isinstance(entry, dict) else {}
    reference_path = paths.get("reference", "")
    target_path = paths.get("target", "")
    valid_path = paths.get("valid_mask", "")
    if reference_path and target_path and valid_path:
        reference_file = config.root / reference_path
        target_file = config.root / target_path
        valid_file = config.root / valid_path
        if all(path.exists() for path in (reference_file, target_file, valid_file)):
            reference = load_image_rgb(reference_file)
            target = load_image_rgb(target_file)
            valid = np.asarray(Image.open(valid_file).convert("L"), dtype=np.uint8)
            metrics = dict(entry.get("alignment", {}))
            metrics.pop("notes_list", None)
            return (
                reference,
                target,
                valid,
                metrics,
                bool(entry.get("stats", {}).get("target_resized", False)),
                "precomputed aligned outputs",
            )
    if pair.record_a is None or pair.record_b is None:
        raise AlignmentError("No usable screenshot records for this pair.")
    reference_file = pair.record_a.absolute_path(config)
    target_file = pair.record_b.absolute_path(config)
    reference_mtime = reference_file.stat().st_mtime_ns
    target_mtime = target_file.stat().st_mtime_ns
    reference, target, valid, metrics, resized = _align_cached(
        str(reference_file),
        str(target_file),
        reference_mtime,
        target_mtime,
        config.processing.alignment,
    )
    return reference, target, valid, metrics, resized, "live alignment from originals"


def _tile_view(config: AppConfig, scan: ScanResult, index: dict[str, Any]) -> None:
    selection = _select_tile(scan, index)
    if selection is None:
        return
    tile_id, pair = selection
    tiles = index.get("tiles", {}) if isinstance(index, dict) else {}
    entry = tiles.get(tile_id, {}) if isinstance(tiles, dict) else {}
    if not isinstance(entry, dict):
        entry = {}

    st.subheader(f"Tile {tile_id}")
    st.caption(
        f"{pair.year_a} reference | {pair.year_b} target | Pair status: {pair.status}"
    )
    if pair.notes:
        st.info(pair.notes)

    with st.expander("Adjustable comparison thresholds", expanded=False):
        controls = st.columns(4)
        pixel_threshold = controls[0].slider(
            "Pixel difference threshold",
            0,
            255,
            int(config.dashboard.default_pixel_threshold),
        )
        ssim_threshold = controls[1].slider(
            "SSIM threshold (higher = more sensitive)",
            0.0,
            1.0,
            float(config.dashboard.default_ssim_threshold),
            step=0.01,
        )
        combine_mode = controls[2].selectbox(
            "Combine conditions",
            ["and", "or"],
            index=0 if config.processing.diff.combine_mode == "and" else 1,
            help="'and' requires both thresholds; 'or' accepts either.",
        )
        min_region = controls[3].number_input(
            "Minimum region area (px)",
            min_value=0,
            max_value=100000,
            value=int(config.processing.diff.min_region_area),
            step=10,
        )
        st.caption(
            "Changing these values recomputes the difference map live. It does not modify "
            "the saved outputs until you re-run processing with changed config.yaml values."
        )

    if not pair.processable and not entry.get("paths"):
        st.warning("This pair is not processable. Fix the validation issues shown below.")
        _show_pair_issues(pair)
        return

    try:
        (
            reference,
            target,
            valid_mask,
            alignment_metrics,
            target_resized,
            source,
        ) = _load_tile_images(config, pair, entry)
    except (AlignmentError, OSError, ValueError, DataFileError) as exc:
        st.error(f"Could not load/align this tile: {exc}")
        _show_pair_issues(pair)
        return

    diff_settings = replace(
        config.processing.diff,
        pixel_threshold=int(pixel_threshold),
        ssim_threshold=float(ssim_threshold),
        combine_mode=combine_mode,
        min_region_area=int(min_region),
    )
    try:
        analysis = analyze_pair(
            reference,
            target,
            valid_mask,
            diff_settings,
            target_resized=target_resized,
        )
    except DiffError as exc:
        st.error(f"Could not compare this tile: {exc}")
        return

    alignment_reliable = alignment_metrics.get("alignment_reliable")
    alignment_method = alignment_metrics.get("alignment_method")
    if alignment_reliable is False and alignment_method not in {None, "", "none"}:
        st.warning(
            "Alignment was flagged unreliable. Treat the change map as indicative only and "
            "review the raw imagery. " + str(alignment_metrics.get("alignment_notes", ""))
        )
    elif alignment_method == "none":
        st.caption("Alignment is disabled in config.yaml; images are compared as captured.")
    flags = list(analysis.stats.get("image_flags", []))
    if target_resized:
        flags.append("target_resized")
    if flags:
        st.caption("Flags: " + ", ".join(dict.fromkeys(flags)))

    metrics = st.columns(5)
    metrics[0].metric("Changed % of valid", f"{analysis.stats['changed_pct_of_valid']:.2f}%")
    metrics[1].metric("Mean abs diff", f"{analysis.stats['mean_abs_diff']:.2f}")
    metrics[2].metric("SSIM mean", f"{analysis.stats['ssim_mean']:.3f}")
    metrics[3].metric("Candidate regions", analysis.stats["region_count"])
    metrics[4].metric("Valid overlap", f"{analysis.stats['valid_pct']:.1f}%")
    st.caption(f"Image source: {source}. {analysis.stats['width']}x{analysis.stats['height']}.")

    tabs = st.tabs(["Side by side", "Change map", "Statistics", "Regions", "Metadata", "Downloads"])
    with tabs[0]:
        left, right = st.columns(2)
        left.image(analysis.reference, caption=f"{pair.year_a} reference", width="stretch")
        right.image(analysis.target, caption=f"{pair.year_b} target", width="stretch")
        st.image(
            analysis.annotated,
            caption="Candidate regions (image differences only)",
            width="stretch",
        )
    with tabs[1]:
        choice = st.radio(
            "Layer",
            ["Heatmap", "SSIM dissimilarity", "Change mask", "Overlay", "Annotated"],
            horizontal=True,
        )
        chosen = {
            "Heatmap": analysis.heatmap,
            "SSIM dissimilarity": analysis.ssim_heatmap,
            "Change mask": analysis.change_mask,
            "Overlay": analysis.overlay,
            "Annotated": analysis.annotated,
        }[choice]
        st.image(chosen, caption=choice, width="stretch")
        st.caption(
            "Brighter/red colours indicate larger measured image differences. Mask is white "
            "where both pixel and SSIM thresholds are satisfied (or either, if combine=or)."
        )
    with tabs[2]:
        stats_frame = pd.DataFrame(
            [{"measure": key, "value": _format_stat(value)} for key, value in analysis.stats.items()]
        )
        st.dataframe(stats_frame, hide_index=True, width="stretch")
        st.markdown("**Alignment**")
        alignment_frame = pd.DataFrame(
            [{"measure": key, "value": _format_stat(value)} for key, value in alignment_metrics.items()]
        )
        st.dataframe(alignment_frame, hide_index=True, width="stretch")
    with tabs[3]:
        if analysis.regions:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "region": region.label,
                            "x": region.x,
                            "y": region.y,
                            "width": region.width,
                            "height": region.height,
                            "area_px": region.area,
                            "mean_abs_diff": round(region.mean_abs_diff, 2),
                            "mean_ssim": round(region.mean_ssim, 3),
                        }
                        for region in analysis.regions
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("No candidate regions pass the current thresholds and minimum area.")
    with tabs[4]:
        metadata_rows = []
        for record in (pair.record_a, pair.record_b):
            if record is None:
                continue
            metadata_rows.append(
                {
                    "year": record.year,
                    "filename": record.filename,
                    "imagery_date": record.imagery_date or "(unknown)",
                    "image_capture_date": record.image_capture_date or "(unknown)",
                    "latitude": record.latitude or "(unknown)",
                    "longitude": record.longitude or "(unknown)",
                    "metadata_status": record.metadata_status,
                    "metadata_notes": record.metadata_notes,
                }
            )
        st.dataframe(pd.DataFrame(metadata_rows), hide_index=True, width="stretch")
        st.caption(
            "Imagery dates and coordinates are user-entered, never inferred. Use "
            "`python -m src.cli metadata ...` and re-validate after edits."
        )
    with tabs[5]:
        comparison = render_comparison(
            analysis,
            f"{pair.year_a} reference",
            f"{pair.year_b} target",
            [
                f"Thresholds: pixel >= {pixel_threshold}, SSIM < {ssim_threshold:.2f}, "
                f"combine={combine_mode}, min region {min_region}px",
                "Measured image differences only - not confirmed geographic change.",
            ],
        )
        download_cols = st.columns(3)
        download_cols[0].download_button(
            "Current comparison PNG",
            data=_png_bytes(comparison),
            file_name=f"{tile_id}_current_comparison.png",
            mime="image/png",
        )
        download_cols[1].download_button(
            "Current change mask PNG",
            data=_png_bytes(analysis.change_mask),
            file_name=f"{tile_id}_current_change_mask.png",
            mime="image/png",
        )
        download_cols[2].download_button(
            "Current overlay PNG",
            data=_png_bytes(analysis.overlay),
            file_name=f"{tile_id}_current_overlay.png",
            mime="image/png",
        )
        stats_payload = {
            "tile_id": tile_id,
            "years": [pair.year_a, pair.year_b],
            "thresholds": {
                "pixel_threshold": pixel_threshold,
                "ssim_threshold": ssim_threshold,
                "combine_mode": combine_mode,
                "min_region_area": min_region,
            },
            "alignment": alignment_metrics,
            "stats": analysis.stats,
            "regions": [
                {
                    "index": region.index,
                    "x": region.x,
                    "y": region.y,
                    "width": region.width,
                    "height": region.height,
                    "area": region.area,
                    "mean_abs_diff": region.mean_abs_diff,
                    "mean_ssim": region.mean_ssim,
                }
                for region in analysis.regions
            ],
        }
        st.download_button(
            "Current statistics JSON",
            data=json.dumps(stats_payload, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            file_name=f"{tile_id}_current_stats.json",
            mime="application/json",
        )
        st.markdown("**Saved batch outputs**")
        paths = entry.get("paths", {}) if isinstance(entry, dict) else {}
        saved_cols = st.columns(3)
        keys = list(paths)
        for index_key, key in enumerate(keys):
            path = config.root / paths[key]
            with saved_cols[index_key % 3]:
                _download_file(
                    path,
                    f"Saved {key}",
                    "image/png" if path.suffix.lower() == ".png" else "application/octet-stream",
                    key=f"saved_{tile_id}_{key}",
                )


def _show_pair_issues(pair: PairInfo) -> None:
    rows = []
    for record in (pair.record_a, pair.record_b):
        if record is None:
            continue
        rows.append(
            {
                "year": record.year,
                "filename": record.filename,
                "exists": record.file_exists,
                "validation_status": record.validation_status,
                "validation_notes": record.validation_notes,
                "metadata_status": record.metadata_status,
            }
        )
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _format_stat(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return value


def _quality(config: AppConfig, scan: ScanResult, index: dict[str, Any]) -> None:
    st.subheader("Data quality and missing files")
    rows = _row_frame(scan)
    pair_frame = _pair_frame(scan)
    tabs = st.tabs(["Issues", "Missing", "Unrecognized", "Metadata gaps", "Processed flags", "Reports"])

    with tabs[0]:
        if rows.empty:
            st.info("No manifest rows.")
        else:
            issues = rows[
                (rows["validation_status"] != "valid") | (rows["metadata_status"] != "complete")
            ]
            st.dataframe(issues, hide_index=True, width="stretch")
            st.download_button(
                "Download dataset issues CSV",
                data=_frame_bytes(issues),
                file_name="dataset_issues.csv",
                mime="text/csv",
            )
    with tabs[1]:
        missing = rows[rows["validation_status"] == "missing_file"] if not rows.empty else rows
        st.dataframe(missing, hide_index=True, width="stretch")
        if not pair_frame.empty:
            bad_pairs = pair_frame[~pair_frame["processable"]]
            st.markdown("**Non-processable pairs**")
            st.dataframe(bad_pairs, hide_index=True, width="stretch")
    with tabs[2]:
        if scan.unrecognized:
            st.dataframe(pd.DataFrame(scan.unrecognized), hide_index=True, width="stretch")
        else:
            st.success("No unrecognized files.")
        st.caption(
            "Expected names are T###_YYYY.png (or another configured extension) inside the "
            "matching year folder. Files outside the tile range or with mismatched years are "
            "reported here, never silently ignored."
        )
    with tabs[3]:
        if rows.empty:
            st.info("No manifest rows.")
        else:
            gaps = rows[rows["metadata_status"] != "complete"]
            st.dataframe(gaps, hide_index=True, width="stretch")
            st.caption(
                "Imagery dates must be transcribed from Google Earth Pro. Blank dates are "
                "unknown, not inferred."
            )
    with tabs[4]:
        flags = _flag_frame(index)
        if flags.empty:
            st.info("No processed results. Run `python -m src.cli process` first.")
        else:
            only_flagged = flags[
                (flags["flags"].astype(str).str.len() > 0)
                | (flags["alignment_reliable"].astype(str).str.lower() == "false")
                | (flags["error"].astype(str).str.len() > 0)
            ]
            st.dataframe(only_flagged, hide_index=True, width="stretch")
            st.download_button(
                "Download processed quality flags CSV",
                data=_frame_bytes(only_flagged),
                file_name="quality_flags.csv",
                mime="text/csv",
            )
    with tabs[5]:
        if st.button("Regenerate all reports"):
            with st.spinner("Scanning and writing reports..."):
                manager = DatasetManager(config)
                new_scan = manager.scan(save=True)
                generate_reports(config, scan=new_scan)
                st.session_state["scan"] = new_scan
            st.success("Reports regenerated.")
        report_files = [
            ("Validation report (Markdown)", config.paths.reports / "validation_report.md", "text/markdown"),
            ("Processing report (Markdown)", config.paths.reports / "processing_report.md", "text/markdown"),
            ("Dataset issues (CSV)", config.paths.reports / "dataset_issues.csv", "text/csv"),
            ("Missing files (CSV)", config.paths.reports / "missing_files.csv", "text/csv"),
            ("Metadata gaps (CSV)", config.paths.reports / "metadata_gaps.csv", "text/csv"),
            ("Change summary (CSV)", config.paths.reports / "change_summary.csv", "text/csv"),
            ("Quality flags (CSV)", config.paths.reports / "quality_flags.csv", "text/csv"),
        ]
        columns = st.columns(2)
        for index_file, (label, path, mime) in enumerate(report_files):
            with columns[index_file % 2]:
                _download_file(path, label, mime, key=f"report_{path.name}")


def _downloads(config: AppConfig, scan: ScanResult, index: dict[str, Any]) -> None:
    st.subheader("Downloads and exports")
    rows = _row_frame(scan)
    pairs = _pair_frame(scan)
    flags = _flag_frame(index)
    columns = st.columns(2)
    with columns[0]:
        st.markdown("**Dataset tables**")
        st.download_button(
            "Manifest rows CSV",
            data=_frame_bytes(rows),
            file_name="tile_manifest.csv",
            mime="text/csv",
        )
        st.download_button(
            "Pair summary CSV",
            data=_frame_bytes(pairs),
            file_name="pair_summary.csv",
            mime="text/csv",
        )
        if not flags.empty:
            st.download_button(
                "Processed results CSV",
                data=_frame_bytes(flags),
                file_name="processing_results.csv",
                mime="text/csv",
            )
        _download_file(
            config.paths.tile_stats,
            "Tile statistics CSV",
            "text/csv",
            key="download_tile_stats",
        )
        _download_file(
            config.paths.results_index,
            "Results index JSON",
            "application/json",
            key="download_results_json",
        )
    with columns[1]:
        st.markdown("**Reports**")
        for key, path, mime in [
            ("download_validation_md", config.paths.reports / "validation_report.md", "text/markdown"),
            ("download_processing_md", config.paths.reports / "processing_report.md", "text/markdown"),
            ("download_issues_csv", config.paths.reports / "dataset_issues.csv", "text/csv"),
            ("download_change_csv", config.paths.reports / "change_summary.csv", "text/csv"),
            ("download_flags_csv", config.paths.reports / "quality_flags.csv", "text/csv"),
        ]:
            _download_file(path, path.name, mime, key=key)
        st.caption("Use the Data quality → Reports tab to regenerate report files.")


def _changed_pairs(config: AppConfig, index: dict[str, Any]) -> None:
    st.subheader("Keep only changed 2019/2024 pairs")
    st.caption(
        "Copies only the original before/after screenshots for tiles whose measured difference "
        "exceeds the threshold. Originals in raw_screenshots/ are never moved or deleted."
    )
    tiles = index.get("tiles", {}) if isinstance(index, dict) else {}
    processed = (
        [entry for entry in tiles.values() if isinstance(entry, dict) and entry.get("status") == "processed"]
        if isinstance(tiles, dict)
        else []
    )
    if not processed:
        st.info(
            "No processed tile pairs yet. Add paired screenshots, run validation and processing, "
            "then return to this page."
        )
        return

    controls = st.columns(4)
    threshold = controls[0].number_input(
        "Changed % of valid overlap (must be exceeded)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
    )
    min_overlap = controls[1].number_input(
        "Minimum valid overlap %",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
    )
    include_unreliable = controls[2].checkbox("Include unreliable alignment", value=False)
    keep_stale = controls[3].checkbox("Keep previously exported copies", value=False)

    preview_rows = []
    for tile_id, entry in sorted(tiles.items()):
        if not isinstance(entry, dict) or entry.get("status") != "processed":
            continue
        stats = entry.get("stats", {})
        alignment = entry.get("alignment", {})
        method = str(alignment.get("alignment_method", ""))
        reliable = alignment.get("alignment_reliable") is True or method == "none"
        changed = _as_float(stats.get("changed_pct_of_valid"))
        valid = _as_float(stats.get("valid_pct"))
        if (reliable or include_unreliable) and changed > threshold and valid >= min_overlap:
            preview_rows.append(
                {
                    "tile_id": tile_id,
                    "changed_pct_of_valid": changed,
                    "ssim_mean": stats.get("ssim_mean", ""),
                    "valid_pct": valid,
                    "alignment_method": method,
                    "alignment_reliable": alignment.get("alignment_reliable", ""),
                }
            )
    st.markdown(f"**Tiles currently qualifying: {len(preview_rows)}**")
    if preview_rows:
        st.dataframe(pd.DataFrame(preview_rows), hide_index=True, width="stretch")
    else:
        st.caption("No processed tiles exceed the current settings.")

    if st.button("Export changed pairs", type="primary"):
        try:
            result = export_changed_pairs(
                config,
                threshold=float(threshold),
                min_valid_overlap=float(min_overlap),
                include_unreliable=include_unreliable,
                keep_stale=keep_stale,
            )
        except ExportError as exc:
            st.error(str(exc))
        else:
            st.success(
                f"Exported {result['exported']} pair(s); skipped {result['skipped']}; "
                f"removed stale copies: {result['removed_stale']}."
            )
            st.write(f"Folder: `{result['output']}`")
            st.write(f"Tile list: `{result['changes_csv']}`")
            for year in config.years:
                folder = Path(result["output"]) / str(year)
                if folder.exists():
                    st.caption(f"{year}: {folder}")
            st.warning(
                "These are measured image differences, not confirmed geographic change. "
                "Review the comparison outputs and source imagery."
            )

    st.divider()
    st.markdown("**Previously exported tile list**")
    _download_file(
        Path(config.root) / "changes_only" / "changes.csv",
        "changes.csv",
        "text/csv",
        key="changed_pairs_changes_csv",
    )
    _download_file(
        Path(config.root) / "changes_only" / "skipped.csv",
        "skipped.csv",
        "text/csv",
        key="changed_pairs_skipped_csv",
    )


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    config = _get_config()
    st.sidebar.title("Google Earth changes")
    st.sidebar.caption(
        "Measured image differences between paired historical screenshots. "
        "Not confirmed geographic change."
    )
    if st.sidebar.button("Refresh data"):
        _refresh()
    st.sidebar.markdown(
        "**Manual capture reminder**\n\nSelect historical imagery yourself in Google Earth Pro, "
        "then save the map image as `T###_YYYY.png`. Dates are transcribed manually."
    )
    st.sidebar.caption(f"Project root: {config.root}")

    try:
        scan = _get_scan(config)
    except (DataFileError, OSError, ValueError) as exc:
        st.error(f"Could not scan the dataset: {exc}")
        st.stop()
    index = _load_index(config)

    page = st.sidebar.radio(
        "Page", ["Overview", "Tile explorer", "Changed pairs", "Data quality", "Downloads"]
    )
    st.title(config.project_name)
    if page == "Overview":
        _overview(config, scan, index)
    elif page == "Tile explorer":
        _tile_view(config, scan, index)
    elif page == "Changed pairs":
        _changed_pairs(config, index)
    elif page == "Data quality":
        _quality(config, scan, index)
    else:
        _downloads(config, scan, index)


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import add_screenshot, make_texture
from src.capture_assistant import (
    CaptureError,
    build_capture_plan,
    capture_countdown,
    save_screenshot_clipboard,
    save_screenshot_file,
)
from src.config import AppConfig
from src.dataset import DatasetManager
from src.processing import artefact_paths, run_processing
from src.reporting import generate_reports
from src.utils import verify_written_image


def _two_year_change(config: AppConfig) -> None:
    reference = make_texture(120, 90, seed=21, low=160, high=230)
    target = reference.copy()
    target[25:55, 35:85] = 0
    add_screenshot(config, "T001", 2019, reference)
    add_screenshot(config, "T001", 2024, target)


def test_processing_writes_verified_outputs_and_index(project: AppConfig) -> None:
    _two_year_change(project)
    index = run_processing(project)
    entry = index["tiles"]["T001"]
    assert entry["status"] == "processed"
    assert entry["stats"]["changed_pct_of_valid"] > 0
    assert entry["alignment"]["alignment_method"] == "none"

    paths = artefact_paths(project, "T001")
    for key in (
        "reference",
        "target",
        "comparison",
        "diff",
        "heatmap",
        "ssim_map",
        "ssim_heatmap",
        "mask",
        "valid_mask",
        "overlay",
        "annotated",
    ):
        assert paths[key].exists(), key
        assert paths[key].stat().st_size > 0
    verify_written_image(paths["comparison"])
    verify_written_image(paths["reference"], 120, 90)

    payload = json.loads(project.paths.results_index.read_text(encoding="utf-8"))
    assert payload["tiles"]["T001"]["status"] == "processed"
    assert payload["summary"]["processed"] == 1
    assert project.paths.tile_stats.exists()
    stats_text = project.paths.tile_stats.read_text(encoding="utf-8")
    assert "T001" in stats_text
    assert "changed_pct_of_valid" in stats_text


def test_processing_marks_missing_pairs_without_outputs(project: AppConfig) -> None:
    add_screenshot(project, "T001", 2019, seed=31)  # type: ignore[call-arg]
    index = run_processing(project, tile_ids=["T002"])
    entry = index["tiles"]["T002"]
    assert entry["status"] == "not_processable"
    assert entry["error"].startswith("missing_both")
    paths = artefact_paths(project, "T002")
    assert not paths["comparison"].exists()


def test_processing_skips_existing_and_force_recomputes(project: AppConfig) -> None:
    _two_year_change(project)
    first = run_processing(project)
    assert first["summary"]["run_counts"]["processed"] == 1
    second = run_processing(project)
    assert second["summary"]["run_counts"]["skipped"] == 1
    forced = run_processing(project, force=True)
    assert forced["summary"]["run_counts"]["processed"] == 1


def test_reports_are_generated_from_synthetic_results(project: AppConfig) -> None:
    _two_year_change(project)
    managers_scan = DatasetManager(project).scan(save=True)
    index = run_processing(project)
    paths = generate_reports(project, scan=managers_scan, index=index)
    for key in ("validation_markdown", "processing_markdown", "issues", "missing", "unrecognized"):
        assert paths[key].exists()
        assert paths[key].stat().st_size > 0
    validation = paths["validation_markdown"].read_text(encoding="utf-8")
    processing = paths["processing_markdown"].read_text(encoding="utf-8")
    assert "Dataset validation report" in validation
    assert "Processing report" in processing
    assert "not confirm geographic" in processing
    assert "T001" in processing


def test_capture_plan_lists_missing_slots(project: AppConfig) -> None:
    add_screenshot(project, "T001", 2019, seed=41)  # type: ignore[call-arg]
    add_screenshot(project, "T001", 2024, seed=42)  # type: ignore[call-arg]
    slots = build_capture_plan(project)
    assert len(slots) == 6
    missing = {(slot.tile_id, slot.year) for slot in slots if not slot.exists}
    assert ("T001", 2019) not in missing
    assert ("T002", 2019) in missing


def test_countdown_capture_writes_canonical_file_and_refuses_overwrite(
    project: AppConfig,
) -> None:
    def fake_grab(**kwargs: object) -> Image.Image:
        return Image.new("RGB", (160, 120), (80, 140, 200))

    destination = capture_countdown(
        project,
        "T002",
        2019,
        countdown=0,
        sleep=lambda seconds: None,
        grab=fake_grab,
    )
    assert destination.name == "T002_2019.png"
    assert destination.exists()
    verify_written_image(destination, 160, 120)

    try:
        capture_countdown(
            project,
            "T002",
            2019,
            countdown=0,
            sleep=lambda seconds: None,
            grab=fake_grab,
        )
    except CaptureError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Expected capture to refuse overwriting an original")

    destination = capture_countdown(
        project,
        "T002",
        2019,
        countdown=0,
        overwrite=True,
        sleep=lambda seconds: None,
        grab=fake_grab,
    )
    assert destination.exists()


def test_capture_rejects_flat_screen(project: AppConfig) -> None:
    def flat_grab(**kwargs: object) -> Image.Image:
        return Image.new("RGB", (160, 120), (20, 20, 20))

    try:
        capture_countdown(
            project,
            "T001",
            2024,
            countdown=0,
            sleep=lambda seconds: None,
            grab=flat_grab,
        )
    except CaptureError as exc:
        assert "flat colour" in str(exc)
    else:
        raise AssertionError("Expected flat screen capture to be rejected")


def test_save_screenshot_file_copies_canonical_name_and_preserves_hash(
    project: AppConfig, tmp_path: Path
) -> None:
    source = tmp_path / "my capture.jpg"
    Image.fromarray(make_texture(120, 90, seed=51), mode="RGB").save(source, format="JPEG")
    destination = save_screenshot_file(project, "T001", 2019, source)
    expected = project.paths.raw_year_dir(2019) / "T001_2019.jpg"
    assert destination == expected
    assert destination.exists()
    assert destination.read_bytes() == source.read_bytes()
    assert source.exists()

    try:
        save_screenshot_file(project, "T001", 2019, source)
    except CaptureError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("Expected existing original to be protected")


def test_save_screenshot_clipboard_writes_png(project: AppConfig) -> None:
    def fake_clipboard() -> Image.Image:
        return Image.fromarray(make_texture(140, 100, seed=52), mode="RGB")

    destination = save_screenshot_clipboard(
        project, "T002", 2024, grab=fake_clipboard
    )
    assert destination.name == "T002_2024.png"
    assert destination.exists()
    verify_written_image(destination, 140, 100)


def test_save_screenshot_rejects_bad_inputs(project: AppConfig, tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    try:
        save_screenshot_file(project, "T001", 2019, missing)
    except CaptureError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("Expected missing source to be rejected")

    invalid = tmp_path / "invalid.png"
    invalid.write_text("not an image", encoding="utf-8")
    try:
        save_screenshot_file(project, "T001", 2019, invalid)
    except CaptureError as exc:
        assert "Invalid screenshot" in str(exc)
    else:
        raise AssertionError("Expected invalid image to be rejected")

    with pytest.raises(CaptureError):
        save_screenshot_file(project, "T999", 2019, invalid)


def test_capture_rejects_unconfigured_tile_and_year(project: AppConfig) -> None:
    def fake_grab(**kwargs: object) -> Image.Image:
        return Image.new("RGB", (160, 120), (80, 140, 200))

    for tile, year in (("T999", 2019), ("T001", 2030)):
        try:
            capture_countdown(
                project,
                tile,
                year,
                countdown=0,
                sleep=lambda seconds: None,
                grab=fake_grab,
            )
        except CaptureError:
            pass
        else:
            raise AssertionError(f"Expected rejection for {tile} {year}")

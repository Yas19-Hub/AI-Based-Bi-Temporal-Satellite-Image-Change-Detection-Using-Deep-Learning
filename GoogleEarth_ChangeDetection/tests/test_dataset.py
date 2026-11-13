from pathlib import Path

import pytest

from src.config import AppConfig
from src.dataset import DatasetManager, init_project, upsert_override
from src.utils import DataFileError


def test_full_expected_grid_has_200_slots(full_project: AppConfig) -> None:
    scan = DatasetManager(full_project).scan(save=True)
    assert len(scan.rows) == 200
    assert len(scan.pairs) == 100
    assert scan.summary["expected_screenshots"] == 200
    assert scan.summary["missing_files"] == 200
    assert all(row.validation_status == "missing_file" for row in scan.rows)
    assert full_project.paths.manifest.exists()
    lines = full_project.paths.manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 201


def test_pairing_and_missing_slots(project: AppConfig) -> None:
    from conftest import add_screenshot

    add_screenshot(project, "T001", 2019, seed=1)  # type: ignore[call-arg]
    add_screenshot(project, "T001", 2024, seed=2)  # type: ignore[call-arg]
    add_screenshot(project, "T002", 2019, seed=3)  # type: ignore[call-arg]
    scan = DatasetManager(project).scan(save=True)

    assert scan.summary["valid_files"] == 3
    assert scan.summary["missing_files"] == 3
    assert scan.pairs["T001"].status == "complete"
    assert scan.pairs["T001"].processable is True
    assert scan.pairs["T002"].status == "missing_2024"
    assert scan.pairs["T002"].processable is False
    assert scan.pairs["T003"].status == "missing_both"
    rows_003 = [row for row in scan.rows if row.tile_id == "T003"]
    assert len(rows_003) == 2
    assert all(row.file_exists is False for row in rows_003)


def test_invalid_image_is_reported_not_processed(project: AppConfig) -> None:
    bad = project.paths.raw_year_dir(2019) / "T001_2019.png"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("this is not an image", encoding="utf-8")
    from conftest import add_screenshot

    add_screenshot(project, "T001", 2024, seed=2)  # type: ignore[call-arg]
    scan = DatasetManager(project).scan(save=True)

    record = next(row for row in scan.rows if row.tile_id == "T001" and row.year == 2019)
    assert record.validation_status == "invalid_image"
    assert "Unreadable image" in record.validation_notes
    assert scan.pairs["T001"].status == "invalid_2019"
    assert scan.pairs["T001"].processable is False


def test_duplicate_valid_file_blocks_pair_without_hiding_it(project: AppConfig) -> None:
    from conftest import add_screenshot

    add_screenshot(project, "T001", 2019, seed=1)  # type: ignore[call-arg]
    add_screenshot(project, "T001", 2019, seed=2, filename="T001_2019.bmp")  # type: ignore[call-arg]
    add_screenshot(project, "T001", 2024, seed=3)  # type: ignore[call-arg]
    scan = DatasetManager(project).scan(save=True)

    rows = [row for row in scan.rows if row.tile_id == "T001" and row.year == 2019]
    assert len(rows) == 2
    assert [row.is_primary for row in rows].count(True) == 1
    assert any(row.validation_status == "duplicate" for row in rows)
    assert scan.pairs["T001"].status == "duplicate_2019"
    assert scan.pairs["T001"].processable is False


def test_unrecognized_and_year_mismatch_are_reported(project: AppConfig) -> None:
    from conftest import add_screenshot

    add_screenshot(project, "T001", 2019, seed=1)  # type: ignore[call-arg]
    raw_2019 = project.paths.raw_year_dir(2019)
    add_screenshot(project, "T001", 2024, seed=2, filename="T001_2024.png")  # type: ignore[call-arg]
    (raw_2019 / "random.png").write_bytes(b"x")
    add_screenshot(project, "T999", 2019, seed=3)  # type: ignore[call-arg]
    add_screenshot(project, "T002", 2024, seed=4, filename="T002_2019.png")  # type: ignore[call-arg]
    add_screenshot(project, "T002", 2024, seed=5, filename="T002_2024.png")  # type: ignore[call-arg]

    scan = DatasetManager(project).scan(save=True)
    reasons = {item["reason"] for item in scan.unrecognized}
    assert "unrecognized_filename" in reasons
    assert "unexpected_tile" in reasons
    assert "year_folder_mismatch" in reasons
    assert scan.summary["unrecognized_files"] == 3
    # A misnamed 2024 image in the 2024 folder is recognized, but T002 2019 is absent.
    assert scan.pairs["T002"].status == "missing_2019"


def test_metadata_is_persisted_and_never_inferred(project: AppConfig) -> None:
    from conftest import add_screenshot

    add_screenshot(project, "T001", 2019, seed=1)  # type: ignore[call-arg]
    add_screenshot(project, "T001", 2024, seed=2)  # type: ignore[call-arg]
    scan = DatasetManager(project).scan(save=True)
    record = next(row for row in scan.rows if row.tile_id == "T001" and row.year == 2019)
    assert record.imagery_date == ""
    assert record.latitude == ""
    assert record.metadata_status == "missing"

    upsert_override(
        project,
        "T001",
        2019,
        imagery_date="2019-06-15",
        image_capture_date="2026-10-21",
        latitude="51.5",
        longitude="-0.12",
        notes="observed in Google Earth Pro",
    )
    scan = DatasetManager(project).scan(save=True)
    record = next(row for row in scan.rows if row.tile_id == "T001" and row.year == 2019)
    assert record.imagery_date == "2019-06-15"
    assert record.latitude == "51.5"
    assert record.longitude == "-0.12"
    assert record.metadata_status == "complete"
    assert "Override note" in record.metadata_notes

    scan = DatasetManager(project).scan(save=True)
    record = next(row for row in scan.rows if row.tile_id == "T001" and row.year == 2019)
    assert record.imagery_date == "2019-06-15"
    assert record.metadata_status == "complete"

    other_year = next(row for row in scan.rows if row.tile_id == "T001" and row.year == 2024)
    assert other_year.imagery_date == ""
    assert other_year.metadata_status == "missing"


def test_invalid_override_values_are_rejected(project: AppConfig) -> None:
    with pytest.raises(ValueError, match="Invalid imagery_date"):
        upsert_override(project, "T001", 2019, imagery_date="summer 2019")
    with pytest.raises(ValueError, match="Invalid latitude"):
        upsert_override(project, "T001", 2019, latitude="999")
    with pytest.raises(ValueError, match="not one of the configured years"):
        upsert_override(project, "T001", 2030, imagery_date="2030")


def test_expected_dimensions_are_enforced(tmp_path: Path) -> None:
    from conftest import add_screenshot, make_texture, write_config

    config = write_config(tmp_path, expected_dimensions=(120, 90))
    add_screenshot(config, "T001", 2019, make_texture(120, 90, seed=1))
    add_screenshot(config, "T001", 2024, make_texture(100, 80, seed=2))
    scan = DatasetManager(config).scan(save=True)
    record = next(row for row in scan.rows if row.tile_id == "T001" and row.year == 2024)
    assert record.validation_status == "unexpected_dimensions"
    assert scan.pairs["T001"].status == "invalid_2024"


def test_init_creates_templates_and_manifest(project: AppConfig) -> None:
    result = init_project(project)
    assert project.paths.manifest.exists()
    assert project.paths.pair_summary.exists()
    assert project.paths.tile_registry.exists()
    assert project.paths.manual_overrides.exists()
    assert project.paths.review_status.exists()
    assert result["summary"]["expected_screenshots"] == 6

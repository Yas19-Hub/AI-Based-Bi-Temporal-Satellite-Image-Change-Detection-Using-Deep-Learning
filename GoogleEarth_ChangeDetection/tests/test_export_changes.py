import json
from pathlib import Path

import pytest

from conftest import add_screenshot, make_texture
from src.config import AppConfig
from src.export_changes import ExportError, export_changed_pairs
from src.processing import run_processing
from src.utils import hash_file


def _prepare_two_tiles(config: AppConfig) -> None:
    reference = make_texture(120, 90, seed=71, low=170, high=235)
    changed = reference.copy()
    changed[15:75, 30:80] = 0
    add_screenshot(config, "T001", 2019, reference)
    add_screenshot(config, "T001", 2024, changed)
    unchanged = make_texture(120, 90, seed=72, low=170, high=235)
    add_screenshot(config, "T002", 2019, unchanged)
    add_screenshot(config, "T002", 2024, unchanged.copy())


def test_export_writes_only_pairs_above_threshold(project: AppConfig) -> None:
    _prepare_two_tiles(project)
    run_processing(project)
    result = export_changed_pairs(project, threshold=10.0)

    assert result["exported"] == 1
    output = Path(result["output"])
    year_a, year_b = project.years
    copied_a = output / str(year_a) / f"T001_{year_a}.png"
    copied_b = output / str(year_b) / f"T001_{year_b}.png"
    assert copied_a.exists()
    assert copied_b.exists()
    assert not (output / str(year_a) / f"T002_{year_a}.png").exists()

    source_a = project.paths.raw_year_dir(year_a) / f"T001_{year_a}.png"
    source_b = project.paths.raw_year_dir(year_b) / f"T001_{year_b}.png"
    assert hash_file(copied_a) == hash_file(source_a)
    assert hash_file(copied_b) == hash_file(source_b)
    assert source_a.exists() and source_b.exists()

    changes = (output / "changes.csv").read_text(encoding="utf-8")
    skipped = (output / "skipped.csv").read_text(encoding="utf-8")
    assert "T001" in changes
    assert "T002" not in changes
    assert "T002" in skipped
    assert "below_threshold" in skipped


def test_reimport_with_higher_threshold_removes_stale_copies(project: AppConfig) -> None:
    _prepare_two_tiles(project)
    run_processing(project)
    first = export_changed_pairs(project, threshold=10.0)
    output = Path(first["output"])
    assert (output / "2019" / "T001_2019.png").exists()

    second = export_changed_pairs(project, threshold=40.0)
    assert second["exported"] == 0
    assert second["removed_stale"] == 2
    assert not (output / "2019" / "T001_2019.png").exists()
    assert not (output / "2024" / "T001_2024.png").exists()
    assert "T001" not in (output / "changes.csv").read_text(encoding="utf-8")


def test_keep_stale_preserves_previous_copies(project: AppConfig) -> None:
    _prepare_two_tiles(project)
    run_processing(project)
    export_changed_pairs(project, threshold=10.0)
    second = export_changed_pairs(project, threshold=40.0, keep_stale=True)
    assert second["exported"] == 0
    assert second["removed_stale"] == 0
    output = Path(second["output"])
    assert (output / "2019" / "T001_2019.png").exists()


def test_export_requires_processing_results(project: AppConfig) -> None:
    with pytest.raises(ExportError, match="No processed tile pairs"):
        export_changed_pairs(project)
    run_processing(project)
    with pytest.raises(ExportError, match="No processed tile pairs"):
        export_changed_pairs(project)


def test_export_rejects_invalid_ranges(project: AppConfig) -> None:
    run_processing(project)
    with pytest.raises(ExportError, match="Threshold"):
        export_changed_pairs(project, threshold=101)
    with pytest.raises(ExportError, match="Minimum valid overlap"):
        export_changed_pairs(project, min_valid_overlap=-1)


def test_export_manifest_tracks_files(project: AppConfig) -> None:
    _prepare_two_tiles(project)
    run_processing(project)
    result = export_changed_pairs(project, threshold=10.0)
    manifest = json.loads(
        (Path(result["output"]) / ".export_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["threshold_pct"] == 10.0
    assert len(manifest["files"]) == 2
    assert all(path.startswith(("2019/", "2024/")) for path in manifest["files"])

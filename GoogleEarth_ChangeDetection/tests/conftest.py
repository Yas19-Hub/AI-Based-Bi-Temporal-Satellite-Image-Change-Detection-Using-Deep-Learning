"""Shared synthetic test fixtures. Tests never touch the real dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

from src.config import AppConfig, load_config


def write_config(
    root: Path,
    *,
    tile_start: int = 1,
    tile_end: int = 3,
    years: tuple[int, int] = (2019, 2024),
    expected_dimensions: tuple[int, int] | None = None,
    validation: dict[str, Any] | None = None,
    processing: dict[str, Any] | None = None,
) -> AppConfig:
    validation_settings: dict[str, Any] = {
        "min_width": 10,
        "min_height": 10,
        "max_file_size_mb": 10,
    }
    if validation:
        validation_settings.update(validation)
    processing_settings: dict[str, Any] = {
        "resize_mismatched": True,
        "save_aligned": True,
        "skip_existing": True,
        "alignment": {"enabled": False, "method": "none"},
        "diff": {
            "pixel_threshold": 20,
            "ssim_threshold": 0.90,
            "combine_mode": "and",
            "min_region_area": 10,
            "morphology_kernel": 3,
            "ssim_win_size": 5,
            "max_annotated_regions": 20,
        },
    }
    if processing:
        for key, value in processing.items():
            if isinstance(value, dict) and isinstance(processing_settings.get(key), dict):
                processing_settings[key].update(value)
            else:
                processing_settings[key] = value
    raw = {
        "project": {
            "name": "Synthetic change detection test",
            "years": list(years),
            "tiles": {"start": tile_start, "end": tile_end},
            "expected_dimensions": (
                None
                if expected_dimensions is None
                else {"width": expected_dimensions[0], "height": expected_dimensions[1]}
            ),
        },
        "validation": validation_settings,
        "processing": processing_settings,
        "logging": {"level": "WARNING"},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return load_config(root=root)


def make_texture(
    width: int = 120,
    height: int = 90,
    seed: int = 0,
    low: int = 80,
    high: int = 220,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    texture = rng.integers(low, high + 1, size=(height, width, 3), dtype=np.uint8)
    return texture


def save_png(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB" if array.ndim == 3 else "L").save(path)
    return path


def add_screenshot(
    config: AppConfig,
    tile_id: str,
    year: int,
    array: np.ndarray | None = None,
    *,
    filename: str | None = None,
    seed: int | None = None,
) -> Path:
    if array is None:
        array = make_texture(seed=seed if seed is not None else int(tile_id[1:]) + year)
    name = filename or f"{tile_id}_{year}.png"
    destination = config.paths.raw_year_dir(year) / name
    return save_png(destination, array)


@pytest.fixture
def project(tmp_path: Path) -> AppConfig:
    return write_config(tmp_path)


@pytest.fixture
def full_project(tmp_path: Path) -> AppConfig:
    return write_config(tmp_path / "full", tile_start=1, tile_end=100, validation={"min_width": 1, "min_height": 1})

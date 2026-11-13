"""Safe, manual capture helpers for Google Earth Pro screenshots.

This module never controls Google Earth Pro and never scrapes imagery services.
It only prepares target filenames, reports which slots are missing, and can take
a countdown screenshot of the user's own screen after they have navigated the
application manually.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageGrab

from .config import AppConfig
from .dataset import DatasetManager, FileRecord
from .logging_setup import get_logger
from .utils import (
    hash_file,
    inspect_image,
    save_image,
    utc_now,
    verify_written_image,
)

LOGGER = get_logger(__name__)


class CaptureError(RuntimeError):
    """Raised when a screenshot cannot be captured safely."""


@dataclass
class CaptureSlot:
    tile_id: str
    year: int
    expected_filename: str
    exists: bool
    validation_status: str
    relative_path: str
    imagery_date: str

    def to_row(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "year": self.year,
            "expected_filename": self.expected_filename,
            "exists": self.exists,
            "validation_status": self.validation_status,
            "relative_path": self.relative_path,
            "imagery_date": self.imagery_date,
        }


def build_capture_plan(config: AppConfig, year: int | None = None) -> list[CaptureSlot]:
    """Return every configured tile/year slot with its current discovery status."""
    if year is not None and year not in config.years:
        raise CaptureError(f"Year {year} is not configured. Use one of {config.years}.")
    scan = DatasetManager(config).scan(save=True)
    by_slot: dict[tuple[str, int], FileRecord] = {}
    for row in scan.rows:
        if row.is_primary:
            by_slot[(row.tile_id, row.year)] = row
    selected_years = (year,) if year is not None else config.years
    slots: list[CaptureSlot] = []
    for tile_id in config.tile_ids:
        for slot_year in selected_years:
            record = by_slot.get((tile_id, slot_year))
            if record is None:
                slots.append(
                    CaptureSlot(
                        tile_id=tile_id,
                        year=slot_year,
                        expected_filename=f"{tile_id}_{slot_year}.png",
                        exists=False,
                        validation_status="missing_file",
                        relative_path="",
                        imagery_date="",
                    )
                )
                continue
            slots.append(
                CaptureSlot(
                    tile_id=tile_id,
                    year=slot_year,
                    expected_filename=f"{tile_id}_{slot_year}.png",
                    exists=record.file_exists,
                    validation_status=record.validation_status,
                    relative_path=record.relative_path,
                    imagery_date=record.imagery_date,
                )
            )
    return slots


def next_missing_tiles(config: AppConfig, year: int | None = None, limit: int = 20) -> list[str]:
    missing = [
        slot.tile_id
        for slot in build_capture_plan(config, year=year)
        if not slot.exists or slot.validation_status == "missing_file"
    ]
    return list(dict.fromkeys(missing))[:limit]


def capture_countdown(
    config: AppConfig,
    tile_id: str,
    year: int,
    *,
    countdown: int = 5,
    overwrite: bool = False,
    region: tuple[int, int, int, int] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    grab: Callable[..., Any] = ImageGrab.grab,
) -> Path:
    """Take a screenshot after a countdown, saving it under the canonical filename.

    The user must manually navigate Google Earth Pro and choose the historical
    imagery. This function does not interact with the application or any network
    service, and it never records an imagery date on its own.
    """
    if year not in config.years:
        raise CaptureError(f"Year {year} is not configured. Use one of {config.years}.")
    tile = tile_id.strip().upper()
    if tile not in config.tile_ids:
        raise CaptureError(f"Tile {tile_id!r} is outside the configured tile range.")
    destination = config.paths.raw_year_dir(year) / f"{tile}_{year}.png"
    if destination.exists() and not overwrite:
        raise CaptureError(
            f"{destination} already exists. Originals are protected; pass overwrite explicitly "
            "if you really intend to replace it."
        )
    if countdown < 0 or countdown > 60:
        raise CaptureError("Countdown must be between 0 and 60 seconds.")

    LOGGER.info(
        "Capturing %s %s in %d seconds. Keep Google Earth Pro positioned with the desired "
        "historical imagery visible and hide any private screen content.",
        tile,
        year,
        countdown,
    )
    for remaining in range(countdown, 0, -1):
        LOGGER.info("Capturing in %d...", remaining)
        sleep(1.0)

    try:
        screenshot = grab(bbox=region, include_layered_windows=False) if region else grab()
    except Exception as exc:  # noqa: BLE001 - OS-specific grab failures
        raise CaptureError(f"Screen capture failed: {exc}") from exc
    rgb = np.asarray(screenshot.convert("RGB"), dtype=np.uint8).copy()
    if rgb.size == 0:
        raise CaptureError("Screen capture returned an empty image.")
    if int(rgb.max()) - int(rgb.min()) < 2:
        raise CaptureError(
            "Screen capture appears to be a single flat colour; check that Google Earth Pro "
            "is visible and the screen is unlocked."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_image(destination, rgb)
    verify_written_image(destination, rgb.shape[1], rgb.shape[0])
    LOGGER.info(
        "Saved %s. Now record the imagery date you observed (from the Google Earth Pro "
        "historical imagery display) using the metadata command; it is not inferred.",
        destination,
    )
    return destination


def _validate_slot(config: AppConfig, tile_id: str, year: int) -> tuple[str, Path]:
    if year not in config.years:
        raise CaptureError(f"Year {year} is not configured. Use one of {config.years}.")
    tile = tile_id.strip().upper()
    if tile not in config.tile_ids:
        raise CaptureError(f"Tile {tile_id!r} is outside the configured tile range.")
    return tile, config.paths.raw_year_dir(year)


def _verify_capture_source(config: AppConfig, source: Path) -> dict[str, Any]:
    if not source.is_file():
        raise CaptureError(f"Screenshot source does not exist: {source}")
    extension = source.suffix.lower()
    allowed = tuple(ext.lower() for ext in config.validation.allowed_extensions)
    if extension not in allowed:
        raise CaptureError(
            f"Unsupported screenshot extension {extension!r}. Allowed: {', '.join(allowed)}"
        )
    info = inspect_image(
        source,
        min_width=config.validation.min_width,
        min_height=config.validation.min_height,
    )
    if not info["ok"]:
        raise CaptureError(f"Invalid screenshot {source}: {info['error']}")
    return info


def save_screenshot_file(
    config: AppConfig,
    tile_id: str,
    year: int,
    source: Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy an existing screenshot into raw_screenshots/YYYY with its canonical name."""
    tile, folder = _validate_slot(config, tile_id, year)
    source = Path(source).expanduser()
    _verify_capture_source(config, source)
    destination = folder / f"{tile}_{year}{source.suffix.lower()}"
    if source.resolve() == destination.resolve():
        return destination
    if destination.exists() and not overwrite:
        raise CaptureError(
            f"{destination} already exists. Originals are protected; pass overwrite explicitly "
            "if you really intend to replace it."
        )
    folder.mkdir(parents=True, exist_ok=True)
    source_hash = hash_file(source)
    temp = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temp)
        if hash_file(temp) != source_hash:
            raise CaptureError(f"Copy verification failed for {source}.")
        os.replace(temp, destination)
        if hash_file(destination) != source_hash:
            raise CaptureError(f"Saved file verification failed for {destination}.")
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass
    verify_written_image(destination)
    LOGGER.info("Saved screenshot to %s", destination)
    return destination


def save_screenshot_clipboard(
    config: AppConfig,
    tile_id: str,
    year: int,
    *,
    overwrite: bool = False,
    grab: Callable[..., Any] = ImageGrab.grabclipboard,
) -> Path:
    """Save an image currently on the clipboard as a canonical PNG screenshot."""
    tile, folder = _validate_slot(config, tile_id, year)
    try:
        clipboard = grab()
    except Exception as exc:  # noqa: BLE001 - OS-specific clipboard failures
        raise CaptureError(f"Could not read the clipboard: {exc}") from exc
    if isinstance(clipboard, list):
        if len(clipboard) != 1:
            raise CaptureError(
                "The clipboard contains multiple files. Copy a single screenshot or use --source."
            )
        return save_screenshot_file(
            config, tile, year, Path(clipboard[0]), overwrite=overwrite
        )
    if not isinstance(clipboard, Image.Image):
        raise CaptureError(
            "No image found on the clipboard. Copy a screenshot first or use --source."
        )
    destination = folder / f"{tile}_{year}.png"
    if destination.exists() and not overwrite:
        raise CaptureError(
            f"{destination} already exists. Originals are protected; pass overwrite explicitly "
            "if you really intend to replace it."
        )
    rgb = np.asarray(clipboard.convert("RGB"), dtype=np.uint8).copy()
    if rgb.shape[1] < config.validation.min_width or rgb.shape[0] < config.validation.min_height:
        raise CaptureError(
            f"Clipboard image is {rgb.shape[1]}x{rgb.shape[0]}, smaller than the required "
            f"{config.validation.min_width}x{config.validation.min_height}."
        )
    folder.mkdir(parents=True, exist_ok=True)
    save_image(destination, rgb)
    verify_written_image(destination)
    LOGGER.info("Saved clipboard screenshot to %s", destination)
    return destination


def capture_instructions(config: AppConfig, year: int, tile_ids: Sequence[str]) -> str:
    tiles = ", ".join(tile_ids) if tile_ids else "(none missing)"
    folder = config.paths.raw_year_dir(year)
    return (
        f"Manual capture checklist for {year}\n"
        f"1. Open Google Earth Pro and navigate to each tile's saved placemark or coordinates.\n"
        f"2. Open Historical Imagery and select a date whose year is {year}.\n"
        f"3. Confirm the visible imagery date; write it down or enter it later. Never guess.\n"
        f"4. Save/export only the map image to: {folder}\n"
        f"5. Name it T###_{year}.png, for example T001_{year}.png.\n"
        f"6. Do not edit the original screenshots after capture; keep them immutable.\n"
        f"7. Register metadata with: python -m src.cli metadata --tile T001 --year {year} "
        f"--imagery-date YYYY-MM-DD\n"
        f"8. Re-run: python -m src.cli validate\n"
        f"Missing tiles: {tiles}\n"
        f"Generated: {utc_now()}"
    )

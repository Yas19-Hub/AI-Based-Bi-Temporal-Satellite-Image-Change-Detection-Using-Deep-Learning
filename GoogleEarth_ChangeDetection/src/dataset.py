"""Dataset discovery, validation, pairing and metadata management."""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import AppConfig, ensure_directories
from .utils import (
    DataFileError,
    display_path,
    hash_file,
    inspect_image,
    normalize_tile_id,
    parse_coordinate,
    read_csv,
    utc_now,
    validate_capture_date,
    validate_imagery_date,
    write_csv,
)

LOGGER = logging.getLogger("gecd.dataset")

MANIFEST_COLUMNS = [
    "tile_id",
    "year",
    "expected_filename",
    "filename",
    "relative_path",
    "file_exists",
    "in_expected_set",
    "is_primary",
    "file_size_bytes",
    "file_modified_utc",
    "width",
    "height",
    "image_format",
    "image_mode",
    "sha256",
    "imagery_date",
    "image_capture_date",
    "capture_source",
    "latitude",
    "longitude",
    "metadata_status",
    "validation_status",
    "validation_notes",
    "metadata_notes",
    "pair_status",
    "pair_notes",
    "last_validated_utc",
    "updated_utc",
]

MANUAL_FIELDS = [
    "imagery_date",
    "image_capture_date",
    "capture_source",
    "latitude",
    "longitude",
]

OVERRIDE_COLUMNS = [
    "tile_id",
    "year",
    "imagery_date",
    "image_capture_date",
    "latitude",
    "longitude",
    "notes",
]

REGISTRY_COLUMNS = ["tile_id", "label", "latitude", "longitude", "notes"]
REVIEW_COLUMNS = ["tile_id", "review_status", "reviewer", "reviewed_utc", "notes"]
REVIEW_STATUSES = {"pending", "reviewed", "unreliable"}

VALID = "valid"
MISSING_FILE = "missing_file"
INVALID_IMAGE = "invalid_image"
INVALID_DIMENSIONS = "invalid_dimensions"
UNEXPECTED_DIMENSIONS = "unexpected_dimensions"
FILE_TOO_LARGE = "file_too_large"
DUPLICATE = "duplicate"

FILENAME_PATTERN = re.compile(r"^([Tt]\d{3})_(\d{4})(\.[A-Za-z0-9]+)$")
IGNORED_SYSTEM_FILES = {".gitkeep", "thumbs.db", "desktop.ini"}


class DatasetError(RuntimeError):
    """Raised for unrecoverable dataset problems."""


@dataclass
class FileRecord:
    tile_id: str
    year: int
    expected_filename: str
    filename: str = ""
    relative_path: str = ""
    file_exists: bool = False
    in_expected_set: bool = True
    is_primary: bool = False
    file_size_bytes: int | None = None
    file_modified_utc: str = ""
    width: int | None = None
    height: int | None = None
    image_format: str = ""
    image_mode: str = ""
    sha256: str = ""
    imagery_date: str = ""
    image_capture_date: str = ""
    capture_source: str = ""
    latitude: str = ""
    longitude: str = ""
    metadata_status: str = "missing"
    validation_status: str = MISSING_FILE
    validation_notes: str = ""
    metadata_notes: str = ""
    pair_status: str = ""
    pair_notes: str = ""
    last_validated_utc: str = ""
    updated_utc: str = ""

    @property
    def valid(self) -> bool:
        return self.validation_status == VALID

    def absolute_path(self, config: AppConfig) -> Path:
        if not self.relative_path:
            raise DatasetError(f"No file path recorded for {self.tile_id} ({self.year}).")
        candidate = Path(self.relative_path)
        return candidate if candidate.is_absolute() else config.root / candidate

    def to_row(self) -> dict[str, Any]:
        return {
            "tile_id": self.tile_id,
            "year": self.year,
            "expected_filename": self.expected_filename,
            "filename": self.filename,
            "relative_path": self.relative_path,
            "file_exists": self.file_exists,
            "in_expected_set": self.in_expected_set,
            "is_primary": self.is_primary,
            "file_size_bytes": self.file_size_bytes if self.file_size_bytes is not None else "",
            "file_modified_utc": self.file_modified_utc,
            "width": self.width if self.width is not None else "",
            "height": self.height if self.height is not None else "",
            "image_format": self.image_format,
            "image_mode": self.image_mode,
            "sha256": self.sha256,
            "imagery_date": self.imagery_date,
            "image_capture_date": self.image_capture_date,
            "capture_source": self.capture_source,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "metadata_status": self.metadata_status,
            "validation_status": self.validation_status,
            "validation_notes": self.validation_notes,
            "metadata_notes": self.metadata_notes,
            "pair_status": self.pair_status,
            "pair_notes": self.pair_notes,
            "last_validated_utc": self.last_validated_utc,
            "updated_utc": self.updated_utc,
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "FileRecord":
        return cls(
            tile_id=str(row.get("tile_id", "")),
            year=_to_int(row.get("year")) or 0,
            expected_filename=str(row.get("expected_filename", "")),
            filename=str(row.get("filename", "")),
            relative_path=str(row.get("relative_path", "")),
            file_exists=_to_bool(row.get("file_exists")),
            in_expected_set=_to_bool(row.get("in_expected_set")),
            is_primary=_to_bool(row.get("is_primary")),
            file_size_bytes=_to_int(row.get("file_size_bytes")),
            file_modified_utc=str(row.get("file_modified_utc", "")),
            width=_to_int(row.get("width")),
            height=_to_int(row.get("height")),
            image_format=str(row.get("image_format", "")),
            image_mode=str(row.get("image_mode", "")),
            sha256=str(row.get("sha256", "")),
            imagery_date=str(row.get("imagery_date", "")),
            image_capture_date=str(row.get("image_capture_date", "")),
            capture_source=str(row.get("capture_source", "")),
            latitude=str(row.get("latitude", "")),
            longitude=str(row.get("longitude", "")),
            metadata_status=str(row.get("metadata_status", "missing")),
            validation_status=str(row.get("validation_status", MISSING_FILE)),
            validation_notes=str(row.get("validation_notes", "")),
            metadata_notes=str(row.get("metadata_notes", "")),
            pair_status=str(row.get("pair_status", "")),
            pair_notes=str(row.get("pair_notes", "")),
            last_validated_utc=str(row.get("last_validated_utc", "")),
            updated_utc=str(row.get("updated_utc", "")),
        )


@dataclass
class PairInfo:
    tile_id: str
    year_a: int
    year_b: int
    record_a: FileRecord | None
    record_b: FileRecord | None
    status: str
    processable: bool
    notes: str
    resized_planned: bool = False
    duplicate_years: tuple[int, ...] = ()
    invalid_years: tuple[int, ...] = ()
    missing_years: tuple[int, ...] = ()

    def record_for(self, year: int) -> FileRecord | None:
        if year == self.year_a:
            return self.record_a
        if year == self.year_b:
            return self.record_b
        return None


@dataclass
class ScanResult:
    rows: list[FileRecord]
    pairs: dict[str, PairInfo]
    unrecognized: list[dict[str, str]]
    summary: dict[str, Any]


def _to_int(value: Any) -> int | None:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _nonempty(value: Any) -> bool:
    return str(value if value is not None else "").strip() != ""


def _split_notes(*notes: str) -> str:
    return "; ".join(dict.fromkeys(part.strip() for part in notes if part and part.strip()))


def pair_summary_columns(config: AppConfig) -> list[str]:
    year_a, year_b = config.years
    return [
        "tile_id",
        "status",
        "processable",
        "year_a",
        "year_b",
        f"file_{year_a}",
        f"file_{year_b}",
        f"validation_{year_a}",
        f"validation_{year_b}",
        f"width_{year_a}",
        f"height_{year_a}",
        f"width_{year_b}",
        f"height_{year_b}",
        f"imagery_date_{year_a}",
        f"imagery_date_{year_b}",
        f"capture_date_{year_a}",
        f"capture_date_{year_b}",
        f"metadata_status_{year_a}",
        f"metadata_status_{year_b}",
        "latitude",
        "longitude",
        "resized_planned",
        "duplicate_years",
        "invalid_years",
        "missing_years",
        "validation_notes",
        "metadata_notes",
        "pair_notes",
    ]


def load_manifest(config: AppConfig) -> list[FileRecord]:
    rows = read_csv(config.paths.manifest, columns=MANIFEST_COLUMNS)
    return [FileRecord.from_row(row) for row in rows]


def load_overrides(config: AppConfig) -> list[dict[str, str]]:
    return read_csv(config.paths.manual_overrides, columns=OVERRIDE_COLUMNS)


def load_registry(config: AppConfig) -> dict[str, dict[str, str]]:
    rows = read_csv(config.paths.tile_registry, columns=REGISTRY_COLUMNS)
    return {row["tile_id"].upper(): row for row in rows if row.get("tile_id")}


def load_review_status(config: AppConfig) -> dict[str, dict[str, str]]:
    rows = read_csv(config.paths.review_status, columns=REVIEW_COLUMNS)
    return {row["tile_id"].upper(): row for row in rows if row.get("tile_id")}


def set_review_status(
    config: AppConfig,
    tile_id: str,
    status: str,
    reviewer: str = "",
    notes: str | None = None,
) -> None:
    normalized_tile = normalize_tile_id(tile_id)
    status = status.strip().lower()
    if status not in REVIEW_STATUSES:
        raise ValueError(f"Review status must be one of {sorted(REVIEW_STATUSES)}.")
    rows = read_csv(config.paths.review_status, columns=REVIEW_COLUMNS)
    now = utc_now()
    updated = False
    for row in rows:
        if row["tile_id"].upper() == normalized_tile:
            row["review_status"] = status
            if reviewer:
                row["reviewer"] = reviewer
            if notes is not None:
                row["notes"] = notes
            row["reviewed_utc"] = "" if status == "pending" else now
            updated = True
            break
    if not updated:
        rows.append(
            {
                "tile_id": normalized_tile,
                "review_status": status,
                "reviewer": reviewer,
                "reviewed_utc": "" if status == "pending" else now,
                "notes": notes or "",
            }
        )
    rows.sort(key=lambda item: item["tile_id"])
    write_csv(config.paths.review_status, rows, columns=REVIEW_COLUMNS)


def upsert_override(
    config: AppConfig,
    tile_id: str,
    year: int | None,
    *,
    imagery_date: str | None = None,
    image_capture_date: str | None = None,
    latitude: str | None = None,
    longitude: str | None = None,
    notes: str | None = None,
) -> None:
    """Create or update a manual metadata override. Blank values clear a field."""
    normalized_tile = normalize_tile_id(tile_id)
    if year is not None and year not in config.years:
        raise ValueError(f"Year {year} is not one of the configured years {config.years}.")
    values = {
        "imagery_date": imagery_date,
        "image_capture_date": image_capture_date,
        "latitude": latitude,
        "longitude": longitude,
    }
    for field_name, value in values.items():
        if value is None:
            continue
        if field_name == "imagery_date":
            _, error = validate_imagery_date(value)
        elif field_name == "image_capture_date":
            _, error = validate_capture_date(value)
        elif field_name == "latitude":
            _, error = parse_coordinate(value, "latitude")
        else:
            _, error = parse_coordinate(value, "longitude")
        if error:
            raise ValueError(f"Invalid {field_name}: {error}")

    rows = read_csv(config.paths.manual_overrides, columns=OVERRIDE_COLUMNS)
    year_text = "" if year is None else str(year)
    match = None
    for row in rows:
        if row["tile_id"].upper() == normalized_tile and str(row.get("year", "")) == year_text:
            match = row
            break
    if match is None:
        match = {
            "tile_id": normalized_tile,
            "year": year_text,
            "imagery_date": "",
            "image_capture_date": "",
            "latitude": "",
            "longitude": "",
            "notes": "",
        }
        rows.append(match)
    for field_name, value in values.items():
        if value is not None:
            match[field_name] = str(value).strip()
    if notes is not None:
        match["notes"] = notes.strip()
    rows.sort(key=lambda item: (item["tile_id"], item.get("year", "")))
    write_csv(config.paths.manual_overrides, rows, columns=OVERRIDE_COLUMNS)


class DatasetManager:
    """Discover screenshots, validate them, preserve metadata and build pairs."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.rows: list[FileRecord] = []
        self.pairs: dict[str, PairInfo] = {}
        self.unrecognized: list[dict[str, str]] = []
        self.summary: dict[str, Any] = {}

    def scan(self, save: bool = True) -> ScanResult:
        ensure_directories(self.config)
        previous = read_csv(self.config.paths.manifest, columns=MANIFEST_COLUMNS)
        overrides = load_overrides(self.config)
        registry = load_registry(self.config)

        candidates, unrecognized = self._discover()
        self.unrecognized = unrecognized

        rows: list[FileRecord] = []
        for tile_id in self.config.tile_ids:
            for year in self.config.years:
                rows.extend(self._records_for_slot(tile_id, year, candidates.get((tile_id, year), [])))

        self._apply_manual_metadata(rows, previous, overrides, registry)
        pairs = self._build_pairs(rows)
        self.pairs = {pair.tile_id: pair for pair in pairs}

        for row in rows:
            pair = self.pairs.get(row.tile_id)
            if pair is not None:
                row.pair_status = pair.status
                row.pair_notes = pair.notes

        rows.sort(
            key=lambda item: (
                item.tile_id,
                item.year,
                not item.is_primary,
                item.filename.lower(),
            )
        )
        self.rows = rows
        self.summary = self._build_summary(rows, pairs, unrecognized)

        for entry in unrecognized:
            LOGGER.warning("Unrecognized file %s: %s", entry["relative_path"], entry["reason"])
        for status, count in sorted(Counter(row.validation_status for row in rows).items()):
            LOGGER.info("Validation status %-22s %d", status, count)
        LOGGER.info(
            "Dataset: %d/%d screenshots valid, %d/%d tiles complete, %d processable pairs",
            self.summary["valid_files"],
            self.summary["expected_screenshots"],
            self.summary["complete_pairs"],
            self.summary["expected_tiles"],
            self.summary["processable_pairs"],
        )

        if save:
            self.save()
        return ScanResult(rows=rows, pairs=self.pairs, unrecognized=unrecognized, summary=self.summary)

    def save(self) -> None:
        write_csv(
            self.config.paths.manifest,
            [row.to_row() for row in self.rows],
            columns=MANIFEST_COLUMNS,
        )
        write_csv(
            self.config.paths.pair_summary,
            [self._pair_row(pair) for pair in self.pairs.values()],
            columns=pair_summary_columns(self.config),
        )

    def _discover(self) -> tuple[dict[tuple[str, int], list[Path]], list[dict[str, str]]]:
        expected_tiles = set(self.config.tile_ids)
        candidates: dict[tuple[str, int], list[Path]] = {}
        unrecognized: list[dict[str, str]] = []
        allowed = tuple(ext.lower() for ext in self.config.validation.allowed_extensions)

        for folder_year in self.config.years:
            directory = self.config.paths.raw_year_dir(folder_year)
            if not directory.exists():
                LOGGER.warning("Year folder does not exist: %s", directory)
                continue
            for path in sorted(directory.iterdir()):
                if not path.is_file() or path.name.lower() in IGNORED_SYSTEM_FILES:
                    continue
                match = FILENAME_PATTERN.fullmatch(path.name)
                if not match:
                    unrecognized.append(
                        {
                            "relative_path": display_path(path, self.config.root),
                            "reason": "unrecognized_filename",
                            "detail": (
                                f"Rename to the pattern T###_{folder_year}.png "
                                f"(for example T001_{folder_year}.png)."
                            ),
                        }
                    )
                    continue
                tile_id = match.group(1).upper()
                file_year = int(match.group(2))
                extension = match.group(3).lower()
                if file_year != folder_year:
                    unrecognized.append(
                        {
                            "relative_path": display_path(path, self.config.root),
                            "reason": "year_folder_mismatch",
                            "detail": f"Filename year {file_year} does not match folder {folder_year}.",
                        }
                    )
                    continue
                if tile_id not in expected_tiles:
                    unrecognized.append(
                        {
                            "relative_path": display_path(path, self.config.root),
                            "reason": "unexpected_tile",
                            "detail": f"Tile {tile_id} is outside the configured tile range.",
                        }
                    )
                    continue
                if extension not in allowed:
                    unrecognized.append(
                        {
                            "relative_path": display_path(path, self.config.root),
                            "reason": "unsupported_extension",
                            "detail": f"Extension {extension} is not in the allowed list.",
                        }
                    )
                    continue
                candidates.setdefault((tile_id, folder_year), []).append(path)
        return candidates, unrecognized

    def _records_for_slot(
        self, tile_id: str, year: int, paths: Sequence[Path]
    ) -> list[FileRecord]:
        expected_filename = f"{tile_id}_{year}.png"
        if not paths:
            now = utc_now()
            return [
                FileRecord(
                    tile_id=tile_id,
                    year=year,
                    expected_filename=expected_filename,
                    validation_status=MISSING_FILE,
                    validation_notes=(
                        "No supported screenshot found for this tile/year. "
                        f"Expected a file such as {expected_filename}."
                    ),
                    last_validated_utc=now,
                    updated_utc=now,
                )
            ]

        extension_order = {
            extension.lower(): index
            for index, extension in enumerate(self.config.validation.allowed_extensions)
        }
        ordered = sorted(
            paths,
            key=lambda path: (extension_order.get(path.suffix.lower(), 999), path.name.lower()),
        )
        inspected: list[tuple[Path, dict[str, Any], str, str]] = []
        for path in ordered:
            info = inspect_image(
                path,
                min_width=self.config.validation.min_width,
                min_height=self.config.validation.min_height,
            )
            status = VALID
            note = ""
            file_size = path.stat().st_size
            if file_size <= 0:
                status = INVALID_IMAGE
                note = "File is empty."
            elif file_size > self.config.validation.max_file_size_bytes:
                status = FILE_TOO_LARGE
                note = (
                    f"File is {file_size / (1024 * 1024):.1f} MB, above the configured "
                    f"{self.config.validation.max_file_size_mb} MB limit."
                )
            elif not info["ok"]:
                error = str(info.get("error") or "")
                status = INVALID_DIMENSIONS if "smaller than the required" in error else INVALID_IMAGE
                note = error
            elif self.config.expected_dimensions is not None:
                expected_width, expected_height = self.config.expected_dimensions
                if (info["width"], info["height"]) != (expected_width, expected_height):
                    status = UNEXPECTED_DIMENSIONS
                    note = (
                        f"Image is {info['width']}x{info['height']}, but expected "
                        f"{expected_width}x{expected_height}."
                    )
            if info["format"] and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                expected_format = "PNG" if path.suffix.lower() == ".png" else "JPEG"
                if info["format"] != expected_format:
                    note = _split_notes(
                        note,
                        f"Extension is {path.suffix} but file content is {info['format']}.",
                    )
            inspected.append((path, info, status, note))

        valid_paths = [item for item in inspected if item[2] == VALID]
        primary_path = (valid_paths[0] if valid_paths else inspected[0])[0]
        duplicate_paths = [item[0] for item in valid_paths if item[0] != primary_path]

        records: list[FileRecord] = []
        now = utc_now()
        for path, info, status, note in inspected:
            is_primary = path == primary_path
            if path in duplicate_paths:
                status = DUPLICATE
                note = _split_notes(
                    note,
                    f"Additional valid file for {tile_id} {year}; primary is {primary_path.name}. "
                    "Keep exactly one file per tile/year to make the pair unambiguous.",
                )
            elif not is_primary:
                note = _split_notes(
                    note, f"Not selected as primary (primary: {primary_path.name})."
                )
            size = path.stat().st_size
            modified = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            digest = ""
            try:
                digest = hash_file(path)
            except OSError as exc:
                note = _split_notes(note, f"Could not hash file: {exc}")
            records.append(
                FileRecord(
                    tile_id=tile_id,
                    year=year,
                    expected_filename=expected_filename,
                    filename=path.name,
                    relative_path=display_path(path, self.config.root),
                    file_exists=True,
                    in_expected_set=True,
                    is_primary=is_primary,
                    file_size_bytes=size,
                    file_modified_utc=modified,
                    width=info["width"],
                    height=info["height"],
                    image_format=str(info["format"] or ""),
                    image_mode=str(info["mode"] or ""),
                    sha256=digest,
                    validation_status=status,
                    validation_notes=note,
                    last_validated_utc=now,
                    updated_utc=now,
                )
            )
        return records

    def _apply_manual_metadata(
        self,
        rows: list[FileRecord],
        previous: Sequence[Mapping[str, str]],
        overrides: Sequence[Mapping[str, str]],
        registry: Mapping[str, Mapping[str, str]],
    ) -> None:
        exact: dict[tuple[str, int, str], Mapping[str, str]] = {}
        by_slot: dict[tuple[str, int], list[Mapping[str, str]]] = {}
        for old in previous:
            old_tile = str(old.get("tile_id", "")).upper()
            old_year = _to_int(old.get("year"))
            if not old_tile or old_year is None:
                continue
            old_filename = str(old.get("filename", ""))
            exact[(old_tile, old_year, old_filename)] = old
            by_slot.setdefault((old_tile, old_year), []).append(old)

        override_by_tile: dict[str, list[Mapping[str, str]]] = {}
        for override in overrides:
            tile = str(override.get("tile_id", "")).upper()
            if tile:
                override_by_tile.setdefault(tile, []).append(override)

        for record in rows:
            selected: dict[str, str] = {}
            old = exact.get((record.tile_id, record.year, record.filename))
            if old is not None:
                for field_name in MANUAL_FIELDS:
                    if _nonempty(old.get(field_name)):
                        selected[field_name] = str(old[field_name]).strip()
            slot_rows = by_slot.get((record.tile_id, record.year), [])
            for field_name in MANUAL_FIELDS:
                if field_name in selected:
                    continue
                values = {
                    str(candidate[field_name]).strip()
                    for candidate in slot_rows
                    if _nonempty(candidate.get(field_name))
                }
                if len(values) == 1:
                    selected[field_name] = values.pop()

            for field_name in MANUAL_FIELDS:
                setattr(record, field_name, selected.get(field_name, ""))

            notes: list[str] = []
            if not record.latitude and not record.longitude:
                reg = registry.get(record.tile_id)
                if reg and _nonempty(reg.get("latitude")) and _nonempty(reg.get("longitude")):
                    record.latitude = str(reg["latitude"]).strip()
                    record.longitude = str(reg["longitude"]).strip()
                    notes.append("Coordinates taken from tile_registry.csv.")

            applied = False
            for override in override_by_tile.get(record.tile_id, []):
                override_year = str(override.get("year", "")).strip()
                if override_year and _to_int(override_year) != record.year:
                    continue
                for field_name in MANUAL_FIELDS:
                    if _nonempty(override.get(field_name)):
                        setattr(record, field_name, str(override[field_name]).strip())
                        applied = True
                if _nonempty(override.get("notes")):
                    notes.append(f"Override note: {str(override['notes']).strip()}")
            if applied:
                record.capture_source = record.capture_source or "manual_override"

            record.metadata_status, metadata_notes = self._assess_metadata(record)
            record.metadata_notes = _split_notes(record.metadata_notes, *notes, metadata_notes)

    def _assess_metadata(self, record: FileRecord) -> tuple[str, str]:
        notes: list[str] = []
        invalid = False

        if record.imagery_date:
            normalized, error = validate_imagery_date(record.imagery_date)
            if error:
                invalid = True
                notes.append(f"Invalid imagery date '{record.imagery_date}': {error}")
            else:
                record.imagery_date = normalized
                if not normalized.startswith(str(record.year)):
                    notes.append(
                        f"Imagery date {normalized} does not fall in the {record.year} folder year; "
                        "verify it was transcribed correctly."
                    )
        if record.image_capture_date:
            normalized_capture, error = validate_capture_date(record.image_capture_date)
            if error:
                invalid = True
                notes.append(f"Invalid image capture date: {error}")
            else:
                record.image_capture_date = normalized_capture

        latitude, lat_error = parse_coordinate(record.latitude, "latitude")
        longitude, lon_error = parse_coordinate(record.longitude, "longitude")
        if lat_error:
            invalid = True
            notes.append(lat_error)
        if lon_error:
            invalid = True
            notes.append(lon_error)
        if _nonempty(record.latitude) and not _nonempty(record.longitude):
            notes.append("Latitude is set but longitude is missing.")
        if _nonempty(record.longitude) and not _nonempty(record.latitude):
            notes.append("Longitude is set but latitude is missing.")

        has_imagery = bool(record.imagery_date)
        has_coords = latitude is not None and longitude is not None
        if invalid:
            status = "invalid"
        elif has_imagery and has_coords:
            status = "complete"
        elif has_imagery or has_coords or _nonempty(record.image_capture_date):
            status = "partial"
        else:
            status = "missing"
        if record.imagery_date and not record.imagery_date.startswith(str(record.year)):
            status = "review" if status == "complete" else status
        return status, _split_notes(*notes)

    def _build_pairs(self, rows: Sequence[FileRecord]) -> list[PairInfo]:
        by_slot_primary: dict[tuple[str, int], FileRecord] = {}
        duplicates: dict[tuple[str, int], list[FileRecord]] = {}
        invalids: dict[tuple[str, int], list[FileRecord]] = {}
        for row in rows:
            if row.is_primary:
                by_slot_primary[(row.tile_id, row.year)] = row
            if row.validation_status == DUPLICATE:
                duplicates.setdefault((row.tile_id, row.year), []).append(row)
            elif row.validation_status not in {VALID, MISSING_FILE}:
                invalids.setdefault((row.tile_id, row.year), []).append(row)

        year_a, year_b = self.config.years
        pairs: list[PairInfo] = []
        for tile_id in self.config.tile_ids:
            raw_a = by_slot_primary.get((tile_id, year_a))
            raw_b = by_slot_primary.get((tile_id, year_b))
            rec_a = raw_a if raw_a and raw_a.file_exists else None
            rec_b = raw_b if raw_b and raw_b.file_exists else None
            valid_a = rec_a if rec_a and rec_a.valid else None
            valid_b = rec_b if rec_b and rec_b.valid else None
            dup_years = tuple(year for year in (year_a, year_b) if (tile_id, year) in duplicates)
            invalid_years = tuple(
                year for year in (year_a, year_b) if (tile_id, year) in invalids
            )
            missing_years = tuple(
                year for year in (year_a, year_b) if by_slot_primary.get((tile_id, year)) is None
            )

            notes: list[str] = []
            for year in dup_years:
                names = ", ".join(item.filename for item in duplicates[(tile_id, year)])
                notes.append(
                    f"Duplicate valid files for {year}: {names}. Keep exactly one file per tile/year."
                )
            for year in invalid_years:
                for item in invalids[(tile_id, year)]:
                    notes.append(f"{year}: {item.validation_notes or item.validation_status}")
            for year in missing_years:
                notes.append(f"Missing supported screenshot for {year}.")

            resized = False
            dimension_mismatch = False
            if valid_a and valid_b:
                if (valid_a.width, valid_a.height) != (valid_b.width, valid_b.height):
                    dimension_mismatch = True
                    if self.config.processing.resize_mismatched:
                        resized = True
                        notes.append(
                            f"Dimensions differ ({valid_a.width}x{valid_a.height} vs "
                            f"{valid_b.width}x{valid_b.height}); the {year_b} image will be resized "
                            f"to the {year_a} reference size for comparison."
                        )
                    else:
                        notes.append(
                            f"Dimensions differ ({valid_a.width}x{valid_a.height} vs "
                            f"{valid_b.width}x{valid_b.height}) and resizing is disabled."
                        )
                if _nonempty(valid_a.latitude) and _nonempty(valid_b.latitude):
                    lat_a, _ = parse_coordinate(valid_a.latitude, "latitude")
                    lat_b, _ = parse_coordinate(valid_b.latitude, "latitude")
                    lon_a, _ = parse_coordinate(valid_a.longitude, "longitude")
                    lon_b, _ = parse_coordinate(valid_b.longitude, "longitude")
                    if (
                        lat_a is not None
                        and lat_b is not None
                        and lon_a is not None
                        and lon_b is not None
                        and (abs(lat_a - lat_b) > 1e-6 or abs(lon_a - lon_b) > 1e-6)
                    ):
                        notes.append(
                            "Latitude/longitude differ between the two years; verify both captures "
                            "show the same location."
                        )

            if rec_a is None and rec_b is None:
                status = "missing_both"
            elif rec_a is None or rec_b is None:
                missing_year = year_a if rec_a is None else year_b
                status = f"missing_{missing_year}"
            elif not rec_a.valid and not rec_b.valid:
                status = "invalid_both"
            elif not rec_a.valid or not rec_b.valid:
                invalid_year = year_a if not rec_a.valid else year_b
                status = f"invalid_{invalid_year}"
            elif dup_years:
                status = "duplicate_both" if len(dup_years) == 2 else f"duplicate_{dup_years[0]}"
            elif dimension_mismatch:
                status = "complete_resized" if resized else "dimension_mismatch"
            else:
                status = "complete"

            processable = bool(
                valid_a
                and valid_b
                and (not dup_years or not self.config.validation.skip_ambiguous_duplicates)
                and (not dimension_mismatch or self.config.processing.resize_mismatched)
            )
            pairs.append(
                PairInfo(
                    tile_id=tile_id,
                    year_a=year_a,
                    year_b=year_b,
                    record_a=rec_a,
                    record_b=rec_b,
                    status=status,
                    processable=processable,
                    notes=_split_notes(*notes),
                    resized_planned=resized,
                    duplicate_years=dup_years,
                    invalid_years=invalid_years,
                    missing_years=missing_years,
                )
            )
        return pairs

    def _pair_row(self, pair: PairInfo) -> dict[str, Any]:
        year_a, year_b = pair.year_a, pair.year_b
        rec_a, rec_b = pair.record_a, pair.record_b
        lat = (rec_a.latitude if rec_a and rec_a.latitude else rec_b.latitude if rec_b else "")
        lon = (rec_a.longitude if rec_a and rec_a.longitude else rec_b.longitude if rec_b else "")
        return {
            "tile_id": pair.tile_id,
            "status": pair.status,
            "processable": pair.processable,
            "year_a": year_a,
            "year_b": year_b,
            f"file_{year_a}": rec_a.filename if rec_a else "",
            f"file_{year_b}": rec_b.filename if rec_b else "",
            f"validation_{year_a}": rec_a.validation_status if rec_a else MISSING_FILE,
            f"validation_{year_b}": rec_b.validation_status if rec_b else MISSING_FILE,
            f"width_{year_a}": rec_a.width if rec_a and rec_a.width else "",
            f"height_{year_a}": rec_a.height if rec_a and rec_a.height else "",
            f"width_{year_b}": rec_b.width if rec_b and rec_b.width else "",
            f"height_{year_b}": rec_b.height if rec_b and rec_b.height else "",
            f"imagery_date_{year_a}": rec_a.imagery_date if rec_a else "",
            f"imagery_date_{year_b}": rec_b.imagery_date if rec_b else "",
            f"capture_date_{year_a}": rec_a.image_capture_date if rec_a else "",
            f"capture_date_{year_b}": rec_b.image_capture_date if rec_b else "",
            f"metadata_status_{year_a}": rec_a.metadata_status if rec_a else "missing",
            f"metadata_status_{year_b}": rec_b.metadata_status if rec_b else "missing",
            "latitude": lat,
            "longitude": lon,
            "resized_planned": pair.resized_planned,
            "duplicate_years": pair.duplicate_years,
            "invalid_years": pair.invalid_years,
            "missing_years": pair.missing_years,
            "validation_notes": _split_notes(
                rec_a.validation_notes if rec_a else "", rec_b.validation_notes if rec_b else ""
            ),
            "metadata_notes": _split_notes(
                rec_a.metadata_notes if rec_a else "", rec_b.metadata_notes if rec_b else ""
            ),
            "pair_notes": pair.notes,
        }

    def _build_summary(
        self,
        rows: Sequence[FileRecord],
        pairs: Sequence[PairInfo],
        unrecognized: Sequence[Mapping[str, str]],
    ) -> dict[str, Any]:
        statuses = Counter(row.validation_status for row in rows)
        pair_statuses = Counter(pair.status for pair in pairs)
        expected_screenshots = len(self.config.tile_ids) * len(self.config.years)
        expected_tiles = len(self.config.tile_ids)
        complete_pairs = sum(1 for pair in pairs if pair.status in {"complete", "complete_resized"})
        processable_pairs = sum(1 for pair in pairs if pair.processable)
        missing_rows = [
            row for row in rows if row.validation_status == MISSING_FILE and row.in_expected_set
        ]
        invalid_rows = [
            row
            for row in rows
            if row.validation_status
            in {INVALID_IMAGE, INVALID_DIMENSIONS, UNEXPECTED_DIMENSIONS, FILE_TOO_LARGE}
        ]
        duplicate_rows = [row for row in rows if row.validation_status == DUPLICATE]
        valid_rows = [row for row in rows if row.validation_status == VALID]
        return {
            "expected_tiles": expected_tiles,
            "expected_screenshots": expected_screenshots,
            "valid_files": len(valid_rows),
            "missing_files": len(missing_rows),
            "invalid_files": len(invalid_rows),
            "duplicate_files": len(duplicate_rows),
            "unrecognized_files": len(unrecognized),
            "complete_pairs": complete_pairs,
            "processable_pairs": processable_pairs,
            "missing_pairs": sum(
                1
                for pair in pairs
                if pair.status.startswith("missing") or pair.status.startswith("invalid")
            ),
            "duplicate_pairs": sum(1 for pair in pairs if pair.duplicate_years),
            "completion_pct": round(100.0 * complete_pairs / expected_tiles, 2)
            if expected_tiles
            else 0.0,
            "file_completion_pct": round(100.0 * len(valid_rows) / expected_screenshots, 2)
            if expected_screenshots
            else 0.0,
            "validation_statuses": dict(statuses),
            "pair_statuses": dict(pair_statuses),
        }


def init_project(config: AppConfig) -> dict[str, Any]:
    """Create folders and safe metadata templates, then build the initial manifest."""
    ensure_directories(config)
    created: list[str] = []
    for year in config.years:
        directory = config.paths.raw_year_dir(year)
        directory.mkdir(parents=True, exist_ok=True)
        keep = directory / ".gitkeep"
        if not keep.exists():
            keep.write_text("", encoding="utf-8")

    if not config.paths.tile_registry.exists():
        write_csv(
            config.paths.tile_registry,
            [
                {
                    "tile_id": tile_id,
                    "label": "",
                    "latitude": "",
                    "longitude": "",
                    "notes": "",
                }
                for tile_id in config.tile_ids
            ],
            columns=REGISTRY_COLUMNS,
        )
        created.append(str(config.paths.tile_registry))
    if not config.paths.manual_overrides.exists():
        write_csv(config.paths.manual_overrides, [], columns=OVERRIDE_COLUMNS)
        created.append(str(config.paths.manual_overrides))
    if not config.paths.review_status.exists():
        write_csv(config.paths.review_status, [], columns=REVIEW_COLUMNS)
        created.append(str(config.paths.review_status))

    manager = DatasetManager(config)
    result = manager.scan(save=True)
    created.extend([str(config.paths.manifest), str(config.paths.pair_summary)])
    return {"created": created, "summary": result.summary}

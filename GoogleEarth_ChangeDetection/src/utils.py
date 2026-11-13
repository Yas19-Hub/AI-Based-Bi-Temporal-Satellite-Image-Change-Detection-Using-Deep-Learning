"""Shared helpers for safe file I/O, dates, images and hashing."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = False

TILE_PATTERN = re.compile(r"^[Tt](\d{3})_(\d{4})$")
IMAGERY_DATE_PATTERN = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")


class DataFileError(RuntimeError):
    """Raised when a metadata or derived file cannot be parsed safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_tile_id(number: int) -> str:
    if not 1 <= number <= 999:
        raise ValueError(f"Tile number out of range: {number}")
    return f"T{number:03d}"


def tile_number(tile_id: str) -> int:
    match = re.fullmatch(r"[Tt](\d{3})", str(tile_id).strip())
    if not match:
        raise ValueError(f"Invalid tile id: {tile_id!r}")
    return int(match.group(1))


def normalize_tile_id(tile_id: str) -> str:
    return format_tile_id(tile_number(tile_id))


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_with_retry(temp: Path, destination: Path, attempts: int = 5) -> None:
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            os.replace(temp, destination)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.2 * (attempt + 1))
    if temp.exists():
        try:
            temp.unlink()
        except OSError:
            pass
    raise DataFileError(f"Could not replace {destination}: {last_error}")


def _temp_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    temp = _temp_path(path)
    try:
        temp.write_bytes(data)
        _replace_with_retry(temp, path)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n"
    atomic_write_text(path, text)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, np.bool_):
        return "true" if bool(value) else "false"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list, set)):
        return ";".join(str(item) for item in value)
    return value


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> None:
    if isinstance(rows, pd.DataFrame):
        frame = rows.copy()
    else:
        normalized = [{key: _csv_value(value) for key, value in row.items()} for row in rows]
        frame = pd.DataFrame(normalized)
    if columns is not None:
        for column in columns:
            if column not in frame.columns:
                frame[column] = ""
        frame = frame.loc[:, list(columns)]
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].map(_csv_value)
    temp = _temp_path(path)
    try:
        frame.to_csv(temp, index=False, encoding="utf-8", lineterminator="\n")
        _replace_with_retry(temp, path)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def read_csv(path: Path, columns: Sequence[str] | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return []
    except Exception as exc:
        raise DataFileError(f"Could not read {path}: {exc}") from exc
    if columns is not None:
        for column in columns:
            if column not in frame.columns:
                frame[column] = ""
        frame = frame.loc[:, list(columns)]
    return [
        {str(key): str(value).strip() for key, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise DataFileError(f"Could not read JSON file {path}: {exc}") from exc


def validate_imagery_date(value: str) -> tuple[str, str | None]:
    """Return (normalized value, error). Blank means unknown, not a date."""
    text = str(value or "").strip()
    if not text:
        return "", None
    match = IMAGERY_DATE_PATTERN.fullmatch(text)
    if not match:
        return text, "Expected YYYY, YYYY-MM or YYYY-MM-DD."
    year, month, day = match.groups()
    try:
        if month is None:
            datetime(int(year), 1, 1)
        elif day is None:
            datetime(int(year), int(month), 1)
        else:
            datetime(int(year), int(month), int(day))
    except ValueError:
        return text, "Date is not a valid calendar date."
    return text, None


def validate_capture_date(value: str) -> tuple[str, str | None]:
    text = str(value or "").strip()
    if not text:
        return "", None
    candidate = text.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(candidate)
    except ValueError:
        try:
            datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            return text, "Expected ISO date/time (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ)."
    return text, None


def parse_coordinate(value: str, kind: str) -> tuple[float | None, str | None]:
    text = str(value or "").strip()
    if not text:
        return None, None
    try:
        number = float(text)
    except ValueError:
        return None, f"{kind} must be a number."
    if kind.lower().startswith("lat") and not -90.0 <= number <= 90.0:
        return None, "Latitude must be between -90 and 90."
    if kind.lower().startswith("lon") and not -180.0 <= number <= 180.0:
        return None, "Longitude must be between -180 and 180."
    return number, None


def inspect_image(path: Path, min_width: int = 1, min_height: int = 1) -> dict[str, Any]:
    """Fully decode an image and return metadata or a clear validation error."""
    result: dict[str, Any] = {
        "ok": False,
        "width": None,
        "height": None,
        "format": None,
        "mode": None,
        "error": None,
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                width, height = image.size
                result.update(
                    width=int(width),
                    height=int(height),
                    format=image.format,
                    mode=image.mode,
                    ok=True,
                )
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        result["error"] = f"Unreadable image: {exc}"
        return result
    except Image.DecompressionBombWarning as exc:
        result["error"] = f"Image exceeds Pillow decompression safety limit: {exc}"
        return result
    except Image.DecompressionBombError as exc:
        result["error"] = f"Image exceeds Pillow decompression safety limit: {exc}"
        return result

    if result["width"] is None or result["height"] is None:
        result["error"] = "Image has no readable dimensions."
        result["ok"] = False
    elif result["width"] < min_width or result["height"] < min_height:
        result["error"] = (
            f"Image is {result['width']}x{result['height']}, smaller than the required "
            f"{min_width}x{min_height}."
        )
        result["ok"] = False
    return result


def load_image_rgb(path: Path, max_dimension: int | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        rgb = image.convert("RGB")
        if max_dimension and max(rgb.size) > max_dimension:
            scale = max_dimension / float(max(rgb.size))
            size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
            rgb = rgb.resize(size, Image.Resampling.LANCZOS)
        rgb.load()
        return np.asarray(rgb, dtype=np.uint8).copy()


def save_image(path: Path, array: np.ndarray) -> None:
    """Atomically save a uint8 RGB or grayscale array as PNG."""
    if array.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {array.dtype}.")
    if array.ndim == 2:
        image = Image.fromarray(array, mode="L")
    elif array.ndim == 3 and array.shape[2] == 3:
        image = Image.fromarray(array, mode="RGB")
    else:
        raise ValueError(f"Unsupported image shape: {array.shape}")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    atomic_write_bytes(path, buffer.getvalue())


def verify_written_image(
    path: Path, expected_width: int | None = None, expected_height: int | None = None
) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise DataFileError(f"Expected output was not written: {path}")
    info = inspect_image(path)
    if not info["ok"]:
        raise DataFileError(f"Output image failed validation ({path}): {info['error']}")
    if expected_width is not None and info["width"] != expected_width:
        raise DataFileError(f"Output width {info['width']} != expected {expected_width}: {path}")
    if expected_height is not None and info["height"] != expected_height:
        raise DataFileError(f"Output height {info['height']} != expected {expected_height}: {path}")


def display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))

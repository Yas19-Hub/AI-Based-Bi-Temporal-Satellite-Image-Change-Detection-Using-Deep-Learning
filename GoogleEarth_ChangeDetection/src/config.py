"""Configuration loading, validation and path management."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILENAME = "config.yaml"
TILE_ID_MIN = 1
TILE_ID_MAX = 999


class ConfigError(ValueError):
    """Raised when configuration values are invalid."""


@dataclass(frozen=True)
class PathSettings:
    root: Path
    raw_screenshots: Path
    metadata: Path
    processed: Path
    aligned: Path
    comparisons: Path
    change_maps: Path
    reports: Path
    logs: Path
    manifest: Path
    pair_summary: Path
    manual_overrides: Path
    tile_registry: Path
    review_status: Path
    results_index: Path
    tile_stats: Path

    def raw_year_dir(self, year: int) -> Path:
        return self.raw_screenshots / str(year)

    def relative_to_root(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(path)


def build_paths(root: Path, overrides: dict[str, Any] | None = None) -> PathSettings:
    root = Path(root).resolve()
    values: dict[str, Any] = {
        "raw_screenshots": "raw_screenshots",
        "metadata": "metadata",
        "processed": "processed",
        "aligned": "processed/aligned",
        "comparisons": "processed/comparisons",
        "change_maps": "processed/change_maps",
        "reports": "reports",
        "logs": "logs",
        "manifest": "metadata/tile_manifest.csv",
        "pair_summary": "metadata/pair_summary.csv",
        "manual_overrides": "metadata/manual_overrides.csv",
        "tile_registry": "metadata/tile_registry.csv",
        "review_status": "metadata/review_status.csv",
        "results_index": "processed/results_index.json",
        "tile_stats": "processed/tile_stats.csv",
    }
    if overrides:
        unknown = set(overrides) - set(values)
        if unknown:
            raise ConfigError(f"Unknown path setting(s): {', '.join(sorted(unknown))}")
        values.update(overrides)

    resolved: dict[str, Path] = {}
    for key, value in values.items():
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"Path '{key}' must be a string or path.")
        candidate = Path(value)
        resolved[key] = candidate if candidate.is_absolute() else (root / candidate)
    return PathSettings(root=root, **{k: v.resolve() for k, v in resolved.items()})


@dataclass(frozen=True)
class ValidationSettings:
    allowed_extensions: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".tif",
        ".tiff",
        ".bmp",
    )
    min_width: int = 200
    min_height: int = 200
    max_file_size_mb: int = 100
    skip_ambiguous_duplicates: bool = True

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@dataclass(frozen=True)
class AlignmentSettings:
    enabled: bool = True
    method: str = "auto"
    max_features: int = 8000
    min_inliers: int = 20
    min_inlier_ratio: float = 0.15
    min_overlap: float = 0.55
    max_scale_change: float = 0.25
    max_rotation_deg: float = 3.0
    ratio_test: float = 0.75
    ransac_threshold: float = 5.0
    ecc_enabled: bool = True
    ecc_max_dimension: int = 1000
    ecc_iterations: int = 200
    ecc_epsilon: float = 0.00001
    ecc_min_correlation: float = 0.5


@dataclass(frozen=True)
class DiffSettings:
    pixel_threshold: int = 30
    ssim_threshold: float = 0.85
    combine_mode: str = "and"
    min_region_area: int = 150
    morphology_kernel: int = 5
    ssim_win_size: int = 7
    max_annotated_regions: int = 40
    exclude_border_px: int = 0
    normalize_brightness: bool = False
    heatmap_percentile: float = 99.0


@dataclass(frozen=True)
class ProcessingSettings:
    resize_mismatched: bool = True
    save_aligned: bool = True
    skip_existing: bool = True
    alignment: AlignmentSettings = AlignmentSettings()
    diff: DiffSettings = DiffSettings()


@dataclass(frozen=True)
class DashboardSettings:
    default_pixel_threshold: int = 30
    default_ssim_threshold: float = 0.85
    display_max_width: int = 900
    cache_ttl_seconds: int = 5


@dataclass(frozen=True)
class LoggingSettings:
    level: str = "INFO"
    max_bytes: int = 5 * 1024 * 1024
    backup_count: int = 3


@dataclass(frozen=True)
class AppConfig:
    paths: PathSettings
    project_name: str
    years: tuple[int, int]
    tile_ids: tuple[str, ...]
    expected_dimensions: tuple[int, int] | None
    validation: ValidationSettings
    processing: ProcessingSettings
    dashboard: DashboardSettings
    logging: LoggingSettings

    @property
    def root(self) -> Path:
        return self.paths.root

    @property
    def tile_start(self) -> int:
        return int(self.tile_ids[0][1:])

    @property
    def tile_end(self) -> int:
        return int(self.tile_ids[-1][1:])

    def with_root(self, root: Path) -> "AppConfig":
        return replace(self, paths=build_paths(root))


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ConfigError(f"'{name}' must be true or false, got {value!r}.")


def _as_int(value: Any, name: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{name}' must be an integer, got {value!r}.")
    if minimum is not None and value < minimum:
        raise ConfigError(f"'{name}' must be >= {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ConfigError(f"'{name}' must be <= {maximum}, got {value}.")
    return value


def _as_float(
    value: Any, name: str, minimum: float | None = None, maximum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"'{name}' must be a number, got {value!r}.")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ConfigError(f"'{name}' must be >= {minimum}, got {result}.")
    if maximum is not None and result > maximum:
        raise ConfigError(f"'{name}' must be <= {maximum}, got {result}.")
    return result


def _as_mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"'{name}' must be a mapping.")
    return value


def _tile_ids_from_config(project: dict[str, Any]) -> tuple[str, ...]:
    if "tiles" in project:
        tiles = _as_mapping(project["tiles"], "project.tiles")
        start = _as_int(tiles.get("start", 1), "project.tiles.start", TILE_ID_MIN, TILE_ID_MAX)
        end = _as_int(tiles.get("end", 100), "project.tiles.end", start, TILE_ID_MAX)
        return tuple(f"T{number:03d}" for number in range(start, end + 1))
    if "tile_ids" in project:
        raw_ids = project["tile_ids"]
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ConfigError("'project.tile_ids' must be a non-empty list.")
        ids: list[str] = []
        for raw in raw_ids:
            tile = str(raw).strip().upper()
            if not (len(tile) == 4 and tile.startswith("T") and tile[1:].isdigit()):
                raise ConfigError(f"Invalid tile id {raw!r}; expected values like T001.")
            ids.append(tile)
        if len(set(ids)) != len(ids):
            raise ConfigError("'project.tile_ids' contains duplicates.")
        return tuple(sorted(ids))
    return tuple(f"T{number:03d}" for number in range(1, 101))


def _build_validation(raw: dict[str, Any]) -> ValidationSettings:
    defaults = ValidationSettings()
    extensions = raw.get("allowed_extensions", list(defaults.allowed_extensions))
    if not isinstance(extensions, list) or not extensions:
        raise ConfigError("'validation.allowed_extensions' must be a non-empty list.")
    normalized: list[str] = []
    for extension in extensions:
        ext = str(extension).strip().lower()
        if not ext.startswith("."):
            ext = "." + ext
        if not ext.isascii() or len(ext) < 2:
            raise ConfigError(f"Invalid extension {extension!r}.")
        normalized.append(ext)
    return ValidationSettings(
        allowed_extensions=tuple(dict.fromkeys(normalized)),
        min_width=_as_int(raw.get("min_width", defaults.min_width), "validation.min_width", 1),
        min_height=_as_int(raw.get("min_height", defaults.min_height), "validation.min_height", 1),
        max_file_size_mb=_as_int(
            raw.get("max_file_size_mb", defaults.max_file_size_mb),
            "validation.max_file_size_mb",
            1,
        ),
        skip_ambiguous_duplicates=_as_bool(
            raw.get("skip_ambiguous_duplicates", defaults.skip_ambiguous_duplicates),
            "validation.skip_ambiguous_duplicates",
        ),
    )


def _build_alignment(raw: dict[str, Any]) -> AlignmentSettings:
    defaults = AlignmentSettings()
    method = str(raw.get("method", defaults.method)).strip().lower()
    if method not in {"auto", "feature", "ecc", "none"}:
        raise ConfigError("'processing.alignment.method' must be auto, feature, ecc or none.")
    return AlignmentSettings(
        enabled=_as_bool(raw.get("enabled", defaults.enabled), "processing.alignment.enabled"),
        method=method,
        max_features=_as_int(
            raw.get("max_features", defaults.max_features),
            "processing.alignment.max_features",
            500,
        ),
        min_inliers=_as_int(
            raw.get("min_inliers", defaults.min_inliers),
            "processing.alignment.min_inliers",
            4,
        ),
        min_inlier_ratio=_as_float(
            raw.get("min_inlier_ratio", defaults.min_inlier_ratio),
            "processing.alignment.min_inlier_ratio",
            0.0,
            1.0,
        ),
        min_overlap=_as_float(
            raw.get("min_overlap", defaults.min_overlap),
            "processing.alignment.min_overlap",
            0.1,
            1.0,
        ),
        max_scale_change=_as_float(
            raw.get("max_scale_change", defaults.max_scale_change),
            "processing.alignment.max_scale_change",
            0.01,
            0.9,
        ),
        max_rotation_deg=_as_float(
            raw.get("max_rotation_deg", defaults.max_rotation_deg),
            "processing.alignment.max_rotation_deg",
            0.5,
            45.0,
        ),
        ratio_test=_as_float(
            raw.get("ratio_test", defaults.ratio_test),
            "processing.alignment.ratio_test",
            0.5,
            0.95,
        ),
        ransac_threshold=_as_float(
            raw.get("ransac_threshold", defaults.ransac_threshold),
            "processing.alignment.ransac_threshold",
            0.5,
            20.0,
        ),
        ecc_enabled=_as_bool(
            raw.get("ecc_enabled", defaults.ecc_enabled), "processing.alignment.ecc_enabled"
        ),
        ecc_max_dimension=_as_int(
            raw.get("ecc_max_dimension", defaults.ecc_max_dimension),
            "processing.alignment.ecc_max_dimension",
            200,
        ),
        ecc_iterations=_as_int(
            raw.get("ecc_iterations", defaults.ecc_iterations),
            "processing.alignment.ecc_iterations",
            10,
        ),
        ecc_epsilon=_as_float(
            raw.get("ecc_epsilon", defaults.ecc_epsilon),
            "processing.alignment.ecc_epsilon",
            1e-9,
            1.0,
        ),
        ecc_min_correlation=_as_float(
            raw.get("ecc_min_correlation", defaults.ecc_min_correlation),
            "processing.alignment.ecc_min_correlation",
            -1.0,
            1.0,
        ),
    )


def _build_diff(raw: dict[str, Any]) -> DiffSettings:
    defaults = DiffSettings()
    combine_mode = str(raw.get("combine_mode", defaults.combine_mode)).strip().lower()
    if combine_mode not in {"and", "or"}:
        raise ConfigError("'processing.diff.combine_mode' must be 'and' or 'or'.")
    kernel = _as_int(
        raw.get("morphology_kernel", defaults.morphology_kernel),
        "processing.diff.morphology_kernel",
        1,
        31,
    )
    if kernel % 2 == 0:
        raise ConfigError("'processing.diff.morphology_kernel' must be odd.")
    ssim_window = _as_int(
        raw.get("ssim_win_size", defaults.ssim_win_size),
        "processing.diff.ssim_win_size",
        3,
        31,
    )
    if ssim_window % 2 == 0:
        raise ConfigError("'processing.diff.ssim_win_size' must be odd.")
    return DiffSettings(
        pixel_threshold=_as_int(
            raw.get("pixel_threshold", defaults.pixel_threshold),
            "processing.diff.pixel_threshold",
            0,
            255,
        ),
        ssim_threshold=_as_float(
            raw.get("ssim_threshold", defaults.ssim_threshold),
            "processing.diff.ssim_threshold",
            0.0,
            1.0,
        ),
        combine_mode=combine_mode,
        min_region_area=_as_int(
            raw.get("min_region_area", defaults.min_region_area),
            "processing.diff.min_region_area",
            0,
        ),
        morphology_kernel=kernel,
        ssim_win_size=ssim_window,
        max_annotated_regions=_as_int(
            raw.get("max_annotated_regions", defaults.max_annotated_regions),
            "processing.diff.max_annotated_regions",
            1,
            500,
        ),
        exclude_border_px=_as_int(
            raw.get("exclude_border_px", defaults.exclude_border_px),
            "processing.diff.exclude_border_px",
            0,
            500,
        ),
        normalize_brightness=_as_bool(
            raw.get("normalize_brightness", defaults.normalize_brightness),
            "processing.diff.normalize_brightness",
        ),
        heatmap_percentile=_as_float(
            raw.get("heatmap_percentile", defaults.heatmap_percentile),
            "processing.diff.heatmap_percentile",
            50.0,
            100.0,
        ),
    )


def _build_processing(raw: dict[str, Any]) -> ProcessingSettings:
    defaults = ProcessingSettings()
    return ProcessingSettings(
        resize_mismatched=_as_bool(
            raw.get("resize_mismatched", defaults.resize_mismatched),
            "processing.resize_mismatched",
        ),
        save_aligned=_as_bool(
            raw.get("save_aligned", defaults.save_aligned), "processing.save_aligned"
        ),
        skip_existing=_as_bool(
            raw.get("skip_existing", defaults.skip_existing), "processing.skip_existing"
        ),
        alignment=_build_alignment(_as_mapping(raw.get("alignment"), "processing.alignment")),
        diff=_build_diff(_as_mapping(raw.get("diff"), "processing.diff")),
    )


def _build_config(raw: dict[str, Any], root: Path) -> AppConfig:
    project = _as_mapping(raw.get("project"), "project")
    years_raw = project.get("years", [2019, 2024])
    if not isinstance(years_raw, list) or len(years_raw) != 2:
        raise ConfigError("'project.years' must contain exactly two years.")
    years = tuple(_as_int(year, "project.years", 1900, 2200) for year in years_raw)
    if years[0] == years[1]:
        raise ConfigError("'project.years' must contain two different years.")
    years = (years[0], years[1])  # type: ignore[assignment]

    expected_dimensions: tuple[int, int] | None = None
    expected_raw = project.get("expected_dimensions")
    if expected_raw is not None:
        expected_map = _as_mapping(expected_raw, "project.expected_dimensions")
        width = _as_int(expected_map.get("width"), "project.expected_dimensions.width", 1)
        height = _as_int(expected_map.get("height"), "project.expected_dimensions.height", 1)
        expected_dimensions = (width, height)

    dashboard_raw = _as_mapping(raw.get("dashboard"), "dashboard")
    dashboard_defaults = DashboardSettings()
    logging_raw = _as_mapping(raw.get("logging"), "logging")
    logging_defaults = LoggingSettings()
    level = str(logging_raw.get("level", logging_defaults.level)).upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError("'logging.level' must be DEBUG, INFO, WARNING, ERROR or CRITICAL.")

    paths = build_paths(root, _as_mapping(raw.get("paths"), "paths"))
    return AppConfig(
        paths=paths,
        project_name=str(project.get("name", "Google Earth Historical Image Change Detection")),
        years=years,
        tile_ids=_tile_ids_from_config(project),
        expected_dimensions=expected_dimensions,
        validation=_build_validation(_as_mapping(raw.get("validation"), "validation")),
        processing=_build_processing(_as_mapping(raw.get("processing"), "processing")),
        dashboard=DashboardSettings(
            default_pixel_threshold=_as_int(
                dashboard_raw.get("default_pixel_threshold", dashboard_defaults.default_pixel_threshold),
                "dashboard.default_pixel_threshold",
                0,
                255,
            ),
            default_ssim_threshold=_as_float(
                dashboard_raw.get("default_ssim_threshold", dashboard_defaults.default_ssim_threshold),
                "dashboard.default_ssim_threshold",
                0.0,
                1.0,
            ),
            display_max_width=_as_int(
                dashboard_raw.get("display_max_width", dashboard_defaults.display_max_width),
                "dashboard.display_max_width",
                300,
                4000,
            ),
            cache_ttl_seconds=_as_int(
                dashboard_raw.get("cache_ttl_seconds", dashboard_defaults.cache_ttl_seconds),
                "dashboard.cache_ttl_seconds",
                0,
                3600,
            ),
        ),
        logging=LoggingSettings(
            level=level,
            max_bytes=_as_int(
                logging_raw.get("max_bytes", logging_defaults.max_bytes),
                "logging.max_bytes",
                1024,
            ),
            backup_count=_as_int(
                logging_raw.get("backup_count", logging_defaults.backup_count),
                "logging.backup_count",
                0,
                100,
            ),
        ),
    )


def load_config(
    config_path: str | Path | None = None, root: str | Path | None = None
) -> AppConfig:
    """Load configuration from YAML, falling back to validated defaults."""
    root_path = Path(root).resolve() if root is not None else PROJECT_ROOT
    path = Path(config_path) if config_path is not None else root_path / DEFAULT_CONFIG_FILENAME
    if not path.is_absolute():
        path = (root_path / path).resolve()

    raw: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"Could not parse {path}: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path} must contain a YAML mapping.")
        raw = loaded
    return _build_config(raw, root_path)


def config_as_dict(config: AppConfig) -> dict[str, Any]:
    """Serialize the config to a YAML-friendly mapping."""
    relative = lambda path: config.paths.relative_to_root(path)
    return {
        "project": {
            "name": config.project_name,
            "years": list(config.years),
            "tiles": {"start": config.tile_start, "end": config.tile_end},
            "expected_dimensions": (
                None
                if config.expected_dimensions is None
                else {
                    "width": config.expected_dimensions[0],
                    "height": config.expected_dimensions[1],
                }
            ),
        },
        "paths": {
            "raw_screenshots": relative(config.paths.raw_screenshots),
            "metadata": relative(config.paths.metadata),
            "processed": relative(config.paths.processed),
            "aligned": relative(config.paths.aligned),
            "comparisons": relative(config.paths.comparisons),
            "change_maps": relative(config.paths.change_maps),
            "reports": relative(config.paths.reports),
            "logs": relative(config.paths.logs),
            "manifest": relative(config.paths.manifest),
            "pair_summary": relative(config.paths.pair_summary),
            "manual_overrides": relative(config.paths.manual_overrides),
            "tile_registry": relative(config.paths.tile_registry),
            "review_status": relative(config.paths.review_status),
            "results_index": relative(config.paths.results_index),
            "tile_stats": relative(config.paths.tile_stats),
        },
        "validation": {
            "allowed_extensions": list(config.validation.allowed_extensions),
            "min_width": config.validation.min_width,
            "min_height": config.validation.min_height,
            "max_file_size_mb": config.validation.max_file_size_mb,
            "skip_ambiguous_duplicates": config.validation.skip_ambiguous_duplicates,
        },
        "processing": {
            "resize_mismatched": config.processing.resize_mismatched,
            "save_aligned": config.processing.save_aligned,
            "skip_existing": config.processing.skip_existing,
            "alignment": {
                "enabled": config.processing.alignment.enabled,
                "method": config.processing.alignment.method,
                "max_features": config.processing.alignment.max_features,
                "min_inliers": config.processing.alignment.min_inliers,
                "min_inlier_ratio": config.processing.alignment.min_inlier_ratio,
                "min_overlap": config.processing.alignment.min_overlap,
                "max_scale_change": config.processing.alignment.max_scale_change,
                "max_rotation_deg": config.processing.alignment.max_rotation_deg,
                "ratio_test": config.processing.alignment.ratio_test,
                "ransac_threshold": config.processing.alignment.ransac_threshold,
                "ecc_enabled": config.processing.alignment.ecc_enabled,
                "ecc_max_dimension": config.processing.alignment.ecc_max_dimension,
                "ecc_iterations": config.processing.alignment.ecc_iterations,
                "ecc_epsilon": config.processing.alignment.ecc_epsilon,
                "ecc_min_correlation": config.processing.alignment.ecc_min_correlation,
            },
            "diff": {
                "pixel_threshold": config.processing.diff.pixel_threshold,
                "ssim_threshold": config.processing.diff.ssim_threshold,
                "combine_mode": config.processing.diff.combine_mode,
                "min_region_area": config.processing.diff.min_region_area,
                "morphology_kernel": config.processing.diff.morphology_kernel,
                "ssim_win_size": config.processing.diff.ssim_win_size,
                "max_annotated_regions": config.processing.diff.max_annotated_regions,
                "exclude_border_px": config.processing.diff.exclude_border_px,
                "normalize_brightness": config.processing.diff.normalize_brightness,
                "heatmap_percentile": config.processing.diff.heatmap_percentile,
            },
        },
        "dashboard": {
            "default_pixel_threshold": config.dashboard.default_pixel_threshold,
            "default_ssim_threshold": config.dashboard.default_ssim_threshold,
            "display_max_width": config.dashboard.display_max_width,
            "cache_ttl_seconds": config.dashboard.cache_ttl_seconds,
        },
        "logging": {
            "level": config.logging.level,
            "max_bytes": config.logging.max_bytes,
            "backup_count": config.logging.backup_count,
        },
    }


def ensure_directories(config: AppConfig) -> None:
    directories: Iterable[Path] = (
        config.paths.raw_screenshots,
        *(config.paths.raw_year_dir(year) for year in config.years),
        config.paths.metadata,
        config.paths.processed,
        config.paths.aligned,
        config.paths.comparisons,
        config.paths.change_maps,
        config.paths.reports,
        config.paths.logs,
        config.paths.manifest.parent,
        config.paths.results_index.parent,
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def write_default_config(destination: Path) -> bool:
    """Write a commented default config unless one already exists."""
    if destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    config = load_config(root=destination.parent)
    text = (
        "# Google Earth Historical Image Change Detection - configuration\n"
        "# All relative paths are resolved against the project root.\n\n"
        + yaml.safe_dump(config_as_dict(config), sort_keys=False, allow_unicode=True)
    )
    destination.write_text(text, encoding="utf-8")
    return True

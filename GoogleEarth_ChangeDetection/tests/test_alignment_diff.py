import cv2
import numpy as np
import pytest

from conftest import make_texture
from src.alignment import AlignmentError, align_pair, validate_transform
from src.config import AlignmentSettings, DiffSettings
from src.diff import DiffError, analyze_pair, render_comparison


def test_disabled_alignment_returns_identity_and_full_mask() -> None:
    reference = make_texture(100, 80, seed=1)
    target = make_texture(100, 80, seed=2)
    result = align_pair(
        reference, target, AlignmentSettings(enabled=False, method="none")
    )
    assert result.method == "none"
    assert result.status == "none"
    assert result.reliable is False
    assert result.aligned_target.shape == reference.shape
    assert np.array_equal(result.aligned_target, target)
    assert np.all(result.valid_mask == 255)


def test_mismatched_dimensions_resize_or_raise() -> None:
    reference = make_texture(100, 80, seed=1)
    target = make_texture(50, 40, seed=2)
    result = align_pair(
        reference, target, AlignmentSettings(enabled=False, method="none"), resize_mismatched=True
    )
    assert result.resized is True
    assert result.aligned_target.shape == reference.shape

    with pytest.raises(AlignmentError):
        align_pair(
            reference,
            target,
            AlignmentSettings(enabled=False, method="none"),
            resize_mismatched=False,
        )


def test_feature_alignment_recovers_synthetic_translation() -> None:
    rng = np.random.default_rng(1)
    reference = rng.integers(0, 255, (360, 480, 3), dtype=np.uint8)
    reference = cv2.GaussianBlur(reference, (5, 5), 0)
    matrix = np.float32([[1, 0, 10], [0, 1, -6]])
    target = cv2.warpAffine(
        reference, matrix, (480, 360), borderMode=cv2.BORDER_REFLECT
    )
    result = align_pair(reference, target, AlignmentSettings(method="feature"))
    assert result.reliable is True
    assert result.method == "feature"
    assert result.inliers >= 20
    assert result.overlap > 0.8


def test_ecc_alignment_recovers_synthetic_translation() -> None:
    yy, xx = np.mgrid[0:360, 0:480]
    reference = np.stack(
        [
            (xx * 0.2 + 20 * np.sin(yy / 17)).astype(np.uint8),
            (yy * 0.25 + 20 * np.cos(xx / 23)).astype(np.uint8),
            ((xx + yy) * 0.1).astype(np.uint8),
        ],
        axis=2,
    )
    matrix = np.float32([[1, 0, 8], [0, 1, -5]])
    target = cv2.warpAffine(
        reference, matrix, (480, 360), borderMode=cv2.BORDER_REFLECT
    )
    result = align_pair(reference, target, AlignmentSettings(method="ecc"))
    assert result.reliable is True
    assert result.method.startswith("ecc")
    assert result.ecc_correlation > 0.9
    assert result.overlap > 0.8


def test_transform_validation_rejects_extreme_scale_and_rotation() -> None:
    settings = AlignmentSettings(
        enabled=True,
        method="feature",
        max_scale_change=0.25,
        max_rotation_deg=3.0,
    )
    ok, notes, metrics = validate_transform(np.eye(3), (100, 100), (100, 100), settings)
    assert ok is True
    assert notes == []
    assert metrics["scale_change"] == pytest.approx(1.0)

    scale = np.diag([4.0, 4.0, 1.0])
    ok, notes, _ = validate_transform(scale, (100, 100), (100, 100), settings)
    assert ok is False
    assert any("Scale change" in note for note in notes)

    angle = np.deg2rad(45)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ok, notes, _ = validate_transform(rotation, (100, 100), (100, 100), settings)
    assert ok is False
    assert any("Rotation" in note for note in notes)


def test_identical_images_have_zero_change() -> None:
    image = make_texture(100, 80, seed=4)
    mask = np.full(image.shape[:2], 255, dtype=np.uint8)
    result = analyze_pair(image, image.copy(), mask, DiffSettings(ssim_win_size=5))
    assert result.stats["changed_pixels"] == 0
    assert result.stats["changed_pct_of_valid"] == 0.0
    assert result.stats["mean_abs_diff"] == 0.0
    assert result.stats["ssim_mean"] == pytest.approx(1.0)
    assert result.stats["region_count"] == 0


def test_change_rectangle_is_detected_and_masked() -> None:
    reference = make_texture(120, 90, seed=5, low=180, high=240)
    target = reference.copy()
    target[30:60, 40:90] = 0
    mask = np.full(reference.shape[:2], 255, dtype=np.uint8)
    settings = DiffSettings(
        pixel_threshold=20,
        ssim_threshold=0.9,
        combine_mode="and",
        min_region_area=10,
        morphology_kernel=3,
        ssim_win_size=5,
    )
    result = analyze_pair(reference, target, mask, settings)
    assert result.stats["changed_pixels"] > 0
    assert result.stats["changed_pct_of_valid"] > 0
    assert result.stats["region_count"] >= 1
    assert np.count_nonzero(result.change_mask) > 0
    assert result.heatmap.shape == reference.shape
    assert result.overlay.shape == reference.shape
    assert result.annotated.shape == reference.shape
    region = result.regions[0]
    assert region.width >= 40
    assert region.height >= 20


def test_border_exclusion_reduces_valid_area() -> None:
    reference = make_texture(100, 100, seed=6)
    mask = np.full((100, 100), 255, dtype=np.uint8)
    result = analyze_pair(
        reference,
        reference.copy(),
        mask,
        DiffSettings(exclude_border_px=10, ssim_win_size=5),
    )
    assert result.stats["valid_pixels"] == 80 * 80
    assert result.stats["border_excluded_pct"] == pytest.approx(36.0)


def test_invalid_difference_inputs_raise() -> None:
    image = make_texture(100, 80, seed=7)
    mask = np.full((80, 100), 255, dtype=np.uint8)
    with pytest.raises(DiffError):
        analyze_pair(image, make_texture(90, 80, seed=8), mask, DiffSettings())
    with pytest.raises(DiffError):
        analyze_pair(image, image, np.full((10, 10), 255, dtype=np.uint8), DiffSettings())
    tiny = make_texture(2, 2, seed=9)
    with pytest.raises(DiffError):
        analyze_pair(
            tiny,
            tiny,
            np.full((2, 2), 255, dtype=np.uint8),
            DiffSettings(),
        )


def test_comparison_renderer_produces_rgb_panel() -> None:
    reference = make_texture(120, 90, seed=10)
    target = reference.copy()
    target[20:40, 20:50] = 0
    result = analyze_pair(
        reference,
        target,
        np.full((90, 120), 255, dtype=np.uint8),
        DiffSettings(min_region_area=10, ssim_win_size=5),
    )
    comparison = render_comparison(result, "2019 reference", "2024 target", ["test footer"])
    assert comparison.ndim == 3
    assert comparison.shape[2] == 3
    assert comparison.dtype == np.uint8
    assert comparison.shape[1] > 120

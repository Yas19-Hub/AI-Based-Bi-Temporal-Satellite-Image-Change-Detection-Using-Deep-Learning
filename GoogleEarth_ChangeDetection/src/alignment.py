"""Optional image alignment using ORB feature matching or ECC registration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .config import AlignmentSettings
from .logging_setup import get_logger

LOGGER = get_logger(__name__)

MIN_FEATURE_MATCH_COVERAGE = 0.10
FEATURE_MAX_DIMENSION = 1600


class AlignmentError(RuntimeError):
    """Raised when images cannot be prepared for comparison."""


@dataclass
class AlignmentResult:
    reference: np.ndarray
    aligned_target: np.ndarray
    valid_mask: np.ndarray
    method: str
    status: str
    reliable: bool
    notes: list[str] = field(default_factory=list)
    homography: np.ndarray | None = None
    inliers: int = 0
    inlier_ratio: float = 0.0
    match_coverage: float = 0.0
    overlap: float = 0.0
    ecc_correlation: float = 0.0
    resized: bool = False
    scale_change: float = 1.0
    rotation_deg: float = 0.0

    def metrics(self) -> dict[str, Any]:
        return {
            "alignment_method": self.method,
            "alignment_status": self.status,
            "alignment_reliable": self.reliable,
            "alignment_inliers": self.inliers,
            "alignment_inlier_ratio": round(float(self.inlier_ratio), 4),
            "alignment_match_coverage": round(float(self.match_coverage), 4),
            "alignment_overlap": round(float(self.overlap), 4),
            "alignment_ecc_correlation": round(float(self.ecc_correlation), 4),
            "alignment_scale_change": round(float(self.scale_change), 4),
            "alignment_rotation_deg": round(float(self.rotation_deg), 3),
            "alignment_resized": self.resized,
            "alignment_notes": "; ".join(self.notes),
        }


def align_pair(
    reference: np.ndarray,
    target: np.ndarray,
    settings: AlignmentSettings,
    resize_mismatched: bool = True,
) -> AlignmentResult:
    """Align target (later year) to the reference (earlier year) canvas."""
    if reference.ndim != 3 or reference.shape[2] != 3 or target.ndim != 3 or target.shape[2] != 3:
        raise AlignmentError("Alignment expects two RGB images.")
    if reference.dtype != np.uint8 or target.dtype != np.uint8:
        raise AlignmentError("Alignment expects uint8 images.")

    resized = False
    notes: list[str] = []
    if target.shape[:2] != reference.shape[:2]:
        if not resize_mismatched:
            raise AlignmentError(
                f"Image dimensions differ ({reference.shape[1]}x{reference.shape[0]} vs "
                f"{target.shape[1]}x{target.shape[0]}) and resizing is disabled."
            )
        target_height, target_width = target.shape[:2]
        target = cv2.resize(
            target,
            (reference.shape[1], reference.shape[0]),
            interpolation=(
                cv2.INTER_AREA
                if target_width * target_height > reference.shape[1] * reference.shape[0]
                else cv2.INTER_LANCZOS4
            ),
        )
        resized = True
        notes.append(
            f"Target resized from {target_width}x{target_height} to "
            f"{reference.shape[1]}x{reference.shape[0]} for comparison."
        )

    if not settings.enabled or settings.method == "none":
        notes.append("Alignment was not applied; screenshots are compared at the reference size.")
        return AlignmentResult(
            reference=reference,
            aligned_target=target,
            valid_mask=np.full(reference.shape[:2], 255, dtype=np.uint8),
            method="none",
            status="none",
            reliable=False,
            notes=notes,
            resized=resized,
        )

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)

    attempts: list[str] = []
    if settings.method in {"auto", "feature"}:
        feature_result = _try_feature(reference, target, reference_gray, target_gray, settings)
        if feature_result is not None and feature_result.reliable:
            feature_result.resized = resized
            feature_result.notes = notes + feature_result.notes
            return feature_result
        if feature_result is not None:
            attempts.append(
                "feature alignment rejected: "
                + ("; ".join(feature_result.notes) or "low quality")
            )
        else:
            attempts.append("feature matching failed")

    if settings.method in {"auto", "ecc"} and settings.ecc_enabled:
        ecc_result = _try_ecc(reference, target, settings)
        if ecc_result is not None and ecc_result.reliable:
            ecc_result.resized = resized
            ecc_result.notes = notes + ecc_result.notes
            return ecc_result
        if ecc_result is not None:
            attempts.append(
                "ECC alignment rejected: " + ("; ".join(ecc_result.notes) or "low quality")
            )
        else:
            attempts.append("ECC registration failed")

    notes.append(
        "No reliable alignment transform was found"
        + (f" ({'; '.join(attempts)})" if attempts else "")
        + "; the target is compared without geometric correction. Review this tile carefully."
    )
    return AlignmentResult(
        reference=reference,
        aligned_target=target,
        valid_mask=np.full(reference.shape[:2], 255, dtype=np.uint8),
        method="identity_fallback",
        status="identity_fallback",
        reliable=False,
        notes=notes,
        resized=resized,
    )


def _detection_scale(shape: tuple[int, ...], max_dimension: int) -> tuple[float, tuple[int, int]]:
    height, width = shape[:2]
    scale = min(1.0, max_dimension / float(max(height, width)))
    return scale, (max(1, round(width * scale)), max(1, round(height * scale)))


def _gray_for_features(gray: np.ndarray, scale: float, size: tuple[int, int]) -> np.ndarray:
    if scale != 1.0:
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _try_feature(
    reference: np.ndarray,
    target: np.ndarray,
    reference_gray: np.ndarray,
    target_gray: np.ndarray,
    settings: AlignmentSettings,
) -> AlignmentResult | None:
    scale, size = _detection_scale(reference_gray.shape, FEATURE_MAX_DIMENSION)
    ref_small = _gray_for_features(reference_gray, scale, size)
    tgt_small = _gray_for_features(target_gray, scale, size)

    orb = cv2.ORB_create(nfeatures=settings.max_features)
    keypoints_ref, descriptors_ref = orb.detectAndCompute(ref_small, None)
    keypoints_tgt, descriptors_tgt = orb.detectAndCompute(tgt_small, None)
    if descriptors_ref is None or descriptors_tgt is None:
        LOGGER.info(
            "ORB could not compute descriptors (reference=%s, target=%s)",
            len(keypoints_ref),
            len(keypoints_tgt),
        )
        return None
    if len(keypoints_ref) < 10 or len(keypoints_tgt) < 10:
        LOGGER.info("Too few ORB keypoints (%d vs %d)", len(keypoints_ref), len(keypoints_tgt))
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    try:
        knn = matcher.knnMatch(descriptors_tgt, descriptors_ref, k=2)
    except cv2.error as exc:
        LOGGER.warning("Feature matching error: %s", exc)
        return None
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < settings.ratio_test * second.distance:
            good.append(first)
    if len(good) < max(8, settings.min_inliers):
        LOGGER.info("Too few good ORB matches: %d", len(good))
        return None

    source = np.float32([keypoints_tgt[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    destination = np.float32([keypoints_ref[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(
        source,
        destination,
        cv2.RANSAC,
        settings.ransac_threshold,
        maxIters=5000,
        confidence=0.999,
    )
    if homography is None or mask is None:
        LOGGER.info("findHomography returned no model.")
        return None

    scaling = np.array(
        [[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    upscaling = np.array(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    full_homography = scaling @ homography @ upscaling

    inlier_count = int(mask.sum())
    inlier_ratio = inlier_count / float(len(good))
    points = source[mask.ravel() == 1].reshape(-1, 2)
    match_coverage = 0.0
    if len(points) >= 3:
        hull = cv2.convexHull(points.astype(np.float32))
        match_coverage = float(cv2.contourArea(hull)) / float(
            tgt_small.shape[0] * tgt_small.shape[1]
        )

    transform_ok, transform_notes, transform_metrics = validate_transform(
        full_homography, reference_gray.shape, target_gray.shape, settings
    )
    aligned, valid_mask = _warp_target(target, full_homography, reference_gray.shape)
    overlap = _mask_overlap(valid_mask)

    notes: list[str] = list(transform_notes)
    reliable = (
        transform_ok
        and inlier_count >= settings.min_inliers
        and inlier_ratio >= settings.min_inlier_ratio
        and match_coverage >= MIN_FEATURE_MATCH_COVERAGE
        and overlap >= settings.min_overlap
    )
    if inlier_count < settings.min_inliers:
        notes.append(f"Only {inlier_count} inliers (minimum {settings.min_inliers}).")
    if inlier_ratio < settings.min_inlier_ratio:
        notes.append(
            f"Inlier ratio {inlier_ratio:.2f} is below the minimum {settings.min_inlier_ratio:.2f}."
        )
    if match_coverage < MIN_FEATURE_MATCH_COVERAGE:
        notes.append(f"Matched feature coverage {match_coverage:.2f} is too localised.")
    if overlap < settings.min_overlap:
        notes.append(
            f"Warp overlaps only {overlap:.0%} of the reference (minimum {settings.min_overlap:.0%})."
        )
    if not reliable:
        LOGGER.info("Rejected feature alignment: %s", "; ".join(notes))

    return AlignmentResult(
        reference=reference,
        aligned_target=aligned,
        valid_mask=valid_mask,
        method="feature",
        status="aligned" if reliable else "rejected",
        reliable=reliable,
        notes=notes,
        homography=full_homography,
        inliers=inlier_count,
        inlier_ratio=inlier_ratio,
        match_coverage=match_coverage,
        overlap=overlap,
        scale_change=transform_metrics["scale_change"],
        rotation_deg=transform_metrics["rotation_deg"],
    )


def _try_ecc(
    reference: np.ndarray, target: np.ndarray, settings: AlignmentSettings
) -> AlignmentResult | None:
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    tgt_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    scale, size = _detection_scale(ref_gray.shape, settings.ecc_max_dimension)
    ref_small = cv2.resize(ref_gray, size, interpolation=cv2.INTER_AREA) if scale != 1.0 else ref_gray
    tgt_small = cv2.resize(tgt_gray, size, interpolation=cv2.INTER_AREA) if scale != 1.0 else tgt_gray
    ref_float = ref_small.astype(np.float32) / 255.0
    tgt_float = tgt_small.astype(np.float32) / 255.0
    ref_float = cv2.GaussianBlur(ref_float, (5, 5), 0)
    tgt_float = cv2.GaussianBlur(tgt_float, (5, 5), 0)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        settings.ecc_iterations,
        settings.ecc_epsilon,
    )
    warp_small = np.eye(2, 3, dtype=np.float32)
    correlation = -1.0
    motion_used = ""
    for motion, label in (
        (cv2.MOTION_EUCLIDEAN, "euclidean"),
        (cv2.MOTION_AFFINE, "affine"),
    ):
        candidate = np.eye(2, 3, dtype=np.float32)
        try:
            cc, candidate = cv2.findTransformECC(
                ref_float, tgt_float, candidate, motion, criteria, None, 5
            )
        except cv2.error as exc:
            LOGGER.info("ECC %s failed: %s", label, exc)
            continue
        warp_small = candidate
        correlation = float(cc)
        motion_used = label
        break
    if not motion_used:
        return None

    # findTransformECC returns a transform usable with WARP_INVERSE_MAP.
    # Convert it from detection resolution to full resolution before applying it.
    scale_matrix = np.array(
        [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    inverse_scale = np.array(
        [[1.0 / scale, 0.0, 0.0], [0.0, 1.0 / scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    warp_full_3 = np.eye(3, dtype=np.float64)
    warp_full_3[:2, :] = warp_small
    warp_full_3 = inverse_scale @ warp_full_3 @ scale_matrix
    warp_full = warp_full_3[:2, :]

    aligned, valid_mask = _warp_target(
        target, warp_full, reference.shape[:2], inverse=True
    )
    overlap = _mask_overlap(valid_mask)

    target_to_reference = np.eye(3, dtype=np.float64)
    try:
        target_to_reference[:2, :] = cv2.invertAffineTransform(warp_full)
    except cv2.error:
        return None
    transform_ok, transform_notes, transform_metrics = validate_transform(
        target_to_reference, reference.shape[:2], target.shape[:2], settings
    )
    notes = list(transform_notes)
    reliable = (
        transform_ok
        and correlation >= settings.ecc_min_correlation
        and overlap >= settings.min_overlap
    )
    if correlation < settings.ecc_min_correlation:
        notes.append(
            f"ECC correlation {correlation:.3f} is below the minimum "
            f"{settings.ecc_min_correlation:.3f}."
        )
    if overlap < settings.min_overlap:
        notes.append(
            f"Warp overlaps only {overlap:.0%} of the reference (minimum {settings.min_overlap:.0%})."
        )
    if not reliable:
        LOGGER.info("Rejected ECC alignment: %s", "; ".join(notes))

    return AlignmentResult(
        reference=reference,
        aligned_target=aligned,
        valid_mask=valid_mask,
        method=f"ecc_{motion_used}",
        status="aligned" if reliable else "rejected",
        reliable=reliable,
        notes=notes,
        homography=target_to_reference,
        overlap=overlap,
        ecc_correlation=correlation,
        scale_change=transform_metrics["scale_change"],
        rotation_deg=transform_metrics["rotation_deg"],
    )


def _warp_target(
    target: np.ndarray,
    homography: np.ndarray,
    reference_shape: tuple[int, ...],
    inverse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = reference_shape[:2]
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, np.array([[0.0, 0.0, 1.0]])])
    if matrix.shape != (3, 3):
        raise AlignmentError(f"Expected a 3x3 or 2x3 transform, got {matrix.shape}.")
    flags = cv2.INTER_LINEAR + (cv2.WARP_INVERSE_MAP if inverse else 0)
    aligned = cv2.warpPerspective(
        target,
        matrix,
        (width, height),
        flags=flags,
        borderMode=cv2.BORDER_REPLICATE,
    )
    ones = np.full(target.shape[:2], 255, dtype=np.uint8)
    valid = cv2.warpPerspective(
        ones,
        matrix,
        (width, height),
        flags=flags,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = (valid > 200).astype(np.uint8) * 255
    valid = cv2.erode(valid, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return aligned, valid


def _mask_overlap(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def validate_transform(
    homography: np.ndarray,
    reference_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    settings: AlignmentSettings,
) -> tuple[bool, list[str], dict[str, float]]:
    """Validate a target-to-reference transform without applying it."""
    notes: list[str] = []
    metrics = {"scale_change": 0.0, "rotation_deg": 0.0, "determinant": 0.0}
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return False, ["Transform is not a finite 3x3 matrix."], metrics
    if abs(matrix[2, 2]) < 1e-12:
        return False, ["Transform has a near-zero homogeneous scale."], metrics
    matrix = matrix / matrix[2, 2]
    determinant = float(np.linalg.det(matrix))
    metrics["determinant"] = determinant
    if determinant <= 0.05 or determinant > 5.0:
        notes.append(f"Transform determinant {determinant:.3f} is outside the safe range.")
    scale_change = math.sqrt(abs(determinant))
    metrics["scale_change"] = scale_change
    if abs(scale_change - 1.0) > settings.max_scale_change:
        notes.append(
            f"Scale change {scale_change:.3f} exceeds the allowed "
            f"±{settings.max_scale_change:.3f}."
        )
    rotation = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    metrics["rotation_deg"] = rotation
    if abs(rotation) > settings.max_rotation_deg:
        notes.append(
            f"Rotation {rotation:.2f}° exceeds the allowed ±{settings.max_rotation_deg:.2f}°."
        )
    if abs(matrix[2, 0]) > 0.002 or abs(matrix[2, 1]) > 0.002:
        notes.append("Transform contains excessive perspective distortion.")

    height, width = reference_shape[:2]
    target_height, target_width = target_shape[:2]
    corners = np.array(
        [
            [0.0, 0.0, 1.0],
            [float(target_width), 0.0, 1.0],
            [float(target_width), float(target_height), 1.0],
            [0.0, float(target_height), 1.0],
        ],
        dtype=np.float64,
    )
    projected = (matrix @ corners.T).T
    if np.any(np.abs(projected[:, 2]) < 1e-12):
        return False, notes + ["Transform projects a corner to infinity."], metrics
    points = projected[:, :2] / projected[:, 2:3]
    margin_x, margin_y = width * 0.5, height * 0.5
    if (
        points[:, 0].min() < -margin_x
        or points[:, 0].max() > width + margin_x
        or points[:, 1].min() < -margin_y
        or points[:, 1].max() > height + margin_y
    ):
        notes.append("Transformed corners lie far outside the reference frame.")
    return not notes, notes, metrics

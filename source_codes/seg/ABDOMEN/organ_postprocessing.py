"""Organ-specific postprocessing for fetal abdomen multi-label segmentation.

The input is a sigmoid probability array with shape (C, H, W). Structure
channels must follow Config.STRUCTURES. The skin-line channel is not processed
here because infer.py already has a dedicated skin-line pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class OrganRule:
    low_threshold: float = 0.50
    max_components: int | None = None
    close_size: int = 0
    open_size: int = 0
    fill_holes: bool = False


# Starting values only. Tune these on the validation set.
DEFAULT_RULE = OrganRule()
ORGAN_RULES: Mapping[str, OrganRule] = {
    "Stomach":        OrganRule(0.40, 1,    close_size=5, fill_holes=True),
    "Umbilical Vein": OrganRule(0.40, 1,    close_size=3),
    "Vertebrae":      OrganRule(0.45, 1, close_size=3),
    "Aorta":          OrganRule(0.35, 1,    close_size=3),
    "IVC":            OrganRule(0.35, 2,    close_size=3),
    "Adrenal":        OrganRule(0.35, 2,    close_size=3),
    "Placenta":       OrganRule(0.50, 2,    open_size=5, close_size=7,
                                fill_holes=True),
    "Gall Bladder":   OrganRule(0.40, 1,    close_size=3, fill_holes=True),
    "Kidney Cortex":  OrganRule(0.35, 2,    close_size=5, fill_holes=True),
    # Tiny renal-pelvis candidates are handled by the kidney-proximity rule.
    "Renal Pelvis":   OrganRule(0.18, 2,    close_size=3, fill_holes=True),
    "Bladder":        OrganRule(0.45, 1,    close_size=5, fill_holes=True),
    "Cord Insertion": OrganRule(0.40, 2,    close_size=3),
}


def _kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    holes = 1 - flood[1:-1, 1:-1]
    return np.maximum(mask, holes).astype(np.uint8)


def _seeded_components(
    probability: np.ndarray,
    rule: OrganRule,
    allowed_region: np.ndarray | None = None,
) -> list[tuple[np.ndarray, int, float]]:
    """Return components that pass the low threshold."""
    low = probability >= rule.low_threshold
    if allowed_region is not None:
        low &= allowed_region.astype(bool)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        low.astype(np.uint8), connectivity=8
    )
    components = []
    for label in range(1, n_labels):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        confidence = float(probability[component].mean())
        components.append((component, area, confidence))
    return components


def _finish_mask(
    components: list[tuple[np.ndarray, int, float]],
    shape: tuple[int, int],
    rule: OrganRule,
) -> np.ndarray:
    if rule.max_components is not None:
        components = sorted(
            components, key=lambda item: item[1] * item[2], reverse=True
        )[:rule.max_components]

    mask = np.zeros(shape, dtype=np.uint8)
    for component, _, _ in components:
        mask[component] = 1

    if rule.open_size > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _kernel(rule.open_size))
    if rule.close_size > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _kernel(rule.close_size))
    if rule.fill_holes and mask.any():
        mask = _fill_holes(mask)
    return mask


def _renal_pelvis_mask(
    probability: np.ndarray,
    kidney_mask: np.ndarray,
    rule: OrganRule,
) -> np.ndarray:
    # The pelvis should lie inside or very close to kidney cortex. Dilation
    # tolerates imperfect cortex prediction while rejecting distant speckle.
    near_kidney = cv2.dilate(kidney_mask, _kernel(21), iterations=1).astype(bool)
    components = _seeded_components(probability, rule)

    # Kidney cortex can itself be missed. Preserve a pelvis candidate when it
    # is near cortex OR independently very confident.
    kept = []
    for component, area, mean_confidence in components:
        near = bool(np.any(component & near_kidney))
        max_confidence = float(probability[component].max())
        strong_without_kidney = max_confidence >= 0.65 and mean_confidence >= 0.45
        if near or strong_without_kidney:
            kept.append((component, area, mean_confidence))

    return _finish_mask(kept, probability.shape, rule)


def postprocess_structures(
    struct_probs: np.ndarray,
    structure_names: Sequence[str],
    rules: Mapping[str, OrganRule] = ORGAN_RULES,
    plane: str | None = None,
) -> np.ndarray:
    """Return independent binary structure masks with shape (C, H, W)."""
    if struct_probs.ndim != 3 or struct_probs.shape[0] != len(structure_names):
        raise ValueError(
            f"Expected ({len(structure_names)}, H, W), got {struct_probs.shape}"
        )

    result = np.zeros_like(struct_probs, dtype=np.uint8)
    names_to_idx = {name: idx for idx, name in enumerate(structure_names)}

    # Process kidney first because renal-pelvis filtering depends on it.
    order = [name for name in structure_names if name != "Renal Pelvis"]
    if "Renal Pelvis" in names_to_idx:
        order.append("Renal Pelvis")

    for name in order:
        idx = names_to_idx[name]
        probability = struct_probs[idx]
        rule = rules.get(name, DEFAULT_RULE)

        if name == "Renal Pelvis" and plane is not None and plane.upper() != "TK":
            continue

        if name == "Renal Pelvis" and "Kidney Cortex" in names_to_idx:
            kidney = result[names_to_idx["Kidney Cortex"]]
            result[idx] = _renal_pelvis_mask(probability, kidney, rule)
        else:
            components = _seeded_components(probability, rule)
            result[idx] = _finish_mask(components, probability.shape, rule)

    return result


def masks_to_exclusive_label_map(
    masks: np.ndarray,
    probabilities: np.ndarray,
    priority_indices: Sequence[int] = (),
) -> np.ndarray:
    """Convert multi-label masks to infer.py's 1-based exclusive label map.

    Priority channels are written last. This is useful for renal pelvis, which
    can anatomically overlap the merged kidney-cortex mask.
    """
    if masks.shape != probabilities.shape:
        raise ValueError(f"Shape mismatch: {masks.shape} vs {probabilities.shape}")
    masked_probs = probabilities * masks
    foreground = masks.any(axis=0)
    label_map = np.zeros(masks.shape[1:], dtype=np.int32)
    label_map[foreground] = masked_probs.argmax(axis=0)[foreground] + 1
    for index in priority_indices:
        label_map[masks[index] > 0] = index + 1
    return label_map
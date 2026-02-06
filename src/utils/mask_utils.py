from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import cv2


@dataclass
class SavedMaskInfo:
    path: Path
    area_px: int


def mask_area_px(mask: np.ndarray) -> int:
    """Count pixels==1 in a binary mask."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return int(mask.sum())


def save_binary_mask_png(mask: np.ndarray, out_path: Path) -> SavedMaskInfo:
    """
    Save binary mask as PNG (0 or 255).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    mask01 = (mask > 0).astype(np.uint8)
    area = int(mask01.sum())

    cv2.imwrite(str(out_path), (mask01 * 255).astype(np.uint8))
    return SavedMaskInfo(path=out_path, area_px=area)


def compute_void_rate(component_mask: np.ndarray, void_masks: list[np.ndarray]) -> Optional[float]:
    """
    void_rate = total_void_area / component_area
    """
    comp_area = mask_area_px(component_mask)
    if comp_area == 0:
        return None
    void_area = sum(mask_area_px(m) for m in void_masks)
    return float(void_area) / float(comp_area)


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2


@dataclass
class MaskStats:
    area_px: int
    bbox_xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)


def to_uint8_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert a float/bool mask (H,W) to uint8 (0/255).
    """
    if mask.dtype != np.uint8:
        mask = (mask >= threshold).astype(np.uint8) * 255
    return mask


def mask_area(mask_u8: np.ndarray) -> int:
    """
    Compute mask area in pixels (count of non-zero pixels).
    mask_u8: uint8 mask 0/255
    """
    return int(np.count_nonzero(mask_u8))


def bbox_from_mask(mask_u8: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute bounding box (x1,y1,x2,y2) from a binary mask (0/255).
    Returns (0,0,0,0) if empty.
    """
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2)


def compute_stats(mask_u8: np.ndarray) -> MaskStats:
    return MaskStats(area_px=mask_area(mask_u8), bbox_xyxy=bbox_from_mask(mask_u8))


def save_mask_png(mask_u8: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask_u8)

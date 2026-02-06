from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ultralytics import YOLO


@dataclass
class YoloConfig:
    weights_path: str = "yolov8n-seg.pt"
    conf: float = 0.25
    iou: float = 0.7
    device: Optional[str] = None  # e.g. "cpu" or "0"


class YoloPredictor:
    def __init__(self, cfg: YoloConfig) -> None:
        self.cfg = cfg
        self.model = YOLO(cfg.weights_path)

    def predict(self, image_path: str):
        """
        Returns Ultralytics Results list (usually length 1 for 1 image).
        """
        results = self.model.predict(
            source=image_path,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            device=self.cfg.device,
            verbose=False,
        )
        return results

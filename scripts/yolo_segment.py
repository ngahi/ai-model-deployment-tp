from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

from src.models.yolo_predictor import YoloPredictor, YoloConfig
from src.utils.mask_utils import to_uint8_mask, compute_stats, save_mask_png


def draw_bbox(img: np.ndarray, xyxy, label: str) -> None:
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input X-ray image")
    parser.add_argument("--weights", default="yolov8n-seg.pt", help="YOLO seg weights path")
    parser.add_argument("--outdir", default="outputs/yolo", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None, help='"cpu" or "0" (gpu id)')
    args = parser.parse_args()

    image_path = args.image
    outdir = Path(args.outdir)
    out_masks = outdir / "masks"
    out_overlays = outdir / "overlays"
    out_json = outdir / "json"

    out_masks.mkdir(parents=True, exist_ok=True)
    out_overlays.mkdir(parents=True, exist_ok=True)
    out_json.mkdir(parents=True, exist_ok=True)

    # 1) Load image for overlay rendering
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    overlay = img_bgr.copy()

    # 2) Run YOLO
    predictor = YoloPredictor(YoloConfig(
        weights_path=args.weights,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    ))
    results = predictor.predict(image_path)
    r0 = results[0]

    # 3) Extract detections
    names = r0.names  # class_id -> name

    detections: List[Dict[str, Any]] = []
    mask_index = 0

    # Boxes: r0.boxes.xyxy (N,4), r0.boxes.cls (N,), r0.boxes.conf (N,)
    # Masks: r0.masks.data (N,H,W) floats in [0,1] (if segmentation present)
    has_masks = r0.masks is not None and r0.masks.data is not None

    if r0.boxes is None or len(r0.boxes) == 0:
        print("No detections.")
    else:
        xyxy_all = r0.boxes.xyxy.cpu().numpy()
        cls_all = r0.boxes.cls.cpu().numpy().astype(int)
        conf_all = r0.boxes.conf.cpu().numpy()

        masks_all = None
        if has_masks:
            masks_all = r0.masks.data.cpu().numpy()  # (N,H,W)

        for i in range(len(xyxy_all)):
            class_id = int(cls_all[i])
            class_name = names.get(class_id, str(class_id))
            conf = float(conf_all[i])
            xyxy = xyxy_all[i].tolist()

            det: Dict[str, Any] = {
                "id": i,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox_xyxy": [int(x) for x in xyxy],
            }

            # If we have masks, save them + stats
            if masks_all is not None:
                mask_f = masks_all[i]  # (H,W) float
                mask_u8 = to_uint8_mask(mask_f, threshold=0.5)

                stats = compute_stats(mask_u8)
                det["mask_area_px"] = stats.area_px
                det["mask_bbox_xyxy"] = list(stats.bbox_xyxy)

                mask_file = out_masks / f"mask_{i:03d}_{class_name}.png"
                save_mask_png(mask_u8, mask_file)
                det["mask_path"] = str(mask_file)

                # Overlay mask (simple visualization)
                color_layer = np.zeros_like(overlay)
                color_layer[:, :, 1] = (mask_u8 > 0).astype(np.uint8) * 180  # green-ish
                overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.5, 0)

            # Draw bbox
            draw_bbox(overlay, det["bbox_xyxy"], f"{class_name} {conf:.2f}")

            detections.append(det)

    # 4) Save overlay image
    overlay_file = out_overlays / "overlay.png"
    cv2.imwrite(str(overlay_file), overlay)

    # 5) Save JSON summary
    out = {
        "image": image_path,
        "weights": args.weights,
        "detections_count": len(detections),
        "detections": detections,
        "overlay_path": str(overlay_file),
    }
    json_file = out_json / "results.json"
    json_file.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("✅ YOLO segmentation pipeline done")
    print(f"- Overlay: {overlay_file}")
    print(f"- JSON:    {json_file}")
    if len(detections) > 0 and any("mask_path" in d for d in detections):
        print(f"- Masks:   {out_masks}")


if __name__ == "__main__":
    main()

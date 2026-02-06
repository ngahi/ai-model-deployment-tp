from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
WEIGHTS_DIR = ARTIFACTS_DIR / "weights"

YOLO_WEIGHTS = "yolov8n-seg.pt"  # downloaded automatically by ultralytics
SAM_CHECKPOINT = WEIGHTS_DIR / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

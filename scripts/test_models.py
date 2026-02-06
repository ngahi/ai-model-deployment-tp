from pathlib import Path

from src.config.settings import SAM_CHECKPOINT, SAM_MODEL_TYPE
from src.models.sam_loader import load_sam
from src.models.yolo_loader import load_yolo_seg


def main():
    print("Loading YOLO segmentation model...")
    yolo = load_yolo_seg()
    print("YOLO loaded ✅")

    print("Loading SAM model...")
    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {SAM_CHECKPOINT}. "
            "Download it into artifacts/weights/"
        )
    sam = load_sam(SAM_MODEL_TYPE, str(SAM_CHECKPOINT))
    print("SAM loaded ✅")

    # YOLO inference smoke test
    img_path = Path("data/raw/test.jpg")
    if not img_path.exists():
        print("⚠️ Put a test image at data/raw/test.jpg to run YOLO inference.")
        print("Models are loaded correctly ✅")
        return

    print(f"Running YOLO inference on: {img_path}")
    results = yolo(str(img_path))
    print("Inference done ✅")
    print(f"Detections: {len(results)} result object(s)")


if __name__ == "__main__":
    main()

from ultralytics import YOLO


def load_yolo_seg(weights: str = "yolov8n-seg.pt") -> YOLO:
    """
    Load a YOLO segmentation model (Ultralytics).
    weights can be a local path or a model name like 'yolov8n-seg.pt'.
    """
    return YOLO(weights)

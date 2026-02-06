from segment_anything import sam_model_registry


def load_sam(model_type: str, checkpoint_path: str):
    """
    Load Segment Anything Model (SAM).
    model_type: 'vit_b', 'vit_l', 'vit_h'
    checkpoint_path: path to .pth file
    """
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.eval()
    return sam

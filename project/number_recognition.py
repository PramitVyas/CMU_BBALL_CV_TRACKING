import numpy as np
import cv2
from typing import List, Tuple, Dict


# ------------------------------------------------------------
# Cropping utilities
# ------------------------------------------------------------

def pad_and_clip_boxes(xyxy: np.ndarray, frame_w: int, frame_h: int,
                       px: int = 10, py: int = 10) -> np.ndarray:
    """Pad boxes by (px, py) and clip to frame boundaries."""
    boxes = xyxy.copy()
    boxes[:, 0] = np.clip(boxes[:, 0] - px, 0, frame_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1] - py, 0, frame_h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2] + px, 0, frame_w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3] + py, 0, frame_h - 1)
    return boxes


def crop_number_regions(frame: np.ndarray,
                        number_boxes_xyxy: np.ndarray,
                        resolution: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
    """Crop and resize number regions for OCR."""
    h, w, _ = frame.shape
    padded = pad_and_clip_boxes(number_boxes_xyxy, w, h)

    crops = []
    for x1, y1, x2, y2 in padded:
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, resolution, interpolation=cv2.INTER_LINEAR)
        crops.append(crop)

    return crops


# ------------------------------------------------------------
# IoS computation
# ------------------------------------------------------------

def compute_ios_matrix(number_boxes: np.ndarray,
                       player_masks: List[np.ndarray]) -> np.ndarray:
    """
    Compute IoS(number_box, player_mask) for all pairs.
    IoS = intersection_area / min(box_area, mask_area)
    """
    ios = np.zeros((len(number_boxes), len(player_masks)), dtype=float)

    for i, (x1, y1, x2, y2) in enumerate(number_boxes):
        box_area = (x2 - x1) * (y2 - y1)

        # create a binary mask for the number box
        box_mask = np.zeros_like(player_masks[0], dtype=np.uint8)
        box_mask[y1:y2, x1:x2] = 1

        for j, pmask in enumerate(player_masks):
            inter = np.sum((box_mask == 1) & (pmask == 1))
            mask_area = np.sum(pmask == 1)

            denom = min(box_area, mask_area)
            ios[i, j] = inter / denom if denom > 0 else 0.0

    return ios


# ------------------------------------------------------------
# Association logic
# ------------------------------------------------------------

def associate_numbers_with_players(ios_matrix: np.ndarray,
                                   threshold: float = 0.9) -> Dict[int, int]:
    """
    Return mapping: number_index -> player_index
    Only pairs with IoS > threshold are considered.
    One-to-one matching enforced greedily by IoS descending.
    """
    associations = {}
    used_numbers = set()
    used_players = set()

    # get all candidate pairs above threshold
    rows, cols = np.where(ios_matrix > threshold)
    pairs = list(zip(rows.tolist(), cols.tolist()))

    # sort by IoS descending
    pairs.sort(key=lambda rc: ios_matrix[rc[0], rc[1]], reverse=True)

    for n_idx, p_idx in pairs:
        if n_idx in used_numbers:
            continue
        if p_idx in used_players:
            continue

        associations[n_idx] = p_idx
        used_numbers.add(n_idx)
        used_players.add(p_idx)

    return associations


# ------------------------------------------------------------
# OCR wrapper
# ------------------------------------------------------------

def run_ocr_on_crops(crops: List[np.ndarray],
                     model,
                     prompt: str) -> List[str]:
    """Run OCR model on each crop and return digit strings."""
    outputs = []
    for crop in crops:
        pred = model.predict(crop, prompt)[0]
        outputs.append(pred)
    return outputs

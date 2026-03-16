import os
import cv2
import numpy as np
import logging
from ultralytics import YOLO
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# COURT FEATURE CLASS MAPPING
# -------------------------------------------------------------------
COURT_CLASS_MAPPING = {
    0: "Court",
    1: "Sideline",
    2: "Baseline",
    3: "FreeThrowLine",
    4: "ThreePointLine",
    5: "MidcourtLine",
    6: "CenterCircle",
    7: "Paint",
}


class SegmentationProcessor:
    """Process frames through a court-feature segmentation model."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        try:
            self.model = YOLO(self.model_path)
            logger.info(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    # -------------------------------------------------------------------
    # PROCESS FRAME
    # -------------------------------------------------------------------
    def process_frame(
        self, frame: np.ndarray, frame_id: int = None, output_dir: str = None
    ) -> Dict[str, List[Dict]]:

        results = self.model(frame)

        if len(results) == 0:
            logger.warning("No segmentation results produced")
            return {"segmentation_mask": None, "features": {}}

        result = results[0]
        features = {}

        if hasattr(result, 'masks') and result.masks is not None:
            masks = result.masks.data
            if len(masks) > 0:

                classes = result.boxes.cls.cpu().numpy()
                mask_by_class = {}

                for i, mask in enumerate(masks):
                    class_idx = int(classes[i])

                    if hasattr(self.model, 'names') and self.model.names:
                        class_name = self.model.names.get(class_idx, f"class_{class_idx}")
                    else:
                        class_name = COURT_CLASS_MAPPING.get(class_idx, f"class_{class_idx}")

                    if class_name not in mask_by_class:
                        mask_by_class[class_name] = np.zeros(
                            (frame.shape[0], frame.shape[1]), dtype=bool
                        )

                    numpy_mask = mask.cpu().numpy()
                    resized_mask = cv2.resize(
                        numpy_mask.astype(np.uint8),
                        (frame.shape[1], frame.shape[0])
                    )
                    mask_by_class[class_name] |= (resized_mask > 0)

                features = self._extract_features_from_segmentation(mask_by_class)

                colored_mask = np.zeros((*frame.shape[:2], 3), dtype=np.uint8)

                color_map = {
                    "Sideline": (0, 255, 0),
                    "Baseline": (255, 0, 0),
                    "FreeThrowLine": (0, 0, 255),
                    "ThreePointLine": (255, 255, 0),
                    "MidcourtLine": (255, 0, 255),
                    "CenterCircle": (0, 255, 255),
                    "Paint": (128, 0, 128)
                }

                for class_name, mask in mask_by_class.items():
                    if class_name in color_map:
                        colored_mask[mask] = color_map[class_name]

                if output_dir and frame_id is not None:
                    vis_img = self._save_debug_visualizations(
                        frame, mask_by_class, features, output_dir, frame_id
                    )

                return {
                    "segmentation_mask": colored_mask,
                    "features": features,
                    "raw_masks": mask_by_class,
                    "overlay_visualization": (
                        vis_img if output_dir and frame_id is not None else None
                    )
                }

        return {"segmentation_mask": None, "features": {}}

    # -------------------------------------------------------------------
    # EXTRACT FEATURES FROM MASKS
    # -------------------------------------------------------------------
    def _extract_features_from_segmentation(
        self, mask_by_class: Dict[str, np.ndarray]
    ) -> Dict[str, List[Dict]]:

        features = {}

        for class_name, mask in mask_by_class.items():

            mask_uint8 = (mask.astype(np.uint8) * 255)

            if np.sum(mask_uint8) == 0:
                continue

            logger.info(f"Processing mask for class: {class_name}")

            if class_name in [
                "Sideline",
                "Baseline",
                "FreeThrowLine",
                "ThreePointLine",
                "MidcourtLine",
                "Paint"
            ]:
                line_segments = self._extract_line_segments(mask_uint8, class_name)
                if line_segments:
                    features[class_name] = line_segments
                continue

            if class_name == "CenterCircle":
                circle_features = self._extract_circles(mask_uint8)
                if circle_features:
                    features[class_name] = circle_features
                continue

            if class_name == "ThreePointArc":
                arc_segments = self._extract_line_segments(mask_uint8, class_name)
                if arc_segments:
                    features[class_name] = arc_segments
                continue

            generic = self._extract_line_segments(mask_uint8, class_name)
            if generic:
                features[class_name] = generic

        return features

    # -------------------------------------------------------------------
    # EXTRACT LINE SEGMENTS
    # -------------------------------------------------------------------
    def _extract_line_segments(self, binary_mask, class_name=None):

        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = 200
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        features = []
        frame_width = binary_mask.shape[1]
        frame_center_x = frame_width / 2.0

        for contour in contours:
            contour_points = contour.reshape(-1, 2)
            if contour_points.size == 0:
                continue

            sorted_by_y = contour_points[np.argsort(contour_points[:, 1])]
            top_point = sorted_by_y[0]
            bottom_point = sorted_by_y[-1]

            feature = {
                "points": [
                    {"x": int(top_point[0]), "y": int(top_point[1])},
                    {"x": int(bottom_point[0]), "y": int(bottom_point[1])}
                ]
            }

            if class_name == "ThreePointLine":
                centroid_x = float(np.mean(contour_points[:, 0]))

                if centroid_x < frame_center_x:
                    apex = contour_points[np.argmax(contour_points[:, 0])]
                else:
                    apex = contour_points[np.argmin(contour_points[:, 0])]

                feature["apex"] = {
                    "x": int(apex[0]),
                    "y": int(apex[1])
                }

            features.append(feature)

        return features

    # -------------------------------------------------------------------
    # EXTRACT 3PT APEX (UTILITY)
    # -------------------------------------------------------------------
    def _extract_three_point_apex_from_contour(self, contour_points, frame_width):

        contour_points = np.asarray(contour_points)
        centroid_x = np.mean(contour_points[:, 0])
        frame_center_x = frame_width / 2.0

        if centroid_x < frame_center_x:
            apex = contour_points[np.argmax(contour_points[:, 0])]
        else:
            apex = contour_points[np.argmin(contour_points[:, 0])]

        return {"x": int(apex[0]), "y": int(apex[1])}

    # -------------------------------------------------------------------
    # EXTRACT CIRCLES
    # -------------------------------------------------------------------
    def _extract_circles(self, mask: np.ndarray, min_radius: int = 10) -> List[Dict]:

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        circles = []

        for contour in contours:

            if cv2.contourArea(contour) < np.pi * min_radius**2:
                continue

            if len(contour) < 5:
                continue

            try:
                (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
                radius = np.sqrt(major_axis * minor_axis) / 2

                if radius < min_radius:
                    continue

                circles.append({
                    "center": {"x": float(x), "y": float(y)},
                    "radius": float(radius)
                })

            except cv2.error:
                continue

        return circles

    # -------------------------------------------------------------------
    # SAVE DEBUG VISUALIZATIONS
    # -------------------------------------------------------------------
    def _save_debug_visualizations(
        self,
        frame: np.ndarray,
        mask_by_class: Dict[str, np.ndarray],
        features: Dict[str, List[Dict]],
        output_dir: str,
        frame_id: int
    ) -> np.ndarray:

        debug_dir = os.path.join(output_dir, "debug_segmentation")
        os.makedirs(debug_dir, exist_ok=True)

        vis_img = frame.copy()

        colors = {
            "Sideline":       (0, 255, 0,   120),
            "Baseline":       (255, 0, 0,   120),
            "FreeThrowLine":  (0, 0, 255,   120),
            "ThreePointLine": (255, 255, 0, 120),
            "MidcourtLine":   (255, 0, 255, 120),
            "Paint":          (128, 0, 128, 120),
            "CenterCircle":   (0, 255, 255, 120)
        }

        for class_name, mask in mask_by_class.items():
            if class_name not in colors:
                continue

            color = colors[class_name]
            overlay = np.zeros_like(frame, dtype=np.uint8)
            overlay[mask] = color[:3]

            alpha = color[3] / 255.0
            cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)

        for class_name, feats in features.items():
            for feat in feats:

                if "points" in feat:
                    pts = feat["points"]
                    if len(pts) == 2:
                        p1 = (int(pts[0]["x"]), int(pts[0]["y"]))
                        p2 = (int(pts[1]["x"]), int(pts[1]["y"]))
                        cv2.circle(vis_img, p1, 5, (0, 0, 0), -1)
                        cv2.circle(vis_img, p2, 5, (0, 0, 0), -1)
                        cv2.line(vis_img, p1, p2, (0, 0, 0), 2)

                if "center" in feat:
                    cx = int(feat["center"]["x"])
                    cy = int(feat["center"]["y"])
                    cv2.circle(vis_img, (cx, cy), 6, (0, 0, 0), -1)

                if "apex" in feat:
                    ax = int(feat["apex"]["x"])
                    ay = int(feat["apex"]["y"])
                    cv2.circle(vis_img, (ax, ay), 6, (0, 0, 255), -1)

        out_path = os.path.join(debug_dir, f"segmentation_{frame_id:04d}.jpg")
        cv2.imwrite(out_path, vis_img)

        return vis_img

import cv2
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque


class HomographyCalculator:
    """
    Calculates homography transformation between broadcast footage and the 2D court model.
    Maps detected court features in broadcast footage to their corresponding positions in the 2D court.
    """

    def __init__(self, court_coordinates_path: str, broadcast_width: int = 1948, broadcast_height: int = 1042):
        self.court_coordinates_path = court_coordinates_path
        self.broadcast_width = broadcast_width
        self.broadcast_height = broadcast_height

        # Default court dimensions (can be overridden by JSON)
        self.court_width = 2000
        self.court_height = 1255

        # Load court coordinates
        self.court_coordinates = self._load_court_coordinates()

        # Cache for homography matrices
        self.homography_cache = {}

        # Cache for destination points (per frame)
        self.destination_points_cache = {}

        # Base destination points from court coordinates
        self.base_destination_points = self._get_base_destination_points()

        # For homography smoothing
        self.recent_matrices = deque(maxlen=10)
        self.last_valid_matrix = None
        self.matrix_age = 0

        # Logger
        self.logger = logging.getLogger(__name__)

        # Max matrices for smoothing
        self.max_matrices = 5

    # -------------------------------------------------------------------------
    # LOAD COURT COORDINATES
    # -------------------------------------------------------------------------
    def _load_court_coordinates(self) -> Dict:
        if not os.path.exists(self.court_coordinates_path):
            raise FileNotFoundError(f"Court coordinates file not found: {self.court_coordinates_path}")

        with open(self.court_coordinates_path, 'r') as f:
            court_coordinates = json.load(f)

        if 'court_dimensions' in court_coordinates:
            self.court_width = court_coordinates['court_dimensions'].get('width', self.court_width)
            self.court_height = court_coordinates['court_dimensions'].get('height', self.court_height)

        return court_coordinates

    # -------------------------------------------------------------------------
    # BASE DESTINATION POINTS
    # -------------------------------------------------------------------------
    def _get_base_destination_points(self) -> Dict[str, Tuple[float, float]]:
        dest = {}
        court = self.court_coordinates

        # Court corners
        corners = court.get("destination_points", {})
        for name, pt in corners.items():
            dest[name] = (float(pt["x"]), float(pt["y"]))

        # Three‑point corners
        three_corners = court.get("additional_points", {}).get("three_point_corners", {})
        for name, pt in three_corners.items():
            dest[name] = (float(pt["x"]), float(pt["y"]))

        # Free‑throw lines
        ft_lines = court.get("additional_points", {}).get("free_throw_lines", {})
        for side, pts in ft_lines.items():
            for name, pt in pts.items():
                dest[f"free_throw_{side}_{name}"] = (float(pt["x"]), float(pt["y"]))

        # Three‑point arc
        arc = court.get("additional_points", {}).get("three_point_arc", {})
        for name, pt in arc.items():
            dest[name] = (float(pt["x"]), float(pt["y"]))

        # Center circle
        circle = court.get("additional_points", {}).get("center_circle", {})
        for name, pt in circle.items():
            dest[f"center_circle_{name}"] = (float(pt["x"]), float(pt["y"]))

        # Midcourt line
        midcourt = court.get("additional_points", {}).get("midcourt_line", {})
        for name, pt in midcourt.items():
            dest[f"midcourt_{name}"] = (float(pt["x"]), float(pt["y"]))

        # Paint corners
        paint = court.get("additional_points", {}).get("paint_corners", {})
        for name, pt in paint.items():
            dest[f"paint_{name}"] = (float(pt["x"]), float(pt["y"]))

        return dest

    # -------------------------------------------------------------------------
    # DESTINATION POINTS (FRAME-SPECIFIC)
    # -------------------------------------------------------------------------
    def get_destination_points(self, frame_idx: int = None) -> Dict[str, Tuple[float, float]]:
        if frame_idx is None:
            return self.base_destination_points.copy()

        if frame_idx in self.destination_points_cache:
            return self.destination_points_cache[frame_idx].copy()

        valid_indices = sorted(self.destination_points_cache.keys())
        if not valid_indices:
            return self.base_destination_points.copy()

        before_idx = None
        after_idx = None

        for idx in valid_indices:
            if idx <= frame_idx:
                before_idx = idx
            if idx > frame_idx:
                after_idx = idx
                break

        if before_idx is not None and after_idx is not None:
            before_points = self.destination_points_cache[before_idx]
            after_points = self.destination_points_cache[after_idx]

            frame_diff = after_idx - before_idx
            if frame_diff == 0:
                interpolated = before_points.copy()
            else:
                t = (frame_idx - before_idx) / frame_diff
                interpolated = {}
                for key in before_points:
                    if key in after_points:
                        x1, y1 = before_points[key]
                        x2, y2 = after_points[key]
                        x = (1 - t) * x1 + t * x2
                        y = (1 - t) * y1 + t * y2
                        interpolated[key] = (x, y)
                    else:
                        interpolated[key] = before_points[key]

                for key in after_points:
                    if key not in before_points:
                        interpolated[key] = after_points[key]

            self.destination_points_cache[frame_idx] = interpolated
            return interpolated.copy()

        if before_idx is not None:
            pts = self.destination_points_cache[before_idx].copy()
            self.destination_points_cache[frame_idx] = pts
            return pts

        if after_idx is not None:
            pts = self.destination_points_cache[after_idx].copy()
            self.destination_points_cache[frame_idx] = pts
            return pts

        return self.base_destination_points.copy()

    # -------------------------------------------------------------------------
    # SOURCE POINT EXTRACTION
    # -------------------------------------------------------------------------
    def extract_source_points(self, segmentation_features: Dict[str, List[Dict]]) -> Dict[str, Tuple[float, float]]:
        source_points: Dict[str, Tuple[float, float]] = {}

        self.logger.info(f"Raw segmentation features: {json.dumps(segmentation_features, indent=2)}")

        def _points_from_feature(feat: Dict) -> List[Tuple[float, float]]:
            pts = feat.get("points", [])
            if not pts:
                return []
            if isinstance(pts[0], dict):
                return [(float(p["x"]), float(p["y"])) for p in pts]
            return [(float(p[0]), float(p[1])) for p in pts]

        # Sidelines
        sidelines = []
        for feat in segmentation_features.get("Sideline", []):
            pts = _points_from_feature(feat)
            if len(pts) < 2:
                continue
            pts_sorted_y = sorted(pts, key=lambda p: p[1])
            top, bottom = pts_sorted_y[0], pts_sorted_y[-1]
            avg_x = sum(p[0] for p in pts) / len(pts)
            sidelines.append((avg_x, top, bottom))

        if len(sidelines) >= 2:
            sidelines.sort(key=lambda x: x[0])
            left_side = sidelines[0]
            right_side = sidelines[-1]

            source_points["left_top"] = left_side[1]
            source_points["left_bottom"] = left_side[2]
            source_points["right_top"] = right_side[1]
            source_points["right_bottom"] = right_side[2]

        # Free‑throw lines
        ft_lines = []
        for feat in segmentation_features.get("FreeThrowLine", []):
            pts = _points_from_feature(feat)
            if len(pts) < 2:
                continue
            pts_sorted_y = sorted(pts, key=lambda p: p[1])
            top, bottom = pts_sorted_y[0], pts_sorted_y[-1]
            avg_x = sum(p[0] for p in pts) / len(pts)
            ft_lines.append((avg_x, top, bottom))

        if len(ft_lines) >= 2:
            ft_lines.sort(key=lambda x: x[0])
            left_ft = ft_lines[0]
            right_ft = ft_lines[-1]

            source_points["free_throw_left_top"] = left_ft[1]
            source_points["free_throw_left_bottom"] = left_ft[2]
            source_points["free_throw_right_top"] = right_ft[1]
            source_points["free_throw_right_bottom"] = right_ft[2]

        # Paint corners
        paints = []
        for feat in segmentation_features.get("Paint", []):
            pts = _points_from_feature(feat)
            if len(pts) < 2:
                continue
            pts_sorted_y = sorted(pts, key=lambda p: p[1])
            top, bottom = pts_sorted_y[0], pts_sorted_y[-1]
            avg_x = sum(p[0] for p in pts) / len(pts)
            paints.append((avg_x, top, bottom))

        if len(paints) >= 2:
            paints.sort(key=lambda x: x[0])
            left_paint = paints[0]
            right_paint = paints[-1]

            source_points["paint_left_top"] = left_paint[1]
            source_points["paint_left_bottom"] = left_paint[2]
            source_points["paint_right_top"] = right_paint[1]
            source_points["paint_right_bottom"] = right_paint[2]

        # Three‑point corners + arc apex
        three_arcs = []
        for feat in segmentation_features.get("ThreePointLine", []):
            pts = _points_from_feature(feat)
            if len(pts) < 2:
                continue
            pts_sorted_y = sorted(pts, key=lambda p: p[1])
            top_corner, bottom_corner = pts_sorted_y[0], pts_sorted_y[-1]
            avg_x = sum(p[0]) / len(pts)
            apex = feat.get("apex", None)
            apex_pt = (float(apex["x"]), float(apex["y"])) if apex else None
            three_arcs.append((avg_x, top_corner, bottom_corner, apex_pt))

        if len(three_arcs) >= 2:
            three_arcs.sort(key=lambda x: x[0])
            left_arc = three_arcs[0]
            right_arc = three_arcs[-1]

            source_points["left_corner_top"] = left_arc[1]
            source_points["left_corner_bottom"] = left_arc[2]
            source_points["right_corner_top"] = right_arc[1]
            source_points["right_corner_bottom"] = right_arc[2]

            if left_arc[3] is not None:
                source_points["left_arc_top"] = left_arc[3]
            if right_arc[3] is not None:
                source_points["right_arc_top"] = right_arc[3]

        # Center circle
        center_circle_feats = segmentation_features.get("CenterCircle", [])
        center_circle_center = None
        center_circle_radius = None

        if center_circle_feats:
            cc = center_circle_feats[0]
            if "center" in cc:
                cx = float(cc["center"]["x"])
                cy = float(cc["center"]["y"])
                center_circle_center = (cx, cy)
                center_circle_radius = float(cc.get("radius", cc.get("equivalent_radius", 0.0)))

                if center_circle_radius > 0:
                    left_cc = (cx - center_circle_radius, cy)
                    right_cc = (cx + center_circle_radius, cy)
                    source_points["center_circle_left"] = left_cc
                    source_points["center_circle_center"] = center_circle_center
                    source_points["center_circle_right"] = right_cc

        # Midcourt line
        midcourt_feats = segmentation_features.get("MidcourtLine", [])
        if midcourt_feats:
            mc = midcourt_feats[0]
            pts = _points_from_feature(mc)
            if len(pts) >= 2:
                pts_sorted_y = sorted(pts, key=lambda p: p[1])
                top, bottom = pts_sorted_y[0], pts_sorted_y[-1]
                source_points["midcourt_top"] = top
                source_points["midcourt_bottom"] = bottom

                if center_circle_center is not None and center_circle_radius is not None and center_circle_radius > 0:
                    cx, cy = center_circle_center
                    top_cc = (cx, cy - center_circle_radius)
                    bottom_cc = (cx, cy + center_circle_radius)

                    source_points["midcourt_center_circle_center"] = center_circle_center
                    source_points["midcourt_center_circle_top"] = top_cc
                    source_points["midcourt_center_circle_bottom"] = bottom_cc

        self.logger.info(f"Found {len(source_points)} source points:")
        for name, point in source_points.items():
            self.logger.info(f"  {name}: {point}")

        return source_points

    # -------------------------------------------------------------------------
    # CALCULATE HOMOGRAPHY
    # -------------------------------------------------------------------------
    def calculate_homography(self, segmentation_features: Dict[str, List[Dict]], frame_idx: int = None) -> Optional[np.ndarray]:
        source_points = self.extract_source_points(segmentation_features)
        self.logger.info(f"Available source points: {list(source_points.keys())}")

        dest_points = self.get_destination_points(frame_idx)
        self.logger.info(f"Available destination points: {list(dest_points.keys())}")

        source_pts = []
        dest_pts = []

        for name, pt in source_points.items():
            if name in dest_points:
                source_pts.append(pt)
                dest_pts.append(dest_points[name])

        if len(source_pts) < 4:
            self.logger.warning(f"Insufficient matching points: found {len(source_pts)}, need 4")
            return None

        source_pts = np.array(source_pts, dtype=np.float32)
        dest_pts = np.array(dest_pts, dtype=np.float32)

        try:
            matrix, _ = cv2.findHomography(source_pts, dest_pts, 0)

            if matrix is None:
                self.logger.warning("Homography solve returned None")
                return None

            if not self.validate_homography(matrix):
                return None

            if frame_idx is not None:
                self.homography_cache[frame_idx] = matrix
                self.recent_matrices.append(matrix)
                if len(self.recent_matrices) > self.max_matrices:
                    self.recent_matrices.popleft()

            return matrix

        except Exception as e:
            self.logger.error(f"Error calculating homography: {str(e)}")
            return None

    # -------------------------------------------------------------------------
    # VALIDATE HOMOGRAPHY
    # -------------------------------------------------------------------------
    def validate_homography(self, h_matrix):
        if h_matrix is None:
            self.logger.error("Homography matrix is None")
            return False

        if not isinstance(h_matrix, np.ndarray) or h_matrix.shape != (3, 3):
            shape_info = (h_matrix.shape if isinstance(h_matrix, np.ndarray) else type(h_matrix))
            self.logger.error(f"Invalid matrix shape: {shape_info}")
            return False

        corners = np.float32([
            [0, 0],
            [self.broadcast_width, 0],
            [self.broadcast_width, self.broadcast_height],
            [0, self.broadcast_height]
        ])

        transformed_corners = cv2.perspectiveTransform(
            corners.reshape(-1, 1, 2),
            h_matrix
        ).reshape(-1, 2)

        court_w = self.court_width
        court_h = self.court_height

        margin = 0.5
        min_x = -court_w * margin
        max_x = court_w * (1 + margin)
        min_y = -court_h * margin
        max_y = court_h * (1 + margin)

        for i, (x, y) in enumerate(transformed_corners):
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                self.logger.error(f"Corner {i} is out of bounds")
                return False

        original_area = self.broadcast_width * self.broadcast_height
        transformed_area = cv2.contourArea(transformed_corners)
        area_ratio = transformed_area / original_area

        if not (0.02 <= area_ratio <= 10.0):
            self.logger.error(f"Invalid area ratio: {area_ratio:.3f}")
            return False

        original_diag = np.sqrt(self.broadcast_width**2 + self.broadcast_height**2)
        diag1 = np.linalg.norm(transformed_corners[1] - transformed_corners[3])
        diag2 = np.linalg.norm(transformed_corners[0] - transformed_corners[2])
        diag_ratio = max(diag1, diag2) / original_diag

        if not (0.1 <= diag_ratio <= 10.0):
            self.logger.error(f"Invalid diagonal ratio: {diag_ratio:.3f}")
            return False

        return True

    # -------------------------------------------------------------------------
    # AVERAGE MATRIX
    # -------------------------------------------------------------------------
    def get_average_matrix(self) -> Optional[np.ndarray]:
        if not self.recent_matrices:
            return None
        return np.mean([m for m in self.recent_matrices], axis=0)

    # -------------------------------------------------------------------------
    # APPLY HOMOGRAPHY
    # -------------------------------------------------------------------------
    def apply_homography(self, points: List[Tuple[float, float]], homography_matrix: np.ndarray) -> List[Tuple[float, float]]:
        """
        Apply homography transformation to points.

        Args:
            points: List of points to transform (x, y)
            homography_matrix: Homography matrix to apply

        Returns:
            List of transformed points
        """
        if homography_matrix is None or not points:
            return []

        # Convert points to homogeneous coordinates
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

        # Apply transformation
        transformed_pts = cv2.perspectiveTransform(pts, homography_matrix)

        # Convert back to list of tuples
        return [(pt[0][0], pt[0][1]) for pt in transformed_pts]

    # -------------------------------------------------------------------------
    # INTERPOLATE HOMOGRAPHY
    # -------------------------------------------------------------------------
    def interpolate_homography(self, matrix1: np.ndarray, matrix2: np.ndarray, t: float) -> np.ndarray:
        """
        Linearly interpolate between two homography matrices.

        Args:
            matrix1: First homography matrix
            matrix2: Second homography matrix
            t: Interpolation factor (0.0 to 1.0)

        Returns:
            Interpolated homography matrix
        """
        t = max(0.0, min(1.0, t))
        interpolated = (1 - t) * matrix1 + t * matrix2
        interpolated = interpolated / interpolated[2, 2]
        return interpolated

    # -------------------------------------------------------------------------
    # GET HOMOGRAPHY MATRIX (WITH INTERPOLATION)
    # -------------------------------------------------------------------------
    def get_homography_matrix(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get homography matrix for a specific frame, using interpolation if necessary.
        """
        if frame_idx in self.homography_cache:
            return self.homography_cache[frame_idx]

        valid_indices = sorted(self.homography_cache.keys())
        if not valid_indices:
            self.logger.warning("No valid homography matrices in cache for interpolation")
            return None

        before_idx = None
        after_idx = None

        before_indices = [idx for idx in valid_indices if idx <= frame_idx]
        if before_indices:
            before_idx = max(before_indices)

        after_indices = [idx for idx in valid_indices if idx > frame_idx]
        if after_indices:
            after_idx = min(after_indices)

        self.logger.info(f"Looking for homography for frame {frame_idx}")
        self.logger.info(f"Found before_idx={before_idx}, after_idx={after_idx}")

        if before_idx is not None and after_idx is not None:
            matrix1 = self.homography_cache[before_idx]
            matrix2 = self.homography_cache[after_idx]

            frame_diff = after_idx - before_idx
            if frame_diff == 0:
                interpolated = matrix1
            else:
                t = (frame_idx - before_idx) / frame_diff
                self.logger.info(f"TRUE INTERPOLATION: t={t:.3f} between frames {before_idx} and {after_idx}")
                interpolated = self.interpolate_homography(matrix1, matrix2, t)

            self.homography_cache[frame_idx] = interpolated
            self.logger.info(f"Stored interpolated homography matrix for frame {frame_idx}")
            return interpolated

        elif before_idx is not None:
            before_matrix = self.homography_cache[before_idx]
            self.homography_cache[frame_idx] = before_matrix
            return before_matrix

        elif after_idx is not None:
            after_matrix = self.homography_cache[after_idx]
            self.homography_cache[frame_idx] = after_matrix
            return after_matrix

        self.logger.warning(f"No suitable homography matrix found for frame {frame_idx}")
        return None

    # -------------------------------------------------------------------------
    # WARP FRAME
    # -------------------------------------------------------------------------
    def warp_frame(self, frame: np.ndarray, homography_matrix: np.ndarray, court_dims: Tuple[int, int]) -> np.ndarray:
        """
        Warp the broadcast frame to the court perspective using the homography matrix.
        """
        if homography_matrix is None:
            return None

        warped_frame = cv2.warpPerspective(frame, homography_matrix, court_dims)
        return warped_frame

    # -------------------------------------------------------------------------
    # DRAW SEGMENTATION LINES
    # -------------------------------------------------------------------------
    def draw_segmentation_lines(self, frame: np.ndarray, segmentation_features: Dict) -> np.ndarray:
        """
        Draw basketball segmentation features on a frame.
        """
        vis = frame.copy()

        def draw_poly(points, color, label=None):
            pts = [(int(p["x"]), int(p["y"])) for p in points]
            for i in range(len(pts) - 1):
                cv2.line(vis, pts[i], pts[i+1], color, 2)
            if label:
                cv2.putText(vis, label, (pts[0][0] + 5, pts[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        def draw_arc(points, color, label=None):
            pts = [(int(p["x"]), int(p["y"])) for p in points]
            cv2.polylines(vis, [np.array(pts)], False, color, 2)
            if label:
                mid = pts[len(pts)//2]
                cv2.putText(vis, label, (mid[0], mid[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Sidelines
        if "sideline_left" in segmentation_features:
            draw_poly(segmentation_features["sideline_left"], (255, 255, 255), "Sideline L")
        if "sideline_right" in segmentation_features:
            draw_poly(segmentation_features["sideline_right"], (255, 255, 255), "Sideline R")

        # Baselines
        if "baseline_left" in segmentation_features:
            draw_poly(segmentation_features["baseline_left"], (255, 255, 255), "Baseline L")
        if "baseline_right" in segmentation_features:
            draw_poly(segmentation_features["baseline_right"], (255, 255, 255), "Baseline R")

        # Free‑throw lines
        if "free_throw_line_left" in segmentation_features:
            draw_poly(segmentation_features["free_throw_line_left"], (0, 255, 255), "FT Line L")
        if "free_throw_line_right" in segmentation_features:
            draw_poly(segmentation_features["free_throw_line_right"], (0, 255, 255), "FT Line R")

        # Three‑point arcs
        if "three_point_arc_left" in segmentation_features:
            draw_arc(segmentation_features["three_point_arc_left"], (0, 255, 0), "3PT Arc L")
        if "three_point_arc_right" in segmentation_features:
            draw_arc(segmentation_features["three_point_arc_right"], (0, 255, 0), "3PT Arc R")

        # Center circle
        if "center_circle" in segmentation_features:
            circle = segmentation_features["center_circle"]
            if "center" in circle and "radius" in circle:
                cx = int(circle["center"]["x"])
                cy = int(circle["center"]["y"])
                r = int(circle["radius"])
                cv2.circle(vis, (cx, cy), r, (0, 128, 255), 2)
                cv2.circle(vis, (cx, cy), 4, (0, 128, 255), -1)
                cv2.putText(vis, "Center Circle", (cx + 10, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 128, 255), 2)

        # Midcourt line
        if "midcourt_line" in segmentation_features:
            draw_poly(segmentation_features["midcourt_line"], (0, 0, 255), "Midcourt")

        # Paint rectangles
        def draw_paint(side):
            keys = [
                f"paint_{side}_top",
                f"paint_{side}_corner_left",
                f"paint_{side}_bottom",
                f"paint_{side}_corner_right",
                f"paint_{side}_top"
            ]
            if all(k in segmentation_features for k in keys[:-1]):
                pts = [segmentation_features[k] for k in keys]
                pts = [{"x": p["x"], "y": p["y"]} for p in pts]
                draw_poly(pts, (255, 0, 255), f"Paint {side.capitalize()}")

        draw_paint("left")
        draw_paint("right")

        return vis

    # -------------------------------------------------------------------------
    # DRAW VISUALIZATION
    # -------------------------------------------------------------------------
    def draw_visualization(self, frame: np.ndarray, segmentation_features: Dict, homography: Optional[np.ndarray] = None) -> np.ndarray:
        vis_frame = frame.copy()
        vis_frame = self.draw_segmentation_lines(vis_frame, segmentation_features)

        if homography is None:
            cv2.putText(vis_frame, "Homography calculation failed",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2)
        return vis_frame

    # -------------------------------------------------------------------------
    # PROJECT POINT TO COURT
    # -------------------------------------------------------------------------
    def project_point_to_court(self, point: Tuple[float, float], homography_matrix: np.ndarray) -> Optional[Dict[str, float]]:
        try:
            if homography_matrix is None:
                return None

            pts = np.array([[point]], dtype=np.float32)
            projected = cv2.perspectiveTransform(pts, homography_matrix)[0][0]

            x, y = float(projected[0]), float(projected[1])
            return {"x": x, "y": y}

        except Exception as e:
            self.logger.error(f"Error projecting point: {str(e)}")
            return None

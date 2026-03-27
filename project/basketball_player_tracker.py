import cv2
import numpy as np
import torch
import json
import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

from ultralytics import YOLO
from court_segmentation_processor import SegmentationProcessor
from basketball_homography_calculator import HomographyCalculator
from player_detector import PlayerDetector


# ============================================================
# SORT TRACKER IMPLEMENTATION
# ============================================================

from collections import deque

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.bbox = bbox
        self.hits = 1
        self.no_losses = 0
        self.trace = deque(maxlen=10)

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.no_losses = 0
        self.trace.append(bbox)

    def predict(self):
        self.no_losses += 1
        return self.bbox


class Sort:
    def __init__(self, max_age=5, min_hits=1):
        self.max_age = max_age
        self.min_hits = min_hits
        self.trackers = []

    def update(self, detections):
        updated_trackers = []

        for det in detections:
            matched = False
            for trk in self.trackers:
                iou = self._iou(det, trk.bbox)
                if iou > 0.3:
                    trk.update(det)
                    updated_trackers.append(trk)
                    matched = True
                    break

            if not matched:
                new_trk = KalmanBoxTracker(det)
                updated_trackers.append(new_trk)

        self.trackers = [
            t for t in updated_trackers if t.no_losses <= self.max_age
        ]

        return self.trackers

    @staticmethod
    def _iou(bb1, bb2):
        x1 = max(bb1[0], bb2[0])
        y1 = max(bb1[1], bb2[1])
        x2 = min(bb1[2], bb2[2])
        y2 = min(bb1[3], bb2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        return inter / float(area1 + area2 - inter)


# ============================================================
# JSON ENCODER
# ============================================================

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# ============================================================
# PLAYER TRACKER
# ============================================================

class PlayerTracker:
    def __init__(
        self,
        detection_model_path: str,
        output_dir: str = None,
        segmentation_model_path: Optional[str] = None,
        court_coordinates_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.output_dir = output_dir

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.player_detector = PlayerDetector(
            model_path=detection_model_path,
            device=device,
            output_dir=output_dir
        )

        self.segmentation_processor = None
        if segmentation_model_path:
            self.segmentation_processor = SegmentationProcessor(segmentation_model_path, device)

        self.homography_calculator = None
        if court_coordinates_path:
            self.homography_calculator = HomographyCalculator(court_coordinates_path)

        self.tracking_data = {}
        self.logger = logging.getLogger(__name__)

        self.sort_tracker = Sort()

        import numpy as np
        self.H_left = np.load("homography/H_left.npy")
        self.H_right = np.load("homography/H_right.npy")

    # ============================================================
    # METRICS
    # ============================================================

    def calculate_player_metrics(self, current_player, frame_id, prev_frame_data):
        metrics = {
            "speed": 0.0,
            "acceleration": 0.0,
            "implied_orientation": 0.0
        }

        if "court_position" not in current_player:
            return metrics

        current_pos = current_player["court_position"]

        if not prev_frame_data or "players" not in prev_frame_data:
            return metrics

        prev_players = prev_frame_data["players"]
        prev_player = next(
            (p for p in prev_players if p.get("id") == current_player.get("id")),
            None
        )

        if not prev_player or "court_position" not in prev_player:
            return metrics

        prev_pos = prev_player["court_position"]

        dx = current_pos[0] - prev_pos[0]
        dy = current_pos[1] - prev_pos[1]
        speed = float(np.sqrt(dx**2 + dy**2))
        metrics["speed"] = speed

        prev_speed = prev_player.get("speed", 0.0)
        metrics["acceleration"] = speed - prev_speed

        if dx != 0 or dy != 0:
            metrics["implied_orientation"] = float(np.degrees(np.arctan2(dy, dx)))

        return metrics

    def compute_blend_factor(self, frame_id):
        return 0.5

    # ============================================================
    # PROCESS FRAME
    # ============================================================

    def process_frame(self, frame: np.ndarray, frame_id: int, debug_mode: bool = False) -> Dict:
        frame_height, frame_width = frame.shape[:2]

        frame_data = {
            "frame_id": frame_id,
            "timestamp": datetime.now().isoformat(),
            "players": []
        }

        prev_frame_data = self.tracking_data.get(frame_id - 1)

        # ---------------------------------------------------------
        # Segmentation + Homography
        # ---------------------------------------------------------
        if self.segmentation_processor:
            seg_result = self.segmentation_processor.process_frame(
                frame, frame_id, self.output_dir
            )
            frame_data["segmentation_features"] = seg_result

            if self.homography_calculator:
                try:
                    t = self.compute_blend_factor(frame_id)
                    H = self.homography_calculator.interpolate_homography(
                        self.H_left,
                        self.H_right,
                        t)

                    if H is not None:
                        frame_data["homography_matrix"] = H.tolist()
                        frame_data["homography_success"] = True
                    else:
                        H = self.homography_calculator.get_homography_matrix(frame_id)
                        if H is not None:
                            frame_data["homography_matrix"] = H.tolist()
                            frame_data["homography_success"] = True
                            frame_data["homography_interpolated"] = True
                        else:
                            frame_data["homography_success"] = False

                except Exception as e:
                    self.logger.error(f"Error calculating homography: {e}")
                    frame_data["homography_success"] = False

        # ---------------------------------------------------------
        # YOLO Detection
        # ---------------------------------------------------------
        detections = []
        if self.player_detector:
            detections = self.player_detector.process_frame(frame, frame_id)

        det_boxes = [det["bbox"] for det in detections]

        # ---------------------------------------------------------
        # SORT Tracking
        # ---------------------------------------------------------
        trackers = self.sort_tracker.update(det_boxes)

        id_map = {}
        for trk in trackers:
            id_map[tuple(trk.bbox)] = trk.id

        # ---------------------------------------------------------
        # Build Player Entries
        # ---------------------------------------------------------
        for det in detections:
            bbox = det["bbox"]
            pid = id_map.get(tuple(bbox), None)

            player_data = {
                "id": pid,
                "bbox": bbox,
                "confidence": det["confidence"]
            }

            x1, y1, x2, y2 = bbox
            bottom_center = ((x1 + x2) / 2, y2)

            if frame_data.get("homography_success", False):
                try:
                    H = np.array(frame_data["homography_matrix"])
                    court_pos = self.homography_calculator.project_point_to_court(
                        bottom_center, H
                    )
                    if court_pos:
                        player_data["court_position"] = court_pos

                        metrics = self.calculate_player_metrics(
                            current_player=player_data,
                            frame_id=frame_id,
                            prev_frame_data=prev_frame_data
                        )
                        player_data.update(metrics)

                except Exception as e:
                    self.logger.error(f"Error projecting point: {e}")

            frame_data["players"].append(player_data)

        self.tracking_data[frame_id] = frame_data
        return frame_data

    def process_frame_with_roboflow(
            self,
            frame: np.ndarray,
            frame_id: int,
            mask_by_class: Dict[str, np.ndarray],
            players: List[Dict],
            debug_mode: bool = False
        ) -> Dict:

        frame_height, frame_width = frame.shape[:2]

        frame_data = {
            "frame_id": frame_id,
            "timestamp": datetime.now().isoformat(),
            "players": []
        }

        prev_frame_data = self.tracking_data.get(frame_id - 1)

        # ---------------------------------------------------------
        # Homography (interpolated MVP version)
        # ---------------------------------------------------------
        H = None
        if self.homography_calculator:
            try:
                t = self.compute_blend_factor(frame_id)
                H = self.homography_calculator.interpolate_homography(
                    self.H_left,
                    self.H_right,
                    t
                )

                if H is not None:
                    frame_data["homography_matrix"] = H.tolist()
                    frame_data["homography_success"] = True
                else:
                    frame_data["homography_success"] = False

            except Exception as e:
                self.logger.error(f"Error computing interpolated homography: {e}")
                frame_data["homography_success"] = False

        # ---------------------------------------------------------
        # Build Player Entries from Roboflow detections
        # ---------------------------------------------------------
        for det in players:
            bbox = det["bbox"]
            pid = det.get("id")

            player_data = {
                "id": pid,
                "bbox": bbox,
                "confidence": det.get("confidence", 1.0)
            }

            x1, y1, x2, y2 = bbox
            bottom_center = ((x1 + x2) / 2, y2)

            # -----------------------------------------------------
            # Project to court coordinates using interpolated H
            # -----------------------------------------------------
            if frame_data.get("homography_success", False):
                try:
                    pt = np.array([[[bottom_center[0], bottom_center[1]]]], dtype=np.float32)
                    court_pt = cv2.perspectiveTransform(pt, H)[0][0]
                    court_pos = [float(court_pt[0]), float(court_pt[1])]
                    player_data["court_position"] = court_pos

                    # -------------------------------------------------
                    # Metrics (speed, acceleration, orientation)
                    # -------------------------------------------------
                    metrics = self.calculate_player_metrics(
                        current_player=player_data,
                        frame_id=frame_id,
                        prev_frame_data=prev_frame_data
                    )
                    player_data.update(metrics)

                except Exception as e:
                    self.logger.error(f"Error projecting point: {e}")

            frame_data["players"].append(player_data)

        # Save for next frame
        self.tracking_data[frame_id] = frame_data
        return frame_data

    # ============================================================
    # VISUALIZATION
    # ============================================================

    def visualize_frame(self, frame, frame_data, debug_mode=False):
        visualizations = {}
        broadcast_vis = frame.copy()

        homography_success = frame_data.get("homography_success", False)
        segmentation_features = frame_data.get("segmentation_features", {}).get("features", {})
        players = frame_data.get("players", [])
        frame_idx = frame_data.get("frame_id", 0)

        if self.homography_calculator and segmentation_features:
            if hasattr(self.homography_calculator, "draw_visualization"):
                H = None
                if homography_success and "homography_matrix" in frame_data:
                    H = frame_data["homography_matrix"]

                broadcast_vis = self.homography_calculator.draw_visualization(
                    broadcast_vis,
                    segmentation_features,
                    H
                )
            else:
                broadcast_vis = self.homography_calculator.draw_segmentation_lines(
                    broadcast_vis,
                    segmentation_features
                )

                status_text = "Homography: Success" if homography_success else "Homography: Failed"
                cv2.putText(
                    broadcast_vis,
                    status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

        for player in players:
            if "bbox" in player:
                x1, y1, x2, y2 = player["bbox"]
                bottom_center = (int((x1 + x2) / 2), int(y2))

                cv2.circle(broadcast_vis, bottom_center, 4, (255, 0, 0), -1)

                if "id" in player:
                    cv2.putText(
                        broadcast_vis,
                        str(player["id"]),
                        (bottom_center[0] + 10, bottom_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )

        cv2.putText(
            broadcast_vis,
            f"Frame: {frame_idx}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        visualizations["broadcast"] = broadcast_vis
        return visualizations

    # ============================================================
    # SAVE TRACKING DATA
    # ============================================================

    def save_tracking_data(self, output_path: str) -> str:
        serializable_data = {}

        print(f"Preparing tracking data for saving ({len(self.tracking_data)} frames)...")

        for frame_id, frame_data in self.tracking_data.items():
            try:
                players_out = []
                for p in frame_data.get("players", []):
                    player_entry = {
                        "id": p.get("id"),
                        "bbox": p.get("bbox"),
                        "court_position": p.get("court_position"),
                        "speed": p.get("speed"),
                        "acceleration": p.get("acceleration"),
                        "implied_orientation": p.get("implied_orientation")
                    }
                    players_out.append(player_entry)

                serializable_frame = {
                    "frame_id": frame_data.get("frame_id"),
                    "timestamp": frame_data.get("timestamp"),
                    "players": players_out,
                    "homography_success": frame_data.get("homography_success", False)
                }

                if frame_data.get("homography_success", False):
                    serializable_frame["homography_matrix"] = frame_data.get("homography_matrix")

                if "segmentation_features" in frame_data:
                    seg = frame_data["segmentation_features"].get("features", {})
                    serializable_frame["segmentation_features"] = {
                        "features": seg
                    }

                serializable_data[str(frame_id)] = serializable_frame

            except Exception as e:
                print(f"Error processing frame {frame_id}: {str(e)}")
                continue

        try:
            print(f"Saving {len(serializable_data)} frames to {output_path}...")

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(serializable_data, f, indent=2, cls=NumpyEncoder)

            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"Successfully saved tracking data to {output_path} ({file_size/1024:.1f} KB)")
            else:
                print(f"Failed to save tracking data to {output_path}")

            return output_path

        except Exception as e:
            print(f"Error saving tracking data: {str(e)}")
            return None

    def visualize_court(self, frame_data):
        """
        MVP: Draw players on a 2D court map using court_position only.
        """

        # Load your static 2D court image (must match your JSON coordinate system)
        court_img = cv2.imread("basketball-court.png")  # adjust if needed
        if court_img is None:
            raise FileNotFoundError("basketball-court.png not found")

        vis = court_img.copy()

        # Draw each player
        for p in frame_data["players"]:
            if "court_position" not in p:
                continue

            cx, cy = p["court_position"]
            cx = int(cx)
            cy = int(cy)

            # Player dot
            cv2.circle(vis, (cx, cy), 8, (0, 0, 255), -1)

        # Frame label
        cv2.putText(
            vis,
            f"Frame {frame_data['frame_id']}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

        return vis

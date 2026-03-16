import cv2
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging
import time

from segmentation_processor import SegmentationProcessor
from player_detector import PlayerDetector
from homography_calculator import HomographyCalculator
from ultralytics import YOLO


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that can handle numpy arrays and other non-serializable types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(NumpyEncoder, self).default(obj)


class PlayerTracker:
    """
    Main module that integrates all components to track players in basketball broadcast footage.
    """

    def __init__(
        self,
        detection_model_path: str,
        output_dir: str = None,
        segmentation_model_path: Optional[str] = None,
        court_coordinates_path: Optional[str] = None,
        device: str = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
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

    # -------------------------------------------------------------------------
    # METRICS
    # -------------------------------------------------------------------------
    def calculate_player_metrics(
        self,
        current_player: Dict,
        frame_id: int,
        prev_frame_data: Optional[Dict] = None
    ) -> Dict:

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

    # -------------------------------------------------------------------------
    # PROCESS SINGLE FRAME
    # -------------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, frame_id: int, debug_mode: bool = False) -> Dict:

        frame_height, frame_width = frame.shape[:2]

        frame_data = {
            "frame_id": frame_id,
            "timestamp": datetime.now().isoformat(),
            "players": []
        }

        prev_frame_data = self.tracking_data.get(frame_id - 1)

        # ---------------------------------------------------------
        # Step 1: Court segmentation + homography
        # ---------------------------------------------------------
        if self.segmentation_processor:
            seg_result = self.segmentation_processor.process_frame(
                frame, frame_id, self.output_dir
            )
            frame_data["segmentation_features"] = seg_result

            if self.homography_calculator:
                try:
                    H = self.homography_calculator.calculate_homography(
                        seg_result["features"],
                        frame_id
                    )

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
        # Step 2: Player detection
        # ---------------------------------------------------------
        detections = []
        if self.player_detector:
            detections = self.player_detector.process_frame(frame, frame_id)

        # ---------------------------------------------------------
        # Step 3: Process each detected player
        # ---------------------------------------------------------
        for det in detections:
            player_data = {
                "bbox": det["bbox"],
                "confidence": det["confidence"]
            }

            x1, y1, x2, y2 = det["bbox"]
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

    # -------------------------------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------------------------------
    def visualize_frame(
        self,
        frame: np.ndarray,
        frame_data: Dict,
        debug_mode: bool = False
    ) -> Dict[str, np.ndarray]:

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

    # -------------------------------------------------------------------------
    # SAVE TRACKING DATA
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # TRACK PLAYERS (ALTERNATE ENTRY POINT)
    # -------------------------------------------------------------------------
    def track_players(self, frame, frame_idx, timestamp):

        frame_data = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "players": [],
            "homography_success": False
        }

        if self.segmentation_processor:
            seg_result = self.segmentation_processor.process_frame(frame)
            frame_data["segmentation_features"] = seg_result.get("features", {})

            if self.homography_calculator:
                try:
                    H = self.homography_calculator.calculate_homography(
                        seg_result.get("features", {}),
                        frame_idx
                    )

                    if H is not None:
                        frame_data["homography_matrix"] = H.tolist()
                        frame_data["homography_success"] = True
                    else:
                        H = self.homography_calculator.get_homography_matrix(frame_idx)
                        if H is not None:
                            frame_data["homography_matrix"] = H.tolist()
                            frame_data["homography_success"] = True
                            frame_data["homography_interpolated"] = True

                except Exception as e:
                    self.logger.error(f"Error calculating homography: {e}")
                    frame_data["homography_success"] = False

        detections = self.player_detector.process_frame(frame, frame_idx)

        players_out = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bottom_center = ((x1 + x2) / 2, y2)

            player_entry = {
                "bbox": det["bbox"],
                "confidence": det["confidence"]
            }

            if frame_data.get("homography_success", False):
                try:
                    H = np.array(frame_data["homography_matrix"])
                    court_pos = self.homography_calculator.project_point_to_court(
                        bottom_center, H
                    )
                    if court_pos:
                        player_entry["court_position"] = court_pos

                except Exception as e:
                    self.logger.error(f"Error projecting player position: {e}")

            players_out.append(player_entry)

        frame_data["players"] = players_out
        return frame_data

    # -------------------------------------------------------------------------
    # PROCESS VIDEO CLIP
    # -------------------------------------------------------------------------
    def process_video_clip(
        self,
        video_path,
        start_second=0,
        num_seconds=5,
        frame_step=1,
        max_frames=None
    ):
        results = {}

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Could not open video: {video_path}")
                return {"error": f"Could not open video: {video_path}"}

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            start_frame = int(start_second * fps)
            end_frame = (
                int((start_second + num_seconds) * fps)
                if num_seconds > 0 else total_frames
            )

            if max_frames is not None and (end_frame - start_frame) > max_frames:
                end_frame = start_frame + max_frames

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_idx = start_frame
            output_frames = []

            vis_dir = os.path.join(self.output_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)

            while frame_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_idx - start_frame) % frame_step == 0:
                    timestamp = frame_idx / fps

                    frame_data = self.process_frame(frame, frame_idx)

                    if len(self.tracking_data) == 0:
                        players = frame_data["players"]
                        players = sorted(players, key=lambda p: p["bbox"][0])
                        for idx, p in enumerate(players):
                            p["id"] = idx + 1
                    else:
                        prev_players = self.tracking_data[frame_idx - frame_step]["players"]
                        for idx, p in enumerate(frame_data["players"]):
                            p["id"] = prev_players[idx]["id"]

                    self.tracking_data[frame_idx] = frame_data
                    output_frames.append(frame_data)

                    vis = self.visualize_frame(frame, frame_data)
                    vis_path = os.path.join(vis_dir, f"frame_{frame_idx}.jpg")
                    cv2.imwrite(vis_path, vis["broadcast"])

                    self.logger.info(f"Processed frame {frame_idx}")

                frame_idx += 1

                if max_frames is not None and (frame_idx - start_frame) >= max_frames * frame_step:
                    break

            output_file = os.path.join(
                self.output_dir,
                f"tracking_data_{int(time.time())}.json"
            )
            self.save_tracking_data(output_file)

            results = {
                "video_info": {
                    "path": video_path,
                    "fps": fps,
                    "total_frames": total_frames,
                    "processed_frames": len(output_frames)
                },
                "frames": output_frames,
                "tracking_data_file": output_file
            }

            self.logger.info(f"Tracking data saved to {output_file}")

        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            results = {"error": str(e)}

        finally:
            if "cap" in locals() and cap is not None:
                cap.release()

        return results

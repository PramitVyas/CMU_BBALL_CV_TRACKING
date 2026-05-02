import os
import cv2
import argparse
import logging
from datetime import datetime

from basketball_player_tracker import PlayerTracker
from pathlib import Path
from paths import SOURCE_VIDEO_DIRECTORY

import numpy as np


# ---------------------------------------------------------
# PATH RESOLUTION
# ---------------------------------------------------------
def get_video_path(filename: str) -> Path:
    return SOURCE_VIDEO_DIRECTORY / filename


# ---------------------------------------------------------
# COURT POLYGON (BROADCAST SPACE)
# ---------------------------------------------------------
# 5 clicked points, ordered via convex hull:
COURT_POLY = np.array([
    [157, 488],     # left mid
    [880, 127],     # top mid
    [1627, 248],    # right mid
    [1498, 1042],   # bottom right
    [1060, 1043],   # bottom left
], dtype=np.float32)

def build_court_polygon(frame_width: int, frame_height: int) -> np.ndarray:
    # Already in pixel coordinates — no scaling needed
    return COURT_POLY

def is_on_court(bx: float, by: float, court_polygon: np.ndarray) -> bool:
    """
    Returns True if the bottom-center point (bx, by) lies inside the court polygon.
    """
    return cv2.pointPolygonTest(court_polygon, (float(bx), float(by)), False) >= 0


# ---------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Basketball Tracking Pipeline with Court Mask")

    parser.add_argument("--video", type=str, required=True,
                        help="Filename of input video located in the source directory")

    parser.add_argument("--detection_model", type=str, required=True,
                        help="Path to YOLO detection model")

    parser.add_argument("--segmentation_model", type=str, default=None,
                        help="Path to court segmentation model (optional)")

    parser.add_argument("--court_coordinates", type=str, default=None,
                        help="Path to court coordinates JSON")

    parser.add_argument("--output_dir", type=str, default="../outputs_courtmask",
                        help="Directory to save results")

    parser.add_argument("--max_frames", type=int, default=None,
                        help="Optional limit on number of frames to process")

    parser.add_argument("--frame_step", type=int, default=1,
                        help="Process every Nth frame")

    return parser.parse_args()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main_courtmask")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Resolve video path
    # ---------------------------------------------------------
    video_path = get_video_path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ---------------------------------------------------------
    # Open video (to get size for polygon)
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, sample_frame = cap.read()
    if not ret:
        logger.error("Could not read first frame for size.")
        return

    frame_height, frame_width = sample_frame.shape[:2]
    court_polygon = build_court_polygon(frame_width, frame_height)

    # Reset to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    logger.info(f"Video loaded: {video_path}")
    logger.info(f"FPS: {fps}, Total frames: {total_frames}")
    logger.info(f"Frame size: {frame_width}x{frame_height}")

    # ---------------------------------------------------------
    # Initialize tracker
    # ---------------------------------------------------------
    tracker = PlayerTracker(
        detection_model_path=args.detection_model,
        segmentation_model_path=args.segmentation_model,
        court_coordinates_path=args.court_coordinates,
        output_dir=args.output_dir
    )

    frame_idx = 0
    processed = 0

    vis_dir = os.path.join(args.output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.frame_step == 0:

            # ---------------------------------------------------------
            # YOLO + SORT + Homography pipeline
            # ---------------------------------------------------------
            frame_data = tracker.process_frame(
                frame=frame,
                frame_id=frame_idx
            )

            # ---------------------------------------------------------
            # Filter players by court polygon (broadcast space)
            # ---------------------------------------------------------
            filtered_players = []
            for p in frame_data.get("players", []):
                bbox = p.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = bbox
                bx = (x1 + x2) / 2.0
                by = y2  # bottom-center

                if is_on_court(bx, by, court_polygon):
                    filtered_players.append(p)

            frame_data["players"] = filtered_players
            tracker.tracking_data[frame_idx] = frame_data

            # ---------------------------------------------------------
            # Visualization
            # ---------------------------------------------------------
            court_vis = tracker.visualize_court(frame_data)
            cv2.imwrite(os.path.join(args.output_dir, f"court_{frame_idx:06d}.jpg"), court_vis)

            vis = tracker.visualize_frame(frame, frame_data)
            vis_path = os.path.join(vis_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(vis_path, vis["broadcast"])

            logger.info(f"Processed frame {frame_idx} (players kept: {len(filtered_players)})")
            processed += 1

            if args.max_frames is not None and processed >= args.max_frames:
                break

        frame_idx += 1

    cap.release()

    # ---------------------------------------------------------
    # Save tracking JSON
    # ---------------------------------------------------------
    output_json = os.path.join(args.output_dir, "tracking_data.json")
    tracker.save_tracking_data(output_json)

    logger.info(f"Tracking complete. JSON saved to {output_json}")


if __name__ == "__main__":
    main()

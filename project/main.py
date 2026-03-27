import os
import cv2
import argparse
import logging
from datetime import datetime

from basketball_player_tracker import PlayerTracker

from pathlib import Path
from paths import SOURCE_VIDEO_DIRECTORY

import numpy as np
from roboflow import Roboflow

# ---------------------------------------------------------
# ROBOFLOW SETUP
# ---------------------------------------------------------
ROBOFLOW_API_KEY = "BVHtT2IBWuPcY16z0o4V"  # <-- replace with your key

rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project("basketball-player-detection-3-ycjdo")
model = project.version(4).model


# ---------------------------------------------------------
# ROBOFLOW HELPERS
# ---------------------------------------------------------
def run_roboflow_inference(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model.predict(rgb, confidence=40).json()
    return result


def extract_mask_by_class(result, frame_shape):
    """
    Convert Roboflow masks into the exact mask_by_class format
    expected by SegmentationProcessor._extract_features_from_segmentation.
    """
    h, w = frame_shape[:2]
    mask_by_class = {}

    for pred in result.get("predictions", []):
        if "mask" not in pred:
            continue

        class_name = pred.get("class")
        if class_name is None:
            continue

        # Roboflow mask is a 2D list of 0/1
        numpy_mask = np.array(pred["mask"], dtype=np.uint8)

        # Resize to match frame
        resized_mask = cv2.resize(numpy_mask, (w, h)) > 0

        if class_name not in mask_by_class:
            mask_by_class[class_name] = resized_mask
        else:
            mask_by_class[class_name] |= resized_mask

    return mask_by_class


def extract_players(result):
    players = []
    for pred in result.get("predictions", []):
        if pred.get("class") != "player":
            continue

        x = pred["x"]
        y = pred["y"]
        w = pred["width"]
        h = pred["height"]

        # Convert center-width-height → x1,y1,x2,y2
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        players.append({
            "id": pred.get("track_id"),
            "bbox": [x1, y1, x2, y2],
            "confidence": pred.get("confidence", 1.0),
        })
    return players


# ---------------------------------------------------------
# PATH RESOLUTION
# ---------------------------------------------------------
def get_video_path(filename: str) -> Path:
    return SOURCE_VIDEO_DIRECTORY / filename


# ---------------------------------------------------------
# ARGUMENT PARSER
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Basketball Tracking Pipeline")

    parser.add_argument("--video", type=str, required=True,
                        help="Filename of input video located in the source directory")

    parser.add_argument("--detection_model", type=str, required=True,
                        help="Path to player detection model (unused for Roboflow but required by constructor)")

    parser.add_argument("--segmentation_model", type=str, default=None,
                        help="Path to court segmentation model (unused for Roboflow)")

    parser.add_argument("--court_coordinates", type=str, default=None,
                        help="Path to court coordinates JSON")

    parser.add_argument("--output_dir", type=str, default="outputs",
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
    logger = logging.getLogger("main")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Resolve video path
    # ---------------------------------------------------------
    video_path = get_video_path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ---------------------------------------------------------
    # Initialize tracker
    # ---------------------------------------------------------
    tracker = PlayerTracker(
        detection_model_path=args.detection_model,
        segmentation_model_path=args.segmentation_model,
        court_coordinates_path=args.court_coordinates,
        output_dir=args.output_dir
    )

    # ---------------------------------------------------------
    # Open video
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video loaded: {video_path}")
    logger.info(f"FPS: {fps}, Total frames: {total_frames}")

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
            # Roboflow inference
            # ---------------------------------------------------------
            result = run_roboflow_inference(frame)
            mask_by_class = extract_mask_by_class(result, frame.shape)
            players = extract_players(result)

            # ---------------------------------------------------------
            # Process frame using Roboflow outputs
            # ---------------------------------------------------------
            frame_data = tracker.process_frame_with_roboflow(
                frame=frame,
                frame_id=frame_idx,
                mask_by_class=mask_by_class,
                players=players
            )

            # ---------------------------------------------------------
            # Visualization
            # ---------------------------------------------------------
            court_vis = tracker.visualize_court(frame_data)
            cv2.imwrite(os.path.join(args.output_dir, f"court_{frame_idx:06d}.jpg"), court_vis)

            vis = tracker.visualize_frame(frame, frame_data)
            vis_path = os.path.join(vis_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(vis_path, vis["broadcast"])

            logger.info(f"Processed frame {frame_idx}")
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

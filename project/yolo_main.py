import os
import cv2
import argparse
import logging
from datetime import datetime

from basketball_player_tracker import PlayerTracker
from pathlib import Path
from paths import SOURCE_VIDEO_DIRECTORY


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
                        help="Path to YOLO detection model")

    parser.add_argument("--segmentation_model", type=str, default=None,
                        help="Path to court segmentation model (optional)")

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
    # Initialize tracker (YOLO + SORT + Homography)
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
            # YOLO + SORT + Homography pipeline
            # ---------------------------------------------------------
            frame_data = tracker.process_frame(
                frame=frame,
                frame_id=frame_idx
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

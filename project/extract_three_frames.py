import cv2
import os

VIDEO_PATH = "../source/CMU_VS_EMORY_MBB.mp4"   # change this to your CMU video
OUTPUT_LEFT = "static_left.jpg"
OUTPUT_CENTER = "static_center.jpg"
OUTPUT_RIGHT = "static_right.jpg"

# pick three frame indices that roughly span the broadcast pan
CENTER_FRAME = 116640   # 32:24 at 60 FPS
RIGHT_FRAME = 105420    # 29:17 at 60 FPS
LEFT_FRAME = 107520     # 29:52 at 60 FPS

def extract_frame(video_path, frame_idx, output_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = min(frame_idx, total - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame {target}")

    cv2.imwrite(output_path, frame)
    print(f"Saved frame {target} → {output_path}")

def main():
    extract_frame(VIDEO_PATH, LEFT_FRAME, OUTPUT_LEFT)
    extract_frame(VIDEO_PATH, CENTER_FRAME, OUTPUT_CENTER)
    extract_frame(VIDEO_PATH, RIGHT_FRAME, OUTPUT_RIGHT)

if __name__ == "__main__":
    main()

import json
import cv2
import os

OUT_DIR = "../outputs"
VIDEO_FRAME_DIR = os.path.join(OUT_DIR, "vis")   # where main.py saved broadcast frames

# Load tracking data
json_path = os.path.join(OUT_DIR, "tracking_data.json")
with open(json_path, "r") as f:
    data = json.load(f)

# Sort frames numerically
frames = sorted(data.keys(), key=lambda x: int(x))

# ---------------------------------------------------------
# 1. Load one sample frame to get size
# ---------------------------------------------------------
sample_frame_path = os.path.join(VIDEO_FRAME_DIR, f"frame_{int(frames[0]):06d}.jpg")
sample = cv2.imread(sample_frame_path)
if sample is None:
    raise FileNotFoundError(f"Could not load sample frame: {sample_frame_path}")

h, w = sample.shape[:2]
print("Broadcast frame size:", w, h)

# ---------------------------------------------------------
# 2. Create animation video
# ---------------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("animation.mp4", fourcc, 30, (w, h))

# ---------------------------------------------------------
# 3. Loop through frames and draw dots using bbox bottom-center
# ---------------------------------------------------------
for f in frames:
    frame_path = os.path.join(VIDEO_FRAME_DIR, f"frame_{int(f):06d}.jpg")
    frame_img = cv2.imread(frame_path)

    if frame_img is None:
        print("Missing frame:", frame_path)
        continue

    for p in data[f]["players"]:
        bbox = p.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        x = int((x1 + x2) / 2)
        y = int(y2)

        cv2.circle(frame_img, (x, y), 6, (0, 0, 255), -1)

        pid = p.get("id")
        if pid is not None:
            cv2.putText(
                frame_img,
                str(pid),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1
            )

    out.write(frame_img)

out.release()
print("Saved animation.mp4")

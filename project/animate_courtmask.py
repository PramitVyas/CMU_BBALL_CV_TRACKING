import json
import cv2
import os
import numpy as np

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
OUT_DIR = "../outputs_courtmask"
COURT_IMG = "basketball-court.png"
OUTPUT_VIDEO = "animation_courtmask.mp4"
FPS = 30


# ---------------------------------------------------------
# LOAD TRACKING DATA
# ---------------------------------------------------------
json_path = os.path.join(OUT_DIR, "tracking_data.json")
with open(json_path, "r") as f:
    data = json.load(f)

frames = sorted(data.keys(), key=lambda x: int(x))


# ---------------------------------------------------------
# LOAD COURT BACKGROUND
# ---------------------------------------------------------
court_bg = cv2.imread(COURT_IMG)
if court_bg is None:
    raise FileNotFoundError(f"Could not load {COURT_IMG}")

h, w = court_bg.shape[:2]


# ---------------------------------------------------------
# COLLECT ALL BOTTOM-CENTER POINTS FOR NORMALIZATION
# ---------------------------------------------------------
all_x = []
all_y = []

for f in frames:
    for p in data[f]["players"]:
        x1, y1, x2, y2 = p["bbox"]
        bx = (x1 + x2) / 2
        by = y2
        all_x.append(bx)
        all_y.append(by)

min_x, max_x = min(all_x), max(all_x)
min_y, max_y = min(all_y), max(all_y)


def scale_to_court(x, y):
    """Normalize broadcast pixel coords → court PNG coords."""
    sx = int((x - min_x) / (max_x - min_x) * w)
    sy = int((y - min_y) / (max_y - min_y) * h)
    return sx, sy


# ---------------------------------------------------------
# CREATE VIDEO WRITER
# ---------------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))


# ---------------------------------------------------------
# DRAW DOTS ON COURT BACKGROUND
# ---------------------------------------------------------
for f in frames:
    frame_img = court_bg.copy()

    for p in data[f]["players"]:
        x1, y1, x2, y2 = p["bbox"]
        bx = (x1 + x2) / 2
        by = y2

        x, y = scale_to_court(bx, by)

        # Dot
        cv2.circle(frame_img, (x, y), 6, (0, 0, 255), -1)

        # ID label
        pid = p.get("id")
        if pid is not None:
            cv2.putText(frame_img, str(pid), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    out.write(frame_img)

out.release()
print(f"Saved {OUTPUT_VIDEO}")

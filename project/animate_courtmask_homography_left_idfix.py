import json
import cv2
import os
import numpy as np
from scipy.spatial.distance import cdist

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
OUT_DIR = "../outputs_courtmask"
COURT_IMG = "basketball-court.png"
HOMOGRAPHY_PATH = "homography/H_left.npy"
OUTPUT_VIDEO = "test_animation_courtmask_homography_left_idfix.mp4"
FPS = 30

MAX_IDS = 20  # force IDs to be 1–20


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
# LOAD HOMOGRAPHY
# ---------------------------------------------------------
H = np.load(HOMOGRAPHY_PATH)

def apply_homography(x, y, H):
    pt = np.array([x, y, 1.0], dtype=np.float32)
    dst = H @ pt
    if dst[2] == 0:
        return None
    X = dst[0] / dst[2]
    Y = dst[1] / dst[2]
    return int(round(X)), int(round(Y))


# ---------------------------------------------------------
# CREATE VIDEO WRITER
# ---------------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))


# ---------------------------------------------------------
# ID CONTINUITY STATE
# ---------------------------------------------------------
prev_positions = {}   # id -> (x,y)


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
for f in frames:
    frame_img = court_bg.copy()

    # Step 1: compute homography positions for all detections
    detections = []  # list of (orig_id, X, Y)
    for p in data[f]["players"]:
        oid = p.get("id", None)

        x1, y1, x2, y2 = p["bbox"]
        bx = (x1 + x2) / 2.0
        by = y2

        mapped = apply_homography(bx, by, H)
        if mapped is None:
            continue

        X, Y = mapped
        if 0 <= X < w and 0 <= Y < h:
            detections.append((oid, X, Y))

    # ---------------------------------------------------------
    # FIX: remove None IDs before any comparisons
    # ---------------------------------------------------------
    clean = [(oid, X, Y) for (oid, X, Y) in detections if oid is not None]

    # Separate into good (1–20) and bad (>20)
    good = [(oid, X, Y) for (oid, X, Y) in clean if oid <= MAX_IDS]
    bad  = [(oid, X, Y) for (oid, X, Y) in clean if oid > MAX_IDS]

    new_positions = {}
    assigned_ids = set()

    # Step 2: keep good IDs as-is
    for oid, X, Y in good:
        new_positions[oid] = (X, Y)
        assigned_ids.add(oid)

    # Step 3: reassign bad IDs using nearest-neighbor to previous frame
    if prev_positions:
        prev_ids_list = np.array(list(prev_positions.keys()))
        prev_xy = np.array([prev_positions[i] for i in prev_ids_list])

        for oid, X, Y in bad:
            cur_xy = np.array([[X, Y]])
            dists = cdist(cur_xy, prev_xy)[0]

            sorted_idx = np.argsort(dists)
            best_id = None

            for idx in sorted_idx:
                candidate = int(prev_ids_list[idx])
                if candidate not in assigned_ids:
                    best_id = candidate
                    break

            if best_id is None:
                continue

            new_positions[best_id] = (X, Y)
            assigned_ids.add(best_id)

    else:
        # First frame: assign sequentially
        next_id = 1
        for oid, X, Y in clean:
            if next_id > MAX_IDS:
                break
            new_positions[next_id] = (X, Y)
            assigned_ids.add(next_id)
            next_id += 1

    # Step 4: draw
    for sid, (X, Y) in new_positions.items():
        cv2.circle(frame_img, (X, Y), 6, (0, 0, 255), -1)
        cv2.putText(frame_img, str(sid), (X + 8, Y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    out.write(frame_img)

    # Step 5: update state
    prev_positions = new_positions.copy()

out.release()
print(f"Saved {OUTPUT_VIDEO}")

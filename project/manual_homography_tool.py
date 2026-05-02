import cv2
import numpy as np
import sys
import json

points = []

def click_event(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: {(x, y)}")
        cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("image", param)

def load_dst(mode):
    # Load your court JSON
    with open("court_coordinates.json", "r") as f:
        court = json.load(f)

    dst_pts = court["destination_points"]
    mid = court["additional_points"]["midcourt_line"]

    left_top     = (dst_pts["left_top"]["x"], dst_pts["left_top"]["y"])
    left_bottom  = (dst_pts["left_bottom"]["x"], dst_pts["left_bottom"]["y"])
    right_top    = (dst_pts["right_top"]["x"], dst_pts["right_top"]["y"])
    right_bottom = (dst_pts["right_bottom"]["x"], dst_pts["right_bottom"]["y"])

    mid_top      = (mid["top"]["x"], mid["top"]["y"])
    mid_bottom   = (mid["bottom"]["x"], mid["bottom"]["y"])

    if mode == "left":
        # P1L, P2L, P3L, P4L
        return np.array([
            left_bottom,   # P1L
            left_top,      # P2L
            mid_top,       # P3L
            mid_bottom     # P4L (anchored)
        ], dtype=np.float32)

    elif mode == "center":
        # P1C, P2C, P3C, P4C
        return np.array([
            left_top,      # P1C
            right_top,     # P2C
            mid_top,       # P3C
            mid_bottom     # P4C (anchored)
        ], dtype=np.float32)

    elif mode == "right":
        # P1R, P2R, P3R, P4R
        return np.array([
            right_bottom,  # P1R
            right_top,     # P2R
            mid_top,       # P3R
            mid_bottom     # P4R (anchored)
        ], dtype=np.float32)

    else:
        raise ValueError("Mode must be one of: left, center, right")

def main():
    if len(sys.argv) != 4:
        print("Usage: python manual_homography_tool.py <image_path> <mode> <output_npy>")
        print("Modes: left, center, right")
        return

    image_path = sys.argv[1]
    mode = sys.argv[2]
    output_path = sys.argv[3]

    dst = load_dst(mode)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    clone = img.copy()
    cv2.imshow("image", clone)
    cv2.setMouseCallback("image", click_event, clone)

    print(f"Mode: {mode}")
    print("Click 4 points in order:")
    if mode == "left":
        print("1. Left baseline × near sideline")
        print("2. Left baseline × far sideline")
        print("3. Midcourt × far sideline")
        print("4. Midcourt (bottommost visible point)")
    elif mode == "center":
        print("1. Left baseline × far sideline")
        print("2. Right baseline × far sideline")
        print("3. Midcourt × far sideline")
        print("4. Midcourt (bottommost visible point)")
    elif mode == "right":
        print("1. Right baseline × near sideline")
        print("2. Right baseline × far sideline")
        print("3. Midcourt × far sideline")
        print("4. Midcourt (bottommost visible point)")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 4:
        raise RuntimeError("You must click exactly 4 points.")

    src = np.array(points, dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    np.save(output_path, H)
    print(f"Saved homography to {output_path}")

if __name__ == "__main__":
    main()

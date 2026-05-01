import cv2
import numpy as np

# Center frames
src_pts = np.array([
    [353, 344],  # left top
    [225, 488],  # left bottom
    [1558, 344],  # right top
    [1685, 488],  # right bottom
], dtype=np.float32)

dst_pts = np.array([
    [500, 487],   # left top
    [500, 770],   # left bottom
    [1500, 487],  # right top
    [1500, 770],  # right bottom
], dtype=np.float32)

H_center, _ = cv2.findHomography(src_pts, dst_pts, method=0)

print("Center")
print(H_center)

# Left frames
src_pts_left = np.array([
    [717, 279],  # left top
    [562, 363],  # left bottom
    [1001, 344],  # right top
    [867, 460],  # right bottom
], dtype=np.float32)

dst_pts_left = np.array([
    [142, 487],   # left top
    [142, 770],   # left bottom
    [500, 487],  # right top
    [500, 770],  # right bottom
], dtype=np.float32)

H_left, _ = cv2.findHomography(src_pts_left, dst_pts_left, method=0)

print("Left")
print(H_left)

# Right frames
src_pts_right = np.array([
    [914, 345],  # left top
    [1049, 460],  # left bottom
    [1193, 269],  # right top
    [1351, 355],  # right bottom
], dtype=np.float32)

dst_pts_right = np.array([
    [1500, 487],   # left top
    [1500, 770],   # left bottom
    [1858, 487],  # right top
    [1858, 770],  # right bottom
], dtype=np.float32)

H_right, _ = cv2.findHomography(src_pts_right, dst_pts_right, method=0)

print("Right")
print(H_right)

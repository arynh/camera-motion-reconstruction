import cv2
import numpy as np
import matplotlib.pyplot as plt
from features import find_point_correspondences
from essential import calculate_essential_matrix
from reconstruction import estimate_camera_pose

iphone_12_mini_k = np.array(
    [
        [1.3270361480372305e3, 0, 9.6142138175295599e2],
        [0, 1.3325859916429802e3, 5.3765189758345116e2],
        [0, 0, 1],
    ]
)

camera_position = np.array([0, 0, 0, 1], dtype=np.float64)
print(camera_position)

for frame_index in range(0, 57, 5):
    frame_one = cv2.imread(
        "samples/tiles0/f{num}.jpg".format(num=str(frame_index + 1).zfill(4))
    )
    frame_two = cv2.imread(
        "samples/tiles0/f{num}.jpg".format(num=str(frame_index + 6).zfill(4))
    )

    matched_points = find_point_correspondences(frame_one, frame_two)

    height, width, _ = frame_one.shape
    m = height if height > width else width

    E = calculate_essential_matrix(matched_points, iphone_12_mini_k, m, 0.001)
    # print(E / E.max())

    transformation = estimate_camera_pose(E, matched_points, iphone_12_mini_k)
    # print(transformation)
    camera_position = transformation @ camera_position
    print(camera_position)

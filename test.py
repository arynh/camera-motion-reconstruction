import cv2
import numpy as np
import matplotlib.pyplot as plt
from features import find_point_correspondences
from essential import calculate_essential_matrix
from reconstruction import estimate_camera_pose


def vector_to_string(v):
    return "({:.3f}, {:.3f}, {:.3f})".format(v[0] / v[3], v[1] / v[3], v[2] / v[3])


iphone_12_mini_k = np.array(
    [
        [1.3270361480372305e3, 0, 9.6142138175295599e2],
        [0, 1.3325859916429802e3, 5.3765189758345116e2],
        [0, 0, 1],
    ]
)

camera_position = np.array([0, 0, 0, 1], dtype=np.float64)
camera_direction = np.array([0, 0, 1, 1], dtype=np.float64)
print(
    f"frame {0}/{57}; "
    + f"position: {vector_to_string(camera_position)}, "
    + f"direction: {vector_to_string(camera_direction)}"
)

for frame_index in range(0, 57 - 5, 5):
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

    rotation, translation = estimate_camera_pose(E, matched_points, iphone_12_mini_k)
    camera_position = translation @ camera_position
    camera_direction = rotation @ camera_direction
    print(
        f"frame {frame_index + 1}/{57}; "
        + f"position: {vector_to_string(camera_position)}, "
        + f"direction: {vector_to_string(camera_direction)}"
    )

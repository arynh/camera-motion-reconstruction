"""Reconstruct 3D transformation of cameras.
"""
import numpy as np
from scipy.linalg import svd


def estimate_camera_pose(essential_matrix, feature_points, intrinsic_matrix):
    """
    Given an essential matrix (from frame1 to frame2) and a set of
    feature points, estimate the camera pose for camera2.

    :param essential_matrix: Essential matrix between two camera views
    :type essential_matrix: np.ndarray (3 x 3)
    :param feature_points: Feature points in image coordinates
    :type feature_points: np.ndarray (2 x number of features x 2)
    :param intrinsic_matrix: Intrinsic camera matrix
    :type intrinsic_matrix: np.ndarray (3 x 3)
    :returns: Tuple of two transformation matrices, the first
        representing rotation and second representing translation.
    :rtype: Tuple [np.ndarray (4 x 4), np.ndarray (4 x 4)]
    """
    n_points = feature_points.shape[1]

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    U, _, Vt = svd(essential_matrix)

    potential_rotation_matrices = [U @ W @ Vt, U @ W.T @ Vt]

    potential_translation_vectors = [U[-1, :], -U[-1, :]]

    # Assume camera1 is fixed at [I|0]
    camera1_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    rotation_transform = np.eye(4)
    translation_transform = np.eye(4)

    # Track the number of points that appear in front of
    # both cameras. The combination of R and t that results
    # in the most is what we will use for camera2 pose.
    best_count = 0

    for R in potential_rotation_matrices:
        for t in potential_translation_vectors:

            front_count = 0
            for i in range(n_points):
                x = np.append(feature_points[1, i, :], 1)  # [x2, y2, 1]
                x = x @ intrinsic_matrix
                x_world = R.T @ x - R.T @ t

                # ensure world point is in front of camera1
                if x_world[2] < 0:
                    continue

                C2_center_world = -R.T @ t
                C2_view_world = R[-1, :].T

                # check if world point is in front of camera2
                if np.dot(x_world - C2_center_world, C2_view_world) > 0:
                    front_count += 1

            if front_count > best_count:
                rotation_transform[:3, :3] = R
                translation_transform[3, :3] = t
                best_count = front_count

    return rotation_transform, translation_transform

"""
Basic testing file to make sure things don't break.
"""
import pytest
import cv2
import numpy as np

from features import find_point_correspondences
from essential import calculate_essential_matrix


def test_basic():
    """
    Run the tracking functionality on one pair of frames. This is more to
    verify that the code runs than to seek a correct answer.
    """
    frame_one = cv2.imread("assets/test_frame_one.jpg")
    frame_two = cv2.imread("assets/test_frame_two.jpg")

    matched_points = find_point_correspondences(frame_one, frame_one)
    assert matched_points.shape[1] > 100
    print(f"Found {matched_points.shape[1]} point correspondences.")

    intrinsic_matrix = np.array(
        [
            [1.3270361480372305e3, 0, 9.6142138175295599e2],
            [0, 1.3325859916429802e3, 5.3765189758345116e2],
            [0, 0, 1],
        ]
    )

    height, width, _ = frame_one.shape
    largest_dimension = width if width > height else height

    reprojection_error_tolerance = 0.01

    essential_matrix = calculate_essential_matrix(
        matched_points,
        intrinsic_matrix,
        largest_dimension,
        reprojection_error_tolerance,
    )
    assert essential_matrix.shape == (3, 3)
    assert not np.isclose(essential_matrix, np.zeros(essential_matrix.shape)).all()
    print("Estimated essential matrix:")
    print(essential_matrix)

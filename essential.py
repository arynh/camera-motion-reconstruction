"""
Calculate the Essential Matrix between two frames using the instrinic matrix
and a list of the corresponding points between the two frames.
"""
import numpy as np
from scipy.optimize import least_squares


def score_fundamental_matrix(
    F: np.ndarray,
    frame_one_points: np.ndarray,
    frame_two_points: np.ndarray,
    tolerance: float,
) -> int:
    """
    Evaluate the guess for the fundamental matrix by counting how many
    reprojected points fall within a tolerance for reprojection error.

    :param F: guess for fundamental matrix
    :type F: np.ndarray
    :param frame_one_points: points from the first view
    :type frame_one_points: np.ndarray
    :param frame_two_points: points from the second view
    :type frame_two_points: np.ndarray
    :param tolerance: allowed error for a point to be counted as an inlier
    :type tolerance: float
    :return: the count of inliers
    :rtype: int
    """
    point_count = frame_one_points.shape[0]
    assert point_count == frame_two_points.shape[0]

    # points should be in homogenous coordinates
    frame_one_points = np.column_stack((frame_one_points, np.ones(point_count)))
    frame_two_points = np.column_stack((frame_two_points, np.ones(point_count)))

    errors = np.zeros(point_count, dtype=np.float64)
    for point_index in range(point_count):
        # x'.T F x = 0 in a perfect reprojection
        errors[point_index] = np.abs(
            frame_two_points[point_index].T @ F @ frame_one_points[point_index]
        )

    return np.sum(errors > tolerance)


def calculate_essential_matrix(
    corresponding_points: np.ndarray,
    intrinsic_matrix: np.ndarray,
    largest_dimension: int,
    reprojection_error_tolerance: float,
    ransac_iterations: int = 1000,
) -> np.ndarray:
    """
    Estimate the fundemental matrix using RANSAC and then use the intrinsic
    matrix to calculate the essential matrix.

    :param corresponding_points: Corresponding set of features in two frames
    :type corresponding_points: np.ndarray (2 x number of features x 2)
    :param k: Camera's intrinsic matrix, assumed to be the same in both frames
    :type k: np.ndarray (3x3)
    :param reprojection_error_tolerance: maximum allowed error in reprojection
    :type reprojection_error_tolerance: float
    :param ransac_iterations: number of iterations to run ransac, default 1000
    :type ransac_iterations: int
    :param m: value to scale points by m = max(im_width,im_height)
    :type m: int
    :return E: essential matrix
    :rtype: np.ndarray (3 x 3)
    """
    corresponding_points /= largest_dimension  # normalize points
    N = corresponding_points.shape[1]  # get the number of features

    max_inliers = -1
    optimal_f = np.random.rand(3, 3) - 0.5
    A = np.zeros((N, 9), dtype=np.float64)
    for _ in range(ransac_iterations):
        # choose 8 random points for 8-points algorithm
        subset = np.random.choice(N, 8, replace=False)
        frame_one_points = corresponding_points[0, subset]
        frame_two_points = corresponding_points[1, subset]

        for i in range(8):
            x = frame_one_points[i, 0]
            y = frame_one_points[i, 1]
            xp = frame_two_points[i, 0]
            yp = frame_two_points[i, 1]
            A[i] = [
                x * xp,
                x * yp,
                x,
                y * xp,
                y * yp,
                y,
                xp,
                yp,
                1,
            ]  # populate the A matrix

        F = least_squares(lambda f: A @ f, optimal_f.reshape(9), method="lm").x
        F = F.reshape((3, 3))

        inlier_count = score_fundamental_matrix(
            F,
            corresponding_points[0],
            corresponding_points[1],
            reprojection_error_tolerance,
        )
        if inlier_count > max_inliers:
            max_inliers = inlier_count
            optimal_f = F

    F = optimal_f
    F *= largest_dimension  # un-normalize m
    E = intrinsic_matrix.T @ F @ intrinsic_matrix  # compute the essential matrix
    return E

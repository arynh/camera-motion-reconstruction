"""
Calculate the Essential Matrix between two frames using the instrinic matrix
and a list of the corresponding points between the two frames.
"""
import numpy as np


def calculate_essential_matrix(corresponding_points, K):
    """
    Calculate the fundemental matrix and then use the intrinsic matrix to
    calculate the essential matrix.
    :param corresponding_points: Corresponding set of features in two frames
    :type corresponding_points: np.ndarray (2 x number of features x 2)
    :param K: Second frame
    :type K: np.ndarray (3x3)
    :return: [description]
    :rtype: np.ndarray (2 x number of features x 2)
    """

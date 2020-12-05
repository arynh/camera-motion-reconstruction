"""
Find point correspondences between two images we need at least eight
correspondences to be able to continue computing camera motion, so if this is
not acheived, fail. This is implemented using a SIFT feature detector.
"""
import numpy as np
import cv2


def find_point_correspondences(
    frame_one: np.ndarray, frame_two: np.ndarray
) -> np.ndarray:
    """
    Use SIFT features to find corresponding set of features in two frames of a
    video.

    :param frame_one: First frame from monocular color video.
    :type frame_one: np.ndarray (height x width x 3)
    :param frame_two: Second frame
    :type frame_two: np.ndarray (height x width x 3)
    :return: [description]
    :rtype: np.ndarray (2 x number of features x 2)
    """
    pass

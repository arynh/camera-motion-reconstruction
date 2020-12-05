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

    Source: Feature detection and matching is based on example code from
    OpenCV documentation.

    :param frame_one: First frame from monocular color video.
    :type frame_one: np.ndarray (height x width x 3)
    :param frame_two: Second frame
    :type frame_two: np.ndarray (height x width x 3)
    :return: [description]
    :rtype: np.ndarray (2 x number of features x 2)
    """
    frame_one_grey = cv2.cvtColor(frame_one, cv2.COLOR_BGR2GRAY)
    frame_two_grey = cv2.cvtColor(frame_two, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    frame_one_points, frame_one_descriptors = sift.detectAndCompute(
        frame_one_grey, None
    )
    frame_two_points, frame_two_descriptors = sift.detectAndCompute(
        frame_two_grey, None
    )

    # BFMatcher with default params
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(frame_one_descriptors, frame_two_descriptors, k=2)

    # Apply ratio test
    good = []
    ratio = 0.7
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    matches = good

    tracked_features = np.zeros((2, len(matches), 2), dtype=np.float64)

    for match_index, match in enumerate(matches):
        tracked_features[0, match_index] = frame_one_points[match.queryIdx].pt
        tracked_features[1, match_index] = frame_two_points[match.trainIdx].pt

    return tracked_features

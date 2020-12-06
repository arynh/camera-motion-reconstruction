"""Helper functions to plot visualizations.

Includes the following visualization tools: 
    * Plot camera motion on 3D plot
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import cv2

# TODO: maybe also add the plotting SIFT points?

def draw_epipolar_lines(img1, img2, pts1, pts2, F):
    """
    Compute and draw epipolar lines onto two images, associated
    by fundamntal matrix F.

    :param img1: First image
    :type img1: np.ndarray (H x W x 3)
    :param img2: Second image
    :type img2: np.ndarray (H x W x 3)
    :param pts1: Set of points in the first image
    :type pts1: np.ndarray (N x 3)
    :param pts2: Set of points in the second image matching pts1
    :type pts2: np.ndarray (N x 3)
    :param F: Fundamental matrix between img1 and img2
    :type F: np.ndarray (3 x 3)
    """
    pass

def plot_camera_motion(ax_3d, c_pos, c_view):
    """
    Plot points for camera position and arrows for view vectors.

    :param ax_3d: 3D axis to plot onto
    :type ax_3d: matplotlib.axes._subplots.Axes3DSubplot
    :param c_pos: array of N camera positions in world coordinates
    :type c_pos: np.ndarray (N x 3)
    :param c_view: array of N view vectors
    :type c_view: np.ndarray (N x 3)
    """
    assert c_pos.shape == c_view.shape
    ax_3d.quiver(c_pos[:,0], 
              c_pos[:,1], 
              c_pos[:,2],
              c_view[:,0],
              c_view[:,0],
              c_view[:,0],
              length=0.5,
              normalize=True) 
    ax_3d.plot(c_pos[:,0], 
            c_pos[:,1], 
            c_pos[:,2],
            marker='.',
            color='g')

def plot_feature_points(ax_3d, feature_pts, transformation_matrix):
    """
    Plot feature points in world coordinates.

    :param ax_3d: 3D axis to plot onto
    :type ax_3d: matplotlib.axes._subplots.Axes3DSubplot
    :param feature_pts: array of N points in image coordinates
    :type feature_pts: np.ndarray (N x 2)
    :param transformation_matrix: Matrix to map points to world coordinates
    :type transformation_matrix: np.ndarray (3 x 3)
    """
    # TODO?
    pass

def plot_3d_points(ax_3d, pts):
    """
    Plot points in world coordinates.

    :param ax_3d: 3D axis to plot onto
    :type ax_3d: matplotlib.axes._subplots.Axes3DSubplot
    :param pts: array of N points in world coordinates
    :type pts: np.ndarray (N x 3)
    """
    ax_3d.plot(pts[:,0], pts[:,1], pts[:,2])

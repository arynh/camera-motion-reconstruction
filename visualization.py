"""Helper functions to plot visualizations.

Note some of the functions assume that a 3D plot was created 
beforehand so that the axis can be passed in as a parameter. 
Here are the lines to create a 3D plot with matplotlib:
    fig = plt.figure()
    ax = plt.axes(projection='3d')

Includes the following visualization tools: 
    * Plot camera motion on 3D plot
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import cv2


def draw_epipolar_lines(img1, img2, pts1, pts2, F):
    """
    Compute and draw epipolar lines onto two images, associated
    by fundamntal matrix F.

    Source: OpenCV

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

    def drawlines(img1, img2, lines, pts1, pts2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        r, c = img1.shape
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR).astype(np.float32) / 256.0
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR).astype(np.float32) / 256.0

        for r, pt1, pt2 in zip(lines, pts1, pts2):

            color = tuple(np.random.randint(0, 255, 3).tolist())

            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1.astype(np.int)), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2.astype(np.int)), 5, color, -1)
        return img1, img2

    linesLeft = cv2.computeCorrespondEpilines(pts2, 2, F)
    linesLeft = linesLeft.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, linesLeft, pts1, pts2)

    linesRight = cv2.computeCorrespondEpilines(pts1, 1, F)
    linesRight = linesRight.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, linesRight, pts2, pts1)

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()


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
    ax_3d.set_xlim3d(-5, 5)
    ax_3d.set_ylim3d(-5, 5)
    ax_3d.set_zlim3d(-1, 9)
    ax_3d.quiver(
        c_pos[:, 0],
        c_pos[:, 1],
        c_pos[:, 2],
        c_view[:, 0],
        c_view[:, 1],
        c_view[:, 2],
        length=1,
        normalize=True,
    )
    ax_3d.plot(c_pos[:, 0], c_pos[:, 1], c_pos[:, 2], marker=".", color="g")


def plot_feature_points(ax_3d, feature_pts, transformation_matrix):
    """
    Plot feature points in world coordinates.

    :param ax_3d: 3D axis to plot onto
    :type ax_3d: matplotlib.axes._subplots.Axes3DSubplot
    :param feature_pts: array of N points in image coordinates
    :type feature_pts: np.ndarray (N x 2)
    :param transformation_matrix: Matrix to map points to world coordinates
    :type transformation_matrix: np.ndarray (3 x 4)
    """
    pts = np.array([np.append(pt, 1) @ transformation_matrix for pt in feature_pts])
    ax_3d.scatter(pts[:, 0] / pts[:, 3], pts[:, 1] / pts[:, 3], pts[:, 2] / pts[:, 3])


def plot_3d_points(ax_3d, pts, projection_mat=None):
    """
    Plot points in world coordinates.

    :param ax_3d: 3D axis to plot onto
    :type ax_3d: matplotlib.axes._subplots.Axes3DSubplot
    :param pts: array of N points in world coordinates
    :type pts: np.ndarray (N x 3)
    """
    ax_3d.scatter(pts[:, 0], pts[:, 1], pts[:, 2])


if __name__ == "__main__":
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from features import find_point_correspondences
    from essential import calculate_essential_matrix
    from reconstruction import estimate_camera_pose
    from visualization import plot_camera_motion

    def vector_to_string(v):
        return "({:.3f}, {:.3f}, {:.3f})".format(v[0] / v[3], v[1] / v[3], v[2] / v[3])

    iphone_12_mini_k = np.array(
        [
            [1.3270361480372305e3, 0, 9.6142138175295599e2],
            [0, 1.3325859916429802e3, 5.3765189758345116e2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    iphone_12_mini_k = np.array(
        [
            [2.9811618446106363e3, 0.0, 9.8161990924832332e2],
            [0.0, 9.9145760093363208e2, 5.4951893357740482e2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    camera_positions = np.array([[0, 0, 0, 1]], dtype=np.float64)
    camera_directions = np.array([[0, 0, 1, 1]], dtype=np.float64)

    # print camera position and direction for .csv
    print(f"0.0,0.0,0.0", end=",")
    print(f"0.0,0.0,1.0")

    filename = "tiles1"
    n_frames = 77
    step_size = 5
    for frame_index in range(1, n_frames - step_size, step_size):
        frame_one = cv2.imread(
            "samples/{fn}/f{num}.jpg".format(fn=filename, num=str(frame_index).zfill(4))
        )
        frame_two = cv2.imread(
            "samples/{fn}/f{num}.jpg".format(
                fn=filename, num=str(frame_index + step_size).zfill(4)
            )
        )
        # print(frame_index, 'to', frame_index+step_size)

        matched_points = find_point_correspondences(frame_one, frame_two)

        height, width, _ = frame_one.shape
        m = height if height > width else width

        E = calculate_essential_matrix(
            np.copy(matched_points), iphone_12_mini_k, m, 0.001
        )
        # draw_epipolar_lines(
        #     frame_one, frame_two, matched_points[0, :100], matched_points[1, :100], F
        # )

        rotation, translation = estimate_camera_pose(
            E, matched_points, iphone_12_mini_k
        )

        # camera_matrix = np.column_stack((rotation[:3,:3].T, -rotation[:3,:3].T @ translation[:3,3]))

        new_position = -translation @ camera_positions[-1]
        # new_position = -rotation.T @ translation[:,3] + rotation.T @ camera_positions[-1]
        camera_positions = np.vstack((camera_positions, new_position))

        # new_direction = rotation @ camera_directions[-1]
        new_direction = (
            rotation.T @ np.array([0, 0, 1, 1])
            - rotation.T @ translation[:, 3]
            - new_position
        )
        camera_directions = np.vstack((camera_directions, new_direction))

        # print camera position and direction for .csv
        print(f"{new_position[0]},{new_position[1]},{new_position[2]}", end=",")
        print(f"{new_direction[0]},{new_direction[1]},{new_direction[2]}")

        # print(
        #     f"frame {frame_index}/{n_frames}; "
        #     + f"position: {vector_to_string(new_position)}, "
        #     + f"direction: {vector_to_string(new_direction)}"
        # )

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    plot_camera_motion(ax, camera_positions[:, :3], camera_directions[:, :3])
    plt.show()

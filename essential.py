"""
Calculate the Essential Matrix between two frames using the instrinic matrix
and a list of the corresponding points between the two frames.
"""
import numpy as np
from scipy.linalg import svd

k = [[1.3270361480372305e3, 0, 9.6142138175295599e2],
    [0, 1.3325859916429802e3, 5.3765189758345116e2],
    [0, 0, 1],]


def calculate_essential_matrix(cp, k):
    """
    Calculate the fundemental matrix and then use the intrinsic matrix to
    calculate the essential matrix.
    :param cp: Corresponding set of features in two frames
    :type cp: np.ndarray (2 x number of features x 2)
    :param k: Second frame
    :type k: np.ndarray (3x3)
    :return E: essential matrix
    :rtype: np.ndarray (2 x number of features x 2)
    """
    N = len(cp) #get the number of features
    A = np.zeros((N, 9), dtype=float)  # create an empty A matrix
    for i in range(N):
        x = cp[0, i, 0]
        y = cp[0, i, 1]
        xp = cp[1, i, 0]
        yp = cp[1, i, 1]
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
    U, S, V = svd(A)
    F = V[-1, :]  # compute fundemental matrix
    E = np.transpose(k) @ F @ k  # compute the essential matrix
    return E

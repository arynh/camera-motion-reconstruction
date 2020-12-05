import cv2
import matplotlib.pyplot as plt
from features import find_point_correspondences
from essential import calculate_essential_matrix

frame_one = cv2.imread("samples/tiles0/f0001.jpg")
frame_two = cv2.imread("samples/tiles0/f0025.jpg")

matched_points = find_point_correspondences(frame_one, frame_two)

iphone_12_mini_k = [
    [1.3270361480372305e3, 0, 9.6142138175295599e2],
    [0, 1.3325859916429802e3, 5.3765189758345116e2],
    [0, 0, 1],
]

E = calculate_essential_matrix(matched_points, iphone_12_mini_k, 0.1)

print(E)

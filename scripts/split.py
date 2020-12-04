import os
import cv2


def video_to_frames(input_file, output_path):
    """
    Extracts the frames from an input video file
    and saves them as separate frames in an output directory.

    Source: CS 445 Project 5 Utils

    Input:
        input_file: Input video file.
        output_path: Output directorys.
    Output:
        None
    """

    cap = cv2.VideoCapture()
    cap.open(input_file)

    if not cap.isOpened():
        print("Failed to open input video")

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_idx = 0
    while frame_idx < frame_count:
        ret, frame = cap.read()

        if not ret:
            print("Failed to get the frame {}".format(frame_idx))
            return

        out_name = os.path.join(output_path, "f{:04d}.jpg".format(frame_idx + 1))
        ret = cv2.imwrite(out_name, frame)

        if not ret:
            print("Failed to write the frame {}".format(frame_idx))
            return

        frame_idx += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


if __name__ == "__main__":
    from sys import argv

    input_video = argv[1]
    name = input_video.split(".")[0]  # {name}.video
    os.mkdir(name)
    video_to_frames(input_video, name)
    print("Done!")

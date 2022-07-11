import os.path
import numpy as np


DATA_PATH = os.path.join("pose_gesture_data")
actions = np.array(["wave_right", "wave_left", "fist_right", "fist_left"])
# thirty videos worth of data
video_num = 30
# videos are going to be 30 frames in length
video_len = 30

if __name__ == '__main__':
    for action in actions:
        for video in range(video_num):
            os.makedirs(os.path.join(DATA_PATH, action, str(video)))

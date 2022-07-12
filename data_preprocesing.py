from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from folder_setup import actions, video_num, video_len, DATA_PATH

label_map = {label: num for num, label in enumerate(actions)}

videos, labels = [], []
for action in actions:
    for video in range(video_num):
        frame_sequence = []
        for frame_num in range(video_len):
            lm_in_frame = np.load(os.path.join(DATA_PATH, action, str(video), f"{frame_num}.npy"))
            frame_sequence.append(lm_in_frame)
        videos.append(frame_sequence)
        labels.append(label_map[action])

videos = np.array(videos)
labels = to_categorical(labels).astype(int)

videos_train, videos_test, labels_train, labels_test = train_test_split(videos, labels, test_size=0.1)
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from videos_setup import video_num, video_len, DATA_PATH, actions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from videos_setup import actions
import os

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

videos_train, videos_test, labels_train, labels_test = train_test_split(videos, labels, test_size=0.05)

log_dir = os.path.join('Logs')
tensorboard_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=videos.shape[1:]))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(videos_train, labels_train, epochs=2000, callbacks=[tensorboard_callback])
model.save('pose_gesture_model')
# can load the weighted model with model.load_weights('pose_gesture_model)

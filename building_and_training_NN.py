from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from folder_setup import actions
from data_preprocesing import videos, videos_train, labels_train
import os

log_dir = os.path.join('Logs')
tensorboard_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=videos.shape[1:]))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

if __name__ == '__main__':
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(videos_train, labels_train, epochs=2000, callbacks=[tensorboard_callback])
    model.save('pose_gesture_model')
    # can load the weighted model with model.load_weights('pose_gesture_model)

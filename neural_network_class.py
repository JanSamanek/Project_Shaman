import os.path
import numpy as np
import cv2
import holistic_detector as hd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import os
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model


class DataCollector(object):

    def __init__(self):
        pass

class NeuralNetworkDetector(object):

    def __init__(self, *actions, folder_name="neural_network_data", video_num=30, num_of_frames=30):

        self.DATA_PATH = os.path.join(folder_name)
        self.actions = np.array(actions)
        self.video_num = video_num
        self.num_of_frames = num_of_frames

        self.label_map = {label: num for num, label in enumerate(actions)}

        self.videos = []
        self.labels = []

    def create_folders(self):

        for action in self.actions:
            for video in range(self.video_num):
                os.makedirs(os.path.join(self.DATA_PATH, action, str(video)))

    def gather_data(self):

        cap = cv2.VideoCapture(0)
        detector = hd.Detector()

        for action in self.actions:
            # loop through videos
            for video in range(self.video_num):
                # loop through video_length
                for frame_num in range(self.num_of_frames):

                    # reading the image from video capture
                    _, img = cap.read()

                    img = detector.init_landmarks(img)

                    # collection logic
                    if frame_num == 0:
                        cv2.putText(img, 'Starting collecting', (180, 250), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(img, f'Collecting frames for {action}. Video number {video}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow("", img)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(img, f'Collecting frames for {action}. Video number {video}', (15, 12), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.imshow("", img)

                    lm_gesture = detector.get_landmarks()
                    path = os.path.join(self.DATA_PATH, action, str(video), str(frame_num))
                    np.save(path, lm_gesture)

                    # breaks out of the loop if q is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()

    def preprocess_data(self):

        for action in self.actions:
            for video in range(self.video_num):
                frame_sequence = []
                for frame_num in range(self.num_of_frames):
                    lm_in_frame = np.load(os.path.join(self.DATA_PATH, action, str(video), f"{frame_num}.npy"))
                    frame_sequence.append(lm_in_frame)
                self.videos.append(frame_sequence)
                self.labels.append(self.label_map[action])

        self.videos = np.array(self.videos)
        self.labels = to_categorical(self.labels).astype(int)

        self.videos_train, self.videos_test, self.labels_train, self.labels_test = \
            train_test_split(self.videos, self.labels, test_size=0.05)

    def create_neural_network(self):

        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=self.videos.shape[1:]))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))

    def train_neural_network(self):

        log_dir = os.path.join('Logs')
        tensorboard_callback = TensorBoard(log_dir=log_dir)

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model.fit(self.videos_train, self.labels_train, epochs=150, callbacks=[tensorboard_callback])
        self.model.save('pose_gesture_model')

    def real_time_detection(self):

        cap = cv2.VideoCapture(0)
        detector = hd.Detector()

        sequence = []

        while cap.isOpened():
            # reading the image from video capture
            _, img = cap.read()
            img = detector.init_landmarks(img)

            key_points = detector.get_landmarks()
            # prediction logic, waits till 30 frames worth of key points are stacked
            sequence.append(key_points)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = self.model.predict(np.expand_dims(sequence, axis=0))[0]
                print(self.actions[np.argmax(res)])

            cv2.imshow("Vision", img)

            # breaks out of the loop if q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def evaluate_neural_network(self):
        prediction = self.model.predict(self.videos_test)
        prediction_true = np.argmax(self.labels_test, axis=1).tolist()
        prediction = np.argmax(prediction, axis=1).tolist()
        print(multilabel_confusion_matrix(prediction_true, prediction))
        print(accuracy_score(prediction_true, prediction))


def main():
    pass


if __name__ == '__main__':

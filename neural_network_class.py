import os.path
import numpy as np
import cv2
import holistic_detector as hd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model


def gather_data(data_path, *actions, video_num=30, num_of_frames=30):

    actions = np.array(actions)
    data_path = os.path.join(data_path)
    print(f'Creating folder: {data_path}')
    for action in actions:
        for video in range(video_num):
            os.makedirs(os.path.join(data_path, action, str(video)))

    cap = cv2.VideoCapture(0)
    detector = hd.HolisticDetector()

    for action in actions:
        # loop through videos
        for video in range(video_num):
            # loop through video_length
            for frame_num in range(num_of_frames):

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
                path = os.path.join(data_path, action, str(video), str(frame_num))
                np.save(path, lm_gesture)

                # breaks out of the loop if q is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()


class NeuralNetwork(object):

    def __init__(self, *actions):

        self.actions = np.array(actions)
        self.label_map = {label: num for num, label in enumerate(self.actions)}
        self.videos = []
        self.labels = []

        self.model = None
        self.sequence = []

    def prepare_data(self, data_path, video_num=30, num_of_frames=30):

        data_path = os.path.join(data_path)
        print('Preprocessing data...')
        for action in self.actions:
            for video in range(video_num):
                frame_sequence = []
                for frame_num in range(num_of_frames):
                    lm_in_frame = np.load(os.path.join(data_path, action, str(video), f"{frame_num}.npy"))
                    frame_sequence.append(lm_in_frame)
                self.videos.append(frame_sequence)
                self.labels.append(self.label_map[action])

        self.videos = np.array(self.videos)
        self.labels = to_categorical(self.labels).astype(int)

        videos_train, videos_test, labels_train, labels_test = train_test_split(self.videos, self.labels, test_size=0.05)

        return videos_train, labels_train, videos_test, labels_test

    def create_and_train(self, videos_train, labels_train):

        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=self.videos.shape[1:]))
        self.model.add(LSTM(128, return_sequences=True, activation='relu'))
        self.model.add(LSTM(64, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))

        log_dir = os.path.join('Logs')
        tensorboard_callback = TensorBoard(log_dir=log_dir)

        self.model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model.fit(videos_train, labels_train, epochs=100, callbacks=[tensorboard_callback])

    def save(self, folder_name):
        print('Saving neural network model...')
        self.model.save(folder_name)

    def load(self, data_path):
        self.model = load_model(data_path)

    def evaluate(self, videos_test, labels_test):

        prediction = self.model.predict(videos_test)
        prediction_true = np.argmax(labels_test, axis=1).tolist()
        prediction = np.argmax(prediction, axis=1).tolist()

        print("Model's multilabel_confusion_matrix: ")
        print(multilabel_confusion_matrix(prediction_true, prediction))
        print("Model's accuracy_score: ", accuracy_score(prediction_true, prediction))

    def detect_gesture(self, landmarks):

        self.sequence.append(landmarks)
        self.sequence = self.sequence[-30:]

        # prediction logic, waits till 30 frames worth of key points are stacked
        if len(self.sequence) == 30:
            res = self.model.predict(np.expand_dims(self.sequence, axis=0))[0]
            print(self.actions[np.argmax(res)])
            print(res[np.argmax(res)])
            return res[np.argmax(res)]


def main():

    with open('gesture_in_order.txt') as file:
        actions = file.read().split('\n')

    nn = NeuralNetwork(*actions)
    nn.load('gesture_model')
    videos_train, labels_train, videos_test, labels_test = nn.prepare_data('holistic_gesture_data')
    nn.evaluate(videos_test, labels_test)

    cap = cv2.VideoCapture(0)
    detector = hd.HolisticDetector()

    previous_time = 0

    while cap.isOpened():
        # reading the image from video capture
        _, img = cap.read()
        img = detector.init_landmarks(img)

        key_points = detector.get_landmarks()
        nn.detect_gesture(key_points)

        previous_time = hd.display_fps(img, previous_time)
        cv2.imshow("Vision", img)

        # breaks out of the loop if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # with open('gesture_in_order.txt') as file:
    #     actions = file.read().split('\n')
    #
    # nn = NeuralNetwork(*actions)
    # videos_train, labels_train, videos_test, labels_test = nn.prepare_data('holistic_gesture_data')
    # nn.create_and_train(videos_train, labels_train)
    # nn.save('gesture_model')
    # nn.evaluate(videos_test, labels_test)

    main()


